# racecontrol

**The two-tier router that makes [trailbrake](https://github.com/PhilipJohnBasile/trailbrake)
and iliria one usable product.**

trailbrake (dense Qwen3-32B-4bit, ~15-25 tok/s, being pruned
into a fast NO-THINK coding specialist) is the fast default. iliria
(GLM-5.2, 744B MoE, ~1.6 tok/s streamed, in `iliria/c`) is the
deep-reasoning escalation for the hard minority of requests. Both already
speak an OpenAI-compatible HTTP API. This router is a thin, dependency-free
proxy in front of both: one endpoint, a pluggable escalation policy that
decides which tier handles each request, canary/instant-rollback for trailbrake
itself, cross-tier fallback, and a durable decision log.

See **[docs/DESIGN.md](docs/DESIGN.md)** for the full architecture, the
escalation policy (and why it's the shipped default), and the
canary/rollback/fallback/telemetry guardrails. This README is quick-start
only.

## Why this repo, not a subdirectory of trailbrake or iliria

This is a new, standalone, MIT-licensed repo rather than
`trailbrake/router/` (the other suggested location) or a
directory inside `iliria/c`. Reasons, in short (see docs/DESIGN.md's
"Why a new repo" for the full argument):

1. The router imports neither engine's source -- it only speaks HTTP to
   both -- so there is no technical reason to couple its dependency
   footprint or release cadence to either.
2. `trailbrake`'s own README states a deliberately narrow engine
   contract ("spends complexity budget on throughput... "); a multi-backend
   proxy with its own policy/telemetry/config surface does not fit that
   scope, any more than the engine's `mlx`/`tokenizers`/`jinja2` dependencies
   belong in a thin proxy.
3. `trailbrake` was itself deliberately split out of `iliria` for
   the same reason -- a small, independently-versioned contract instead
   of another corner of the research tree. The router, as the
   product's front door, deserves the same narrow, independently-versioned
   treatment, not another corner of a research monorepo.
4. `iliria/c` is a C-engine directory (`glm.c`, `backend_metal.mm`, one
   Python gateway file); a second, unrelated Python HTTP service doesn't
   belong there either.

## Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

## Configure

Copy `config/router.example.toml`, point `base_url` at wherever `trailbrake
serve` and iliria's `run-m5max-serve.sh` are actually listening, and adjust
weights/policy as needed:

```bash
cp config/router.example.toml config/router.toml
racecontrol check-config --config config/router.toml
```

## Run

```bash
racecontrol serve --config config/router.toml
```

The router listens on loopback by default (`127.0.0.1:8100`) and speaks the
same `/v1/chat/completions` / `/v1/completions` / `/v1/models` surface as
both backends, plus `GET /health` and `GET /router/status` for its own
canary/circuit-breaker/decision-count state.

**This repo intentionally never talks to a real model or a GPU.** Every test
here mocks or scripts both backends. Wiring this router up to a live
`trailbrake serve` and a live `ili serve` process is the GPU-gated
integration step described in docs/DESIGN.md's last section -- it has not
been run as part of this change.

## Test

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

## Layout

```
src/router/
  config.py     -- TOML config -> dataclasses (backends, tiers, escalation, fallback)
  policy.py      -- escalation policy: which tier handles a request, and why
  backends.py    -- HTTP client per backend (model-id rewrite, timeouts, canary weighting)
  circuit.py     -- per-backend circuit breaker (automatic, unattended rollback)
  telemetry.py   -- JSONL decision log + live counters
  dispatch.py    -- ties the above into one decide-call-fallback-log pipeline
  server.py      -- the OpenAI-compatible HTTP proxy itself
  cli.py         -- `racecontrol serve|check-config`
docs/DESIGN.md   -- architecture, escalation policy, guardrails
config/router.example.toml
tests/           -- unit tests (fakes/mocks only, no sockets) + a handful of
                     real-socket end-to-end tests against tiny scripted fake
                     backend servers (still no GPU, no real model)
```
