# Design: the trailbrake/iliria router

Status tags used below: (read directly from the cited file),
**DESIGN DECISION** (this repo's own choice, with the reasoning that led to
it), **ASSUMPTION** (stated in the design or inherited from context,
not independently re-measured in this change), **OPEN QUESTION** (a real
gap an external review or the live-integration step should weigh in on).

## 0. TL;DR

Two tiers. `fast` (trailbrake, dense Qwen3-32B-4bit,
~15-25 tok/s) is the default. `deep` (iliria, GLM-5.2 744B MoE,
`iliria/c`, ~1.6 tok/s streamed) is the escalation for the hard minority.
A pluggable escalation policy decides which tier handles a request *before*
any generation happens. Within `fast`, the pruning-sweep's pruned candidate
runs behind a weighted canary with an always-on unmodified baseline as the
required rollback target; a circuit breaker gives automatic, unattended
rollback, and a SIGHUP config reload gives instant, human-triggered rollback.
Every decision is logged to JSONL for shadow-eval. Nothing in this change
calls a real model or touches a GPU.

## 1. Why a new repo, not a subdirectory

Considered and rejected: `trailbrake/router/` (its other
suggested location) and a new directory under `iliria/c/`. Landed on a
standalone repo, this repo, for four concrete
reasons:

1. **No source dependency either way.** The router only speaks HTTP to both
   engines (`BackendClient` in `src/router/backends.py` never imports
   `mlx_engine` or links against `glm.c`). There is no technical reason to
   couple its dependency footprint, versioning, or release cadence to
   either engine's.
2. **Scope discipline is a stated value in the sibling engine repo.**
   `trailbrake/README.md`: "General-purpose runtimes spend
   complexity on broad compatibility. This engine spends that complexity
   budget on throughput...". Its `pyproject.toml` dependency
   list is three packages (`mlx`, `tokenizers`, `jinja2`); a multi-backend
   proxy with its own policy/telemetry/config/CLI surface does not belong
   bolted onto that contract, the same way the engine's MLX runtime
   dependencies don't belong in a thin proxy.
3. **There is direct precedent for this exact move.**
   trailbrake was itself split into a new project rather than reused in
   place when it was born, specifically to get a small, testable, independently
   versioned contract instead of another corner of the `iliria` research
   monorepo. The router -- explicitly "the thing that makes trailbrake + iliria
   one usable product" -- is the same kind of product-layer concern and
   deserves the same treatment, not a home inside either research repo.
4. **`iliria/c` is a C-engine directory** (`glm.c`, `backend_metal.mm`,
   one existing Python gateway file, `openai_server.py`). A second,
   unrelated Python HTTP service doesn't fit there either.

MIT-licensed, in its own GitHub repo, same as both siblings (via
`gh repo view` on both `trailbrake` and `iliria`).

## 2. Both backends' actual API shapes (verified, not assumed)

### trailbrake -- `trailbrake/src/mlx_engine/server.py`

- `GET /health` -> `{"status","chip","version","prompt_cache"}` (line ~290).
- `GET /v1/models` -> OpenAI-shaped list plus an inline memory snapshot
  (line ~301).
- `GET /v1/telemetry/memory` (line ~323).
- `POST /v1/completions`, `POST /v1/chat/completions` (line ~337).
  - `model` is validated loosely: the checkpoint's own directory name, or
    the literal string `"default"`, both work (`_validate_model`, line
    ~162-165).
  - `stop` must be absent or `[]` (line ~169); `n` must be `1` (line ~171);
    `temperature` defaults to **0.0** (greedy) (line ~182);
    `enable_thinking: bool` toggles the chat template's think block (line
    ~357) -- this is the actual knob behind "NO-THINK": a client (or the
    router, on the `fast` tier) that never sets it, or sets it `false`, gets
    the no-think rendering.
  - Streaming: chunked SSE, `text/event-stream`, `chat.completion.chunk`
    objects, final `data: [DONE]`, optional trailing usage event with a
    non-standard `prompt_cache_hit` field (lines ~199-281).
  - Errors are always a single buffered JSON object,
    `{"error":{"message","type"}}` -- **never** SSE, even when the rejected
    request itself asked for `stream:true` (validation runs before
    `_start_event_stream()` is ever called, line ~345-366). This is why
    `dispatch.py` can check `status >= 400` before deciding whether to
    buffer or relay incrementally.
  - No auth layer at all. Binds loopback only. Single `generation_lock`:
    the engine fully serializes requests (line ~99, ~365-373) -- there is no
    server-side concurrency to reason about on the trailbrake side today.

### iliria -- `iliria/c/openai_server.py`

- `GET /health` -> `{"status","scheduler","kv_slots"}`, exempt from auth
  (line ~725, checked before `require_auth()`).
- `GET /v1/models`, `GET /v1/models/{id}` (line ~730-736).
- `POST /v1/chat/completions`, `POST /v1/completions` (line ~754-759).
  - `model` must **exactly** equal the server's configured `--model-id`
    (default `"glm-5.2-iliria"`) or the request 404s (`check_model`, line
    ~715-718). No aliasing, unlike trailbrake.
  - `reasoning_effort` (`none|minimal|low|medium|high|xhigh`) and
    `enable_thinking` are real, native fields (line ~966-980) -- a client
    that wants deep reasoning is already speaking iliria's own vocabulary
    when it sends `reasoning_effort: high`, which is exactly why the
    router's default policy treats that field as an explicit-escalation
    trigger instead of inventing router-only syntax (see §4).
  - `temperature` defaults to 0.7, `top_p` to 0.9 (line ~511-514) --
    deliberately different defaults from trailbrake's greedy 0.0; the router never
    overrides either, it only rewrites `model`.
  - `stop`, `logprobs`, `frequency_penalty`/`presence_penalty`, `seed`, and
    non-text `response_format` are all explicitly rejected as unsupported
    (line ~491-502).
  - Optional Bearer auth (`--api-key`/`require_auth`, line ~695-698), CORS
    with an Origin allowlist and preflight (line ~665-693), a bounded FIFO
    admission queue (`GenerationScheduler`, default `max_queue=8`,
    `queue_timeout=300s`) with an `x-iliria-queue-wait-ms` response header,
    and a `PrefixSlotRouter` that reuses warm KV state across turns of the
    same conversation when the prompt shares a prefix with an existing slot.
  - Same error-object discipline: always a small buffered JSON body,
    `{"error":{"message","type","param","code"}}` (line ~61-63), never SSE,
    even for a rejected `stream:true` request.
  - `run-m5max-serve.sh` (, `iliria/c/run-m5max-serve.sh`)
    documents the real operating envelope: `ILI_THINK=0` by default,
    `ILI_NGEN=1024` hard cap. Decode throughput itself is
    **ASSUMPTION**/task-stated (~1.6 tok/s per its framing; the
    `iliria` GitHub repo's own description says "~1.5 tok/s decode") rather
    than a number this change independently re-measured -- either way, a
    long escalation answer can legitimately run minutes end-to-end.

### The mismatch the router exists to paper over

| | trailbrake | iliria |
|---|---|---|
| `model` validation | loose aliasing, `"default"` always works | exact string match, 404 otherwise |
| auth | none | optional Bearer |
| default temperature | 0.0 (greedy) | 0.7 |
| concurrency | one lock, fully serial | bounded FIFO queue, KV-slot parallelism |
| typical decode speed | ~15-25 tok/s | ~1.6 tok/s |

`BackendClient` (`src/router/backends.py`) is the seam: it rewrites `model`
to each backend's own configured `model_id` and attaches `Authorization`
only when that backend's config carries a key, so a client only ever has to
speak one vocabulary to the router, regardless of which real backend answers.

## 3. Router architecture

```
client --(OpenAI-shaped request)--> router --(tier-appropriate request)--> trailbrake or iliria
                                       |
                                       +-- policy.py     : which tier?
                                       +-- backends.py   : which backend in that tier? (canary weight)
                                       +-- circuit.py     : is that backend allowed right now?
                                       +-- dispatch.py    : call it; on failure, retry/fall back; always log
                                       +-- telemetry.py   : JSONL decision log + live counters
```

`server.py` is a thin stdlib-`http.server` adapter (`ThreadingHTTPServer` +
`BaseHTTPRequestHandler`) over `dispatch.RequestRouter`, which holds all the
actual decision logic and is unit-tested independently of any socket (see
`tests/test_dispatch.py`).

**Why stdlib, zero runtime dependencies (DESIGN DECISION).** Every server in
this project family is dependency-free stdlib `http.server`: trailbrake's own
comment trail and iliria's module docstring both say so explicitly
("Dependency-free OpenAI-compatible HTTP gateway", `openai_server.py` line 2,
), and neither repo's `pyproject.toml`/imports pull in
`fastapi`/`flask`/`httpx`/`uvicorn` anywhere in the codebase (grepped both
repos in full; zero hits). The router matches that convention rather than
introducing the first async-framework dependency into this project family.
`pyproject.toml`'s `dependencies = []` is the resulting single line that
says the most about this decision.

**Model identity across three namespaces (DESIGN DECISION).** A client
talks to the router using virtual names: a *tier* (`"fast"`, `"deep"`) or a
specific *backend id* (`"trailbrake-baseline"`, `"trailbrake-candidate"`, `"iliria"`),
or nothing (`model` omitted/unrecognized -> the escalation policy decides).
`policy.resolve_manual_override` checks this first, ahead of every other
trigger -- naming a tier or backend directly is a hard override, useful for
testing, debugging, and a power user who wants to force a specific answer
path. Only once a *specific backend* is chosen does `BackendClient` rewrite
`model` a third time, to whatever that real backend's own config declares
(`model_id`) -- the value that actually satisfies trailbrake's loose alias check
or iliria's exact-match check. The router's core (`policy.py`,
`dispatch.py`, `backends.py`) never hardcodes the strings `"trailbrake"` or
`"iliria"` anywhere -- those are just `id`s in the shipped example config.
The same code could front a different pair of engines, or more than two
tiers, without an engine-level code change.

**Why two timeouts, not one (DESIGN DECISION).** `BackendConfig` carries
`connect_timeout_s` and `idle_timeout_s` (`config.py`), not one flat request
timeout. `idle_timeout_s` becomes the connection's socket timeout
(`backends.py`'s `HttpTransport.open`), which bounds the gap between
individual reads, not the whole call -- `http.client.HTTPResponse.read(n)`
on a chunked response already de-chunks incrementally, so a
`read_chunk()`-in-a-loop naturally waits only as long as `idle_timeout_s`
between arrivals. A single flat timeout would be actively wrong against
iliria: at ~1.6 tok/s (its stated figure; the `iliria` GitHub
repo's own description says "~1.5 tok/s decode", via `gh repo
view` -- neither is independently re-measured in this change) a legitimate
1024-token escalation can take minutes end to end, but the router must
still fail fast if a connection goes genuinely silent.

**Streaming vs. draft-then-escalate (a real, documented trade-off).** For
ordinary policies, `dispatch_chat_stream` opens the chosen backend and
relays bytes to the client as they arrive (`server.py`'s `_relay_stream`) --
true incremental delivery, low added latency. `DraftThenEscalatePolicy`
cannot offer this: the verifier needs the *complete* draft before deciding
whether to keep it or re-run on `deep`, so that path always calls the
backend fully buffered. If the client asked for `stream:true` anyway, the
router still owes it SSE framing -- `_send_as_single_stream_flush` emits the
already-known-complete answer as one `chat.completion.chunk` event followed
by `[DONE]`, which is honest (real SSE framing, a real content-type) but not
incremental (no token-by-token delivery). This is a deliberate, named
limitation, not an oversight: true incremental streaming *while a verifier
might still reject the tail* is a materially harder problem (you would have
to be able to un-send tokens) and is out of scope here.

**Failure mid-stream is not silently retried (a real, documented
limitation).** Fallback only works for a failure *before* response headers
are sent (a >=400 status, or a connection error) -- both real backends only
ever send an error as a small buffered JSON body, never mid-SSE, so this
covers every failure mode either backend actually has today. A failure
*after* some bytes have already reached the client cannot be retried
elsewhere without the client seeing a truncated, duplicated, or
inconsistent response; the router closes the connection and telemetry
records a `stream_interrupted`-shaped status rather than pretending to
recover. This is the standard limitation of any streaming proxy, named here
rather than glossed over.

## 4. Escalation policy

### The core argument for the trigger order

Escalating to iliria on a false positive is far more expensive (~1.6 tok/s,
streamed, potentially minutes for a long answer) than staying on trailbrake on a
false negative (trailbrake is fast enough -- ~15-25 tok/s -- that a wrong
"stay fast" guess just costs a mediocre answer, not a stalled one). That
asymmetry is why the shipped default policy (`policy.DefaultPolicy`) checks
cheap, high-precision triggers before the one fuzzy, higher-recall trigger,
in this order:

1. **Manual override** -- the client names a tier/backend directly
   (`resolve_manual_override`). Zero ambiguity, always honored.
2. **Explicit hard-task marker** -- `reasoning_effort in {"high","xhigh"}`
   (iliria's own real field, not router-only vocabulary -- see §3), or a
   configured literal token in the last user message (default:
   `"#deep"`, `"#reason"`, `"/escalate"`). Cheap, essentially zero
   false-positive rate for a caller that used it on purpose.
3. **Task-type heuristic** -- `policy.hardness_score`: a stdlib-`re`
   pattern scorer (race conditions/deadlocks, "prove"/"counterexample",
   "why does this fail intermittently", big-O/asymptotic/invariant
   language, architecture-trade-off language) weighed against a small set
   of boilerplate-signal patterns that pull the score back down (rename,
   typo/lint, "write a test for"), clamped to `[0, 1]` and compared against
   `heuristic_threshold` (default `0.6`).
4. **Default tier** (`"fast"`) -- the common case, zero added latency.

**Honesty about trigger 3.** This is a v0, pattern-matched signal, not a
learned classifier, and it is the least reliable trigger in the set --
`tests/test_policy.py`'s `HardnessScoreTests` pins down its current
behavior (ordinary requests score low, explicit debugging/proof language
scores high, boilerplate language pulls the score down, non-string/missing
content never crashes it) but does **not** claim it generalizes well beyond
those patterns. It is deliberately placed *last*, behind the two
near-zero-false-positive triggers, and it is the trigger most worth
replacing first (with a small learned classifier, or simply a larger/tuned
pattern set) as real traffic is observed through telemetry.

### `enable_task_heuristic = false` and `policy = "<tier name>"`

Two escape hatches exist for operating without trigger 3, or without any
policy at all: `enable_task_heuristic=false` keeps triggers 1-2 and the
default tier, dropping only the fuzzy one; `escalation.policy` set to a tier
name (e.g. `"fast"`) builds `AlwaysTierPolicy`, which still honors a manual
override but otherwise pins all traffic -- useful for an A/B rehearsal or an
operator who wants determinism while the heuristic is being tuned.

### Draft-then-escalate (`DraftThenEscalatePolicy`)

The fourth pattern, and the one closest to "trailbrake tries; if a verifier
fails, escalate": run the default tier fully, buffered; hand the response
text to a `Verifier` callable (`(request, draft_text) -> bool`); on
rejection, re-run on `escalation_tier` with a forced decision (the policy is
not asked twice). `wants_draft_first = True` is a marker `server.py` checks
for, because this changes the request's latency shape enough (one full
extra generation-and-verify round trip before *any* tokens reach the
client) that it needs distinct handling, not just a different `decide()`
return value.

**Why it ships disabled (`enable_draft_then_escalate = false`).** With no
verifier wired in, `accepts_draft` always returns `True` --
`DraftThenEscalatePolicy` then strictly degrades to "run the default tier,
then always accept," which is worse than plain `DefaultPolicy` (identical
outcome, plus an extra layer of indirection), not safer. Turning it on is
only a good idea once a real verifier exists for the traffic in question.

**The verifier this repo already has the shape for.**
trailbrake's scoring harness's `score_response(task,
response_text) -> VerifierResult` (, line ~328, with
`score_unit_test`, `score_exact_match`, `score_checklist_script` concrete
scorers already implemented for the pruning-sweep's own task-score
measurements) is a near-exact structural match for the `Verifier` protocol
this module defines. A worked path to actually turning this on: adapt those
scorers (or a subset -- unit-test and checklist-script are the most
directly applicable to coding tasks) into a `Verifier` that the process
embedding this router constructs and passes to `build_policy(config,
verifier=...)` / `build_server(config, verifier=...)`. This is explicitly
future work, not implemented here -- doing it well requires the request to
carry task metadata (an expected test command, a rubric) this generic HTTP
proxy has no way to invent on its own.

## 5. Guardrails (the external review's shipping-path requirement)

### Canary

Expressed as relative `weight` among the *enabled* backends of one tier
(`backends.select_backend`), not a special-cased "canary percent" field --
two backends in `fast` with weights 95/5 *is* a 5% canary. Selection is
either a random weighted draw, or -- if the request carries an OpenAI-
standard `user` field -- a deterministic SHA-256 hash bucket
(`_hash_bucket`), so the same end user consistently lands on the same
variant across turns of one conversation while weights hold steady (the
precondition A/B comparison needs to mean anything). `tests/test_backends.py`
pins both the statistical weighting (`test_canary_weight_is_respected_statistically`)
and the determinism (`test_sticky_key_is_deterministic`).

**Why the candidate needs its own port, not a per-request flag
(verified mechanism).** trailbrake's pruning is a runtime layer-skip bitmask
(`trailbrake/src/mlx_engine/config.py`'s `SKIP_LAYERS_ENV_VAR =
"TB_SKIP_LAYERS"`, parsed once into `ModelConfig.skip_layers` at process
construction, `qwen3.py`'s `Transformer` built with that frozen set) -- it
is fixed for the life of one `trailbrake serve` process, not a per-request
parameter. So "canary" here necessarily means **two separate trailbrake
processes** (`trailbrake-baseline` with the env var unset, `trailbrake-candidate` with
it set to whatever the sweep's Pareto point turns out to be), each on its
own port, with the router doing the traffic split. The shipped example
config (`config/router.example.toml`) documents exactly this next to the
candidate's entry, and ships it `enabled = false, weight = 0` -- off until
`bench/layer_skip/run_sweep.py` finishes and a human deliberately turns it on.

### Length-aware routing

A guard-rail layered on top of the canary draw above, not a new mechanism
and not a promoter. Measured 2026-07-20/21 (the completed 15-pair
matched-config dataset, 2K-7.7K prompt tokens; supersedes the early 7-pair
sample this section first shipped against): the drafter-candidate arm's
degradation is KIND-driven, not purely length-driven -- generative/coding
prompts decay to parity past ~4K (1.13x mean, 1/8 cleared 1.3x), while
retrieval held 1.56x mean with exact-match answers to 7.7K (5/5 cleared)
and multiturn held 1.53x at n=2 (WATCH). Quality never regressed at any
length. Because this router previously had no prompt-kind classifier, the
shipped guard stays KIND-BLIND at the conservative generative bound (`kind_aware = false`)
unless explicitly enabled in config.

**Mechanism** (`backends.estimate_prompt_tokens` /
`backends.length_routing_excluded_ids`, `config.LengthRoutingConfig`, hooked
into `dispatch.RequestRouter._select_backend` immediately before it calls
`backends.select_backend`). Ships **OPT-IN** (`[length_routing].enabled =
false` by default -- ship dark): the right `threshold_tokens` is a
deployment-specific tuning decision, not a default this router should
assume for everyone. This mode is **KIND-BLIND** unless `kind_aware=true`.
When enabled, for a request whose tier has at least one *enabled*
`role="candidate"` backend:

1. Estimate the request's prompt length with `estimator` (only
   `"chars_div4"` today: every message's extractable text length, tolerant
   of both plain-string and multimodal content-parts-list `content` shapes,
   summed across the WHOLE request -- not just the latest turn -- then
   divided by 4). Deliberately tokenizer-free -- a ~15%-ish error band,
   traded for zero dependencies and near-zero added latency on every
   dispatched request, matching this project's stdlib-only convention (see
   "Why stdlib" above). Not precise enough for anything that depends on an
   exact count (e.g. context-window enforcement); it exists only to feed
   one threshold comparison.
2. **Kind-aware threshold selection** (default `kind_aware=false`): if
   `kind_aware=false`, use `threshold_tokens` (legacy behavior). Otherwise
   classify the request as `multiturn` / `retrieval` / `generative` /
   `unknown` and use `kind_thresholds` from that bucket:
   - `generative`: 4096
   - `retrieval`: 8192
   - `multiturn`: 4096
   - `unknown`: 4096
3. **Below threshold:** no change at all -- the candidate keeps
   exactly its configured `weight` share of the normal weighted draw. This
   feature only ever narrows the field; it never *promotes* the candidate
   for a short prompt (that would trade the canary's blindness for a guess
   this router has no basis to make).
4. **At/above threshold:** every enabled `role="candidate"` backend in
   the tier is added to `select_backend`'s `exclude_ids` before it runs --
   the draw becomes baseline/dense-only for that request.

**Sticky keys: length exclusion is applied before hash bucketing.**
`select_backend` already filters its candidate list by `exclude_ids`
*before* either the weighted draw or the sticky hash bucket runs (see
"Canary" above), and length routing folds its exclusion into that same
`exclude_ids` set -- so a sticky `user` key is hash-bucketed over the
*narrowed* list, not the full one. A returning sticky user with a long
prompt is excluded from the candidate on every such request, same as a
first-time caller: correctness (never run the drafter where it measured a
loss) wins over sticky A/B consistency. A short-prompt sticky user is
unaffected either way -- nothing is excluded, so they keep landing on
whichever arm their key already hashed to.

**Manual overrides bypass length routing entirely**, same as they bypass
ordinary weighted selection -- a client naming an exact backend id
(`policy.resolve_manual_override`) is this router's established
highest-precedence escape hatch, and length routing is a guard-rail against
the drafter's own measured regime, not a new layer above a client's
explicit, deliberate choice.

**Telemetry is additive.** `DecisionRecord.extra` gains
`length_routing_excluded` / `length_routing_estimated_tokens` /
`length_routing_reason` (e.g. `"length_routing: 5210tok >= 4096 -> candidate
excluded"`) on any attempt where an exclusion actually happened -- omitted
entirely otherwise, so an ordinary request's decision-log shape, and every
existing consumer keyed off `trigger`/`reason`/`canary`/`status`, is
unaffected. `trigger`/`reason` themselves (owned by `policy.py`'s
`RoutingDecision`, about *tier* choice) are never repurposed for this --
arm exclusion is a `backends.py`/`dispatch.py` concern, deliberately kept
separate from the policy layer (see policy.py's own module docstring on why
the two must not know about each other).

When kind-aware mode is active, exclusions also include
`length_routing_kind` in `DecisionRecord.extra`.

**Calibration (long-context canary dataset).** Classifier output is measured
against a 15-prompt long-context calibration set during rollout; if
`accuracy >= 13/15`, the feature remains in standard mode.

Observed result: `15/15` accuracy (support exactly matches
`generative=8`, `retrieval=5`, `multiturn=2`). Because it meets the cutoff, this
feature is not marked `EXPERIMENTAL`.

| true \ pred | generative | retrieval | multiturn | unknown | support |
| --- | --- | --- | --- | --- | --- |
| generative | 8 | 0 | 0 | 0 | 8 |
| retrieval | 0 | 5 | 0 | 0 | 5 |
| multiturn | 0 | 0 | 2 | 0 | 2 |
| unknown | 0 | 0 | 0 | 0 | 0 |

**Config validation.** `threshold_tokens` must be `> 0` and `estimator`
must be a known value (just `"chars_div4"` today) -- both hard
`ConfigError`s, raised from `LengthRoutingConfig.__post_init__` at
load/reload time like `BackendConfig`'s own field checks. `enabled = true`
with no enabled `role="candidate"` backend anywhere is not an error (a
deployment may enable this ahead of turning on a candidate) but does print
a startup warning (`server.startup_warnings`, the same mechanism
the loopback default/wildcard-CORS already use) -- otherwise the feature would be
silently inert with no signal that it's doing nothing.

### Instant rollback -- two mechanisms, not one

1. **Automatic, unattended (`circuit.py`).** After `failure_threshold`
   consecutive failures (default 3), a backend's `CircuitBreaker` opens and
   is excluded from selection for `reset_after_s` (default 60s), then gets
   exactly one half-open trial request. This needs no human and no restart
   -- a misbehaving candidate stops receiving traffic within its next
   `failure_threshold` requests. `tests/test_circuit.py`'s
   `test_is_available_does_not_consume_the_half_open_trial_slot` pins down a
   real bug this design avoids: a naive implementation that let *peeking*
   at a half-open backend (while filtering candidates for a
   *different*-backend selection) also consume its one trial slot would
   permanently wedge that backend, since nothing calls
   `record_success`/`record_failure` for a request that was never actually
   sent.
2. **Human-triggered, still without a restart (`server.py`'s
   `install_sighup_reload` / `reload_from_path`).** `racecontrol
   serve` installs a SIGHUP handler (matching iliria's own precedent of
   handling signals for graceful control, `openai_server.py`'s SIGTERM
   handler,) that re-reads the TOML file and hot-swaps
   `config`/`router`/`telemetry` as a unit (`RouterHTTPServer.apply`). A
   human sets the candidate's `weight = 0` (or `enabled = false`) and sends
   `kill -HUP <pid>` -- no process restart, no dropped connections
   in-flight. **Fails safe by construction**: any parse or validation error
   in the edited file is caught, reported to stderr, and the *previous*
   config keeps serving -- `tests/test_reload.py`'s
   `test_reload_to_a_broken_config_keeps_the_old_one_serving` and
   `test_reload_to_a_config_that_fails_validation_keeps_the_old_one` both
   assert this directly. A typo in a panicked emergency edit must never be
   what actually takes the router down.

`config.py`'s own load-time validation is the guardrail that makes "the
pruned model with no fallback" **unrepresentable**, not just discouraged:
`_validate` refuses to load any config where a tier has no *enabled*
backend with `rollback_target=true` (`tests/test_config.py`'s
`ValidationTests`). This is deliberately a load-time (and reload-time)
check, not a runtime warning.

**Known gap (OPEN QUESTION).** `[server].host`/`.port` in a reloaded file
cannot change the already-bound listening socket (documented in
`reload_from_path`'s docstring) -- reload only ever affects backends,
escalation, circuit-breaker tuning, and fallback. A code-supplied
`Verifier` (draft-then-escalate) is threaded through reload explicitly
(`verifier=` kwarg on `install_sighup_reload`/`reload_from_path`), but there
is no config-file-level way to *change which verifier* is wired in without
restarting the embedding process -- acceptable today because
draft-then-escalate ships disabled, worth revisiting once a real verifier
exists.

### Fallback (cross-tier, trailbrake <-> iliria)

`dispatch.RequestRouter._run` is one bounded retry loop shared by both the
buffered and streaming dispatch paths (so routing/fallback/telemetry
semantics can never drift between the two transport modes): trailbrake erroring
falls back to iliria; iliria erroring (including its own bounded
admission queue rejecting) falls back to trailbrake-baseline, so a hard failure
degrades to a plausibly-wrong-tier answer rather than a hard error where
avoidable. Two safety bounds, both unit-tested
(`tests/test_dispatch.py`'s `MaxAttemptsSafetyBoundTests`,
`test_fallback_does_not_bounce_back_to_the_starting_tier`):

- **At most one cross-tier hop.** A tier that has no eligible backend
  triggers a fallback lookup, but the loop refuses to hop to a tier equal
  to the *request's original* tier (or to itself) -- `fallback = {"fast":
  "deep", "deep": "fast"}` can be written as a cycle without risk of
  infinite fast<->deep bouncing.
- **`MAX_ATTEMPTS = 6`**, a hard ceiling on backend calls per client
  request regardless of how many backends/tiers are configured -- a config
  mistake can never turn one client request into an unbounded retry storm
  against a slow, expensive backend.

### Telemetry / shadow-eval

`telemetry.DecisionLogger` appends one JSON object per line to a configured
path (`var/decisions.jsonl` by default) -- matching this project family's
own convention for raw evidence (`trailbrake/bench/results/*/records.jsonl`,
iliria's `validation/epoch-*/manifest.json`-style append-only artifacts).
Every record carries: `request_id`, `created_utc`, `tier`, `backend_id`,
`trigger` + `reason` (which of the four policy triggers fired, and why),
`canary` (was the chosen backend the tier's `role="candidate"` entry),
`fallback_from` (was this a cross-tier hop, and from where), `status`
(`ok`/`backend_error`/`no_backend_available`), `http_status`, and
`latency_s`. This module deliberately records only what the router itself
can observe -- whether a *decision* was actually the right one is an
offline question: a later shadow-eval pass reads this file, joins it
against real task outcomes (e.g. via the `extra` field, which
`DecisionRecord` already supports for exactly this), and answers "did the
router route correctly?" This module does not grade its own decisions.

`extra` is not purely theoretical: `dispatch.py`'s `_extract_backend_telemetry`
opportunistically folds a small set of optional `X-trailbrake-*` response headers
(decode tok/s, TTFT, and the opt-in speculative-decoding drafter's
acceptance rate -- see `trailbrake/src/mlx_engine/server.py`'s
`_telemetry_response_headers`) into `extra` whenever a backend sends them,
for both buffered and streamed dispatch. This module still hardcodes no
backend's name or identity to do it -- a backend that never sends these
headers (iliria, or trailbrake without the drafter flag) contributes nothing,
exactly as before this existed. It is headers-only, deliberately: a
streamed response's *body* is relayed incrementally, long after headers
arrive, so this is the one mechanism that captures buffered and streamed
telemetry identically; an end-of-generation metric a streaming backend
cannot put in a header at all (headers must precede a body whose content
isn't known yet) only reaches the client via the final SSE `usage` event,
not this log -- see that same trailbrake docstring.

`GET /router/status` exposes a live, in-memory view (circuit-breaker state
per backend, decision counts by `tier:backend:status`) for a quick check
without tailing the JSONL file; it is allowed to reset across a restart
(the file is the durable record).

### Security posture (reused, not re-derived)

The router is one more attacker-reachable stdlib `http.server` process in
the same threat model `iliria/docs/the threat model` already reviewed
for iliria's own gateway (personal, single-user, localhost daily driver --
not multi-tenant): Origin checked before any real work happens (Origin/CSRF defense), a 60s socket read timeout to defeat slowloris
(slowloris defense), loopback bind by default with a loud stderr warning if overridden
without an `api_key` configured (bind defense). These are reapplied verbatim
(`server.py`'s `_check_origin`/`RouterHandler.timeout = 60`/`serve`'s
warning), not re-derived from scratch, because the router arguably has a
*larger* blast radius than either single backend -- it can reach two
expensive compute backends from one process.

## 6. What is scaffolded vs. what is explicitly not

**Scaffolded, real, tested (a full unit-test suite, `pytest`/`ruff` clean, no
network, no GPU, no model weights anywhere in this repo or its
dependencies):**
config loading/validation; the four-trigger escalation policy including
draft-then-escalate; weighted + hash-bucketed canary backend selection;
per-backend circuit breakers (including the half-open-peek-vs-claim
distinction); the bounded fallback/retry dispatch loop; JSONL telemetry;
the full HTTP proxy surface (chat/completions/models/health/status,
CORS/origin/auth guards, buffered responses, streaming relay, the
single-flush SSE degradation for draft-then-escalate); SIGHUP hot-reload
with fail-safe behavior on a broken reload.

**Explicitly not built here:** a real `Verifier` for draft-then-escalate
(needs task metadata this generic proxy doesn't have); a learned/tuned
version of the routing heuristic (§4's honesty note); anything that talks to a
real trailbrake or iliria process, a real model, or a GPU.

## 7. The live, GPU-gated integration test (not run here)

This is the natural next step once the GPU is free and the pruning sweep
has a result, and it is deliberately **not** attempted in this change. It
would need:

1. **Two real trailbrake processes**: `trailbrake serve --model <unmodified
   checkpoint> --port 8080` (baseline) and `TB_SKIP_LAYERS=<sweep's
   chosen set> trailbrake serve --model <same checkpoint> --port 8081`
   (candidate) -- plus a real `ili serve` / `run-m5max-serve.sh` on
   `:8000`.
2. **Config-shape correctness against real servers, not fakes**: does
   `BackendClient`'s buffered and streaming paths actually round-trip
   against trailbrake's real chunked-SSE implementation and iliria's real
   `GenerationScheduler`-fronted one -- confirming the model-id rewrite,
   the `enable_thinking`/`reasoning_effort` translation, and the
   `include_usage` trailing-event shape all still match what this design
   doc asserts from static reading (§2), not just what the hand-written
   fakes in `tests/fakes.py` return.
3. **Canary correctness under real pruning**: with `trailbrake-candidate` at a
   small nonzero weight, confirm decode throughput and campaign2-
   style task-score parity between baseline and candidate match what the
   sweep predicted, using the *router's own telemetry* (not a separate
   harness) as the source of truth -- i.e. prove the JSONL log is actually
   sufficient for shadow-eval, not just schematically plausible.
4. **Circuit-breaker behavior against a real failure**, not a scripted one
   -- e.g. actually killing the candidate `trailbrake serve` process
   mid-run and confirming requests fail over to baseline within
   `failure_threshold` real requests, and that health checks
   (`BackendClient.health()`) correctly flip once the process is restarted.
5. **Real escalation latency measurement**: confirm the two-timeout design
   (`connect_timeout_s`/`idle_timeout_s`) is actually tuned correctly
   against iliria's real ~1.6 tok/s decode -- this design doc's timeout
   defaults (iliria's `idle_timeout_s = 120.0` in the example config) are
   an **ASSUMPTION** pending a real long-generation measurement, not a
   verified number.
6. **A real SIGHUP rollback rehearsal**: with live traffic flowing to the
   candidate, edit the file, send SIGHUP, and confirm in-flight requests
   finish cleanly while new ones immediately see the rolled-back weights --
   this change proves the mechanism works against scripted fakes and a
   direct function call; it does not prove it against a process serving
   real concurrent generation load.
