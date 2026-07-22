"""CLI entry point: `racecontrol serve --config path/to/router.toml`."""

from __future__ import annotations

import argparse

from . import chat as chat_mod
from .config import load_config, with_server_overrides
from .server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="racecontrol")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="Run the router's HTTP proxy")
    serve_parser.add_argument("--config", required=True, help="Path to a router TOML config file")
    serve_parser.add_argument("--host", default=None, help="Override [server].host from the config file")
    serve_parser.add_argument("--port", type=int, default=None, help="Override [server].port from the config file")
    serve_parser.add_argument(
        "--no-reload-on-sighup", action="store_true",
        help="Disable SIGHUP config-reload (on by default; see docs/DESIGN.md's guardrails section)",
    )

    check_parser = sub.add_parser("check-config", help="Load and validate a config file, then exit")
    check_parser.add_argument("--config", required=True)

    chat_parser = sub.add_parser(
        "chat", help="Interactive streaming chat against a running router (client only, no config)",
    )
    chat_parser.add_argument("--base-url", default=chat_mod.DEFAULT_BASE_URL,
                             help=f"Router base URL (default {chat_mod.DEFAULT_BASE_URL})")
    chat_parser.add_argument("--model", default=chat_mod.DEFAULT_MODEL,
                             help='Model name sent upstream (default "default" -- flows through the policy path)')
    chat_parser.add_argument("--temperature", type=float, default=0.7)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "chat":
        return chat_mod.run_repl(
            base_url=args.base_url, model=args.model, temperature=args.temperature,
        )

    config = load_config(args.config)

    if args.command == "check-config":
        for tier in config.tiers():
            backends = config.backends_for_tier(tier)
            print(f"tier {tier!r}: {[b.id for b in backends if b.enabled]} "
                  f"(disabled: {[b.id for b in backends if not b.enabled]})")
        print(f"escalation policy: {config.escalation.policy!r} "
              f"(default={config.escalation.default_tier!r}, escalation={config.escalation.escalation_tier!r})")
        print("OK")
        return 0

    if args.command == "serve":
        config = with_server_overrides(config, host=args.host, port=args.port)
        config_path = None if args.no_reload_on_sighup else args.config
        serve(config, config_path=config_path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
