"""Router-native error types.

The router's own error responses use the same OpenAI-style error object shape
iliria's gateway already implements (`iliria/c/openai_server.py`'s
`APIError` / `error_object`): ``{"error": {"message", "type", "param",
"code"}}``. That shape is a strict superset of trailbrake's own
(`trailbrake/src/mlx_engine/server.py`'s `_error` only sets `message`
and `type`), so proxying an error verbatim from either backend, or raising one
natively in the router, always looks the same to a client.
"""

from __future__ import annotations


class RouterError(Exception):
    """An error the router raises itself (not proxied from a backend)."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type

    def to_object(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class ConfigError(RouterError):
    """Raised while loading/validating a router TOML config file."""

    def __init__(self, message: str) -> None:
        super().__init__(500, message, code="config_error", error_type="config_error")


class NoBackendAvailable(RouterError):
    """Every backend in the selected tier (and its configured fallback tier)
    is either disabled, circuit-open, or was just tried and failed."""

    def __init__(self, tier: str) -> None:
        super().__init__(
            503,
            f"No healthy backend is available for tier {tier!r} (or its fallback).",
            code="no_backend_available",
            error_type="server_error",
        )


class BackendRequestFailed(RouterError):
    """A backend was reachable but replied with a >=400 status. Carries the
    origin backend id so telemetry/fallback logic can attribute the failure."""

    def __init__(self, backend_id: str, status: int, message: str) -> None:
        super().__init__(status, message, code="backend_error", error_type="server_error")
        self.backend_id = backend_id
