"""Minimal authenticated host bridge for Docker-to-Codex ACP inference.

Run this on the Windows host under the account that owns the Codex login.  The
Docker backend sends only a prompt and JSON schema; the bridge deliberately has
no general ACP, filesystem, command, credential, or proxy endpoint.

The implementation uses only Python's standard library plus ``acp_backend`` so
the host does not need OpenShorts' video/ML dependencies installed.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlsplit

import acp_backend


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 3 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROMPT_CHARS = 2_500_000
MAX_SCHEMA_BYTES = 128 * 1024


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], token: str):
        if len(token) < 32:
            raise ValueError("ACP_BRIDGE_TOKEN must contain at least 32 characters")
        super().__init__(address, BridgeHandler)
        self.bridge_token = token


class BridgeHandler(BaseHTTPRequestHandler):
    server: BridgeServer
    protocol_version = "HTTP/1.1"
    server_version = "OpenShortsACPBridge/1"
    sys_version = ""

    def log_message(self, _format: str, *_args) -> None:
        # Prompts and headers must never enter logs.  Startup/errors are printed
        # by main; routine HTTP access logging is intentionally disabled.
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization") or ""
        prefix = "Bearer "
        supplied = header[len(prefix):] if header.startswith(prefix) else ""
        return hmac.compare_digest(supplied, self.server.bridge_token)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        # A POST body has not been consumed yet. Close the connection so those
        # bytes cannot be mistaken for a second HTTP request.
        self.close_connection = True
        self._send_json(401, {"error": "Unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        target = urlsplit(self.path)
        if target.path != "/health" or target.query:
            self._send_json(404, {"error": "Not found"})
            return
        if not self._require_auth():
            return
        self._send_json(200, {
            "ok": True,
            "provider": "codex-acp",
            "adapterAvailable": acp_backend.command_available(),
        })

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        target = urlsplit(self.path)
        if target.path != "/v1/generate" or target.query:
            self._send_json(404, {"error": "Not found"})
            return
        if not self._require_auth():
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self.close_connection = True
            self._send_json(413, {"error": "Request body is missing or too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return
        error = _validate_request(payload)
        if error:
            self._send_json(422, {"error": error})
            return
        try:
            text = acp_backend.run_text(
                payload["prompt"], payload["schema"], payload.get("model")
            )
        except acp_backend.ACPTimeoutError as exc:
            self._send_json(504, {"error": str(exc)[:500]})
            return
        except acp_backend.ACPError as exc:
            self._send_json(503, {"error": str(exc)[:500]})
            return
        except Exception as exc:
            # Do not serialize a traceback, environment, or child-process data.
            self._send_json(500, {"error": f"ACP bridge internal error: {type(exc).__name__}"})
            return
        if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
            self._send_json(502, {"error": "ACP response exceeded the size limit"})
            return
        self._send_json(200, {"text": text, "cost_analysis": None})


def _validate_request(payload) -> Optional[str]:
    if not isinstance(payload, dict):
        return "Request must be a JSON object"
    if set(payload) - {"prompt", "schema", "model"}:
        return "Request contains unsupported fields"
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt or len(prompt) > MAX_PROMPT_CHARS:
        return "prompt must be non-empty text within the size limit"
    schema = payload.get("schema")
    if not isinstance(schema, dict) or not schema:
        return "schema must be a non-empty JSON object"
    try:
        schema_bytes = len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        return "schema must contain JSON values"
    if schema_bytes > MAX_SCHEMA_BYTES:
        return "schema exceeds the size limit"
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or len(model) > 200):
        return "model must be a short string or null"
    return None


def make_server(host: str, port: int, token: str) -> BridgeServer:
    return BridgeServer((host, port), token)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenShorts Codex ACP host bridge")
    parser.add_argument("--host", default=os.environ.get("ACP_BRIDGE_HOST") or DEFAULT_HOST)
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("ACP_BRIDGE_PORT") or DEFAULT_PORT),
    )
    parser.add_argument(
        "--print-token", action="store_true",
        help="print a new bridge token and exit",
    )
    parser.add_argument(
        "--allow-non-loopback", action="store_true",
        help="required with a non-loopback --host; protect the port with a host firewall",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.print_token:
        print(secrets.token_urlsafe(48))
        return 0
    token = (os.environ.get("ACP_BRIDGE_TOKEN") or "").strip()
    if len(token) < 32:
        print("ACP_BRIDGE_TOKEN must contain at least 32 characters", file=os.sys.stderr)
        return 2
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        if not args.allow_non_loopback:
            print(
                "Refusing a non-loopback bind without --allow-non-loopback. "
                "Restrict the port to Docker with the host firewall.",
                file=os.sys.stderr,
            )
            return 2
        print(
            "WARNING: ACP bridge is not loopback-only; enforce a host firewall and "
            "never expose this port publicly.",
            file=os.sys.stderr,
        )
    server = make_server(args.host, args.port, token)
    print(
        f"OpenShorts ACP bridge listening on http://{args.host}:{server.server_port} "
        f"(adapter={'available' if acp_backend.command_available() else 'missing'})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
