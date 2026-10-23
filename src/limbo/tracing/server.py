"""Loopback HTTP server behind ``/tracing-viewer``.

Standard library only (``ThreadingHTTPServer`` on a daemon thread), so the
viewer adds no dependency and never touches the asyncio loop the TUI runs
on. The server only reads trace files; it has no handle on the agent, so
live and historical sessions go through the same code.

Traces hold full prompts, source code and tool output, so access is
locked down: bound to 127.0.0.1 only, every request needs a random
per-process token (passed once in the URL, then kept in an HttpOnly
cookie), and the Host header must name the loopback address (blocks DNS
rebinding from a web page).

API (all JSON):

- ``GET /api/state`` — the frontend's current session id
- ``GET /api/sessions`` — sessions of the same workdir, newest first
- ``GET /api/sessions/<id>/tree?cursor=N`` — the session tree; when the
  trace still has ``N`` records, ``{"unchanged": true}`` (live polling)
- ``GET /api/sessions/<id>/records/<offset>[?diff=1]`` — one full record;
  with ``diff=1`` an ``llm_request`` also carries the messages new since
  the previous request
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from limbo.sessions import list_sessions
from limbo.trace import trace_path_for
from limbo.tracing.index import TraceIndex
from limbo.tracing.model import build_tree, request_diff

_COOKIE = "limbo_trace"
# The page is a single self-contained file: inline script/style, same-origin
# API only. Lock everything else down.
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"
)
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TREE = re.compile(r"^/api/sessions/([^/]+)/tree$")
_RECORD = re.compile(r"^/api/sessions/([^/]+)/records/(\d+)$")


def _load_page() -> bytes:
    return resources.files("limbo.tracing").joinpath("static/index.html").read_bytes()


class TraceViewer:
    """One loopback server per process; start lazily, reuse, stop on exit."""

    def __init__(
        self,
        session_dir: Path,
        workdir: Path | None,
        current_session: Callable[[], str],
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.session_dir = session_dir
        self.workdir = workdir
        self.current_session = current_session
        self._host = host
        self._port = port
        self.token = secrets.token_urlsafe(18)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._indexes: dict[str, TraceIndex] = {}
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("trace viewer is not running")
        return self._server.server_address[1]

    def start(self) -> None:
        """Bind and serve on a daemon thread; no-op when already running."""
        if self._server is not None:
            return
        viewer = self

        class Handler(_Handler):
            pass

        Handler.viewer = viewer
        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="limbo-trace-viewer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            self._indexes.clear()

    def url(self, session_id: str | None = None) -> str:
        """Page URL carrying the access token (and a session to open)."""
        url = f"http://{self._host}:{self.port}/?token={self.token}"
        if session_id:
            url += f"#{session_id}"
        return url

    # -- data access (called from handler threads) ---------------------------

    def trace_file(self, session_id: str) -> Path | None:
        if not _SESSION_ID.match(session_id) or ".." in session_id:
            return None
        return trace_path_for(self.session_dir / f"{session_id}.jsonl")

    def index(self, session_id: str) -> TraceIndex | None:
        path = self.trace_file(session_id)
        if path is None:
            return None
        with self._lock:
            index = self._indexes.get(session_id)
            if index is None:
                index = self._indexes[session_id] = TraceIndex(path)
        index.refresh()
        return index

    def sessions(self) -> list[dict[str, Any]]:
        current = self.current_session()
        result = []
        for meta in list_sessions(self.session_dir, workdir=self.workdir):
            path = self.trace_file(meta.id)
            result.append(
                {
                    "id": meta.id,
                    "title": meta.title,
                    "model": meta.model,
                    "updated_at": meta.updated_at,
                    "created_at": meta.created_at,
                    "current": meta.id == current,
                    "has_trace": path is not None and path.exists(),
                }
            )
        if current and not any(s["id"] == current for s in result):
            # Brand-new session: no session file saved yet, trace may exist.
            result.insert(0, {"id": current, "title": "", "current": True, "has_trace": True})
        return result


class _Handler(BaseHTTPRequestHandler):
    viewer: TraceViewer
    server_version = "limbo-trace"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # stderr belongs to the TUI

    # -- auth ------------------------------------------------------------------

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        port = self.viewer.port
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _cookie_token(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie: SimpleCookie = SimpleCookie()
        try:
            cookie.load(raw)
        except Exception:  # noqa: BLE001 - malformed cookie = no token
            return None
        morsel = cookie.get(_COOKIE)
        return morsel.value if morsel else None

    # -- responses -------------------------------------------------------------

    def _send(
        self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", _CSP)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # -- routing ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "bad host")
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        token = (query.get("token") or [""])[0] or None
        authorized = secrets.compare_digest(token or self._cookie_token() or "", self.viewer.token)
        if not authorized:
            self._error(
                HTTPStatus.UNAUTHORIZED, "missing or wrong token; reopen with /tracing-viewer"
            )
            return
        try:
            self._route(url.path, query, token)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self, path: str, query: dict[str, list[str]], token: str | None) -> None:
        viewer = self.viewer
        if path == "/":
            if token:
                # Swap the token for the HttpOnly cookie and redirect, so
                # the token does not stay in the address bar or history.
                # The fragment (#session) survives a redirect with no
                # fragment of its own.
                self._send(
                    HTTPStatus.FOUND,
                    b"",
                    "text/plain; charset=utf-8",
                    {
                        "Set-Cookie": (
                            f"{_COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/"
                        ),
                        "Location": "/",
                    },
                )
                return
            self._send(HTTPStatus.OK, _load_page(), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._json({"current": viewer.current_session()})
            return
        if path == "/api/sessions":
            self._json({"sessions": viewer.sessions()})
            return
        match = _TREE.match(path)
        if match:
            index = viewer.index(match.group(1))
            if index is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown session")
                return
            records = index.records
            cursor = (query.get("cursor") or [""])[0]
            if cursor.isdigit() and int(cursor) == len(records):
                self._json({"unchanged": True, "record_count": len(records)})
                return
            tree = build_tree(records)
            tree["current"] = match.group(1) == viewer.current_session()
            self._json(tree)
            return
        match = _RECORD.match(path)
        if match:
            index = viewer.index(match.group(1))
            offset = int(match.group(2))
            record = index.read(offset) if index is not None else None
            if index is None or record is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown record")
                return
            payload: dict[str, Any] = {"record": record}
            if query.get("diff") and record.get("type") == "llm_request":
                previous_offset = index.previous_request(offset)
                previous = index.read(previous_offset) if previous_offset is not None else None
                if record.get("kind") == "compaction":
                    previous = None
                payload["diff"] = request_diff(record, previous)
            self._json(payload)
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")
