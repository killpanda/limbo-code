"""Tests for the loopback trace viewer server (limbo.tracing.server)."""

from __future__ import annotations

import http.client
import json
from pathlib import Path

import pytest

from limbo.sessions import SessionMeta, save_session
from limbo.trace import trace_path_for
from limbo.tracing.server import TraceViewer

SESSION = "20260101-000000-000000-1-abcd"


def write_trace(session_dir: Path, session_id: str, records: list[dict]) -> Path:
    path = trace_path_for(session_dir / f"{session_id}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def viewer(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    save_session(
        session_dir / f"{SESSION}.jsonl",
        SessionMeta(id=SESSION, workdir=str(tmp_path), model="m", title="hello"),
        [],
    )
    write_trace(
        session_dir,
        SESSION,
        [
            {"type": "session_start", "ts": "t", "session_id": SESSION},
            {"type": "user_message", "ts": "t", "content": "q"},
            {
                "type": "llm_request",
                "ts": "t",
                "iteration": 1,
                "body": {"messages": [{"role": "system"}]},
            },
            {"type": "llm_response", "ts": "t", "iteration": 1, "duration": 1, "content": "answer"},
        ],
    )
    v = TraceViewer(session_dir, tmp_path, lambda: SESSION)
    v.start()
    yield v
    v.stop()


def get(
    viewer: TraceViewer,
    path: str,
    *,
    token: bool = True,
    host: str | None = None,
    cookie: str | None = None,
):
    conn = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=5)
    sep = "&" if "?" in path else "?"
    url = path + (f"{sep}token={viewer.token}" if token else "")
    headers = {"Host": host or f"127.0.0.1:{viewer.port}"}
    if cookie:
        headers["Cookie"] = cookie
    conn.request("GET", url, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response, body


def test_requests_without_token_are_rejected(viewer):
    response, _ = get(viewer, "/api/state", token=False)
    assert response.status == 401
    response, _ = get(viewer, "/api/state", token=False, cookie="limbo_trace=wrong")
    assert response.status == 401




def test_token_is_swapped_for_an_httponly_cookie(viewer):
    response, body = get(viewer, "/")
    assert response.status == 302
    assert response.getheader("Location") == "/"
    cookie = response.getheader("Set-Cookie") or ""
    assert f"limbo_trace={viewer.token}" in cookie
    assert "HttpOnly" in cookie
    assert "Content-Security-Policy" in response.headers
    assert response.getheader("X-Frame-Options") == "DENY"
    assert body == b""

    # The cookie now authorizes the page without a token in the URL.
    page, html = get(viewer, "/", token=False, cookie=f"limbo_trace={viewer.token}")
    assert page.status == 200
    assert b"<html" in html.lower()
