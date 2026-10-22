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


def get_json(viewer: TraceViewer, path: str) -> dict:
    response, body = get(viewer, path)
    assert response.status == 200, body
    return json.loads(body)


def test_binds_loopback_only(viewer):
    assert viewer.url().startswith(f"http://127.0.0.1:{viewer.port}/?token=")


def test_requests_without_token_are_rejected(viewer):
    response, _ = get(viewer, "/api/state", token=False)
    assert response.status == 401
    response, _ = get(viewer, "/api/state", token=False, cookie="limbo_trace=wrong")
    assert response.status == 401


def test_page_sets_cookie_and_cookie_authorizes(viewer):
    response, body = get(viewer, "/")
    assert response.status == 200
    assert b"Limbo Trace" in body
    cookie = response.getheader("Set-Cookie")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    response, _ = get(viewer, "/api/state", token=False, cookie=f"limbo_trace={viewer.token}")
    assert response.status == 200


def test_foreign_host_header_is_rejected(viewer):
    # DNS rebinding: a page on evil.example resolving to 127.0.0.1.
    response, _ = get(viewer, "/api/state", host=f"evil.example:{viewer.port}")
    assert response.status == 403


def test_state_and_sessions(viewer):
    assert get_json(viewer, "/api/state") == {"current": SESSION}
    (session,) = get_json(viewer, "/api/sessions")["sessions"]
    assert session["id"] == SESSION and session["title"] == "hello"
    assert session["current"] is True and session["has_trace"] is True


def test_tree_and_live_cursor(viewer):
    tree = get_json(viewer, f"/api/sessions/{SESSION}/tree")
    assert tree["record_count"] == 4
    assert tree["current"] is True
    assert len(tree["turns"]) == 1
    assert get_json(viewer, f"/api/sessions/{SESSION}/tree?cursor=4") == {
        "unchanged": True,
        "record_count": 4,
    }
    write_trace(
        viewer.session_dir, SESSION, [{"type": "turn_end", "ts": "t", "status": "completed"}]
    )
    tree = get_json(viewer, f"/api/sessions/{SESSION}/tree?cursor=4")
    assert tree["record_count"] == 5
    assert tree["turns"][0]["status"] == "completed"


def test_record_with_diff(viewer):
    tree = get_json(viewer, f"/api/sessions/{SESSION}/tree")
    generation = tree["turns"][0]["spans"][0]
    (request_offset,) = generation["attempts"]
    payload = get_json(viewer, f"/api/sessions/{SESSION}/records/{request_offset}?diff=1")
    assert payload["record"]["type"] == "llm_request"
    assert payload["diff"] == {"prefix": 0, "total": 1, "messages": [{"role": "system"}]}
    payload = get_json(viewer, f"/api/sessions/{SESSION}/records/{generation['response_offset']}")
    assert payload["record"]["content"] == "answer"


def test_unknown_session_and_record(viewer):
    response, _ = get(viewer, f"/api/sessions/{SESSION}/records/3")
    assert response.status == 404
    response, _ = get(viewer, "/api/sessions/..%2F..%2Fetc/tree")
    assert response.status == 404
    response, _ = get(viewer, "/api/nope")
    assert response.status == 404


def test_start_is_idempotent_and_stop_releases(tmp_path):
    v = TraceViewer(tmp_path, None, lambda: "x")
    v.start()
    port = v.port
    v.start()
    assert v.port == port
    v.stop()
    assert not v.running
    v.stop()  # idempotent
