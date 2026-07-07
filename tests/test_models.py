from limbo.models import Message, TextChunk, ToolCall, ToolCallEvent, ToolResult


def test_message_creation():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_tool_call_creation():
    tc = ToolCall(id="call_1", name="read", arguments={"path": "main.py"})
    assert tc.name == "read"
    assert tc.arguments["path"] == "main.py"


def test_tool_result_requires_confirmation_defaults_false():
    result = ToolResult(success=True, output="ok")
    assert result.requires_confirmation is False


def test_llm_events_are_frozen_dataclasses():
    chunk = TextChunk(text="hello")
    assert chunk.text == "hello"
    call = ToolCallEvent(id="c1", name="read", arguments={"path": "x.py"})
    assert call.name == "read"
    assert call.arguments["path"] == "x.py"
