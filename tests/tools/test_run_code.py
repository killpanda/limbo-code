"""Tests for the run_code tool (Code Mode transport, deepseek-harness parity)."""

import tempfile
from pathlib import Path

import pytest

from limbo.tools.registry import ToolRegistry
from limbo.tools.run_code import RunCodeTool, ToolCallError


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def registry(workdir):
    return ToolRegistry(workdir=workdir)


@pytest.fixture
def tool(registry):
    t = RunCodeTool(workdir=registry.workdir)
    t.registry = registry
    return t


def test_registry_registers_run_code(registry):
    assert registry.get("run_code") is not None
    # Native mode: run_code is registered but hidden from the model.
    names = {d["function"]["name"] for d in registry.definitions()}
    assert "run_code" not in names
    assert "read" in names


def test_code_mode_collapses_definitions(registry):
    registry.code_mode = True
    defs = registry.definitions()
    assert [d["function"]["name"] for d in defs] == ["run_code"]


def test_all_definitions_excludes_run_code(registry):
    names = {d["function"]["name"] for d in registry.all_definitions()}
    assert "run_code" not in names
    assert "read" in names


def test_reads_file_via_proxy(workdir, tool):
    (workdir / "hello.txt").write_text("code mode works")
    result = tool.run(
        {
            "code": (
                'content = await tools.read(path="hello.txt")\n'
                'return f"got: {content}"'
            ),
            "description": "Read hello.txt and return its contents",
        }
    )
    assert result.success is True
    assert "got: code mode works" in (result.output or "")


def test_print_and_return(tool):
    result = tool.run(
        {
            "code": 'print("step one")\nprint("step two")\nreturn 42',
            "description": "Print two lines and return 42",
        }
    )
    assert result.success is True
    assert "step one" in (result.output or "")
    assert "step two" in (result.output or "")
    assert "42" in (result.output or "")


def test_no_output(tool):
    result = tool.run(
        {"code": "x = 1 + 1", "description": "Compute silently"}
    )
    assert result.success is True
    assert "no output" in (result.output or "")


def test_tool_call_error_is_catchable(workdir, tool):
    (workdir / "x.txt").write_text("hi")
    result = tool.run(
        {
            "code": (
                "try:\n"
                '    await tools.read(path="missing.txt")\n'
                "except ToolCallError as e:\n"
                '    return f"caught {e.toolName}"'
            ),
            "description": "Catch a failing tool call",
        }
    )
    assert result.success is True
    assert "caught read" in (result.output or "")


def test_unhandled_error_fails_with_traceback(tool):
    result = tool.run(
        {
            "code": 'content = await tools.read(path="nope.txt")\nreturn content',
            "description": "Read a missing file",
        }
    )
    assert result.success is False
    assert "File not found" in (result.error or "")
    assert "Traceback" in (result.error or "")


def test_syntax_error_reported(tool):
    result = tool.run(
        {"code": "def broken(", "description": "Syntax error probe"}
    )
    assert result.success is False
    assert "SyntaxError" in (result.error or "")


def test_timeout(tool):
    result = tool.run(
        {
            "code": "import asyncio\nawait asyncio.sleep(30)\nreturn 1",
            "description": "Sleep past the budget",
            "timeout": 0.2,
        }
    )
    assert result.success is False
    assert "timeout" in (result.error or "")


def test_cancel_interrupts_at_next_call(workdir, tool):
    (workdir / "x.txt").write_text("hi")
    # ESC lands before the first tool dispatch: the proxy checks the event.
    tool.cancel()
    result = tool.run(
        {
            "code": 'return await tools.read(path="x.txt")',
            "description": "Should be interrupted",
        }
    )
    assert result.success is False
    assert "interrupted" in (result.error or "")


def test_required_arguments(tool):
    assert tool.run({"code": "pass"}).success is False
    assert tool.run({"description": "d"}).success is False


def test_run_code_not_reachable_inside_program(tool):
    # tools.run_code must raise AttributeError → program fails loudly.
    result = tool.run(
        {
            "code": 'await tools.run_code(code="pass", description="nested")',
            "description": "Try to nest a run",
        }
    )
    assert result.success is False


def test_subscript_access(workdir, tool):
    (workdir / "x.txt").write_text("subscript works")
    result = tool.run(
        {
            "code": (
                'content = await tools["read"](path="x.txt")\nreturn content'
            ),
            "description": "Call via subscript",
        }
    )
    assert result.success is True
    assert "subscript works" in (result.output or "")


def test_gather_parallel_calls(workdir, tool):
    (workdir / "a.txt").write_text("aaa")
    (workdir / "b.txt").write_text("bbb")
    result = tool.run(
        {
            "code": (
                "import asyncio\n"
                "a, b = await asyncio.gather(\n"
                '    tools.read(path="a.txt"),\n'
                '    tools.read(path="b.txt"),\n'
                ")\n"
                'return a + "|" + b'
            ),
            "description": "Read two files concurrently",
        }
    )
    assert result.success is True
    assert "aaa|bbb" in (result.output or "")


def test_missing_registry_is_error():
    t = RunCodeTool(workdir=Path("."))
    result = t.run(
        {"code": "return 1", "description": "No registry attached"}
    )
    assert result.success is False


def test_positional_args_map_to_parameter_order(workdir, tool):
    (workdir / "x.txt").write_text("positional works")
    result = tool.run(
        {
            "code": 'return await tools.read("x.txt")',
            "description": "Read via positional arg",
        }
    )
    assert result.success is True
    assert "positional works" in (result.output or "")


def test_dict_style_call_expands(workdir, tool):
    (workdir / "x.txt").write_text("dict works")
    result = tool.run(
        {
            "code": 'return await tools.read({"path": "x.txt"})',
            "description": "Read via deepseek-style dict",
        }
    )
    assert result.success is True
    assert "dict works" in (result.output or "")


def test_positional_mixed_with_kwargs(workdir, tool):
    (workdir / "x.txt").write_text("line one\nline two\n")
    result = tool.run(
        {
            "code": 'return await tools.read("x.txt", limit=1)',
            "description": "Positional plus keyword mix",
        }
    )
    assert result.success is True
    assert "line one" in (result.output or "")
    assert "line two" not in (result.output or "")


def test_too_many_positional_args_fails(tool):
    result = tool.run(
        {
            # read takes at most 3 params; pass 5.
            "code": 'return await tools.read("a", "b", "c", "d", "e")',
            "description": "Too many positional args",
        }
    )
    assert result.success is False
    assert "too many positional arguments" in (result.error or "")


def test_tool_call_error_type():
    err = ToolCallError("read", "boom")
    assert err.tool_name == "read"
    assert str(err) == "boom"

def test_multiline_string_literal_survives_wrapping(tool):
    """Regression: the async-def wrap must not indent string-literal content.

    The old textwrap.indent wrap added 4 spaces to every body line —
    including lines inside triple-quoted literals — silently corrupting
    file-writing programs (the write-file-with-literal pattern, which is
    how Code Mode is supposed to author files).
    """
    q = "'" * 3  # a triple quote, spelled out to keep source nesting simple
    result = tool.run(
        {
            "code": f"text = {q}\nalpha\n    beta\ngamma\n{q}\nreturn text",
            "description": "Return a multi-line literal verbatim",
        }
    )
    assert result.success is True
    assert result.output == "\nalpha\n    beta\ngamma\n"
