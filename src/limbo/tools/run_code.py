"""The ``run_code`` tool: limbo's Code Mode transport (deepseek-harness parity).

When Code Mode is on (``/code-mode on``), the model sees exactly one tool —
``run_code`` — and every other registered tool is presented as an SDK the
model calls from *inside* a Python program: ``await tools.read(path=...)``.
A sequence that would otherwise take several LLM round trips becomes one
``run_code`` call: the program runs to completion, and only what it prints
or returns enters the conversation.

Execution model:

- The ``code`` argument is the *body* of an async function (top-level
  ``await`` and ``return`` both work). It is compiled and run on a fresh
  event loop in the calling worker thread.
- Two names are bound in the program: ``tools`` (the tool proxy) and
  ``ToolCallError``. Everything else comes from Python builtins only.
- ``tools.<name>(**kwargs)`` forwards keyword arguments verbatim to the
  registered tool and returns its text output as a ``str``. A failed tool
  call raises ``ToolCallError`` (with a ``tool_name`` attribute) so the
  program can handle it and continue; an unhandled exception fails the
  whole run with the traceback and captured output, letting the model
  self-correct on the next iteration.
- ``print``/``stderr`` writes inside the program are captured as logs.
- ESC interruption surfaces as ``asyncio.CancelledError`` at the next tool
  call boundary (the proxy checks a thread-safe event before dispatching);
  in-flight sub-tools run to completion and record real results, matching
  the file-tool philosophy (RFC LIM-53).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import textwrap
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError

if TYPE_CHECKING:
    from limbo.tools.registry import ToolRegistry

# Default wall-clock budget for one program run (seconds). A runaway loop
# must never hang the turn; the model can pass `timeout` to override.
_DEFAULT_TIMEOUT_SECONDS = 120.0

_MAX_RESULT_CHARS = 64 * 1024


class ToolCallError(Exception):
    """Raised inside a run_code program when a tool call fails.

    The program may catch it to handle the failure and continue; unhandled,
    it fails the whole run with the traceback and captured output.
    """

    def __init__(self, tool_name: str, message: str):
        super().__init__(message)
        self.tool_name = tool_name
        self.toolName = tool_name


class _ProgramFailed(Exception):
    """Internal: the program raised; its message is the model-facing error."""


def _render_value(value: Any) -> str:
    """Render a program's return value for the model-facing result text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(value)


class _ToolsProxy:
    """The ``tools`` global a run_code program sees.

    Every registered tool except ``run_code`` itself becomes an async
    callable. Arguments are keyword-only by contract (the SDK renders them
    as named parameters); both attribute calls (``tools.read(...)``) and
    subscript calls (``tools["read"](...)``) are supported. A failed tool
    call raises :class:`ToolCallError`; an ESC interrupt surfaces as
    ``asyncio.CancelledError`` at the next dispatch boundary.
    """

    def __init__(self, registry: ToolRegistry, cancelled: threading.Event):
        self._registry = registry
        self._cancelled = cancelled

    def _binding(self, name: str):
        async def _call(*args: Any, **kwargs: Any) -> str:
            if self._cancelled.is_set():
                raise asyncio.CancelledError("run_code interrupted by user (ESC)")
            # Positional arguments map to the tool's declared parameter
            # order; a lone dict (deepseek-style `tools.name({...})`)
            # expands as keyword arguments. Missing keys surface as the
            # tool's own validation error, keeping one failure contract.
            if args:
                tool = self._registry.get(name)
                if tool is None:
                    raise ToolCallError(name, f"unknown tool: {name}")
                params = list((tool.parameters or {}).get("properties", {}))
                if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                    kwargs = dict(args[0])
                else:
                    if len(args) > len(params):
                        raise ToolCallError(
                            name,
                            f"too many positional arguments ({len(args)}) for "
                            f"{name} (expects at most {len(params)}: "
                            f"{', '.join(params) or 'none'})",
                        )
                    for index, arg in enumerate(args):
                        key = params[index]
                        if key in kwargs:
                            raise ToolCallError(
                                name,
                                f"got multiple values for argument '{key}'",
                            )
                        kwargs[key] = arg
            result = await self._registry.execute(name, kwargs)
            if not result.success:
                raise ToolCallError(name, result.error or f"tool {name} failed")
            return result.output or ""

        return _call

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "run_code":
            raise AttributeError(name)
        return self._binding(name)

    def __getitem__(self, name: str) -> Any:
        return self._binding(name)


class RunCodeTool(BaseTool):
    name = "run_code"
    description = (
        "Execute a Python program against the available tools. Takes two "
        "required arguments: `code`, the BODY of an async Python function "
        "(top-level `await` and `return` work), and `description`, a short "
        "summary of what the program does. Call tools as "
        "`await tools.name(kwarg=value)` (positional args follow the SDK's "
        "declared parameter order; a lone dict also works) per the SDK "
        "declarations in the system prompt. Only what you print or return "
        "comes back — intermediate tool results never enter the "
        "conversation, so extract just what you need."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "The program: the body of an async Python function. "
                    "Use `await tools.<name>(...)` to call tools, "
                    "`print(...)` for logs, and a top-level `return <value>` "
                    "for the final answer."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Clear, concise description of what this program does "
                    "in active voice, 5-10 words (shown in the UI)."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Optional wall-clock budget in seconds "
                    f"(default {_DEFAULT_TIMEOUT_SECONDS:g})."
                ),
            },
        },
        "required": ["code", "description"],
    }

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        # Injected by ToolRegistry after registration: sub-tool dispatch
        # goes through the owning registry so every call shares the same
        # registry view (bash_enabled filtering, etc.).
        self.registry: ToolRegistry | None = None
        # Persistent interrupt flag: cancel() sets it; run() swaps in a
        # per-run event seeded from it (so a pre-flight ESC survives while
        # a previous run's cancel can never leak into the next one).
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal the running program to stop at its next tool boundary."""
        self._cancelled.set()

    def _registry_or_raise(self) -> ToolRegistry:
        if self.registry is None:
            raise ToolError("run_code: registry not attached")
        return self.registry

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        description = arguments.get("description")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                success=False, error="run_code: missing required argument 'code'"
            )
        if not isinstance(description, str) or not description.strip():
            return ToolResult(
                success=False,
                error="run_code: missing required argument 'description'",
            )
        timeout = arguments.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            return ToolResult(success=False, error="run_code: invalid timeout")
        if timeout <= 0:
            return ToolResult(success=False, error="run_code: timeout must be > 0")

        try:
            self._registry_or_raise()
        except ToolError as e:
            return ToolResult(success=False, error=str(e))
        # A cancel() that landed before run() (pre-flight ESC) must not be
        # lost; a cancel from a PREVIOUS run must not leak in. Snapshot the
        # persistent flag, reset it, and hand the run a fresh event.
        pending = self._cancelled.is_set()
        self._cancelled.clear()
        cancelled = threading.Event()
        if pending:
            cancelled.set()
        self._cancelled = cancelled
        logs: list[str] = []
        try:
            value = asyncio.run(
                self._run_program(code, cancelled, logs, timeout)
            )
        except asyncio.CancelledError:
            return ToolResult(
                success=False, error="run_code: program interrupted (ESC)"
            )
        except _ProgramFailed as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:  # noqa: BLE001 - compile errors etc.
            return ToolResult(
                success=False, error=f"run_code: {type(e).__name__}: {e}"
            )
        finally:
            # Hand the flag back to the persistent event for the next run.
            self._cancelled = threading.Event()

        parts = [log for log in logs if log]
        rendered = _render_value(value)
        if rendered:
            parts.append(rendered)
        output = "\n".join(parts) if parts else "(run_code completed with no output)"
        if len(output) > _MAX_RESULT_CHARS:
            output = output[:_MAX_RESULT_CHARS] + "\n[Output truncated.]"
        return ToolResult(success=True, output=output)

    async def _run_program(
        self,
        code: str,
        cancelled: threading.Event,
        logs: list[str],
        timeout: float,
    ) -> Any:
        registry = self._registry_or_raise()
        proxy = _ToolsProxy(registry, cancelled)
        namespace: dict[str, Any] = {
            "tools": proxy,
            "ToolCallError": ToolCallError,
            "__name__": "__limbo_run_code__",
        }
        # The `code` argument is a function BODY: indent it under an async
        # def so top-level `await` and `return` are legal Python.
        wrapped = f"async def _limbo_run_code():\n{textwrap.indent(code, '    ')}"
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                exec(compile(wrapped, "<run_code>", "exec"), namespace)
                coro = namespace["_limbo_run_code"]()
                return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            raise _ProgramFailed(
                f"code run failed (timeout): program exceeded {timeout:g}s"
                f"{self._captured(stdout, stderr)}"
            ) from e
        except Exception as e:  # noqa: BLE001 - the program's own exception
            raise _ProgramFailed(
                f"code run failed ({type(e).__name__}): {e}"
                f"{self._captured(stdout, stderr)}"
                f"\nTraceback:\n{traceback.format_exc()}"
            ) from e
        finally:
            out = stdout.getvalue()
            err = stderr.getvalue()
            if out:
                logs.append(out.rstrip())
            if err:
                logs.append(err.rstrip())

    @staticmethod
    def _captured(stdout: io.StringIO, stderr: io.StringIO) -> str:
        """The captured-output suffix for a failed run's error message."""
        parts = [p for p in (stdout.getvalue(), stderr.getvalue()) if p]
        if not parts:
            return ""
        return "\nCaptured output:\n" + "\n".join(p.rstrip() for p in parts)
