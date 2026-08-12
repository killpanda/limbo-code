"""Execute bash commands tool."""

from __future__ import annotations

import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    BaseTool,
    truncate_tail,
)

# Keep at most this many bytes of each stream in memory; anything beyond is
# only available in the spilled temp file. Aligned with pi's rolling buffer.
MAX_BUFFER_BYTES = DEFAULT_MAX_BYTES * 2

_READ_CHUNK = 65536


def _write_full_output(output: str) -> Path | None:
    """Persist the full command output to a temp file; None on failure."""
    try:
        path = Path(tempfile.gettempdir()) / f"limbo-output-{secrets.token_hex(8)}.log"
        path.write_text(output, encoding="utf-8")
        return path
    except OSError:
        return None


class _StreamCollector:
    """Incrementally collect a process stream on a reader thread.

    Chunks are appended to a rolling in-memory buffer (capped at
    ``MAX_BUFFER_BYTES``); once the combined output exceeds
    ``DEFAULT_MAX_BYTES`` everything is also spilled to a temp file so the
    full output stays retrievable without holding it all in memory —
    unlike the old ``subprocess.run`` implementation, which spilled only
    post-hoc after buffering everything.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._buffered_bytes = 0
        self.total_bytes = 0
        self._lock = threading.Lock()
        self._spill: "_SpillFile | None" = None
        self._thread: threading.Thread | None = None

    def start(self, stream: Any, spill: "_SpillFile") -> None:
        self._spill = spill
        self._thread = threading.Thread(
            target=self._pump, args=(stream,), daemon=True
        )
        self._thread.start()

    def _pump(self, stream: Any) -> None:
        # read1 returns as soon as ANY data is available (unlike read(n),
        # which blocks until the buffer fills or EOF) — that is what makes
        # the collection actually streaming.
        read_chunk = getattr(stream, "read1", None) or stream.read
        while True:
            try:
                data = read_chunk(_READ_CHUNK)
            except (OSError, ValueError):
                break
            if not data:
                break
            text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
            with self._lock:
                self.total_bytes += len(data)
                self._chunks.append(text)
                self._buffered_bytes += len(text.encode("utf-8", errors="replace"))
                while self._buffered_bytes > MAX_BUFFER_BYTES and len(self._chunks) > 1:
                    removed = self._chunks.pop(0)
                    self._buffered_bytes -= len(removed.encode("utf-8", errors="replace"))
            if self._spill is not None:
                self._spill.write(text)

    def finish(self, timeout: float | None = 30.0) -> str:
        """Join the reader thread and return the buffered text.

        The join is time-bounded: a grandchild that escaped the process
        group (setsid) while holding the pipe fd would otherwise keep the
        reader waiting for EOF forever and hang the tool. The thread is a
        daemon, so a leaked one dies with the process.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._lock:
            return "".join(self._chunks)


class _SpillFile:
    """Lazy temp-file writer shared by both stream collectors.

    The file is created on the first write (i.e. as soon as output crosses
    the spill threshold) — small outputs never touch disk.
    """

    def __init__(self, threshold: int = DEFAULT_MAX_BYTES):
        self._threshold = threshold
        self._pending: list[str] = []
        self._pending_bytes = 0
        self._file: Any = None
        self.path: Path | None = None
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            if self._file is None:
                self._pending.append(text)
                self._pending_bytes += len(text.encode("utf-8", errors="replace"))
                if self._pending_bytes <= self._threshold:
                    return
                try:
                    self.path = (
                        Path(tempfile.gettempdir())
                        / f"limbo-output-{secrets.token_hex(8)}.log"
                    )
                    self._file = self.path.open("w", encoding="utf-8")
                except OSError:
                    self.path = None
                    self._pending.clear()
                    return
                for chunk in self._pending:
                    self._file.write(chunk)
                self._pending.clear()
            else:
                try:
                    self._file.write(text)
                except (OSError, ValueError):
                    pass

    def close(self) -> Path | None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
            return self.path


def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process and its whole tree (pi's killProcessTree).

    The shell runs in its own process group (``start_new_session``), so a
    group kill reaches detached children a plain ``proc.kill()`` would
    orphan. Falls back to killing just the shell if the group is gone.
    """
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            proc.kill()
        return
    try:
        os.killpg(proc.pid, 9)  # SIGKILL the whole process group
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass  # already dead


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). If truncated, "
        "the full output is saved to a temp file and its path is shown. "
        "Note: bash is not sandboxed — it can access files outside the working "
        "directory. "
        "For multi-line scripts, write the script to a file with the write "
        "tool and execute that file (e.g. python script.py) instead of using "
        "heredocs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, no default timeout)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path):
        super().__init__(workdir)
        # In-flight processes, for ESC interruption (RFC LIM-53). cancel()
        # runs on the event loop while run() blocks in a worker thread, so
        # both sets are guarded by the same lock; ``_cancelled`` keys are
        # id(proc) to stay correct across pid reuse.
        self._active_procs: set[subprocess.Popen[bytes]] = set()
        self._cancelled: set[int] = set()
        self._procs_lock = threading.Lock()

    def cancel(self) -> None:
        """Kill every in-flight process tree (ESC interrupt, RFC LIM-53).

        Thread-safe. A killed run reports the interruption explicitly
        ("已被用户打断") instead of a bare non-zero exit.
        """
        with self._procs_lock:
            procs = list(self._active_procs)
            self._cancelled.update(id(proc) for proc in procs)
        for proc in procs:
            _kill_process_tree(proc)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command", "")

        if not command:
            return ToolResult(success=False, error="No command provided.")

        # No default timeout and no upper cap (pi parity): a command runs to
        # completion unless the caller passes a timeout.
        timeout = self._parse_timeout(arguments.get("timeout"))
        if isinstance(timeout, ToolResult):
            return timeout

        shell = shutil.which("bash") or "/bin/bash"

        try:
            proc = subprocess.Popen(
                [shell, "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workdir),
                # Own process group so a timeout can kill the whole tree.
                start_new_session=(sys.platform != "win32"),
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"Execution failed: {e}")

        spill = _SpillFile()
        stdout = _StreamCollector()
        stderr = _StreamCollector()
        stdout.start(proc.stdout, spill)
        stderr.start(proc.stderr, spill)

        with self._procs_lock:
            self._active_procs.add(proc)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        with self._procs_lock:
            interrupted = id(proc) in self._cancelled
            self._cancelled.discard(id(proc))
            self._active_procs.discard(proc)

        out_text = stdout.finish()
        err_text = stderr.finish()
        full_path = spill.close()

        output = out_text
        if err_text:
            output += ("\n" if output else "") + f"[stderr]\n{err_text}"

        # Truncate to the tail (errors and final results live at the end).
        # The full output was spilled to disk incrementally while streaming,
        # so the path shown here always holds the complete output.
        truncation = truncate_tail(output)
        if truncation.truncated:
            if full_path is None:
                # Line-limit truncation below the spill threshold: the whole
                # output is still in the rolling buffers, so spill it now.
                full_path = _write_full_output(output)
            hint = (
                f"[Showing lines {truncation.start_line}-"
                f"{truncation.total_lines} of {truncation.total_lines}"
            )
            if full_path is not None:
                hint += f". Full output: {full_path}]"
            else:
                hint += "; full output could not be saved to a temp file]"
            output = f"{truncation.content}\n\n{hint}"

        if interrupted:
            interrupt_msg = "已被用户打断（ESC）"
            output = f"{interrupt_msg}\n{output}" if output else interrupt_msg
            return ToolResult(success=False, output=output, error=interrupt_msg)

        if timed_out:
            timeout_msg = f"Command timed out after {timeout}s and was killed."
            output = f"{timeout_msg}\n{output}" if output else timeout_msg
            return ToolResult(success=False, output=output, error=timeout_msg)

        if proc.returncode != 0:
            exit_msg = f"Command failed with exit code {proc.returncode}."
            output = f"{exit_msg}\n{output}" if output else exit_msg
            return ToolResult(
                success=False,
                output=output,
                error=exit_msg,
            )

        return ToolResult(success=True, output=output or "")

    @staticmethod
    def _parse_timeout(raw: Any) -> float | None | ToolResult:
        """Validate the optional timeout; None means run without a timeout."""
        if raw is None:
            return None
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return ToolResult(success=False, error=f"Invalid timeout: {raw}")
        if not math.isfinite(timeout) or timeout <= 0:
            return ToolResult(
                success=False,
                error="Invalid timeout: must be a positive number of seconds",
            )
        return timeout
