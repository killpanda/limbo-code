"""Cross-platform clipboard reading for the input widget (RFC LIM-17 P2).

``read_clipboard()`` classifies the current OS clipboard into one of four
results — image bytes, file paths, text, or empty/unavailable — so the
input widget can attach images/files and fall back to plain text pastes.

Backends are thin wrappers around OS tools, each with a short timeout and
best-effort semantics (anything unexpected degrades to ``ClipboardEmpty``):

- macOS: ``osascript`` (clipboard info / class extraction) + ``pbpaste``
- Linux: ``wl-paste`` (Wayland) or ``xclip`` (X11)
- Windows: PowerShell ``Get-Clipboard``
"""

from __future__ import annotations

import base64
import binascii
import os
import platform
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

# Subprocess calls must never stall the UI on a hung clipboard helper.
_CMD_TIMEOUT = 5

ATTACHMENTS_DIR = Path.home() / ".limbo" / "attachments"


@dataclass(frozen=True)
class ClipboardImage:
    data: bytes
    ext: str = "png"


@dataclass(frozen=True)
class ClipboardFiles:
    paths: list[Path]


@dataclass(frozen=True)
class ClipboardText:
    text: str


@dataclass(frozen=True)
class ClipboardEmpty:
    """Clipboard unreadable, empty, or unsupported on this platform."""


ClipboardContent = ClipboardImage | ClipboardFiles | ClipboardText | ClipboardEmpty


def read_clipboard() -> ClipboardContent:
    """Classify the current OS clipboard (image > files > text > empty)."""
    system = platform.system()
    try:
        if system == "Darwin":
            return _read_macos()
        if system == "Linux":
            return _read_linux()
        if system == "Windows":
            return _read_windows()
    except (OSError, subprocess.SubprocessError):
        pass
    return ClipboardEmpty()


def save_clipboard_image(data: bytes, ext: str = "png") -> Path:
    """Persist clipboard image bytes under the attachments dir."""
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = ATTACHMENTS_DIR / f"clip-{uuid.uuid4().hex[:12]}.{ext}"
    path.write_bytes(data)
    return path


# -- helpers ------------------------------------------------------------------


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    """Run a clipboard helper; None on any failure (missing tool, error…)."""
    if shutil.which(args[0]) is None:
        return None
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=_CMD_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result


def _run_text(args: list[str]) -> str | None:
    result = _run(args)
    if result is None:
        return None
    return result.stdout.decode("utf-8", "replace")


def _run_bytes(args: list[str]) -> bytes | None:
    result = _run(args)
    if result is None:
        return None
    return result.stdout


# -- macOS ---------------------------------------------------------------------


def _read_macos() -> ClipboardContent:
    info = _run_text(["osascript", "-e", "clipboard info"])
    if info is None:
        return ClipboardEmpty()
    if "«class PNGf»" in info or "«class TIFF»" in info:
        image = _macos_image()
        if image is not None:
            return image
    if "«class furl»" in info:
        files = _macos_files()
        if files is not None:
            return files
    text = _run_text(["pbpaste"])
    if text:
        return ClipboardText(text)
    return ClipboardEmpty()


def _macos_image() -> ClipboardImage | None:
    """Extract PNG data via osascript (returned as a hex dump)."""
    out = _run_text(
        [
            "osascript",
            "-e",
            "set theData to the clipboard as «class PNGf»",
            "-e",
            "return theData",
        ]
    )
    if out is None:
        return None
    match = re.search(r"«data PNGf([0-9A-Fa-f]+)»", out.strip())
    if not match:
        return None
    try:
        return ClipboardImage(binascii.unhexlify(match.group(1)))
    except binascii.Error:
        return None


def _macos_files() -> ClipboardFiles | None:
    # Limitation (deliberate for now): the «class furl» coercion returns a
    # single file reference, so a Finder multi-select copy only yields the
    # first file; the rest are dropped. Documented per LIM-17 review.
    out = _run_text(
        [
            "osascript",
            "-e",
            "set theFiles to the clipboard as «class furl»",
            "-e",
            "return POSIX path of theFiles",
        ]
    )
    if out is None:
        return None
    path = Path(out.strip())
    if path.exists():
        return ClipboardFiles([path])
    return None


# -- Linux ---------------------------------------------------------------------


def _read_linux() -> ClipboardContent:
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        return _read_wl_paste()
    if shutil.which("xclip"):
        return _read_xclip()
    return ClipboardEmpty()


def _read_wl_paste() -> ClipboardContent:
    types = _run_text(["wl-paste", "--list-types"])
    if types is None:
        return ClipboardEmpty()
    mime_types = types.split()
    image_mime = next((t for t in mime_types if t.startswith("image/")), None)
    if image_mime is not None:
        data = _run_bytes(["wl-paste", "--type", image_mime])
        if data:
            ext = image_mime.split("/", 1)[1].split(";")[0] or "png"
            return ClipboardImage(data, ext)
    if "text/uri-list" in mime_types:
        out = _run_text(["wl-paste", "--type", "text/uri-list"])
        files = _paths_from_uri_list(out or "")
        if files:
            return ClipboardFiles(files)
    text = _run_text(["wl-paste", "--no-newline"])
    if text:
        return ClipboardText(text)
    return ClipboardEmpty()


def _read_xclip() -> ClipboardContent:
    types = _run_text(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"])
    if types is None:
        return ClipboardEmpty()
    targets = types.split()
    image_mime = next((t for t in targets if t.startswith("image/")), None)
    if image_mime is not None:
        data = _run_bytes(
            ["xclip", "-selection", "clipboard", "-t", image_mime, "-o"]
        )
        if data:
            ext = image_mime.split("/", 1)[1] or "png"
            return ClipboardImage(data, ext)
    if "text/uri-list" in targets:
        out = _run_text(
            ["xclip", "-selection", "clipboard", "-t", "text/uri-list", "-o"]
        )
        files = _paths_from_uri_list(out or "")
        if files:
            return ClipboardFiles(files)
    text = _run_text(["xclip", "-selection", "clipboard", "-o"])
    if text:
        return ClipboardText(text)
    return ClipboardEmpty()


def _paths_from_uri_list(uri_list: str) -> list[Path]:
    from urllib.parse import unquote, urlparse

    paths = []
    for line in uri_list.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("file://"):
            continue
        path = Path(unquote(urlparse(line).path))
        if path.exists():
            paths.append(path)
    return paths


# -- Windows -------------------------------------------------------------------

_PS_PROBE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$img = [Windows.Forms.Clipboard]::GetImage();"
    "if ($img -ne $null) { Write-Output 'image'; exit };"
    "$files = [Windows.Forms.Clipboard]::GetFileDropList();"
    "if ($files.Count -gt 0) { Write-Output 'files'; exit };"
    "if ([Windows.Forms.Clipboard]::ContainsText()) { Write-Output 'text'; exit };"
    "Write-Output 'empty'"
)

_PS_GET_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$img = [Windows.Forms.Clipboard]::GetImage();"
    "$ms = New-Object IO.MemoryStream;"
    "$img.Save($ms, [Drawing.Imaging.ImageFormat]::Png);"
    "[Convert]::ToBase64String($ms.ToArray())"
)

_PS_GET_FILES = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[Windows.Forms.Clipboard]::GetFileDropList() | ForEach-Object { Write-Output $_ }"
)

_PS_GET_TEXT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "Write-Output ([Windows.Forms.Clipboard]::GetText())"
)


def _powershell(script: str) -> str | None:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        return None
    return _run_text([exe, "-NoProfile", "-NonInteractive", "-Command", script])


def _read_windows() -> ClipboardContent:
    probe = _powershell(_PS_PROBE)
    if probe is None:
        return ClipboardEmpty()
    kind = probe.strip().splitlines()[-1] if probe.strip() else ""
    if kind == "image":
        out = _powershell(_PS_GET_IMAGE)
        if out:
            try:
                return ClipboardImage(base64.b64decode(out.strip()))
            except (binascii.Error, ValueError):
                pass
    elif kind == "files":
        out = _powershell(_PS_GET_FILES)
        if out:
            paths = [Path(line.strip()) for line in out.splitlines() if line.strip()]
            paths = [p for p in paths if p.exists()]
            if paths:
                return ClipboardFiles(paths)
    elif kind == "text":
        out = _powershell(_PS_GET_TEXT)
        if out:
            return ClipboardText(out.rstrip("\r\n"))
    return ClipboardEmpty()
