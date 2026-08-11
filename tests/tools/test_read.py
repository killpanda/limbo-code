import tempfile
from pathlib import Path

import pytest

from limbo.tools.read import MAX_FILE_SIZE, ReadTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_read_success(workdir):
    (workdir / "hello.txt").write_text("hello world")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "hello.txt"})
    assert result.success is True
    assert result.output == "hello world"


def test_read_with_offset_and_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 2, "limit": 2})
    assert result.success is True
    assert result.output == "line2\nline3\n"


def test_read_outside_workdir(workdir, tmp_path):
    """No workdir fence: files outside the workdir are readable."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("external")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": str(outside / "data.txt")})
    assert result.success is True
    assert result.output == "external"


def test_read_dotenv_file(workdir):
    """No sensitive-file list: .env reads like any other file."""
    (workdir / ".env").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": ".env"})
    assert result.success is True
    assert result.output == "SECRET=1"


def test_read_rejects_negative_offset(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": -1})
    assert result.success is False
    assert "positive" in result.error.lower()


def test_read_rejects_negative_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "limit": -1})
    assert result.success is False
    assert "positive" in result.error.lower()


def test_read_follows_symlink_to_outside_file(workdir):
    """No workdir fence: a symlink pointing outside the workdir resolves."""
    import os

    outside = workdir.parent / f"read_outside_{workdir.name}"
    outside.mkdir()
    (outside / "data.txt").write_text("external")

    link = workdir / "outside_link"
    try:
        os.symlink(outside / "data.txt", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "outside_link"})
    assert result.success is True
    assert result.output == "external"


def test_read_broken_symlink_returns_invalid_path(workdir):
    """A broken symlink causes resolve() to raise OSError; the tool must handle it."""
    import os

    link = workdir / "broken_link"
    try:
        os.symlink("does_not_exist", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_read_symlink_loop_returns_invalid_path(workdir):
    """A symlink loop causes resolve() to raise RuntimeError; the tool must handle it."""
    import os

    link = workdir / "loop"
    try:
        os.symlink("loop", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "loop"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_read_truncates_output_by_byte_budget(workdir):
    """Multi-byte characters count against the byte budget, not character count."""
    (workdir / "big.txt").write_text("é" * (60 * 1024))  # 120 KB of UTF-8 bytes
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "big.txt"})
    assert result.success is True
    assert len(result.output.encode("utf-8")) <= 50 * 1024 + 200
    assert "truncated" in result.output.lower()
    assert "Use offset/limit to read more" in result.output


def test_read_truncates_output_by_line_budget(workdir):
    """Head truncation keeps the first 2000 lines and reports the total."""
    (workdir / "lines.txt").write_text("\n".join(f"line-{i}" for i in range(1, 3001)))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt"})
    assert result.success is True
    assert "line-2000" in result.output
    assert "line-2001" not in result.output
    assert "showing lines 1-2000 of 3000" in result.output
    assert "Use offset/limit to read more" in result.output


def test_read_truncation_hint_uses_absolute_line_numbers(workdir):
    """With offset, the hint must refer to file line numbers, not slice-relative ones."""
    (workdir / "lines.txt").write_text("\n".join(f"line-{i}" for i in range(1, 3001)))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 500})
    assert result.success is True
    assert "line-500" in result.output
    assert "line-2499" in result.output
    assert "line-2500" not in result.output
    assert "showing lines 500-2499 of 3000" in result.output


def test_read_allows_offloaded_bash_output_in_tmpdir(workdir):
    """Bash offloading promises the model can retrieve full output with read."""
    offload = Path(tempfile.gettempdir()) / "limbo-output-testdeadbeef.log"
    offload.write_text("full bash output")
    try:
        tool = ReadTool(workdir=workdir)
        result = tool.execute({"path": str(offload)})
        assert result.success is True
        assert result.output == "full bash output"
    finally:
        offload.unlink(missing_ok=True)


def test_read_rejects_files_above_max_size(workdir):
    """Files larger than the generous cap must be refused before reading."""
    big = workdir / "huge.txt"
    big.write_text("x" * (MAX_FILE_SIZE + 1))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "huge.txt"})
    assert result.success is False
    assert "too large" in result.error.lower()


# -- image attachments (pi parity) ----------------------------------------------

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00" * 32
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF_BYTES = b"GIF89a" + b"\x00" * 32
WEBP_BYTES = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 32
BMP_BYTES = b"BM" + b"\x00" * 32


@pytest.mark.parametrize(
    "data, mime",
    [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (GIF_BYTES, "image/gif"),
        (WEBP_BYTES, "image/webp"),
        (BMP_BYTES, "image/bmp"),
    ],
)
def test_read_image_returns_attachment(workdir, data, mime):
    (workdir / "pic.bin").write_bytes(data)
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "pic.bin"})
    assert result.success is True
    assert mime in result.output
    assert result.attachments is not None
    assert len(result.attachments) == 1
    attachment = result.attachments[0]
    assert attachment.kind == "image"
    assert attachment.mime == mime
    assert attachment.path.endswith("pic.bin")


def test_read_image_sniffed_by_magic_bytes_not_extension(workdir):
    (workdir / "photo.txt").write_bytes(PNG_BYTES)
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "photo.txt"})
    assert result.success is True
    assert result.attachments is not None
    assert result.attachments[0].mime == "image/png"


def test_read_oversized_image_rejected(workdir):
    big = workdir / "big.png"
    with big.open("wb") as f:
        f.write(PNG_BYTES)
        f.seek(20 * 1024 * 1024 + 1)
        f.write(b"\x00")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "big.png"})
    assert result.success is False
    assert "too large" in result.error.lower()


def test_read_text_file_has_no_attachments(workdir):
    (workdir / "a.txt").write_text("hello\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "a.txt"})
    assert result.success is True
    assert result.attachments is None
