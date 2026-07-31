import tempfile
from pathlib import Path

import pytest

from limbo.tools.edit import EditTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_edit_success(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 42\ny = 2\n"


def test_edit_requires_exact_match(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x=1", "new_text": "x=42"},
    )
    assert result.success is False
    assert "not found" in result.error.lower()


def test_edit_non_unique(workdir):
    (workdir / "a.py").write_text("x = 1\nx = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
    )
    assert result.success is False
    assert "unique" in result.error.lower()


def test_edit_outside_workdir(workdir):
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "../x.py", "old_text": "a", "new_text": "b"},
    )
    assert result.success is False
    assert "outside" in result.error.lower()


def test_edit_no_change(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 1"},
    )
    assert result.success is False
    assert "no changes" in result.error.lower()


def test_edit_rejects_empty_old_text(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "", "new_text": "x = 42"},
    )
    assert result.success is False
    assert "old_text cannot be empty" in result.error.lower()


def test_edit_handles_os_error(workdir):
    target = workdir / "a.py"
    target.write_text("x = 1\n")
    target.chmod(0o444)
    try:
        tool = EditTool(workdir=workdir)
        result = tool.execute(
            {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
        )
        assert result.success is False
        assert "could not write file" in result.error.lower()
    finally:
        target.chmod(0o644)


def test_edit_handles_read_error(workdir):
    target = workdir / "a.py"
    target.write_text("x = 1\n")
    target.chmod(0o000)
    try:
        tool = EditTool(workdir=workdir)
        result = tool.execute(
            {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
        )
        assert result.success is False
        assert "could not read file" in result.error.lower()
    finally:
        target.chmod(0o644)


def test_edit_rejects_symlink_escape(workdir):
    outside = workdir.parent / "outside_edit_target.py"
    outside.write_text("x = 1\n")
    link = workdir / "escape_link.py"
    link.symlink_to(outside)
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "escape_link.py", "old_text": "x = 1", "new_text": "x = 42"},
    )
    assert result.success is False
    assert "outside" in result.error.lower()
    assert outside.read_text() == "x = 1\n"


def test_edit_broken_symlink_returns_invalid_path(workdir):
    link = workdir / "broken_link.py"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "broken_link.py", "old_text": "a", "new_text": "b"},
    )
    assert result.success is False
    assert "invalid path" in result.error.lower()


# -- edits[] multi-replacement (pi parity) ------------------------------------


def test_edit_multiple_edits_applied_against_original(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "x = 1", "new_text": "x = 10"},
                {"old_text": "z = 3", "new_text": "z = 30"},
            ],
        }
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 10\ny = 2\nz = 30\n"
    assert "2 replacements" in result.output


def test_edit_edits_matched_against_original_not_incrementally(workdir):
    """Each edit sees the ORIGINAL file: an edit may target text another
    edit's new_text would introduce, and still matches the old content."""
    (workdir / "a.py").write_text("aaa\nbbb\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "aaa", "new_text": "bbb"},
                {"old_text": "bbb", "new_text": "ccc"},
            ],
        }
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "bbb\nccc\n"


def test_edit_overlapping_edits_rejected(workdir):
    (workdir / "a.py").write_text("abcdef\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "abcd", "new_text": "wxyz"},
                {"old_text": "cdef", "new_text": "opqr"},
            ],
        }
    )
    assert result.success is False
    assert "overlap" in result.error.lower()
    assert (workdir / "a.py").read_text() == "abcdef\n"


def test_edit_multi_edit_not_found_names_index(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "x = 1", "new_text": "x = 2"},
                {"old_text": "nope", "new_text": "y"},
            ],
        }
    )
    assert result.success is False
    assert "edits[1]" in result.error


def test_edit_multi_edit_non_unique_names_index(workdir):
    (workdir / "a.py").write_text("dup\ndup\nunique\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "unique", "new_text": "u"},
                {"old_text": "dup", "new_text": "d"},
            ],
        }
    )
    assert result.success is False
    assert "edits[1]" in result.error
    assert "unique" in result.error.lower()


def test_edit_rejects_empty_edits_list(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute({"path": "a.py", "edits": []})
    assert result.success is False


def test_edit_multi_edit_rejects_empty_old_text(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": [
                {"old_text": "x = 1", "new_text": "x = 2"},
                {"old_text": "", "new_text": "y"},
            ],
        }
    )
    assert result.success is False
    assert "edits[1]" in result.error


def test_edit_accepts_edits_as_json_string(workdir):
    """Some models serialize the edits array as a JSON string."""
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {
            "path": "a.py",
            "edits": '[{"old_text": "x = 1", "new_text": "x = 2"}]',
        }
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 2\n"


# -- BOM / CRLF normalization ---------------------------------------------------


def test_edit_preserves_bom(workdir):
    (workdir / "a.py").write_bytes("﻿x = 1\n".encode("utf-8"))
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}
    )
    assert result.success is True
    assert (workdir / "a.py").read_bytes() == "﻿x = 2\n".encode("utf-8")


def test_edit_preserves_crlf_line_endings(workdir):
    (workdir / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}
    )
    assert result.success is True
    assert (workdir / "a.py").read_bytes() == b"x = 2\r\ny = 2\r\n"


def test_edit_crlf_file_matches_multiline_lf_old_text(workdir):
    """Models send LF-only old_text; it must still match a CRLF file."""
    (workdir / "a.py").write_bytes(b"x = 1\r\ny = 2\r\nz = 3\r\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1\ny = 2", "new_text": "a\nb"}
    )
    assert result.success is True
    assert (workdir / "a.py").read_bytes() == b"a\r\nb\r\nz = 3\r\n"
