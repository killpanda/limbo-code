"""Tests for attachment policy (how attachments reach the model)."""

from __future__ import annotations

from limbo.attachments import (
    ATTACHMENT_INLINE_MAX_BYTES,
    build_user_content,
    read_inline_text,
)
from limbo.models import Attachment


def test_image_kept_as_block_for_vision_model(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    attachment = Attachment(
        kind="image", name="shot.png", path=str(image), mime="image/png"
    )
    content, images = build_user_content("看图", [attachment], vision=True)
    assert content == "看图"
    assert images == [attachment]


def test_image_degrades_to_path_note_for_non_vision_model(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    attachment = Attachment(
        kind="image", name="shot.png", path=str(image), mime="image/png"
    )
    content, images = build_user_content("看图", [attachment], vision=False)
    assert images == []
    assert str(image) in content
    assert "不支持图像输入" in content


def test_missing_image_noted_but_not_dropped(tmp_path):
    attachment = Attachment(
        kind="image", name="gone.png", path=str(tmp_path / "gone.png")
    )
    content, images = build_user_content("看图", [attachment], vision=True)
    assert images == []
    assert "文件已不存在" in content


def test_small_text_file_inlined(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("remember the milk")
    attachment = Attachment(kind="file", name="note.txt", path=str(note))
    content, images = build_user_content("总结这个文件", [attachment], vision=False)
    assert "remember the milk" in content
    assert images == []


def test_large_or_binary_file_referenced_by_path(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (ATTACHMENT_INLINE_MAX_BYTES + 1))
    attachment = Attachment(kind="file", name="big.bin", path=str(big))
    content, images = build_user_content("分析这个文件", [attachment], vision=False)
    assert str(big) in content
    assert "read 工具" in content
    assert images == []


def test_no_attachments_passes_text_through():
    content, images = build_user_content("plain", [], vision=True)
    assert content == "plain"
    assert images == []


def test_read_inline_text_limits(tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("hello")
    assert read_inline_text(small) == "hello"
    big = tmp_path / "big.txt"
    big.write_text("x" * (ATTACHMENT_INLINE_MAX_BYTES + 1))
    assert read_inline_text(big) is None
    assert read_inline_text(tmp_path / "missing.txt") is None
    assert read_inline_text(tmp_path) is None  # a directory is not a file
