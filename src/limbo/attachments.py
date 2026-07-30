"""Attachment policy: how user attachments reach the model.

Owns the degrade decisions between a submitted attachment and the LLM
request. Image attachments become multimodal blocks only for vision models
(the clients encode them — see ``llm.scaffold.encode_image_data``);
everything else degrades to text: small text files inline, the rest as
path references the model can open with read. Nothing is silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from limbo.models import Attachment

# File attachments at or below this size are inlined into the user message
# (UTF-8 decodable); larger or binary files are referenced by path.
ATTACHMENT_INLINE_MAX_BYTES = 50_000


def read_inline_text(path: Path) -> str | None:
    """The file's text if it qualifies for inline embedding, else None."""
    try:
        if not path.is_file() or path.stat().st_size > ATTACHMENT_INLINE_MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def build_user_content(
    user_input: str, attachments: list[Attachment], *, vision: bool
) -> tuple[str, list[Attachment]]:
    """Combine the typed text with attachments into message content.

    Returns ``(content, images)``: images only when the current model
    supports vision (they become multimodal blocks in the clients);
    everything else degrades to text notes — small text files inline,
    the rest as path references the model can open with read. Nothing
    is silently dropped.
    """
    images: list[Attachment] = []
    notes: list[str] = []
    for attachment in attachments:
        path = Path(attachment.path)
        if attachment.kind == "image":
            if vision and path.exists():
                images.append(attachment)
            elif vision:
                notes.append(f"[图片 {attachment.name} 文件已不存在：{path}]")
            else:
                notes.append(
                    f"[图片 {attachment.name} 已保存到 {path}；"
                    f"当前模型不支持图像输入，无法直接查看]"
                )
            continue
        # File attachment: inline small text files, reference the rest.
        text = read_inline_text(path)
        if text is not None:
            notes.append(f"文件 {attachment.name} 的内容：\n```\n{text}\n```")
        else:
            notes.append(f"[文件 {attachment.name} 位于 {path}，可用 read 工具查看]")
    if not notes:
        return user_input, images
    return user_input + "\n\n" + "\n".join(notes), images
