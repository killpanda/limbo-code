"""The ``ask_user`` tool (daily mode): ask the user and end the turn.

The question (and numbered options) is shown to the user; their next
message is the answer. The turn ends right after the call, so the model
never guesses an answer.
"""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError

MAX_OPTIONS = 6


def format_question(arguments: dict[str, Any]) -> str:
    """What the user sees: the question and its numbered options."""
    lines = [f"❓ {str(arguments.get('question') or '').strip()}"]
    options = arguments.get("options") or []
    if isinstance(options, list):
        lines += [f"  {i}. {o}" for i, o in enumerate(options, 1)]
        if options:
            lines.append("  回复编号选择，或直接输入你的回答")
    return "\n".join(lines)


class AskUserTool(BaseTool):
    name = "ask_user"
    # Changes nothing: safe to replay after a crash and allowed under /plan.
    replay = "safe"
    description = (
        "Ask the user a question when you need their decision or a fact only they "
        "know, instead of guessing. Give up to 6 short options when the choice is "
        "between known alternatives (they may still answer freely). Your turn ends "
        "after this call; their next message is the answer, and a bare number "
        "picks that option."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "One clear question."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Up to {MAX_OPTIONS} short options (optional).",
            },
        },
        "required": ["question"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question must be a non-empty string")
        options = arguments.get("options") or []
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            raise ToolError("options must be a list of strings")
        if len(options) > MAX_OPTIONS:
            raise ToolError(f"at most {MAX_OPTIONS} options")
        return ToolResult(
            success=True,
            output=(
                format_question(arguments)
                + "\n\nAsked the user; end your turn now and wait for their reply."
            ),
            end_turn=True,
        )
