"""Automatic skill review trigger (daily mode /learn)."""

from __future__ import annotations

import pytest

from limbo.agent import (
    CompactionEvent,
    InterruptEvent,
    SteerEvent,
    TextDelta,
    ToolCallRequest,
    ToolResultEvent,
)
from limbo.config import Config, DailySettings
from limbo.conversation import Conversation
from limbo.learn_trigger import LearnTrigger, is_correction, user_texts
from limbo.models import Message, ToolResult
from limbo.subagents.board import Outcome, Task


def _call(name="read", i=0):
    return ToolCallRequest(id=f"c{i}", name=name, arguments={})


def _result(name="read", ok=True, **arguments):
    result = (ToolResult(success=True, output="ok") if ok
              else ToolResult(success=False, error="boom"))
    return ToolResultEvent(id="c", name=name, result=result, arguments=arguments)


def _rounds(trigger: LearnTrigger, n: int, batch: int = 1) -> None:
    for _ in range(n):
        for i in range(batch):
            trigger.observe(_call(i=i))
        for _ in range(batch):
            trigger.observe(_result())
        trigger.observe(TextDelta(text="."))


def test_a_parallel_batch_is_one_round():
    trigger = LearnTrigger(every=10, cooldown=0)
    _rounds(trigger, 3, batch=5)
    assert trigger.rounds == 3


def test_busy_turn_without_a_signal_waits_for_the_fallback():
    trigger = LearnTrigger(every=10, cooldown=0)
    _rounds(trigger, 12)
    assert not trigger.end_turn("done")
    _rounds(trigger, 18)  # 30 = 3 × threshold
    assert trigger.end_turn("done")


@pytest.mark.parametrize("text", ["回答太长了", "以后别用表格", "Don't add emojis", "that's wrong"])
def test_corrections_are_signals(text):
    trigger = LearnTrigger(every=10, cooldown=0)
    _rounds(trigger, 10)
    trigger.observe_user_text(text)
    assert trigger.end_turn("done")


def test_plain_requests_are_not_corrections():
    assert not is_correction("帮我查一下明天北京的天气")
    assert not is_correction("summarize this paper")
    assert not is_correction("hi<system-reminder>don't mention this</system-reminder>")


def test_recovered_failure_and_skill_use_are_signals():
    trigger = LearnTrigger(every=2, cooldown=0)
    trigger.observe(_call("web_fetch"))
    trigger.observe(_result("web_fetch", ok=False))
    trigger.observe(_call("web_fetch"))
    trigger.observe(_result("web_fetch"))
    assert trigger.signals == {"recovered"}
    assert trigger.end_turn("done")

    trigger = LearnTrigger(every=2, cooldown=0)
    trigger.observe(_call())
    trigger.observe(_result("read", path="/x/skills/report/SKILL.md"))
    _rounds(trigger, 1)
    assert trigger.signals == {"skill_used"}
    assert trigger.end_turn("done")


def test_only_clean_turns_trigger():
    for state in ("idle", "error"):
        trigger = LearnTrigger(every=1, cooldown=0)
        _rounds(trigger, 5)
        trigger.observe_user_text("不要这样")
        assert not trigger.end_turn(state)


def test_skill_manage_resets_the_count():
    trigger = LearnTrigger(every=3, cooldown=0)
    _rounds(trigger, 5)
    trigger.observe_user_text("以后都这样")
    trigger.observe(_call("skill_manage"))
    trigger.observe(_result("skill_manage"))
    assert trigger.rounds == 0 and not trigger.signals
    assert not trigger.end_turn("done")


def test_cooldown_counts_turns_since_the_last_review():
    trigger = LearnTrigger(every=1, cooldown=3)
    trigger.review_started("t1")
    for turn in range(3):
        _rounds(trigger, 1)
        trigger.observe_user_text("太长了")
        assert trigger.end_turn("done") == (turn == 2)


def test_compaction_only_runs_are_not_turns():
    trigger = LearnTrigger(every=1, cooldown=3)
    trigger.review_started("t1")
    trigger.observe(CompactionEvent(compacted=True, trigger="manual"))
    trigger.end_turn("done")
    assert trigger.turns_since_review == 0


def _task(summary: str, status="done") -> Task:
    return Task(id="t1", title="复盘", status=status,
                outcome=Outcome(status=status, summary=summary))


def test_nothing_saved_backs_off_and_a_change_resets():
    trigger = LearnTrigger(every=10, cooldown=0)
    for expected in (2, 4, 8, 8):
        trigger.review_started("t1")
        trigger.settle({"t1": _task("Nothing to save.")})
        assert trigger.backoff == expected
    assert trigger.threshold == 80
    trigger.review_started("t1")
    trigger.settle({"t1": _task("weekly-report: add the tone rule")})
    assert trigger.backoff == 1
    trigger.review_started("t1")
    trigger.settle({"t1": _task("", status="failed")})
    assert trigger.backoff == 1


def test_user_texts_skip_subagent_updates():
    messages = [
        Message(role="user", content="<subagent_updates board=\"x\">..."),
        Message(role="assistant", content="不要"),
        Message(role="user", content="以后短一点"),
    ]
    assert user_texts(messages) == ["以后短一点"]


def test_old_config_key_still_read():
    assert DailySettings(learn_after_tool_calls=0).learn_after_rounds == 0
    assert DailySettings(learn_after_tool_calls=5, learn_after_rounds=7).learn_after_rounds == 7


# -- wired into Conversation.drive ---------------------------------------------


def _conversation(tmp_path, every=2, cooldown=0) -> Conversation:
    config = Config()
    config.daily.learn_after_rounds = every
    config.daily.learn_cooldown_turns = cooldown
    conv = Conversation(config=config, workdir=tmp_path, llm_client=None,  # type: ignore[arg-type]
                        session_dir=tmp_path / "sessions")
    conv.agent.set_daily_mode(True)
    return conv


async def _turn(conv: Conversation, user: str, events):
    async def run():
        conv.agent.messages.append(Message(role="user", content=user))
        for event in events:
            yield event

    await conv.drive(run(), lambda e: None)


def _reviews(conv: Conversation) -> list[Task]:
    assert conv.supervisor is not None
    return [t for t in conv.supervisor.board.tasks.values() if t.profile == "skill-reviewer"]


@pytest.mark.asyncio
async def test_drive_starts_a_review_only_with_a_signal(tmp_path):
    conv = _conversation(tmp_path)
    busy = [_call(), _result(), _call(), _result(), TextDelta(text="done")]
    await _turn(conv, "帮我整理一下资料", busy)
    assert _reviews(conv) == []
    await _turn(conv, "太长了，以后控制在三行", busy)
    assert len(_reviews(conv)) == 1
    conv.close()


@pytest.mark.asyncio
async def test_drive_skips_interrupted_turns_and_closed_headless(tmp_path):
    conv = _conversation(tmp_path)
    busy = [_call(), _result(), _call(), _result()]
    await _turn(conv, "不要用表格", [*busy, InterruptEvent(phase="llm_stream")])
    assert _reviews(conv) == []
    conv.auto_learn = False
    await _turn(conv, "不要用表格", busy)
    assert _reviews(conv) == []
    conv.close()


@pytest.mark.asyncio
async def test_steer_corrections_count(tmp_path):
    conv = _conversation(tmp_path)
    await _turn(conv, "查资料", [_call(), _result(), SteerEvent(id="s", text="别再搜英文的了"),
                               _call(), _result()])
    assert len(_reviews(conv)) == 1
    conv.close()
