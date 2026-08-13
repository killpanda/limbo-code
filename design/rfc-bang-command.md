# RFC: `!` 前缀直接执行 bash 命令（bang command）

- 作者：code-monkey
- 状态：draft
- 日期：2026-08-13

## Problem / Motivation

Limbo 输入框提交的所有内容（非 slash 命令）都会作为聊天消息发给 LLM。日常使用中有一类高频诉求：用户只想快速跑一条 shell 命令看一眼结果（`!git status`、`!ls -la`、`!pytest -x` 的最后几行），并不需要消耗一轮 LLM 对话、也不希望这条命令污染对话上下文。pi 等同类工具有成熟的 `!` 前缀惯例：直接执行、结果内联展示。

现状：

- 输入提交链路：`InputWidget.action_submit()`（`src/limbo/ui/widgets/input.py`）发出 `UserSubmitted` 消息 → `MainScreen.on_user_submitted()`（`src/limbo/ui/screens/main.py`）分派：`_editing_verify` 分支 → `/` slash 命令（`ui/commands.py`）→ `/`-path 回退（`ui/path_input.py`，自动 attach）→ steer 队列 / 新 turn。
- bash 执行能力已有：`BashTool`（`src/limbo/tools/bash.py`）提供流式收集、tail 截断 + spill 临时文件、超时、进程组 kill（ESC 打断）。
- 结果展示已有：`ChatWidget.add_tool_card()` + `ToolCard`（`src/limbo/ui/widgets/`），状态机 running → success | error | cancelled，`command` 是 header 摘要的优先键（`_SUMMARY_KEYS`）。

即：执行与展示的基础设施都已存在，缺的只是输入层的 `!` 分派和一条"不进 agent 历史的本地执行"通路。

## Non-Goals

- 不把命令/结果写入 agent 对话历史，不作为上下文发给 LLM（包括后续 turn）；不进 session JSONL，resume 后不可见。
- 不做 bang 结果的 LLM 后处理（不让模型解释输出）。
- v1 不支持 ESC 取消正在运行的 bang 命令（命令跑完为止； bash 工具无默认超时，与 LLM 侧行为一致）。
- 不改 slash 命令、`/`-path 回退、steer 队列的任何现有行为。
- 不做 bang 输出流式刷新（卡片在命令结束后一次性渲染，复用 ToolResult 路径；`BashTool.run()` 本身就是收集完才返回）。
- 不处理 `/new`、`/sessions` 切换后仍在运行的 bang 的卡片归属（旧卡片随 `chat.clear()` 移除，迟到的结果静默丢弃，见技术视角）；进程本身由 `on_unmount` 兜底（app 退出时 kill）。

## Goal

可验证的成功标准：

1. 输入 `!echo hello` 回车：不产生 LLM 请求（`agent.messages` 不增长、trace 无 `llm_request`），对话流出现一张 bash 工具卡片，header 显示命令，展开可见 stdout `hello`。
2. 输入 `!false`（或任意非零退出命令）：卡片呈 error 态，展开 body 含完整输出及 `Command failed with exit code N.` 信息。
3. 输入孤立的 `!`（或 `!` 后全是空白）：不执行任何进程，对话流出现用法提示（info 行），不进历史。
4. 输入以空格开头的 ` !foo`：与 `/` 的 `force_text` 转义一致，作为普通聊天消息发给 LLM。
5. agent turn 运行中（busy）提交 `!cmd`：立即并行执行，不进 steer 队列、不打断 turn；turn 的 ESC 打断（`registry.cancel_active()`）不会误杀 bang 进程，且 `InterruptEvent` 触发的 `chat.cancel_running_tool_cards()` **不会把仍在运行的 bang 卡片标记为已取消**（卡片保持 running，命令结束后正常转 success/error）。
6. `!` 命令进入输入框的 ↑↓ 历史（走现有 `action_submit` 路径即可），长输出复用 bash 工具的截断 + spill 提示。
7. app 退出时仍在运行的 bang 进程被 kill，不泄漏孤儿进程。
8. `make check` 全绿，新增 UI 测试覆盖上述 1–5 及 Test Plan 列出的场景。

存疑点（goal unclear，待马越确认，默认按"不做"实现）：

- 是否需要像 pi 那样把 `!` 命令的输出**附加进后续对话上下文**（让模型看得见）？本任务描述明确"不当作聊天消息发给 LLM"，默认结果也不进历史；若后续想要"执行并告知模型"，可作为独立增量（例如 `!!` 变体或开关）。
- bang 命令是否需要写 trace（`trace.log("bang_command", ...)`）做审计？默认做（一行日志，成本极低），但若认为 trace 只记录 LLM 会话事件可去掉。

## Product / Tech Design

### 用户视角

- 输入框敲 `!` 开头的单行/多行文本并回车 → 不发送给 LLM，直接以 bash 执行（工作目录 = Limbo workdir），对话流中先出现用户消息行（`!cmd` 原样回显，与 slash 命令一致），随后出现工具卡片：`… bash <command>` → 结束后 `✓`（退出码 0）或 `✗`（非零/异常），点击展开看输出；`ctrl+o` 全局展开/收起同样适用。
- `!` 后无内容 → info 提示 `用法：! <bash 命令>`。
- 想发送字面 `!` 开头的文本给 LLM：前面加一个空格（与 `/` 的既有转义一致，hint 文案与错误提示沿用同一套话术）。
- busy（turn/compact 进行中）时 bang 照常可用，行为类似 `/btw`、`/2048`：独立于当前任务。
- 输入框 hint 行文案追加 `! 命令`（`compose()` 中的 Static）。

### 技术视角

**分派层：MainScreen，与 `/` 完全同构。** `!` 解析放在 `MainScreen.on_user_submitted()`，不放在 InputWidget / pump / agent：

- InputWidget 保持"提交原文"的单一职责（刚重构完，不再加分支）；它现有的 `force_text` 转义判断从"仅 `/`"扩展为"`/` 或 `!`"（`ui/widgets/input.py` 的 `action_submit` 一处改动：`raw.lstrip()[:1] in ("/", "!")`）。
- pump/agent 完全不感知 bang：不进 `TurnPump`（无 single-flight 约束、无 steer 交互），`agent.py` 零改动。
- 分派顺序（`on_user_submitted` 内）：`_editing_verify` 分支 **优先于** bang 分支——注意该分支内 slash 与 bang 的语义并不对称：slash 输入是「退出编辑模式、作为命令执行」，而一切非 slash 文本（包括 `!pytest`）都按字面成为验收命令（edit 模式的本意就是编辑一条 shell 命令，`!` 在其中没有特殊含义；`!pytest` 作为验收命令大概率执行失败，用户自证，与现状「任何文本原样成为验收命令」一致）。之后是 bang 分支（`text.startswith("!") and not event.force_text`）；再之后才是现有 `/` 分支。`!` 与 `/` 首字符互斥，与 slash 命令、`/`-path 回退**无优先级冲突**；slash menu 只匹配 `/` 开头，也不受影响。

**执行：复用 `BashTool`，但用 screen 持有的独立实例。** 关键决策——不复用 `agent.registry` 里的那个 `BashTool`：

- turn 的 ESC 打断走 `registry.cancel_active()` → `BashTool.cancel()` 会杀掉该实例所有在途进程；若 bang 复用同一实例，用户打断 turn 会误杀自己手动跑的命令。screen 持有 `BashTool(workdir=self.workdir)` 私有实例即天然隔离。
- 不受 `[tools] bash_enabled` 影响：该开关控制的是"暴露给 LLM 的工具面"，bang 是用户本人的显式动作（与 LLM 自主调用 bash 的风险模型不同）。
- 调用方式：`await asyncio.to_thread(bang_tool.execute, {"command": cmd})`（与 `ToolRegistry.execute` 相同的模式，bash 不在 mutation queue 内，无需锁），包在 `run_worker` 里，每次提交一个独立 worker，天然支持并行多条 bang。
- 超时：不传 `timeout`（与 bash 工具默认一致，无超时）；截断、spill、`[stderr]` 标注全部继承。
- worker 兜底：`_run_bang` 整体包 `try/except Exception`——`BashTool.execute()` 只捕获 `ToolError`，其余异常（如 trace 写入失败、意外的 OSError）不应让 worker 静默死掉而卡片永远 running；异常时 `card.set_error(f"内部错误：{e}")`。
- 生命周期：`MainScreen.on_unmount` 增加 `self._bang_tool.cancel()`（进程组 kill），app 退出时不泄漏在途 bang 进程——现有 `on_unmount` 只关 LLM client 和 trace，bang 是唯一新增的外部进程来源。

**结果进对话流：复用 ToolCard，无新消息类型。**

- 回显：`chat.add_user_message(text)`（`!cmd` 原文，**不带 attachments**——粘贴/附件是给 LLM 消息的载荷，bang 不发给 LLM；`InputWidget` 展开粘贴标记后的纯文本照常作为命令的一部分，但若用户先 attach 了图片再敲 `!cmd`，attachment 被静默丢弃）。
- 卡片：`chat.add_tool_card(tool_id=f"bang-{uuid}", name="bash", arguments={"command": cmd}, agent_owned=False)` —— `ToolCard._summary()` 的 `_SUMMARY_KEYS` 已含 `command`，header 摘要零改动。
- **bang 卡片标记（B1 修复）**：`add_tool_card` 增加 `agent_owned: bool = True` 参数，落到 `ToolCard` 的同名属性；`ChatWidget.cancel_running_tool_cards()` 只把 `agent_owned` 的 running 卡片置 cancelled。理由（相比 screen 侧维护 bang tool_id 集合）：「running 卡片 ⇔ 等待 agent 工具结果」是 chat 组件自己的不变量，标记随卡片生命周期走（`tool_cards` 字典、`clear()` 一并清理），screen 侧平行维护一个集合会在 `/new`、`/sessions` 切换等 clear 路径上产生同步负担和漂移风险。这样 turn 被打断时 bang 卡片保持 running，`set_cancelled` 的「no result will ever arrive」不变量不被打破——bang 的结果一定会在进程结束时到达并正常转 success/error。
- 结果映射：`result.success` → `card.set_success(result.output or "")`；失败（非零退出/超时/启动异常）→ `card.set_error(result.output or result.error or "Command failed.")`。注意失败时要用 `output` 而非 `error` 填 body：`BashTool` 把 `Command failed with exit code N.` 置于 output 首行、完整 stdout/stderr 在后，只显示 `error` 会丢掉全部输出。
- 状态栏：v1 不动（卡片自带 running 态 `…`）；busy 时也不去抢 turn 的状态文案。
- 迟到结果的容忍：`/new`、`/sessions` 切换后 bang 仍在跑，旧卡片已被 `chat.clear()` 卸载；此时 `set_success`/`set_error` 走 `ToolCard` 现有的 `NoMatches` 静默路径，不崩溃、结果丢弃（符合「bang 不进历史」的定位）。
- ESC 路由：不变。bang 进程属独立实例，turn ESC 不会波及；v1 也没有"ESC 取消 bang"的路径（Non-Goal）。

**持久化 / trace：** 不进 `agent.messages`、不进 session JSONL（resume 后无 bang 痕迹，与"历史不重建工具卡片"的既有策略一致）。trace 追加一条 `bang_command` 事件（command/exit_code/success），在 worker 完成处 best-effort 写入：worker 启动时捕获当时的 `self.agent.trace` 对象（而不是完成时再取 `self.agent.trace`，避免 `/new` 后写到新会话的 trace）；写入本身包 try/except，因为旧 agent `close()` 后 trace 句柄已释放。

**hint 行：** 文案追加 `· ! 命令`。该行是单行 `Static`，窄终端下超长截断/换行是既有行为，本次只加几个字符，不引入新的折行问题；如快照因此变化按既有 `--snapshot-update` 流程处理。

**涉及文件清单：**

| 文件 | 改动 |
| --- | --- |
| `src/limbo/ui/screens/main.py` | `on_user_submitted` 增加 bang 分支；新增 `_run_bang(command)` worker（含 try/except 兜底）；`__init__` 持有私有 `BashTool` 实例；`on_unmount` 增加 `cancel()`；hint 文案 |
| `src/limbo/ui/widgets/input.py` | `force_text` 转义条件扩展到 `!`（一处） |
| `src/limbo/ui/widgets/chat.py` | `add_tool_card()` 增加 `agent_owned` 参数；`cancel_running_tool_cards()` 跳过非 agent-owned 卡片 |
| `src/limbo/ui/widgets/tool_card.py` | `ToolCard` 增加 `agent_owned` 属性（默认 True） |
| `src/limbo/tools/bash.py` | 不改（仅被实例化复用） |
| `src/limbo/pump.py` / `agent.py` / `models.py` | 不改 |
| `tests/ui/test_bang_ui.py`（新增） | 见 Test Plan |
| `tests/ui/test_widgets.py` 或 chat 相关测试 | `cancel_running_tool_cards` 跳过 bang 卡片的单元用例 |
| `tests/ui/test_input_paste.py` 或 input 相关测试 | `force_text` 扩展的回归用例 |

**兼容性：** 无迁移。唯一行为变化：此前以 `!` 开头的消息会发给 LLM，现在被本地拦截执行；转义 hatch（前导空格）保留旧行为。

## Test Plan

自动化（`tests/ui/test_bang_ui.py`，textual `run_test()` pilot，参照 `test_path_input.py` / `test_steer_ui.py` 的既有模式；LLM client 用现有 mock/fixture）：

1. `!echo hello` → 无 `llm_request`（断言 mock client 零调用 / `agent.messages` 长度不变），聊天流含 bash 卡片且 body 含 `hello`。
2. `!exit 3` → 卡片 error 态，body 含 `exit code 3`。
3. 孤立 `!` → info 提示出现，无卡片、无进程。
4. ` !echo hi`（前导空格）→ 走普通消息路径（mock client 收到文本 `!echo hi`）。
5. busy 时提交 `!echo hi` → steer 队列计数不变，卡片照常完成；同时断言对 agent 侧 `BashTool` 实例无交互（隔离性，可用 monkeypatch spy）。
6. **B1 回归**：bang 运行中打断一个 turn（注入 `InterruptEvent` 或走 ESC 路径）→ `cancel_running_tool_cards()` 后 bang 卡片仍为 running 态，agent 侧 running 卡片被置 cancelled；bang 进程结束后卡片正常转 success。另配一个 widget 级单元测试：`add_tool_card(agent_owned=False)` 的卡片不被 `cancel_running_tool_cards()` 触碰。
7. `_editing_verify` 模式下提交 `!pytest -x` → 不执行 bang、不出现 bash 卡片，文本按字面成为验收命令（走 `_accept_verify`，断言 `pump.status().verify_command == "!pytest -x"`）。
8. worker 兜底：monkeypatch 私有 `BashTool.execute` 抛非 `ToolError` 异常 → 卡片转 error 态，无未捕获 worker 异常。
9. 输入历史：提交 `!ls` 后按 ↑ 能召回。

手测路径：

- `!git status`、`!ls -la`（成功卡片 + 展开）；`!grep -r xxx /nonexistent`（非零退出卡片）；超长输出（`!seq 1 5000`）验证截断 + spill 提示；turn 运行中敲 bang 验证并行、ESC 打断 turn 后 bang 卡片不被误标 cancelled 且进程跑完；`!sleep 60` 后退出 app 验证无孤儿进程。

不需要 snapshot 测试：卡片样式复用 ToolCard 现有快照，无新视觉元素（若 hint 文案改动影响现有快照，按既有流程 `--snapshot-update` 并审查 diff）。
