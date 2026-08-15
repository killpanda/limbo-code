# RFC LIM-60: Headless CLI 模式（Herdr 可控）

状态：实现随本 RFC 同 PR 落地
依赖：LIM-20（steer）、LIM-40（goal 闭环）、LIM-53（interrupt）、integrations/herdr（生命周期上报）

## 背景

Limbo 目前只有 Textual TUI 一个前端。TUI 对 Herdr 编排有三个结构性障碍（依据
herdr v0.8.0 `skills/herdr/SKILL.md` 的控制模型）：

1. **TUI 跑在 alternate screen**。`herdr agent read` 只能读主屏缓冲 + host
   scrollback；离开 alternate screen 的行不可恢复，完整回答拿不回来。
2. **TUI 的输入框是私有控件**。`herdr agent prompt` 往 pane 写"文本 + Enter"，
   在 TUI 下依赖焦点、输入框状态、kitty keyboard 等一堆巧合才能正常工作。
3. **交互控件不可见**。VerifyPicker 这类模态在屏幕文本里语义不明确，Herdr 无法
   可靠地把 `blocked` 归因到"等哪个决策"。

Herdr 实际控制一个 agent 的手段只有四种：

| 手段 | 命令 |
|---|---|
| 提交文本（原子发送 + Enter，遵守 pane 的 bracketed-paste 模式） | `herdr agent prompt <name> "..." --wait` |
| 逻辑按键 | `herdr agent send-keys <name> esc\|ctrl+c` |
| 读屏（visible / recent / recent-unwrapped / detection） | `herdr agent read <name>` |
| 生命周期状态同步（agent 自上报，我方已有 `HerdrReporter`） | `herdr agent wait <name> --until blocked` |

**结论：headless 模式必须是一个跑在主屏上的、行导向的纯文本 REPL**，
而不是 NDJSON-RPC——Herdr 没有到 pane 内进程的结构化 IPC 通道。
（NDJSON 机器协议作为未来扩展预留，见"非目标"。）

## 目标

`limbo --headless`：无 TUI 的前端，与 TUI 并列，复用全部领域层：

```
herdr agent prompt ──► stdin (行) ──► HeadlessFrontend ──► TurnPump ──► Agent
herdr agent read   ◄── stdout (纯文本渲染) ◄── PumpEvent 流
herdr agent wait   ◄── HerdrReporter(working/idle/blocked)   ← 已有机制
```

- `pump.py` / `agent.py` / `goal.py` / `sessions.py` / `integrations/` **零改动**；
  headless 只是又一个 frontend adapter（`pump.py` docstring 本就预留了这个角色）。
- 不 import textual（与 pump 同规约，测试锁死）。
- 会话文件与 TUI 完全互通：`--continue` / `--resume` / `/new` 同一套。

## 启动

```bash
limbo --headless [--workdir DIR] [--session-dir DIR] [--continue|--resume ID] [--model M]
```

`app.py` 复用现有的 config 加载、API key 校验、resume 解析；
`--headless` 走 `headless.run_headless()`，否则维持 Textual 路径。

## 输入（stdin）

**TTY 模式**（Herdr pane 内）：termios cbreak + 自行回显（关 ECHO，否则
`herdr agent read` 在屏上看不到输入内容——读屏是 Herdr 唯一的回读手段）。

- 启动时写 `\e[?2004h` 开启 bracketed paste，退出时 `\e[?2004l` 恢复。
  `herdr agent prompt` 检测到 2004 后会把多行文本包在 `\e[200~ … \e[201~` 里
  原子发送——粘贴块整体进入行缓冲，内部换行不触发提交；herdr 随后发送的
  Enter 完成提交。
- Enter 提交当前行；Backspace 编辑（CJK 宽字符感知）；可打印 UTF-8 字符回显。
- **Esc**（孤立 `0x1b`，与转义序列以 80ms 超时区分）→ 中断（= TUI 的 Esc）。
- **Ctrl+C**（cbreak + 关 ISIG，到达为字节 `0x03`）→ busy 时中断；idle 时退出。
- **Ctrl+D** 空行 → 退出。
- 未知转义序列（方向键、Alt 组合）丢弃，不污染行缓冲。

**管道模式**（stdin 非 TTY，如脚本/测试）：逐行读取，每行一条提交；
EOF = 等当前 turn 跑完后退出（`echo "任务" | limbo --headless` 语义自然）。

## 输出（stdout，纯文本渲染）

PumpEvent → 行的映射（全部走主屏，`herdr agent read --source recent-unwrapped`
可完整回读）：

| 事件 | 渲染 |
|---|---|
| TextDelta / ThinkingDelta | 原文流式写出 |
| ToolCallRequest | `⚙ name(参数摘要)` 一行（参数截断 100 字符） |
| ToolResultEvent | `✓/✗` + 输出（成功截断到 60 行；错误全文） |
| ErrorEvent | `✗ message` |
| SteerEvent | `➤ 插话已注入: …` |
| InterruptEvent | `⏹ 已打断` |
| CompactionEvent | 压缩结果一行（同 TUI 文案） |
| Goal* 事件 | 同 TUI 的中文文案（验收轮次/通过/耗尽/恢复） |
| UsageUpdate | 累积 tokens，turn 结束时随分隔行打印 |
| turn 结束 | `── limbo idle · 累计 tokens N ──` 分隔行 |

忙时提交的普通文本 → `agent.steer()`（与 TUI 输入框行为一致）；
忙时收到改写历史的命令（/goal /new /model /compact）→ 拒绝并提示（同 TUI）。

## 斜杠命令（v1）

`/goal <text>` · `/goal clear` · `/goal`（查询）· `/compact` · `/model [id]`
（无参=显示当前）· `/new` · `/help` · `/quit`
解析复用 `goal.parse_goal_args` / `model_switch.prepare_model_switch` +
`swap_llm_client`；未知命令报错行。Skills 调用不进 v1（见非目标）。

## Goal 验收确认：唯一的 blocked 场景

TUI 用 VerifyPicker 模态；headless 用"下一条输入即答案"：

1. 提议轮结束（goal active 且无 verify_command）→ 解析最后一条 assistant 消息的
   `<verify_proposal>`（复用 `parse_verify_proposal`）。
2. 打印编号候选 + 操作说明，向 Herdr 上报 `blocked`（message="等待确认 /goal
   验收命令"）。Herdr 侧 `herdr agent wait <name> --until blocked` 此刻返回，
   编排方 `agent read` 读到候选列表后 `agent prompt <name> "1"` 决策。
3. 下一条提交被消费为验收答案，**不进入对话历史**：
   - `1..N` → 采纳该候选；`/skip`（或空行）→ 跳过，保持单轮模式；
   - 其他任意一行 → 作为编辑后的验收命令（对应 TUI 的"编辑"分支）；
   - 此间 Esc/Ctrl+C → 等同 `/skip`；其他斜杠命令 → 报错提示，保持等待。
4. 采纳后 `pump.set_verify(cmd)` + 跑 `build_initial_prompt` 首轮，回报 `working`。

`<none/>`（模型判断无客观验收）→ 提示并保持单轮，不进 blocked。
提议轮期间收到 EOF（管道关闭）→ 打印候选后自动跳过、退出。

## 生命周期上报（复用 integrations）

| 时点 | 上报 |
|---|---|
| 启动就绪 | `report_session(id, path)` + `idle` |
| turn/compact 开始 | `working` |
| turn 结束 | `idle` |
| 等待验收确认 | `blocked` |
| `/new` | `report_session(...)` |
| 退出/信号 | `release()`（复用 `install_exit_hooks`，含 SIGHUP/SIGTERM/atexit） |

测试环境注意：conftest 已剥离 HERDR_* 环境变量；headless 测试显式注入
录制用 fake reporter，不碰真实环境。

## 中断语义（RFC LIM-53 平移）

Esc / busy 时 Ctrl+C → `agent.interrupt()` + `pump.cancel_verify()`。
被中断的 turn 不自动续跑（pump 既有语义）；剩余 steer 留在队列，
下一次提交时作为头部输入拼接。

## 测试接缝（预先约定）

1. **`HeadlessFrontend.run(inputs, out)`**：inputs 为 InputEvent 异步迭代器
   （Submit/Interrupt/Eof），out 为 `str -> None` 收集器。配真实 Agent +
   脚本化 FakeLLMClient（沿用 test_pump.py 模式）+ 录制 fake reporter。
   覆盖：单轮渲染、工具调用渲染、忙时 steer、interrupt、
   goal 提议→blocked→确认/跳过/自定义命令、/compact、/model、/new、
   EOF 等待 turn 完成后退出、忙时命令拒绝。
2. **行解析**：`parse_submission` / `parse_verify_answer` /
   `TerminalLineParser`（bracketed paste、Esc 超时、退格、Ctrl+C/D）——
   纯函数/纯字节测试。
3. **入口**：`--headless` flag 路由到 run_headless（沿用 test_cli.py mock 模式）。

不在接缝内：termios 字节级读写循环（手工验证）、真实 Herdr 联调。

## 非目标（v1）

- NDJSON 机器协议（`--json`）：等出现非 Herdr 编排方再做；渲染/输入已按
  可替换的 out/in 接缝隔离。
- Skills 斜杠调用、`/sessions` 切换器、`/export`、`!` bang 命令、图片粘贴。
- `[headless]` 配置段（渲染行数上限等先硬编码合理默认值）。

## 验收命令

```bash
make check
```

手工联调（Herdr pane 内）：

```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
herdr agent start worker --kind limbo --pane <pane-id> -- --headless
herdr agent prompt worker "读一下 AGENTS.md 并总结架构" --wait --timeout 120000
herdr agent read worker --source recent-unwrapped --lines 80
```
