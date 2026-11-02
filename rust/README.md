# limbo（Rust 版）

这是 Limbo 终端编程 agent 的 Rust 重写版，和 Python 版放在同一个仓库里。它读写同一个 `~/.limbo`：配置、`auth.json` 和 JSONL 会话两边通用，所以可以随时在两个版本之间切换、互相 `--resume`。

写这个版本的另一个目的是**学 Rust**。所以代码里有不少面向学习者的注释，会解释所有权、`async`、trait 对象这类概念在这里为什么这样用。

## 跑起来

```bash
cd rust
cargo run --release                 # 打开 TUI
cargo run --release -- -c           # 继续当前目录最近的会话
cargo run --release -- -r <id>      # 恢复指定会话（id 可以只写前缀）
cargo run --release -- -p "解释一下 src/agent.rs"   # 只跑一轮，把回答打印出来
cargo run --release -- --headless   # 纯文本 REPL（加 --json 输出事件流）
cargo run --release -- models       # 列出可用模型
cargo run --release -- sessions     # 列出当前目录的会话（--all 列出全部）
cargo test                          # 跑测试
```

需要 Rust 1.88 以上（edition 2024）。`grep` 工具依赖系统里装好的 `rg`（ripgrep）。

## TUI 操作

| 按键 | 作用 |
| --- | --- |
| Enter / Shift+Enter（或 Ctrl+J） | 发送 / 换行 |
| Esc、Ctrl+C | 打断当前回合；空闲时连按两次 Ctrl+C 退出 |
| agent 忙时按 Enter | 把消息排进队列，作为“插话”发给 agent |
| ↑ / ↓ | 翻输入历史 |
| PageUp / PageDown、Ctrl+↑/↓ | 滚动对话 |
| Shift+Tab | 切换思考强度 |
| Ctrl+O / Ctrl+T | 展开或折叠工具输出 / 显示或隐藏思考过程 |
| Ctrl+R | 会话列表 |
| `/` | 命令面板：`/model` `/thinking` `/plan` `/compact` `/fork` `/handoff` `/sessions` `/export` `/theme` `/mouse` `/new` … |

主题有 `neon`、`aurora`、`ember` 三套，用 `/theme` 切换，或者在配置里写 `[ui] theme = "aurora"`。动效可以用 `[ui] motion = "full" | "reduced" | "off"` 调整（通过 SSH 连接时默认是 reduced）。

## 代码地图（推荐的阅读顺序）

| 文件 | 内容 | 能学到的 Rust 知识 |
| --- | --- | --- |
| `src/models.rs` | 消息、工具调用这些数据类型 | `serde` 派生、`enum`、`Option` |
| `src/config.rs`、`auth.rs` | 读取 `~/.limbo/config.toml` | 错误处理、`toml_edit` 保留注释改写 |
| `src/llm/` | 三种 LLM 协议，以及共用的 SSE 解析 | trait 对象 `Box<dyn Decoder>`、`mpsc` 通道、流 |
| `src/tools/` | bash/read/write/edit/ls/find/grep | `#[async_trait]`、`Arc<dyn Tool>`、`tokio::select!`、进程组 |
| `src/session.rs`、`history.rs` | JSONL 日志、崩溃恢复 | `#[serde(flatten)]` 保留未知字段、原子写文件 |
| `src/agent.rs` | agent 主循环 | 把 `Agent` *move* 进任务再通过 `JoinHandle` 取回所有权 |
| `src/subagent.rs` | `task` 工具（子 agent） | 递归调用中的所有权和深度限制 |
| `src/tui/` | ratatui 界面和 tachyonfx 动效 | 即时模式 UI、按条目缓存渲染、CJK 宽字符换行 |

## 和 Python 版的兼容

- 配置：`[llm]`、`[ui]`、`[tools]`、`[compaction]`、`[subagents]`、`[providers.*]` 的含义保持一致，`LIMBO_HOME` 和 `LIMBO_AUTH_FILE` 也照样生效。`/model` 改模型时会保留配置文件里的注释。
- 会话：记录类型相同（`meta`、`meta_patch`、`compaction`、`tool_intent`、`queue`、`task`）。遇到不认识的字段会原样保留，不认识的记录会跳过。
- 模型目录：和 Python 版同步，包括 DeepSeek、Kimi、Kimi For Coding、GLM 和 Codex（Responses）。

## 暂时没有移植的功能

daily 模式和每日技能、`/goal` 循环、code mode、tracing 查看器、2048、粘贴图片和剪贴板、`web_search`，以及事件溯源的子 agent 看板（这里换成了更简单的 `task` 工具，嵌套深度限制为 1）。
