# 跟着 Limbo 学 Rust

Limbo 最早是 Python 写的，后来整体用 Rust 重写，现在仓库里只有 Rust 版。重写的另一个目的是**学 Rust**。所以代码里有不少面向学习者的注释，会解释所有权、`async`、trait 对象这类概念在这里为什么这样用。

## 跑起来

```bash
cargo run --release                 # 打开 TUI
cargo run --release -- -c           # 继续当前目录最近的会话
cargo run --release -- -r <id>      # 恢复指定会话（id 可以只写前缀）
cargo run --release -- -p "解释一下 src/agent.rs"   # 只跑一轮，把回答打印出来
cargo run --release -- --headless   # 纯文本 REPL（加 --json 输出事件流）
cargo run --release -- daily        # daily 模式（非编码助手，在 [daily] workdir 里），等同 --daily
cargo run --release -- models [词]  # 列出可用模型（可按名字过滤）
cargo run --release -- auth login [提供商]   # 保存 API key 到 ~/.limbo/auth.json（auth list / auth logout）
cargo run --release -- sessions     # 列出当前目录的会话（--all 列出全部）
cargo test                          # 跑测试
```

需要 Rust 1.88 以上（edition 2024）。`grep` 工具依赖系统里装好的 `rg`（ripgrep）。

第一次运行时如果找不到任何 API key，会直接在终端里提示粘贴一个（输入不回显，先联网验证再保存）。

## TUI 操作

| 按键 | 作用 |
| --- | --- |
| Enter / Shift+Enter（或 Ctrl+J） | 发送 / 换行 |
| Alt+Enter | 排进队列，当前任务完成后再发（follow-up） |
| Ctrl+V / Ctrl+Y | 从剪贴板粘贴（图片、文件路径或文本）/ 复制最后一条回复 |
| Ctrl+G | 子代理面板（任务板 + 选中任务的记录） |
| Esc、Ctrl+C | 打断当前回合；空闲时连按两次 Ctrl+C 退出 |
| agent 忙时按 Enter | 把消息排进队列，作为“插话”发给 agent |
| ↑ / ↓ | 翻输入历史 |
| PageUp / PageDown、Ctrl+↑/↓ | 滚动对话 |
| Shift+Tab | 切换思考强度 |
| Ctrl+O / Ctrl+T | 展开或折叠工具输出 / 显示或隐藏思考过程 |
| Ctrl+R | 会话列表 |
| `/` | 命令面板：`/model` `/thinking` `/plan` `/goal` `/queue` `/code-mode` `/daily-mode` `/memory` `/learn` `/btw` `/agents` `/task` `/tracing-viewer` `/copy` `/2048` `/compact` `/fork` `/handoff` `/sessions` `/export` `/theme` `/mouse` `/new` … |

在 `/model` 里选一个还没有 key 的模型，会弹出登录框，粘贴 key 后验证、保存、再切换。最近用过的模型排在最前面；Shift+Tab 调的思考强度会写回配置。

主题有 `neon`、`aurora`、`ember`、`dawn`（daily 模式默认）和 `omarchy`（跟随 Omarchy 桌面主题），用 `/theme` 切换，或者在配置里写 `[ui] theme = "aurora"`。动效可以用 `[ui] motion = "full" | "reduced" | "off"` 调整（通过 SSH 连接时默认是 reduced）。

## 代码地图（推荐的阅读顺序）

| 文件 | 内容 | 能学到的 Rust 知识 |
| --- | --- | --- |
| `src/models.rs` | 消息、工具调用这些数据类型 | `serde` 派生、`enum`、`Option` |
| `src/config.rs`、`auth.rs` | 读取 `~/.limbo/config.toml` | 错误处理、`toml_edit` 保留注释改写 |
| `src/llm/` | 三种 LLM 协议，以及共用的 SSE 解析 | trait 对象 `Box<dyn Decoder>`、`mpsc` 通道、流 |
| `src/tools/` | bash/read/write/edit/ls/find/grep | `#[async_trait]`、`Arc<dyn Tool>`、`tokio::select!`、进程组 |
| `src/session.rs`、`history.rs` | JSONL 日志、崩溃恢复 | `#[serde(flatten)]` 保留未知字段、原子写文件 |
| `src/agent.rs` | agent 主循环 | 把 `Agent` *move* 进任务再通过 `JoinHandle` 取回所有权 |
| `src/pump.rs`、`goal.rs` | 一次任务的外层循环：`/goal` 验收闭环、follow-up 队列 | 用 `&mut Agent` 串起多轮，`async` 函数里的早退出 |
| `src/subagents/` | 子代理任务板：计划、依赖、优先级、worktree、通知 | `Arc::new_cyclic` + `Weak` 自引用、`Mutex` 里的共享状态、宏生成工具 |
| `src/learn.rs`、`memory.rs`、`skills.rs` | daily 模式的记忆和自动复盘 | 正则 + `OnceLock` 懒初始化 |
| `src/trace_viewer.rs` | `/tracing-viewer`：本机 HTTP 服务 + 浏览器页面 | 只用 `std::net` 和线程写一个小 HTTP 服务 |
| `src/integrations.rs` | Herdr、OSC 7501 状态上报 | `unsafe` 调 libc、信号处理 |
| `src/tui/` | ratatui 界面和 tachyonfx 动效 | 即时模式 UI、按条目缓存渲染、CJK 宽字符换行 |

## 从 Python 版迁移过来

- 配置、`auth.json` 和 JSONL 会话格式都和原来的 Python 版一致，`~/.limbo` 里的旧数据可以直接用，旧会话也能 `--resume`。
- 有一处有意的不同：`/goal` 第一轮验收失败后，Python 版提示“第 2 轮验收未通过”，Rust 版改成了正确的“第 1 轮”。
