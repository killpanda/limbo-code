# 子代理（Subagents）

主 agent 可以把工作拆成任务，交给后台子代理并行完成；你继续和主 agent
对话，子代理的结果、提问和失败会回到主 agent 那里。你也可以随时直接查看
和控制每个子任务。

## 一眼看懂

```
你 ──对话──▶ 主 agent ──plan_tasks──▶ 任务板（t1 t2 t3 …）
                ▲                         │ reconcile：依赖就绪 → 派发
                │ <subagent_updates>      ▼
                └──── 结果 / 提问 / 告警 ── 子代理（各自独立上下文与会话文件）
你 ──/task、/agents、@名字──▶ 任务板 / 子代理（主 agent 会被告知）
```

- **任务板**（`subagents/board.py`）是唯一真相：任务、依赖、等待、打点、
  提问和结果。每次变更都作为 `{"type":"task",...}` 记录追加进主会话文件，
  重启时重放；会话重写时写一条 `snapshot` 折叠历史。
- **Supervisor**（`subagents/supervisor.py`）根据任务板做调度：依赖都结束
  的任务变为就绪，按优先级 / 已有进度 / 关键路径派发到空闲槽位
  （`max_parallel`），并把通知投递给主 agent。
- **子代理**是同进程里的另一个 `Agent`，有自己的会话文件
  `<sessions>/<主会话>.agents/<任务id>-<名字>.jsonl`，崩溃后可以接着跑。

## 自动创建

- 启动时对工作区做一次不调用模型的扫描（语言、文件数、workspace 子包、
  检测到的验证命令、git 状态），作为 `<project_facts>` 写进主 agent 的
  system prompt，和可用角色列表 `<available_agents>` 一起引导它判断什么时候
  值得拆分。
- 默认 `auto = "suggest"`：主 agent 调 `plan_tasks` 后，计划会弹出给你确认
  （TUI 弹窗；headless 里用 `/task approve all`），确认前不会开始。
  `auto = "on"` 直接开始，`auto = "off"` 不给主 agent 任务工具。
- 角色（profile）决定子代理的工具、模型和提示：内置 `explore`（只读）、
  `worker`、`reviewer`（只读）、`verifier`（不改文件）。用户级
  `~/.limbo/agents/<name>.md` 和项目级 `.agents/agents/<name>.md` 可以覆盖或
  新增；`/agents init` 会根据扫描结果为每个 workspace 子包生成一个
  `<包名>-worker`，并生成跑验证命令的 `project-verifier`。

角色文件格式：

```markdown
---
name: api-worker
description: 只改 packages/api/ 下的代码
when_to_use: 任务只涉及 API 时
tools: all            # read_only | all | no_edit | read,grep,bash（白名单）
model: inherit        # inherit | cheap | 具体模型 id
max_iterations: 40
isolation: shared     # shared | worktree
context: brief        # brief（只看任务说明）| fork（带上主对话）
inherit_prompt: false # 仅 fork：沿用主会话的 system prompt 和工具列表
scope_paths: packages/api/**
---
给子代理的额外指令。
```

`inherit_prompt: true`（内置的 skill-reviewer 就是这样）让 fork 子代理逐字节沿用主会话的 system prompt 和工具定义，第一次请求就能命中 prompt 缓存。工具限制只在执行侧生效：主会话的工具都会展示给它，但不在白名单里的调用会失败。它的角色说明和指令放在第一条消息里；它没有上面那五个任务工具，最后一条纯文本回复就是结果。这个开关只在子代理和主会话用同一个模型时生效，换了模型缓存本来就是冷的，会回到普通 fork。

## 打点与任务编排

子代理有五个工具：

| 工具 | 作用 |
|------|------|
| `checkpoint(phase, summary)` | 打点：进入新阶段时记录进度（每 8 次工具调用也会自动打点） |
| `need(on=[…] \| new={…}, reason, policy)` | 发现依赖：`wait` 挂起直到依赖结束，带着结果继续；`cancel_self` 判断自己已无必要；`ask_main` 交给主 agent 决定 |
| `ask_main(question)` | 需要主 agent / 用户决定的问题，挂起直到回答 |
| `finish(summary, files_changed)` | 交付，摘要就是主 agent 读到的全部 |
| `propose_task(title, brief)` | 建议新任务，由主 agent 批准（子代理不能自己再拆，深度固定为 1） |

规则：

- **依赖边**（`depends_on` / `need`）：依赖失败时默认 `notify`——等待方照常
  继续并收到失败原因，自己决定；`on_dep_fail = "cascade"` 改为连带取消。
  `join = "fail_fast"` 让等待方在第一个依赖失败时就醒来。
- **归属边**（`owner`）：父任务取消或失败，它名下的任务一律取消。
- 加依赖时检测环，形成循环会被拒绝；所有任务互相等待（死锁）时，主 agent
  会收到告警。
- 共享工作区时，写任务用 `scope_paths` 认领路径，其他子代理 `edit`/`write`
  被认领的文件会被拦下并提示用 `need` 等待。需要更强隔离时用
  `isolation: worktree`（独立 git worktree 和分支，完成后
  `task_control merge` 合并）。

## 通讯

- **子代理 → 主 agent**：通知合并成一条 `<subagent_updates>` 插话。主 agent
  忙时在下一个一致点注入；空闲时：提问和告警立即唤醒主 agent，结果等这一波
  子代理都结束后一起交付（`wake_main_on` 可调）。每条通知只交付一次，崩溃
  后不会重复。
- **主 agent → 子代理**：`task_status`、`task_send`（运行中的子代理以插话收到，
  其他状态进收件箱、继续时送达）、`answer`、`task_control`（pause / resume /
  cancel / retry / reprioritize / edit / approve / reject / merge）。所以你可以
  直接对主 agent 说"先暂停前端那个任务"。
- **你 → 子代理**：

| 输入 | 作用 |
|------|------|
| `/agents` | 子代理面板：实时任务板、打点、最近记录；`p` 暂停 `r` 恢复 `c` 取消 `t` 重试 `a`/`x` 批准/拒绝 `m` 发消息 |
| `/task` · `/task <id>` | 任务板 / 某个任务详情 |
| `/task <pause\|resume\|cancel\|retry\|approve\|reject\|merge> <id\|all> [说明]` | 直接控制；`/task retry <id> fresh` 丢弃旧上下文重来 |
| `/task prio <id> <n>` · `/task edit <id> <新说明>` · `/task send <id> <消息>` | 调优先级、改说明、发消息 |
| `@worker-1 <消息>` | 直接对某个子代理说话 |
| `/agents scan` · `/agents init` | 项目画像与可用角色 / 写入建议的项目角色 |

  这些命令忙时也可用，你的每个直接操作都会以 `info` 通知告诉主 agent。

## 重启与恢复

进程退出时运行中的任务不写结果。下次打开会话时，它们被标为"已暂停（进程
重启时中断）"并提示 `/task resume all`（`on_resume = "auto"` 则直接继续）。
恢复的子代理从自己的会话文件接着跑，并被告知先检查工作区现状。

headless 模式下输入关闭（EOF）后，会等后台子代理和随之而来的主 agent 总结
轮次结束再退出；暂停和待确认的任务不等。`--json` 输出额外的 `task`、
`subagent_notice`、`plan_proposed` 行。

## 配置

```toml
[subagents]
enabled = true                 # false：完全关闭
auto = "suggest"               # off | suggest | on
max_parallel = 3               # 同时运行的子代理
max_tasks = 12                 # 未结束任务上限
task_token_budget = 400000     # 单任务 token 预算（0 = 不限）：只算新增 token（总量减去缓存命中），超出后当前这批工具跑完、下一次请求前收尾
cheap_model = ""               # model: cheap 的角色用它（空 = 主模型）
wake_main_on = ["question", "alert", "wave_done"]
on_dep_fail = "notify"         # notify | cascade
on_resume = "ask"              # ask | auto | pause
default_isolation = "shared"   # shared | worktree
result_chars = 2000            # 交给主 agent 的结果摘要长度
auto_checkpoint_every = 8      # 每 N 次工具调用自动打点（0 = 关）
```
