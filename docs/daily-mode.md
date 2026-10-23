# Daily mode

日常非 coding 工作（调研、读网页、写东西、整理笔记）用的会话姿态。内核和
coding 模式完全一样（agent loop、会话、压缩、子代理、/goal、skills），只换三样：

- **系统提示词**：通用个人助理口吻，先搜后答、给出处、产出落到文件。
- **工具**：在原有工具之上加 `web_fetch`（读网页/PDF）、`memory`（长期记忆）、
  `session_search`（翻过往会话）、`ask_user`（向你提问）、`skill_manage`（自己改进 skill）。
- **记忆注入**：会话开始时把 `~/.limbo/memory/` 的快照放进提示词。

## 进入和切换

```bash
limbo daily              # = limbo --daily：工作目录切到 [daily] workdir，新会话直接是 daily
limbo daily --continue   # 接上一个 daily 会话
```

会话内 `/daily-mode on|off`（无参切换），状态栏显示 `☀daily`。模式随会话保存，
恢复时还原；`/new` 沿用当前模式。切换会重建系统提示词（前缀缓存失效一次，和 `/model` 一样）。

## 记忆

两个纯文本文件，可以直接打开编辑，条目之间用单独一行 `§` 分隔：

| 文件 | 内容 |
|---|---|
| `~/.limbo/memory/USER.md` | 你是谁：身份、偏好、风格 |
| `~/.limbo/memory/NOTES.md` | agent 的笔记：稳定事实、约定、你纠正过的事 |

记忆只进 daily 会话的提示词，coding 会话完全读不到。

- 模型用 `memory` 工具写入（add / patch / remove，一次调用可批量、原子执行），超过字符上限时整批拒绝。
- 写入立刻落盘（上一版留在 `*.bak`），但本会话的提示词不变，从下一个会话起生效。
- `/memory` 查看内容和用量；`/memory edit [user|notes]` 用 `$VISUAL`/`$EDITOR` 打开（headless 下只显示路径）。

- 每 `memory_nudge_turns` 轮（默认 10）没写过记忆时，会在你的消息后附一句提醒，让模型考虑要不要存。

## Skills

daily 会话按这个顺序收集 skill，同名时后者覆盖前者：

1. **内置起步包**（`src/limbo/daily_skills/`，从 Hermes 移植，MIT）：pdf、docx、xlsx、powerpoint、
   grounded-citations、blocked-page-recovery、meeting-action-items、document-to-action-items、
   weekly-review-planning、youtube-content、arxiv。文档类 skill 需要的 Python 包按 skill 里的说明装。
2. 用户 skill `~/.limbo/skills`、项目 skill `<workdir>/.agents/skills`
3. **daily skill** `~/.limbo/daily/skills/<name>/SKILL.md`

coding 会话只看 2，看不到起步包和 daily skill。

## 自学习（/learn）

- 模型可以用 `skill_manage` 新建或修改 skill（create / patch / write_file，一次批量、全成或全不成）。
  所有写入都落在 `~/.limbo/daily/skills`：改内置或用户 skill 时先把它复制过来再改，原文件不动；改前留 `.bak`。
- `/learn` 在后台开一个 `skill-reviewer` 子代理（带上当前对话、只能读 skill 和调 `skill_manage`、用 cheap_model），
  把这次对话里的纠正、偏好和新技巧写进 skill，结果作为子代理通知回到对话里。
- 自动触发：一轮结束时，如果距上次改 skill 已经有 `learn_after_tool_calls`（默认 10）次工具调用，就自动开一次复盘；设 0 关掉。
- 需要子代理开着（`[subagents] enabled = true`）。

## 其他工具

- `session_search`：不带索引地扫描会话文件，按关键词找过往 daily 会话（`include_coding` 也搜 coding），
  再用 `session_id` + `around` 读上下文。
- `ask_user`：模型给出问题和最多 6 个选项后本轮结束，你回复编号或直接回答。
- 子代理角色 `researcher`：只读调研，daily 模式下能读网页正文，交回带出处的结论。daily 会话里的子代理
  也是 daily 姿态，但 `memory`、`ask_user`、`skill_manage`、`session_search` 只留给主 agent。

## web_fetch

打开 http(s) 链接，返回正文的 Markdown（标题、列表、链接；导航、脚本、页脚去掉，
有 `<article>`/`<main>` 时只取正文）。长页面按 `fetch_max_chars` 分段，带 `offset` 继续读。
PDF 链接在装了 `pypdf` 时直接转成文本（`pip install 'limbo-code[daily]'`）。超长页面除了分段返回，
全文还会存到 `<workdir>/fetched/`，可以用 read 翻。

## 配置

```toml
[daily]
workdir = "~/limbo-daily"
model = ""                    # 留空沿用 [llm] model；只对 limbo daily 启动生效
memory = true
user_chars = 1500
notes_chars = 2500
memory_nudge_turns = 10
learn_after_tool_calls = 10
fetch_max_chars = 20000
fetch_timeout = 30.0
```
