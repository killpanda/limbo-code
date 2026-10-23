# Daily mode

日常非 coding 工作（调研、读网页、写东西、整理笔记）用的会话姿态。内核和
coding 模式完全一样（agent loop、会话、压缩、子代理、/goal、skills），只换三样：

- **系统提示词**：通用个人助理口吻，先搜后答、给出处、产出落到文件。
- **工具**：在原有工具之上加 `web_fetch`（打开链接读正文）和 `memory`（长期记忆）。
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

## Skills

daily 有自己的 skill 目录 `~/.limbo/daily/skills/<name>/SKILL.md`：

- daily 会话能看到：用户 skill（`~/.limbo/skills`）+ 项目 skill（`<workdir>/.agents/skills`）+ daily skill。
  同名时后者覆盖前者（daily 优先）。
- coding 会话看不到 daily skill，系统提示词和 `/` 命令菜单里都不出现。

## web_fetch

打开 http(s) 链接，返回正文的 Markdown（标题、列表、链接；导航、脚本、页脚去掉，
有 `<article>`/`<main>` 时只取正文）。长页面按 `fetch_max_chars` 分段，带 `offset` 继续读。
PDF 暂不支持（提示用 bash 下载后本地转换）。

## 配置

```toml
[daily]
workdir = "~/limbo-daily"
model = ""                    # 留空沿用 [llm] model；只对 limbo daily 启动生效
memory = true
user_chars = 1500
notes_chars = 2500
fetch_max_chars = 20000
fetch_timeout = 30.0
```
