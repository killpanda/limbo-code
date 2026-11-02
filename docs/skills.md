# Skill 支持

Limbo 支持 Claude Code / pi 风格的 skill：一个包含 `SKILL.md` 的目录。
Skill 通过两条路径生效（双轨）：

- **模型自主触发（渐进式披露）**：system prompt 末尾注入 `<available_skills>`
  目录（仅 name/description/location），任务匹配 description 时模型用
  read 工具按需加载 SKILL.md 正文。frontmatter 设
  `disable-model-invocation: true` 的 skill 不进目录。
- **斜杠命令显式触发**：`/<name> [args]` 把 skill 正文 + 参数注入为当轮
  prompt。

frontmatter 校验遵循 Agent Skills 规范：name ≤64 字符、仅小写字母/数字/
连字符（不得以连字符开头/结尾、不含连续连字符），description 必填
（≤1024 字符）。不合规的 skill 告警跳过；命名冲突（项目覆盖用户）打
warning 日志。

## Skill 格式

```
<skills-dir>/<name>/SKILL.md
```

```markdown
---
name: tdd
description: Test-driven development workflow.
---

# 指令正文，调用时作为 prompt 注入对话
```

- frontmatter 读顶层的 `name`、`description`、`disable-model-invocation`；值可以是普通、
  带引号（`"..."` / `'...'`）、缩进续行或 YAML 块标量（`>` / `|`）。嵌套在 `metadata:` 等
  下面的同名键不算
- description 超过 1024 字符时截断后照常列出（不再跳过）
- 无 frontmatter 时：name 取目录名，整个文件作为正文

## 发现路径（同名时后者覆盖前者）

| 来源 | 路径 |
|------|------|
| 内置 daily 起步包（仅 daily 会话） | `assets/daily_skills/`（编进二进制，从 Hermes 移植） |
| 全局（与其他 agent 共用，`npx skills add -g` 装在这里） | `~/.agents/skills/<name>/SKILL.md` |
| 用户级 | `~/.limbo/skills/<name>/SKILL.md` |
| 项目级 | `<workdir>/.agents/skills/<name>/SKILL.md` |
| Daily（仅 daily 会话） | `~/.limbo/daily/skills/<name>/SKILL.md` |

内置斜杠命令（`/sessions` 等）优先于同名 skill。

## 交互

- 输入 `/` 弹出命令菜单，skills 与内置命令一同列出（skill 带 `[skill]` 标记）
- 继续输入实时过滤；菜单在每次弹出时重新扫描，运行中新增的 skill 立即可用；
  每轮开始前若 skill 有增删改，也会重建 system prompt 里的 `<available_skills>` 目录
- 选中 skill：
  - **部分匹配 + Enter / Tab** → 补全为 `/<name> `，可继续输入参数
  - **完整输入 `/<name>` + Enter** → 直接无参调用
- 调用时聊天区显示紧凑形式 `❯ /tdd implement login`，实际发给 LLM 的
  prompt 为 skill 正文 + skill 文件路径（供解析相对引用）+ 用户参数

## 代码结构

- `src/skills.rs` — `parse_skill_md` / `discover_skills` / `validate_skill`，以及内置起步包的解压，纯逻辑无 UI 依赖
- `src/tui/app.rs` — 命令面板把内置命令和 skill 合在一起列出，内置命令优先；选中 skill 时拼出发给模型的 prompt
- `src/tools/skill_manage.rs` — daily 模式下模型改写 skill 的工具
