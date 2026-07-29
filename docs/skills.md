# Skill 支持

Limbo 支持 Claude Code / pi 风格的 skill：一个包含 `SKILL.md` 的目录，
通过斜杠命令调用。

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

- frontmatter 为简单的 `key: value` 行（支持 `name`、`description`）
- 无 frontmatter 时：name 取目录名，整个文件作为正文

## 发现路径（同名时后者覆盖前者）

| 来源 | 路径 |
|------|------|
| 用户级 | `~/.limbo/skills/<name>/SKILL.md` |
| 项目级 | `<workdir>/.agents/skills/<name>/SKILL.md` |

内置斜杠命令（`/sessions` 等）优先于同名 skill。

## 交互

- 输入 `/` 弹出命令菜单，skills 与内置命令一同列出（skill 带 `[skill]` 标记）
- 继续输入实时过滤；菜单在每次弹出时重新扫描，运行中新增的 skill 立即可用
- 选中 skill：
  - **部分匹配 + Enter / Tab** → 补全为 `/<name> `，可继续输入参数
  - **完整输入 `/<name>` + Enter** → 直接无参调用
- 调用时聊天区显示紧凑形式 `❯ /tdd implement login`，实际发给 LLM 的
  prompt 为 skill 正文 + skill 文件路径（供解析相对引用）+ 用户参数

## 代码结构

- `src/limbo/skills.py` — `parse_skill_md` / `discover_skills`，纯逻辑无 UI 依赖
- `src/limbo/ui/commands.py` — `SlashCommandRegistry`（统一菜单元数据与分发），
  `SlashCommand.kind`（`builtin` | `skill`）
- `src/limbo/ui/screens/main.py` — `_slash_candidates()` 委托 registry 合并内置命令与
  skills、`_find_skill()` / `_invoke_skill()`
