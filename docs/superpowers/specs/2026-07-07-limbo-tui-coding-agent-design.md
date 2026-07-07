# Limbo: 极简 TUI Coding Agent 设计文档

> 状态：设计中，待实现
> 目标：一个受 [Pi](https://pi.dev) 启发的、基于 Python + Textual 的终端 AI 编程助手。

---

## 1. 项目定位

Limbo 是一个**持久会话式的终端 AI 编程助手**。用户在终端中启动后进入一个 TUI 界面，通过自然语言与 AI 协作完成代码阅读、修改、测试等任务。

参考产品：Pi（pi.dev）、Claude Code、kimi-code。

核心原则：
- **极简**：MVP 只保留最必要的功能。
- **可控**：所有文件写操作必须经用户确认。
- **模型驱动**：把复杂度放在工具实现里，用精准的提示词和工具描述引导模型行为。

---

## 2. 技术栈

- **语言**：Python 3.11+
- **TUI 框架**：Textual
- **LLM 接口**：OpenAI 兼容 API，默认以 DeepSeek 为推荐 provider
- **配置格式**：TOML
- **测试**：pytest + Textual Pilot + respx/aioresponses

---

## 3. 总体架构

```
┌─────────────────────────────────────────┐
│                  ui/                    │
│  Textual 应用：布局、事件、确认弹窗       │
└──────────────┬──────────────────────────┘
               │ Message / ToolCall / ToolResult
┌──────────────▼──────────────────────────┐
│                agent/                   │
│  维护会话历史，编排 LLM 调用与工具执行     │
└──────────────┬──────────────────────────┘
               │ ToolCall / ToolResult
┌──────────────▼──────────────────────────┐
│                tools/                   │
│  read / bash / edit / write / grep /    │
│  find / ls                              │
└─────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│                 llm/                    │
│  OpenAI 兼容客户端：流式输出、function    │
│  calling、tool_call 解析                 │
└─────────────────────────────────────────┘
```

模块间通过明确的数据结构传递：
- `Message`：统一的消息格式（system/user/assistant/tool_result）
- `ToolCall`：模型请求调用的工具
- `ToolResult`：工具执行结果
- `LLMEvent`：LLM 流式事件（text_chunk / tool_call_start / tool_call_end）

---

## 4. UI 布局与交互

### 4.1 主界面

```
┌────────────┬──────────────────────┬─────────────┐
│  会话/文件  │                      │  文件预览/  │
│  上下文栏   │      聊天区          │  命令输出   │
│   (~25%)   │       (~50%)         │   (~25%)    │
├────────────┴──────────────────────┴─────────────┤
│  [输入框...]                                    │
└─────────────────────────────────────────────────┘
```

- **左侧栏**：会话标题、最近读取文件、工具调用状态。
- **中间主区**：聊天消息流，用户消息在右，AI 消息在左，支持 Markdown、代码块、流式打字效果。工具调用结果以折叠卡片嵌入。
- **右侧栏**：实时显示 `read`/`bash` 的输出内容。
- **底部**：多行输入框 + 快捷键提示。

### 4.2 确认流程

- `write` 调用：暂停后续调用，弹出 diff 确认框，用户选择「应用 / 拒绝 / 编辑后应用」。
- `edit` 调用：第一期同样要求确认，后续可考虑默认通过。
- 高危 `bash` 命令（如 `rm`、`git reset --hard`、输出重定向覆盖）：执行前二次确认。

### 4.3 快捷键

| 快捷键 | 动作 |
|--------|------|
| `Ctrl+C` | 退出（有待确认修改时提示） |
| `Ctrl+R` | 重新生成最后一条 AI 回复 |
| `Ctrl+N` | 新建会话 |
| `/` | 聚焦输入框 |
| `Tab` | 在左/中/右面板间切换焦点 |
| `,` 或 `:settings` | 打开设置 |

---

## 5. 会话与数据流

1. 用户提交消息 → UI 发给 `Agent`。
2. `Agent` 追加消息，调用 `LLMClient.chat(messages, tools)` 流式请求。
3. LLM 返回普通文本 → UI 直接渲染。
4. LLM 返回 `tool_calls` → `Agent` 按顺序执行：
   - 调用对应 tool
   - 每个 tool 返回 `ToolResult`
   - 包装成 `tool_result` 消息追加到历史
5. 若包含未确认的 `write`/`edit`，UI 弹出确认框；用户批准后真正落盘。
6. `Agent` 再次调用 LLM，循环直到 LLM 不再调用工具或达到 `max_iterations`（默认 10）。
7. 最终回复渲染，会话历史自动保存到 `~/.limbo/sessions/`（JSONL）。MVP 只做自动保存，会话恢复/列表为后续功能。

---

## 6. 工具定义

MVP 提供 7 个工具，按使用频率排序给 LLM：

### 6.1 `read`

```json
{
  "name": "read",
  "description": "Read the contents of a file. Use offset/limit for large files.",
  "parameters": {
    "path": "Path to the file (relative or absolute)",
    "offset": "Line number to start from (1-indexed, optional)",
    "limit": "Maximum number of lines to read (optional)"
  }
}
```

- 输出截断到 2000 行或 512KB。
- 禁止读取工作目录外文件和敏感文件（SSH key、`.env` 等）；敏感文件列表可配置。

### 6.2 `bash`

```json
{
  "name": "bash",
  "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
  "parameters": {
    "command": "Bash command to execute",
    "timeout": "Timeout in seconds (optional, default 30)"
  }
}
```

- 默认超时 30 秒。
- 高危命令二次确认。

### 6.3 `edit`

```json
{
  "name": "edit",
  "description": "Make surgical edits to a file by replacing exact text. old_text must match exactly.",
  "parameters": {
    "path": "Path to the file",
    "old_text": "Exact text to find and replace",
    "new_text": "New text to replace with"
  }
}
```

- 目标文件必须在工作目录内。
- `old_text` 不唯一时报错，提示模型扩大上下文。
- 第一期只做精确匹配，保留后续增加模糊匹配的接口。

### 6.4 `write`

```json
{
  "name": "write",
  "description": "Create or overwrite a file. Use only for new files or complete rewrites.",
  "parameters": {
    "path": "Path to the file",
    "content": "Content to write"
  }
}
```

- **必须用户确认**后才能落盘。
- 禁止写出工作目录外。

### 6.5 `grep`

```json
{
  "name": "grep",
  "description": "Search file contents for a pattern. Respects .gitignore.",
  "parameters": {
    "pattern": "Search pattern (regex or literal string)",
    "path": "Directory or file to search (default: current directory)",
    "glob": "Filter files by glob pattern",
    "ignore_case": "Case-insensitive search (default: false)",
    "fixed_string": "Treat pattern as literal string (default: false)",
    "context": "Lines to show before/after each match",
    "limit": "Maximum matches (default: 100)"
  }
}
```

### 6.6 `find`

```json
{
  "name": "find",
  "description": "Find files by glob pattern. Respects .gitignore.",
  "parameters": {
    "pattern": "Glob pattern, e.g. 'src/**/*.ts'",
    "path": "Directory to search (default: current directory)",
    "limit": "Maximum results (default: 1000)"
  }
}
```

### 6.7 `ls`

```json
{
  "name": "ls",
  "description": "List directory contents. Includes dotfiles.",
  "parameters": {
    "path": "Directory to list (default: current directory)",
    "limit": "Maximum entries (default: 500)"
  }
}
```

### 6.8 工具返回值

```python
class ToolResult:
    success: bool
    output: str | None
    error: str | None
    requires_confirmation: bool = False
```

---

## 7. 系统提示词结构

参考 Pi 的设计，系统提示词分为四层：

```
You are an expert coding assistant operating inside limbo, a coding agent harness.
You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands
- edit: Make surgical edits to files (find exact text and replace)
- write: Create or overwrite files
- grep: Search file contents for patterns (respects .gitignore)
- find: Find files by glob pattern (respects .gitignore)
- ls: List directory contents

Guidelines:
- Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)
- Use read to examine files before editing. You must use this tool instead of cat or sed.
- Use edit for precise changes (old_text must match exactly)
- Use write only for new files or complete rewrites
- Be concise in your responses
- Show file paths clearly when working with files

Current date: {date}
Current working directory: {cwd}
```

可选追加项目级提示词：
- `~/.limbo/AGENTS.md`：全局个人偏好
- `./AGENTS.md`：项目特定上下文

---

## 8. 配置

### 8.1 文件位置

- `~/.limbo/config.toml`：全局配置
- `~/.limbo/sessions/`：会话历史（JSONL）
- `./AGENTS.md`（可选）：项目级系统提示词

### 8.2 配置示例

```toml
[llm]
provider = "openai"  # OpenAI 兼容接口，MVP 只实现该 provider
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
temperature = 0.2
max_iterations = 10

[ui]
theme = "dark"
confirm_writes = true
confirm_edits = true

[safety]
dangerous_commands = ["rm", "git reset --hard", ">"]
sensitive_files = [".env", "id_rsa", "id_ed25519"]
```

---

## 9. LLM 客户端抽象

```python
class LLMClient:
    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> AsyncIterator[LLMEvent]:
        ...
```

`LLMEvent` 类型：
- `TextChunk(text: str)`
- `ToolCallStart(id: str, name: str, arguments: dict)`
- `ToolCallEnd(id: str)`

MVP 实现一个 `OpenAICompatibleClient`，通过 `base_url` 和 `api_key` 支持 DeepSeek 等兼容接口。

---

## 10. 错误处理

| 场景 | 处理 |
|------|------|
| LLM API 失败 | UI 显示红色提示，提供重试；不把错误喂给模型 |
| 工具执行失败 | 返回给模型，让模型决定下一步 |
| bash 超时 | 自动 kill，返回超时信息 |
| 超大输出 | 截断并提示模型可继续读取 |
| 死循环 | `max_iterations` 上限保护 |
| 未保存修改退出 | 提示用户是否丢弃 |
| 路径越界 | 所有文件工具先 resolve 并检查是否在工作目录内 |

---

## 11. 测试策略

- **单元测试**：每个 tool 的成功路径和常见失败。
- **Agent 循环测试**：用 mock LLM client 模拟完整会话循环。
- **LLM client 测试**：用 `respx`/`aioresponses` mock 流式响应和 tool_call 提取。
- **TUI 测试**：Textual `Pilot` 覆盖主流程：输入 → 渲染 → 确认 → 应用。
- **静态检查**：ruff + mypy（可选）。

---

## 12. 后续可扩展（非 MVP）

- 模糊匹配 `edit`（BOM、行尾符、智能引号容错）
- 多 provider 支持（Claude、Gemini）
- 会话恢复 `/resume`
- 技能/插件系统
- 背景进程管理
- MCP 工具集成

---

## 13. 验收标准

MVP 完成时，用户应能：

1. 在任意目录运行 `limbo` 启动 TUI。
2. 输入自然语言任务，AI 能调用 read/bash/grep/find/ls 探索代码库。
3. AI 能提出文件修改，用户在确认框中查看 diff 后选择应用或拒绝。
4. 会话历史自动保存，重启后可继续。
5. 通过配置文件指定 DeepSeek API key 和模型。
