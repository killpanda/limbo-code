# Limbo UI 极简设计方案

## 一、设计思路

参考 Claude Code、Aider、oterm 等终端 AI 工具，核心原则：**聊天就是一切**。

- 不保留左侧文件树（文件读取结果直接展示在对话里）
- 不保留右侧预览面板（代码块内联在消息中）
- 不保留工具状态栏（工具调用状态直接出现在对话流中）
- 所有信息内联到聊天消息，无需额外面板

## 二、布局

只有一个区域：**全宽聊天 + 底部输入栏**。可选的极简顶部标题栏。

```
┌────────────────────────────────────────────────────────────┐
│ 🌀 Limbo  ·  deepseek-chat  ·  ○ Idle                       │  ← 极简标题栏（可选）
├────────────────────────────────────────────────────────────┤
│                                                            │
│  You                                                       │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 帮我分析这个项目的架构                               │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  🤖 Assistant                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 我来分析一下代码结构...                              │   │
│  │                                                        │   │
│  │  ┌─▶ read src/limbo/config.py                        │   │
│  │  │  ┌────────────────────────────────────────────┐   │   │
│  │  │  │ from pydantic import BaseModel            │   │   │
│  │  │  │ class Config(BaseModel):                  │   │   │
│  │  │  │     model: str = "deepseek-chat"          │   │   │
│  │  │  └────────────────────────────────────────────┘   │   │
│  │  └─✓ Completed                                       │   │
│  │                                                        │   │
│  │  ┌─▶ edit src/limbo/ui/widgets/chat.py               │   │
│  │  │  ┌─ diff ──────────────────────────────────────┐  │   │
│  │  │  │ -    color: $text-accent;                  │  │   │
│  │  │  │ +    color: $text;                         │  │   │
│  │  │  │ -    text-align: right;                    │  │   │
│  │  │  │ +    text-align: left;                      │  │   │
│  │  │  └────────────────────────────────────────────┘  │   │
│  │  └─✓ Applied (confirmed)                           │   │
│  │                                                        │   │
│  │ 基于以上分析，项目架构是：                            │   │
│  │                                                        │   │
│  │  ```python                                             │   │
│  │  class Agent:                                         │   │
│  │      def run(self): ...                               │   │
│  │  ```                                                  │   │
│  │                                                        │   │
│  │ 3 个优化建议：                                        │   │
│  │ 1. ...                                               │   │
│  │ 2. ...                                               │   │
│  │ 3. ...                                               │   │
│  │                                    ▋                  │   │  ← 流式光标
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  System                                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │ ⚠️ bash 命令需要确认：                               │   │
│  │ rm -rf node_modules                                  │   │
│  │ [确认]  [取消]                                       │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ...                                                       │
│                                                            │
│                                                            │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  💬  Ask anything...                      [Ctrl+Enter]  ⏎ │  ← 输入栏
└────────────────────────────────────────────────────────────┘
```

## 三、关键设计细节

### 1. 极简标题栏（一行，可隐藏）

- 左：`🌀 Limbo` 品牌名（accent 色）
- 中：`deepseek-chat` 模型名（muted）
- 右：`○ Idle` 状态（`○` 绿色 = idle / `◐` 黄色 = thinking / `●` 蓝色 = running / `⬤` 橙色 = waiting）
- 标题栏可以配置隐藏，通过快捷键切换显示

```css
HeaderBar {
    height: 1;
    background: $surface;
    border-bottom: solid $border;
    color: $text-muted;
    padding: 0 2;
    display: none;  /* 默认隐藏，按 Alt+H 显示 */
}
```

### 2. 聊天区域（全宽）

消息列表改为 `VerticalScroll`，不使用左右气泡，而是使用**块级消息**（类似 Telegram/Discord 的终端风格）：

```css
ChatWidget {
    width: 100%;
    height: 1fr;
    padding: 0 1;
    background: $background;
}

/* 消息块 */
.message-block {
    margin: 1 0;
    width: 100%;
}

.message-header {
    height: 1;
    color: $text-muted;
    text-style: bold;
    margin-bottom: 1;
}

.message-body {
    padding: 1 2;
    border-left: thick $border;
}

/* 用户消息 */
.user .message-header { color: $primary; }
.user .message-body { border-left-color: $primary; }

/* 助手消息 */
.assistant .message-header { color: $success; }
.assistant .message-body { border-left-color: $success; }

/* 系统/错误消息 */
.system .message-header { color: $warning; }
.system .message-body { border-left-color: $warning; background: $surface; }
```

### 3. 工具调用卡片（内联）

工具调用结果直接嵌套在 Assistant 消息中，作为可展开折叠的小卡片：

```
┌─▶ read src/limbo/config.py          ┐
│ ┌────────────────────────────────┐  │
│ │ 1  from pydantic import BaseModel│  │
│ │ 2  class Config(BaseModel):    │  │
│ │ 3      model: str = "deepseek"  │  │
│ └────────────────────────────────┘  │
└─✓ Completed                         ┘
```

- 状态：`▶ Calling...` → `✓ Completed` / `✗ Failed`
- 点击标题可展开/收起内容
- 默认只显示标题 + 状态（节省空间）
- 展开时显示完整输出（代码块带语法高亮）

### 4. 编辑/写文件确认（内联）

当需要用户确认时，直接在对话流中展示 Diff + 确认按钮：

```
System
┌────────────────────────────────────────┐
│ ⚠️  edit src/limbo/ui/widgets/chat.py  │
│ 需要确认                               │
│ ┌─ diff ─────────────────────────────┐ │
│ │ -  old line                       │ │
│ │ +  new line                       │ │
│ └────────────────────────────────────┘ │
│                                        │
│ [ 确认 (Y) ]    [ 取消 (N) ]           │
└────────────────────────────────────────┘
```

- 不弹 Modal，直接在消息流中展示
- 用户按 Y/N 或点击按钮即可响应
- 确认后自动继续对话流

### 5. 代码块（内联）

Assistant 回复中的代码块直接展示，带语法高亮和行号：

```css
.code-block {
    background: $background;
    border: solid $border;
    padding: 1 2;
    margin: 1 0;
}

.code-block-header {
    height: 1;
    color: $text-muted;
    border-bottom: solid $border;
    margin-bottom: 1;
}

.code-block pre {
    color: $text;
    text-style: none;
}
```

### 6. 流式输出光标

Assistant 消息最后显示闪烁块光标 `▋`：

```css
.streaming-cursor {
    color: $success;
    text-style: blink;
}
```

### 7. 输入栏（底部固定）

```css
InputArea {
    height: auto;
    min-height: 3;
    max-height: 10;
    background: $surface;
    border-top: solid $border;
    padding: 0 1;
    color: $text;
}

/* 占位符 */
InputArea::placeholder {
    color: $text-muted;
    text-style: italic;
}
```

- 多行输入（Enter 发送，Shift+Enter 换行）
- 右下角显示 `Ctrl+Enter` 快捷键提示
- 禁用状态时显示灰色遮罩

## 四、与现有方案对比

| 特性 | 原三栏方案 | 极简方案 |
|------|-----------|---------|
| 文件树 | 左侧固定面板 | 取消，文件在对话中引用 |
| 预览面板 | 右侧固定面板 | 取消，文件内容内联到代码块 |
| 工具状态 | 底部状态栏 | 内联到消息流中的卡片 |
| 确认弹窗 | 全屏 Modal | 内联到对话中（可配置） |
| 布局 | 5 区域 | 2 区域（聊天 + 输入） |
| 信息密度 | 分散在多个面板 | 全部集中在聊天流 |

## 五、交互模式

### 状态流转

```
User 输入 → 输入框 disabled → 显示 "Thinking..." 状态
                                    ↓
                              LLM 流式输出
                                    ↓
                              工具调用 → 内联展示工具卡片
                                    ↓
                              需要确认 → 内联展示 Diff + 按钮
                                    ↓
                              用户确认 → 继续输出
                                    ↓
                              输入框 enabled
```

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Enter` | 发送消息 |
| `Shift+Enter` | 输入换行 |
| `Alt+H` | 切换标题栏显示/隐藏 |
| `Ctrl+C` | 中断当前 Agent 运行 |
| `↑` | 历史消息输入（可选） |

## 六、实现建议

### 1. CSS 文件精简

相比三栏方案，CSS 可以减少 50% 以上。只需要：
- `message-block` 样式
- `tool-card` 样式（可展开）
- `code-block` 样式（语法高亮）
- `input-area` 样式
- `header-bar` 样式（可选）

### 2. Widget 结构简化

```
MainScreen
├── HeaderBar (可选, Hidden)
├── ChatWidget (VerticalScroll)
│   ├── MessageBlock (user)
│   ├── MessageBlock (assistant)
│   │   ├── ToolCard (read)
│   │   ├── ToolCard (edit)
│   │   ├── CodeBlock (python)
│   │   └── StreamingCursor
│   ├── MessageBlock (system / confirm)
│   └── ...
└── InputWidget (TextArea)
```

### 3. 确认流程优化

原方案：Modal → 阻塞 → 点击按钮 → 关闭 Modal → 继续
极简方案：在对话流中插入 `ConfirmBlock` → 用户按 Y/N → 自动替换为结果 → 继续

### 4. 代码高亮

使用 Rich 的 `Syntax` 在 `CodeBlock` 和 `ToolCard` 中实现：

```python
from rich.syntax import Syntax
from rich.text import Text

# 在工具卡片中
syntax = Syntax(content, lexer, theme="github-dark", line_numbers=True)
```

## 七、设计优先级

### P0 — 核心体验
1. 单栏布局（聊天 + 输入）
2. 消息块样式（左边框区分角色）
3. 代码块语法高亮
4. 流式输出光标

### P1 — 工具可视化
5. 工具调用卡片（可展开）
6. 内联确认 Diff（非 Modal）
7. 状态指示器（标题栏或消息内）

### P2 — 精致细节
8. 可隐藏标题栏
9. 消息时间戳
10. 历史输入记忆

---

*此方案完全基于对话流，不引入任何独立面板。实现更轻量，阅读体验更自然。*
