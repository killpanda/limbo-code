# Limbo UI 重新设计方案

## 一、设计定位

**目标**: 打造一款开发者向的、现代且高效的终端 AI Coding Agent 界面。视觉上参考 VS Code、Cursor、Claude Code CLI 等工具，追求「IDE 级」的精致感，而非传统的聊天机器人 UI。

**核心原则**:
- **暗色优先**: 开发者长时间注视屏幕，深灰蓝底色降低视觉疲劳
- **信息层级清晰**: 通过颜色、边框、间距区分核心内容区与辅助信息
- **代码友好**: 代码块、文件路径、工具输出必须拥有最佳可读性
- **状态透明**: Agent 的每一个动作（思考中、调用工具、等待确认）都应有明确的视觉反馈

---

## 二、现有 UI 问题诊断

| 模块 | 现状 | 问题 |
|------|------|------|
| **整体布局** | 三列等宽，无 Header/Footer | 缺乏品牌感和全局状态区，信息扁平无焦点 |
| **Sidebar** | 标题 + 纯文本列表 + 状态 | 无图标、无层级、文件列表难以扫描 |
| **Chat** | Static 文本堆叠，无样式 | 用户/Agent 消息无区分，代码无高亮，阅读困难 |
| **Input** | 3行 TextArea，默认样式 | 无现代感，无占位提示，发送入口不直观 |
| **Preview** | 标题 + 内容文本 | 无代码语法高亮、无文件类型标识、无行号 |
| **Confirm** | 基础 Modal | 尚可，但缺乏 diff 视觉对比 |
| **状态反馈** | 仅 Sidebar 底部一行文字 | Agent 思考、流式输出、工具链无进度可视化 |

---

## 三、布局重构

### 新布局：「Header + 三栏 + Footer」

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🌀 Limbo  ·  deepseek-chat  ·  ● Running  ·  /Users/edy/project            │  ← Header Bar
├──────────┬─────────────────────────────────────────────┬──────────────────┤
│          │                                             │                  │
│  📁 FILES│  You                                        │  📄 preview.py   │
│  ─────── │  ┌─────────────────────────────────────┐    │  ─────────────── │
│  main.py │  │ 帮我分析一下这个项目               │    │  1  def foo():   │
│  config  │  └─────────────────────────────────────┘    │  2      pass     │
│  src/    │                                             │  3               │
│  tests/  │  🤖 Assistant (streaming...)              │                  │
│  ...     │  ┌─────────────────────────────────────┐    │                  │
│          │  │ 我来分析代码结构...                │    │                  │
│  ─────── │  │                                     │    │                  │
│  🔧 Tools│  │  [read: src/limbo/config.py]       │    │                  │
│  read    │  │  [read: src/limbo/agent.py]        │    │                  │
│  edit    │  │                                     │    │                  │
│  bash    │  │  ```python                         │    │                  │
│  grep    │  │  # 分析结果...                     │    │                  │
│  ...     │  │  ```                               │    │                  │
│          │  └─────────────────────────────────────┘    │                  │
│          │                                             │                  │
├──────────┴─────────────────────────────────────────────┴──────────────────┤
│  💬 Message Limbo...                                    [Ctrl+Enter] Send │  ← Input Bar
└─────────────────────────────────────────────────────────────────────────────┘
```

### 区域定义

| 区域 | 宽度 | 功能 |
|------|------|------|
| **Header** | 100% | 品牌标识、当前模型、Agent 状态指示器、工作目录 |
| **Sidebar** | 22-24 字符/约 20% | 文件树（最近访问）、工具快捷列表、会话信息 |
| **Chat** | 弹性 50-55% | 消息流、工具调用卡片、代码块、流式打字效果 |
| **Preview** | 弹性 25-28% | 文件内容预览、工具输出、Diff 对比（带语法高亮） |
| **Input** | 100% | 多行输入、占位符、发送快捷键提示 |

---

## 四、配色系统（暗色主题）

基于 GitHub Dark / VS Code Dark+ 的调色板，适配 Textual 的终端色域。

### 基础色板

| Token | Hex | Textual 变量 | 用途 |
|-------|-----|-------------|------|
| `bg-base` | `#0d1117` | `$background` | 应用底层背景 |
| `bg-panel` | `#161b22` | `$surface` | 面板、卡片、输入框背景 |
| `bg-hover` | `#21262d` | — | 悬停状态、当前行高亮 |
| `bg-active` | `#1f6feb` | `$primary` | 当前选中、激活状态 |
| `border` | `#30363d` | `$border` | 分割线、边框 |
| `border-focus` | `#58a6ff` | — | 聚焦时的边框高亮 |
| `text-primary` | `#c9d1d9` | `$text` | 主要文本 |
| `text-secondary` | `#8b949e` | `$text-muted` | 次要文本、时间戳、路径 |
| `text-accent` | `#58a6ff` | `$text-accent` | 链接、强调、Agent 名称 |
| `success` | `#238636` | `$success` | 成功、Apply 按钮、Agent 消息左侧线 |
| `warning` | `#d29922` | `$warning` | 警告、等待确认 |
| `error` | `#da3633` | `$error` | 错误、Reject 按钮 |
| `info` | `#1f6feb` | — | 信息提示、用户消息左侧线 |

### 消息气泡配色

| 角色 | 背景 | 左侧边框 | 文本颜色 |
|------|------|---------|---------|
| User | `bg-panel` (#161b22) | `info` (#1f6feb) 3px | `text-primary` |
| Assistant | `bg-panel` (#161b22) | `success` (#238636) 3px | `text-primary` |
| System/Error | `bg-panel` (#161b22) | `error` (#da3633) 3px | `error` |
| Tool Card | `bg-base` (#0d1117) | `border` (#30363d) 1px | `text-secondary` |

---

## 五、组件设计规范

### 1. Header Bar

```css
HeaderBar {
    height: 1;
    background: $surface;
    border-bottom: solid $border;
    color: $text-muted;
    padding: 0 2;
}
```

**内容**（左 → 右）:
- `🌀 Limbo` — 品牌名（`text-accent` + bold）
- `deepseek-chat` — 当前模型名（`text-muted`）
- `● Running` / `○ Idle` — 状态指示器（`success` / `text-muted`）
- 右对齐：`/Users/edy/project` — 工作目录（`text-muted`，截断显示）

### 2. Sidebar（左侧栏）

**分区 A — 文件树**（最近访问）:
```css
.sidebar-section-title {
    color: $text-muted;
    text-style: bold;
    padding: 1 0 0 1;
    border-bottom: solid $border;
}
.sidebar-file-item {
    color: $text;
    padding: 0 1;
    height: 1;
}
.sidebar-file-item:hover {
    background: $bg-hover;
}
.sidebar-file-item.active {
    background: $bg-active;
    color: white;
}
```

**文件项前缀**:
- 使用图标或字符：`📄` 文件、`📁` 目录、`🐍` .py、`⚙️` .toml 等
- 过长路径截断，hover 显示完整路径（Tooltip 或状态栏）

**分区 B — 工具列表**:
- 列出可用工具：`read`, `edit`, `write`, `bash`, `grep`, `find`, `ls`
- 每个工具带一个颜色小圆点，表示是否需要确认（🔴 需确认 / 🟢 直接执行）

**分区 C — 状态**:
- Agent 当前状态：`Idle` / `Thinking` / `Running: read` / `Waiting for confirmation`
- 使用动态色：Idle=灰，Thinking=黄，Running=蓝，Waiting=橙

### 3. ChatWidget（消息流）

**整体容器**:
```css
ChatWidget {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
    scrollbar-color: $border;
    scrollbar-color-hover: $text-muted;
}
```

**消息气泡**:
```css
.message-bubble {
    background: $surface;
    border-left: outer $accent;  /* User=info, Assistant=success */
    padding: 1 2;
    margin: 1 0;
    width: auto;
    max-width: 90%;
}
.message-header {
    color: $text-muted;
    text-style: bold;
    height: 1;
    margin-bottom: 1;
}
```

**消息结构**:
```
┌─ User ───────────────────────┐
│ 帮我分析这个项目             │
│                             │
│ 12:34                       │
└─────────────────────────────┘
```

**工具调用卡片**（嵌入在 Assistant 消息中）:
```css
.tool-card {
    background: $background;
    border: solid $border;
    padding: 1;
    margin: 1 0;
    width: 100%;
}
.tool-card-header {
    color: $text-accent;
    text-style: bold;
    height: 1;
}
.tool-card-body {
    color: $text-muted;
    max-height: 10;
    overflow: auto;
}
```

**工具卡片结构**:
```
┌─ [read] src/limbo/config.py ──────────────────┐
│  ▶ Calling...                                  │
│  ✓ Completed  (点击展开)                       │
│  ────────────────────────────────────────────── │
│  1 | from pydantic import BaseModel            │
│  2 | ...                                       │
└────────────────────────────────────────────────┘
```

**流式输出效果**:
- Assistant 消息底部显示一个闪烁光标 `▌` 或 `…` 表示正在生成
- 使用 `text-style: blink` 或定时替换字符实现

### 4. InputWidget（输入区）

```css
InputArea {
    height: auto;
    min-height: 3;
    max-height: 10;
    background: $surface;
    border: solid $border;
    border-top: tall $border;
    padding: 0 1;
    color: $text;
}
InputArea:focus {
    border: solid $border-focus;
}
```

**内容**:
- 多行文本输入（Enter 发送，Shift+Enter 换行）
- 占位符文本：`💬 Message Limbo...`（`text-muted`，italic）
- 右下角提示：`[Ctrl+Enter] Send`（`text-muted`）

**发送按钮**（可选增强）:
- 在输入框右侧增加一个 `⏎` 或 `Send` 按钮，支持鼠标点击发送

### 5. FilePreviewWidget（预览面板）

```css
FilePreviewWidget {
    width: 1fr;
    height: 1fr;
    background: $background;
    border-left: solid $border;
    padding: 0;
}
.preview-header {
    height: 1;
    background: $surface;
    color: $text-accent;
    text-style: bold;
    padding: 0 1;
    border-bottom: solid $border;
}
.preview-content {
    padding: 0 1;
    scrollbar-color: $border;
}
```

**内容**:
- 顶部 Header：文件路径 + 文件类型标签（如 `Python` `Bash` `Text`）
- 内容区：带行号的代码显示（使用 Rich 的 Syntax 高亮）
- 空状态：显示 `Select a file or run a tool to see output here`（居中，muted）

**Diff 模式**（ConfirmDialog 内嵌）:
```
  - old_line
  + new_line
```
- 删除行前缀 `─` 红色
- 新增行前缀 `+` 绿色
- 上下文行前缀 ` ` 灰色

### 6. ConfirmDialog（确认弹窗）

**优化**:
```css
ConfirmDialog > Vertical {
    width: 90;
    height: auto;
    max-height: 85vh;
    border: thick $border;
    background: $surface;
    padding: 1 2;
}
ConfirmDialog #diff-body {
    max-height: 60vh;
    background: $background;
    border: solid $border;
    padding: 1;
}
```

**内容**:
- 标题：`Apply edit?` / `Apply write?` / `Apply bash?`
- 副标题：文件路径（高亮显示）
- 主体：Unified diff 展示（红绿对比）
- 警告区：bash 安全警告（`warning` 色，带 ⚠️ 图标）
- 按钮：`[Apply]` (success) `[Reject]` (error) `[Show Full]` (optional)

---

## 六、交互状态设计

### Agent 状态机可视化

| 状态 | 视觉表现 | 位置 |
|------|---------|------|
| **Idle** | 灰色圆点 `○` + 文字 `Idle` | Header + Sidebar |
| **Thinking** | 黄色脉冲 `◐` 旋转 + `Thinking...` | Header + Sidebar |
| **Running** | 蓝色圆点 `●` + `Running: <tool>` | Header + Sidebar |
| **Streaming** | 绿色打字光标 `▌` 闪烁 | Chat 消息底部 |
| **Waiting Confirm** | 橙色圆点 `◉` + `Needs confirmation` | Header + Sidebar + 弹窗 |
| **Error** | 红色圆点 `◉` + `Error` | Header + Sidebar |

### 流式输出动画

在 Assistant 消息的最后一个字符后，持续显示一个闪烁的块光标：

```python
# 在 Textual 中可用定时器更新
self._cursor_visible = True
# 每 500ms 切换：text + "▌" / text + " "
```

或更简单的方案：使用 `text-style: blink` 在 Rich Text 中设置一个闪烁的 `▌` 字符。

### 工具调用展开/收起

工具卡片默认显示 Header + 状态（`▶ Calling...` / `✓ Done`），点击或按 Enter 展开显示完整输出。节省 Chat 区域垂直空间。

---

## 七、Textual CSS 实现建议

### 全局 CSS 文件

建议创建 `src/limbo/ui/styles.css` 并在 `LimboApp` 中引用：

```python
class LimboApp(App[None]):
    CSS_PATH = "styles.css"  # Textual 自动加载
```

### 关键 CSS 片段

```css
/* === 全局 === */
* {
    scrollbar-color: $border;
    scrollbar-color-hover: $text-muted;
    scrollbar-color-active: $text-accent;
}

/* === Header === */
HeaderBar {
    height: 1;
    background: $surface;
    border-bottom: solid $border;
    color: $text-muted;
    padding: 0 2;
}
HeaderBar #brand {
    color: $text-accent;
    text-style: bold;
}
HeaderBar #status-dot {
    color: $success;
    text-style: blink;
}

/* === Sidebar === */
SidebarWidget {
    width: 24;
    background: $surface;
    border-right: solid $border;
    padding: 0;
}
SidebarWidget .section-title {
    color: $text-muted;
    text-style: bold;
    padding: 1 1 0 1;
    border-bottom: solid $border;
}
SidebarWidget .file-item {
    color: $text;
    padding: 0 1;
    height: 1;
}
SidebarWidget .file-item:hover {
    background: $primary 10%;
}

/* === Chat === */
ChatWidget {
    width: 1fr;
    height: 1fr;
    background: $background;
    padding: 0 1;
}
.user-message {
    background: $surface;
    border-left: outer $primary;
    padding: 1 2;
    margin: 1 0;
    color: $text;
}
.assistant-message {
    background: $surface;
    border-left: outer $success;
    padding: 1 2;
    margin: 1 0;
    color: $text;
}
.tool-card {
    background: $background;
    border: solid $border;
    padding: 1;
    margin: 1 0;
}

/* === Input === */
InputWidget {
    height: auto;
    min-height: 3;
    max-height: 10;
    background: $surface;
    border: solid $border;
    border-top: tall $border;
    padding: 0 1;
    color: $text;
}
InputWidget:focus {
    border: solid $primary;
}
InputWidget .placeholder {
    color: $text-muted;
    text-style: italic;
}

/* === Preview === */
FilePreviewWidget {
    width: 1fr;
    height: 1fr;
    background: $background;
    border-left: solid $border;
}
FilePreviewWidget .preview-header {
    height: 1;
    background: $surface;
    color: $text-accent;
    text-style: bold;
    padding: 0 1;
    border-bottom: solid $border;
}
```

### 自定义 Widget 建议

为了完全实现设计效果，建议对现有 Widget 做以下调整：

1. **ChatWidget**: 将消息包装为自定义 `MessageBubble` Widget，支持头像、时间戳、代码折叠
2. **SidebarWidget**: 增加 `FileTreeItem` 自定义组件，支持图标前缀和悬停效果
3. **FilePreviewWidget**: 集成 Rich 的 `Syntax` 对象，实现代码高亮和行号
4. **HeaderBar**: 新增一个 `Static` 或 `Horizontal` Widget 作为全局 Header
5. **ToolCard**: 新增可展开/收起的工具调用结果卡片

---

## 八、设计优先级

### P0 — 必须实现（核心体验）
1. 全局暗色主题 + 配色系统
2. 消息气泡（User/Assistant 区分）
3. Header 状态栏 + Sidebar 文件树美化
4. Input 区域视觉优化

### P1 — 重要（提升专业感）
5. 工具调用卡片（展开/收起）
6. Preview 代码语法高亮 + 行号
7. Confirm Dialog Diff 红绿对比
8. 流式输出闪烁光标

### P2 — 加分（精致细节）
9. 文件类型图标（🐍⚙️📄）
10. 工具确认状态圆点（🔴🟢）
11. 消息时间戳
12. 空状态插画/提示

---

## 九、参考与灵感

- **Claude Code CLI**: 工具调用的展开卡片、流式光标、状态指示
- **Cursor Composer**: 代码块的精致展示、文件路径面包屑
- **VS Code Dark+**: 配色方案、代码高亮、面板分割
- **GitHub Copilot Chat**: 消息气泡、代码复制按钮
- **Warp Terminal**: 现代终端的圆角、阴影、区块感

---

*本方案只提供设计规范和原型图，不修改现有代码。实现时请参考各组件的 CSS 代码片段和布局说明。*
