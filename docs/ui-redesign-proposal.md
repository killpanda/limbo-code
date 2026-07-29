# Limbo TUI 重设计方案 — pi 风格极简单栏

> 日期: 2026-07-29 · 现状截图: [assets/limbo-current-ui.png](assets/limbo-current-ui.png)
> 方向: 放弃三栏布局，改为 pi / Claude Code 式**单栏沉浸聊天界面**。

## 一、现状 Review

当前为「左 sidebar / 中 chat / 右 preview」三栏布局，核心问题：

### P0 — 功能性缺陷

1. **Preview 面板不可滚动** — `FilePreviewWidget` 是裸 `Static`，长输出直接溢出裁剪。
2. **输入框存在感过低** — 无边框、无 placeholder，黑底上几乎不可见。
3. **无等待反馈** — LLM 响应期间只有 sidebar 一行小字，看起来像卡死。

### P1 — 布局方向性错误

4. **三栏等宽（各 1fr）** — 聊天区只有 1/3 屏宽，长回复频繁折行；sidebar 只有 4 行文字却占 1/3 宽度。
5. **信息分散** — 工具结果在右侧 preview、状态在左侧 sidebar、对话在中间，视线需要在三栏间跳跃。对话式工具的自然动线是**自上而下一条流**。
6. **无视觉分隔、默认 Header 显示 `LimboApp`**。

### P2 — 消息呈现

7. Assistant 回复无 Markdown 渲染，代码块全是纯文本。
8. 用户消息右对齐 + `You:` 前缀，风格混乱；无时间戳、无轮次分隔。
9. 工具调用/错误以 `[read failed: ...]` 纯文本混入对话流。

> 结论：问题不只是"不好看"，而是**三栏布局本身不适合对话式 agent**。修补 CSS 收益有限，应直接转向单栏方案。

---

## 二、目标设计：pi 风格单栏

参考 pi 的设计哲学：**聊天流是唯一主角，工具调用内联在流里，持久 chrome 只有一个输入框。**

### 布局

```
  ● Limbo                                            deepseek-v4-flash · ~/limbo-code   ← 顶部 1 行状态栏（可省略）

  ❯ 帮我看一下 agent.py 里的确认流程是怎么实现的？                                          ← 用户消息（accent 色，无前缀）

  ● 确认流程是这样的：当工具返回 `requires_confirmation=True` 时…                          ← assistant（Markdown 渲染）

  ⠋ thinking…                                                                             ← 等待指示（出现后消失）

  ── read src/limbo/agent.py ─────────────────────────────────────────────── ✓ 0.1s      ← 工具调用，一行折叠态
  ── edit src/limbo/config.py ────────────────────────────────────────────── ⏸ 等待确认
       ▼ 展开时（点击或快捷键）内联显示 diff/输出，带语法高亮，可滚动
  ── bash rm -rf build/ ─────────────────────────────────────────────────── ✗ 已拒绝

 ╭──────────────────────────────────────────────────────────────────────────────────────╮
 │ 输入消息…                                                                              │  ← 输入框：唯一的持久边框
 ╰──────────────────────────────────────────────────────────────────────────────────────╯
  Enter 发送 · Shift+Enter 换行 · ctrl+o 展开工具输出 · Esc 中断                            ← 底部 1 行 hint
```

### 与现状的映射

| 现状 | 新方案 |
|------|--------|
| 左 sidebar（标题/最近文件/状态） | **删除**。模型名+工作目录移到顶部 1 行状态栏；工具状态内联在聊天流；recent files 移除（需要时用 `read` 即可） |
| 右 preview 面板 | **删除**。工具输出内联为聊天流中的折叠卡片，默认一行摘要，展开显示完整输出（`Collapsible` + 语法高亮 + 可滚动） |
| 中 chat | 占满全宽，Markdown 渲染 |
| 确认弹窗 | 保留 modal（唯一例外），样式精简 |

### 组件设计

1. **状态栏（1 行）**：左 `● Limbo`（spinner 联动 agent 状态），右 `model · workdir`。无 Header 组件，直接用 `Static` + dock。
2. **聊天流 `ChatWidget`**（重写）：
   - 用户消息：`❯ ` 前缀 + `$text-accent` 色，全宽，无气泡；
   - assistant：`textual.widgets.Markdown`，流式用 `Markdown.append()`；
   - 工具卡片：单行 `Static` 摘要（图标 + 工具名 + 参数摘要 + 状态符 ✓/⏸/✗ + 耗时），点击/`ctrl+o` 展开为 `Collapsible` 内的 `RichLog`（语法高亮、可滚动、`max_lines` 上限）；
   - thinking 行：spinner `⠋ thinking…`，首个 `TextDelta` 到达时移除；
   - 新消息自动 `scroll_end()`，用户上翻时暂停跟随。
3. **输入框**：`border: round $accent`（全局唯一的圆角边框，视觉锚点），placeholder `输入消息…`，高度 3→8 自适应，focus 时边框提亮。
4. **底部 hint 行（1 行）**：`Enter 发送 · Shift+Enter 换行 · Esc 中断`，`$text-muted` 色。
5. **确认弹窗**：保留 modal；diff 用 `Syntax` 高亮；按钮标 `[Y] Apply` / `[N] Reject` 并加键绑定；bash 警告独立 `$warning` 色块。

### 样式

新建 `src/limbo/ui/app.tcss` 集中管理，`App.CSS_PATH = "app.tcss"`，移除各 widget 的 `DEFAULT_CSS`。颜色全部走主题变量，预留：

```toml
[ui]
theme = "textual-dark"
```

---

## 三、实施计划

| 步骤 | 内容 | 涉及文件 | 预估 |
|------|------|----------|------|
| 1 | 布局改单栏：删 sidebar/preview 容器，加状态栏+hint 行，输入框样式 | `screens/main.py`、新增 `app.tcss` | 2h |
| 2 | 工具输出内联卡片（单行摘要 + Collapsible 展开，替代 preview 面板） | `screens/main.py`、新增 `widgets/tool_card.py` | 3h |
| 3 | 聊天流升级：Markdown 流式、❯/● 前缀、thinking spinner、自动滚动 | `widgets/chat.py`（重写） | 3h |
| 4 | 确认弹窗精简（高亮 + y/n 快捷键）、主题配置、删 sidebar.py/file_preview.py | `confirm.py`、`config.py` | 2h |

步骤 1+2 落地后即获得 pi 的基本形态；3、4 为体验打磨。`ChatWidget` 重写需同步更新 `tests/ui/test_widgets.py`、`test_confirmation.py`。

## 四、注意事项

- **删除而非改造**：`sidebar.py` 和 `file_preview.py` 直接删除，避免保留两套布局的维护成本。`MainScreen` 中对它们的 `query_one` 调用一并清理。
- **工具卡片是唯一新抽象**：状态机为 `running → success | error | pending_confirm → applied | rejected`，卡片随 `ToolResultEvent` 原地更新（`update()`），不追加新消息。
- **安全性**：assistant 用 `Markdown` widget（不解析任意 markup）；工具输出保持 `Text()` 原样渲染。
- **回归保障**：引入 `pytest-textual-snapshot`，固化布局快照。
- 当前 Textual 8.2.8，`Markdown.append()`、`Collapsible`、`RichLog` 均可用。
