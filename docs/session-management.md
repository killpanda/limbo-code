# Session 管理设计

> 状态：已实现 P0/P1（store、恢复、CLI、斜杠命令、选择器、导出）和 P2 的 `/fork`、`/handoff`。prune 尚未实现。

## 背景

此前 Limbo 只会把会话写入 `~/.limbo/sessions/{timestamp}-{pid}.jsonl`，
没有任何"读"的能力：不能恢复、不能列表、不能切换，文件只增不减。
本设计为 Limbo 增加完整的 session 管理功能。

## 功能分层

| 层级 | 功能 | 状态 |
|------|------|------|
| P0 | `limbo --continue` / `limbo --resume <id>` | ✅ |
| P0 | 会话元数据（meta 行） | ✅ |
| P1 | `/sessions` 选择器（TUI） | ✅ |
| P1 | 自动标题（首条用户消息截断） | ✅ |
| P1 | `/export` 导出 Markdown / JSONL 全量日志 | ✅ |
| P1 | `/new` 开始新会话 | ✅ |
| P2 | `/fork` 会话分叉、`/handoff` 交接 | ✅ |
| P2 | prune 清理旧会话 | ⬜ 未实现 |

## 存储格式

JSONL，首行为 meta，后续为消息行。**向后兼容**：无 meta 行的旧文件可正常加载。

```jsonl
{"type":"meta","id":"20260720-103000-123456-42","workdir":"/path/to/proj","model":"deepseek-v4-pro","title":"修复 grep 工具 bug","created_at":"...","updated_at":"..."}
{"role":"system","content":"..."}
{"role":"user","content":"..."}
{"role":"assistant","content":"...","tool_calls":[...]}
{"role":"tool","tool_call_id":"...","content":"..."}
```

- `id` 即文件名 stem（不含 `.jsonl`），`--resume` 支持 id 前缀匹配
- 每次保存原子重写全量（沿用原有策略），并刷新 `updated_at`
- `title` 取首条 user 消息（折叠空白、截断 50 字符）
- `list_sessions` 只读每个文件的首行，O(1)/文件；旧文件回退用 mtime 排序、标题为空

## 模块划分

```
src/limbo/
├── sessions.py                  # 纯逻辑，无 UI 依赖
│     ├── SessionMeta            # id / workdir / model / title / created_at / updated_at / path
│     ├── save_session(path, meta, messages)
│     ├── load_session(path) -> (SessionMeta, list[Message])
│     ├── list_sessions(dir, workdir=None) -> list[SessionMeta]   # updated_at 倒序
│     ├── latest_session(dir, workdir=None) -> Path | None
│     ├── find_session(dir, id_prefix) -> Path                    # 前缀匹配，歧义报错
│     ├── export_markdown(meta, messages, path)
│     └── export_jsonl(meta, messages, path, trace_path=None)     # 合并 trace 的全量导出
├── trace.py                     # TraceLogger：追加式 JSONL 全链路日志（traces/ 子目录）
├── agent.py                     # Agent(config, llm_client, workdir, session_dir, resume=path)
├── app.py                       # --continue / --resume [ID]
└── ui/
    ├── screens/session_picker.py  # 会话选择 Modal
    └── screens/main.py            # 斜杠命令路由：/sessions /new /export /help
```

## 恢复语义

- **系统消息不恢复**：加载后丢弃文件中第一条 system 消息，用当前环境重新生成
  （AGENTS.md 可能已变更）
- **悬空 tool_calls 修复**：旧会话末尾若有未配对的 `tool_calls`（崩溃导致），
  加载时补占位 tool 消息（`[session restored: tool call interrupted]`），
  避免 OpenAI API 报 400
- **沿用原文件**：resume 不产生新 session 文件，继续写同一文件
- **workdir 跟随会话**：CLI resume 时若 meta 中的 workdir 仍存在，切换过去
- **model 用当前配置**：meta 中记录的原 model 仅作展示

## 过滤与查找

- `--continue` / `/sessions` 默认只列**当前 workdir** 的会话（会话与项目强绑定）
- `limbo --resume <id>` 在全部会话中做前缀匹配；匹配多个时报错并列出候选

## 斜杠命令

输入以 `/` 开头时自动弹出命令菜单（实时过滤，Enter 执行/补全，Tab 补全，
Esc 关闭），不发往 LLM，走命令路由：

| 命令 | 行为 |
|------|------|
| `/sessions` | 弹出会话选择器，Enter 切换，Esc 取消 |
| `/new` | 清空当前对话，开始新会话（新文件） |
| `/export [path]` | 导出会话日志：默认 JSONL 全量日志（`~/.limbo/exports/<id>.jsonl`，含完整 LLM 请求体、token 用量、工具执行、报错）；路径以 `.md` 结尾时导出 Markdown |
| `/tracing-viewer [id\|off]` | 在浏览器查看 trace（见下文「Tracing 网页查看器」）；忙时可用 |
| `/fork [n]` | 把当前会话分叉成新会话并切过去；`n` = 只保留前 n 轮用户对话（省略则全部保留）。原会话文件不变 |
| `/handoff <说明>` | 开一个干净上下文的新会话，以这段说明作为第一条消息发出；提示里带原会话文件路径，模型需要细节时可以自己读 |
| `/help` | 显示命令列表 |

### 分叉与交接

- 子会话是**自包含的拷贝**：`/fork` 把活跃历史（含压缩摘要和对应的
  compaction 记录）截到分叉点写进新文件，加载时不需要读父会话；父会话
  之后的追加也不会影响子会话。
- meta 记录血缘：`parent_session` 为父会话 id，`fork_at` 为复制的最后一条
  父消息 id（`/handoff` 为空）。会话选择器在子会话后面显示 `⑂ <父会话 id>`。
- 子会话不继承 goal、usage 账本和排队消息；Code Mode 等模式会继承。
- headless 同样支持 `/fork [n]` 和 `/handoff <说明>`。

切换会话时聊天区重绘：user / assistant 文本原样渲染，历史 tool 消息
以 info 行省略提示（不重建工具卡片）。

## Trace 全链路日志

每次运行期间，Agent 会把完整执行过程追加写入
`~/.limbo/sessions/traces/<session_id>.trace.jsonl`（独立于会话文件，
append-only，崩溃/打断最多丢失写入中的一条）。`/export` 默认把 meta +
全部 trace 记录 + 最终消息快照合并导出为单个 JSONL。

每行一个 JSON 记录，公共字段：`ts`（ISO 毫秒）、`type`。记录类型：

| type | 关键字段 |
|------|---------|
| `session_start` | `trace_schema`（当前 2）、config 快照（不含 api_key）、limbo/python 版本、是否 resume |
| `user_message` | `turn`、`content` |
| `llm_request` | `turn`、`iteration`、`attempt`（同一次调用的第几次 HTTP 尝试）、`body`（完整请求体：messages 含 system prompt、tools、所有参数） |
| `llm_response` | `started_at`、`duration`、`ttft`、`finish_reason`、`usage`（原始用量，含 provider 缓存字段）、`cached_tokens`（归一化缓存命中）、`content` / `reasoning` 全文及长度、`tool_calls`（id、name、arguments） |
| `llm_error` | `exception_type`、`error`、`traceback` |
| `tool_call` / `tool_result` | `id`、`name`、`arguments`、`success`、`output`/`error`、耗时；崩溃时带 `exception_type` + `traceback` |
| `error` | `kind`（如 max_iterations / length_stop / length_stop_loop）、`message` |
| `turn_end` | `started_at`、`iterations`、`duration`、`status`（completed） |
| `session_save_error` | `error` |

`tool_result`、`compaction` 同样带 `started_at`；`turn_interrupted`（phase
为 llm_stream 时）带已收到的 `content` / `reasoning`。schema 1 的旧文件没有
这些字段，读取方用 `ts - duration` 推算开始时间。

导出文件布局：`meta` 行 → trace 记录（按时间序；trace 缺失时退化为原始
消息）→ `messages_snapshot`（持久化后的完整消息历史）。

## Tracing 网页查看器

`/tracing-viewer` 在本机起一个只读 HTTP 服务（`src/limbo/tracing/`，仅标准库），
在浏览器里按「会话 → 轮次 → LLM 调用 / 工具调用」展示 trace，类似 Langfuse：

- 左栏轮次列表（状态、耗时、token、错误数，可搜索 / 只看出错）；中栏瀑布图
  （LLM 条上浅色段为首 token 时间，重试归入同一调用）；右栏详情（输出、
  reasoning、工具参数与输出、本次请求相对上次新增的消息、完整请求体）。
- 当前会话每秒增量刷新（只读新增字节），也能切换到同 workdir 的历史会话。
- `/tracing-viewer <id 前缀>` 打开历史会话，`/tracing-viewer off` 关闭服务；
  limbo 退出时服务随之关闭。
- 安全：只绑 127.0.0.1、随机端口；URL 带一次性 token（之后存 HttpOnly
  cookie），并校验 Host 头。SSH 远程或无图形界面时不自动开浏览器，只打印
  链接，可用 `ssh -L <port>:127.0.0.1:<port>` 转发。

模块：`index.py`（字节偏移索引 + 摘要，大字段按偏移懒加载）、`model.py`
（扁平记录 → 会话树，纯逻辑）、`server.py`（`TraceViewer`）、`command.py`
（TUI / headless 共用的命令逻辑）、`static/index.html`（单文件前端）。

## 测试接缝（seams）

- `sessions.py`：save/load roundtrip、meta 行、旧格式兼容、列表排序与过滤、
  前缀查找、导出
- `Agent`：resume 后续写同一文件、系统消息重生成、悬空 tool_calls 修复、
  标题自动填充
- CLI：`--continue` / `--resume` 参数解析与分发
- UI：斜杠命令路由、picker 选择、会话切换后渲染
