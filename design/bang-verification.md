# Bang command（`!`）隔离环境验证记录

- 日期：2026-08-13
- 分支：`feat/bang-command`（含 commit `ad2f1bc`）
- RFC：design/rfc-bang-command.md
- 验证方式：真实 `LimboApp`（textual `run_test()` pilot）+ 真实 `BashTool` 子进程（无 mock）；LLM 侧用记录型 stub client，用于断言"无 LLM 请求"并构造一个可被 ESC 打断的长 turn。验证脚本：`/tmp/verify_bang.py`（一次性，未入库）。

## 结果：7/7 通过

| # | 路径 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | `!echo hello` | ✅ | 卡片 `success`，body 含 `hello`；`llm_calls=0`（无任何 LLM 请求）；`agent.messages` 长度保持 1（仅 system），未进历史 |
| 2 | `!false` | ✅ | 卡片 `error` 态，body = `Command failed with exit code 1.`（exit code 信息保留，失败路径填的是 output 而非 error） |
| 3 | 孤立 `!` | ✅ | 对话流出现 `用法：! <bash 命令>（直接执行，不发送给模型）`；未产生 bash 卡片、未发 LLM 请求 |
| 4 | ` !echo hi`（前导空格） | ✅ | 走普通消息路径：LLM client 收到的末条消息 content 恰为 `!echo hi`；未产生 bash 卡片 |
| 5 | `!seq 1 5000`（长输出） | ✅ | 卡片 `success`，body 含 `[Showing lines 3001-5000 of 5000. Full output: /var/folders/.../limbo-output-*.log]`；spill 文件真实存在且含完整 5000 行 |
| 6 | turn 运行中敲 bang + ESC 打断 turn | ✅ | bang（`!sleep 2 && echo survived`）在 turn 运行中立即并行执行（不进 steer 队列，`queued_count=0`）；ESC 打断 turn 后 bang 卡片保持 `running`（未被 `cancel_running_tool_cards()` 误标），2 秒后正常转 `success`，body 含 `survived` |
| 7 | `!sleep 4711` 运行中退出 app | ✅ | 退出前 `pgrep` 可见 `sleep 4711` 进程；app unmount（`on_unmount` → 私有 `BashTool.cancel()`）后 0.2s 内进程消失，无孤儿 |

## 备注

- 验证环境为临时目录 `tempfile.mkdtemp()` 作为 workdir/session_dir，与日常 `~/.limbo` 数据完全隔离。
- 第 7 项初次用 `sleep 60 # comment` 做 pgrep 标记失败：`bash -c` 对单条简单命令会 exec 优化，注释不出现在进程 cmdline 中；改用特殊 sleep 时长做标记后通过（验证脚本问题，非产品问题）。
- 自动化回归：`tests/ui/test_bang_ui.py` 10/10 通过；`make check` 全量 841 passed + ruff + mypy 全绿。
