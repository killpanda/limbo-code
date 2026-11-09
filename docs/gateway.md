# Gateway: limbo in Telegram and WeChat

`limbo gateway` is a long-running process that lets you talk to limbo from
chat apps. It speaks two channels:

- **Telegram**, through the Bot API (long polling; the client is written by
  hand on `reqwest`, no Telegram crate).
- **Personal WeChat**, through Tencent's iLink bot protocol
  (`ilinkai.weixin.qq.com`), the same one the official WeChat ClawBot plugin
  uses. You log in by scanning a QR code; no phone emulator, no hooking.

Every chat gets its own session, in **daily** posture by default (the
general assistant from [daily-mode.md](daily-mode.md)). Only the **owners**
recorded at login can talk to the gateway; messages from anyone else are
ignored without a reply.

## Quick start

```bash
# 1. Telegram: create a bot with @BotFather, then
limbo channels login telegram --token 123456:ABC...
#    and send your bot any private message within 5 minutes: that makes
#    you its owner.

# 2. WeChat: show a QR code in the terminal and scan it with WeChat
limbo channels login weixin
#    The account that scans is the owner.

limbo channels status        # what is configured
limbo gateway                # run in the foreground (Ctrl-C stops)
limbo gateway install        # or run it as a background service
```

The gateway needs an LLM key like any other limbo run (`limbo auth login`).

## In the chat

Write as you would in the TUI. While a reply is running, a new message
*steers* the running turn; `/queue <text>` waits for it to finish instead.
The usual slash commands work (`/new`, `/compact`, `/model`, `/btw`,
`/daily-mode`, `/help`, …), plus a few chat-only ones:

| Command | What it does |
|---|---|
| `/stop` | interrupt the running turn |
| `/sessions` | list this chat's sessions |
| `/resume <id>` | switch back to an older session of this chat |
| `/status` | session id, model, posture, tokens |

Photos, files and voice messages are downloaded into
`<session dir>/media/<session>/` and attached to your message (images go
inline to vision models). WeChat voice arrives as the server's transcript.
The agent can send files back with the `send_file` tool ("send me the
report as a file").

When the agent asks a multiple-choice question (`ask_user`), Telegram shows
buttons; WeChat shows a numbered list and you answer with the number.

### Differences between the channels

| | Telegram | WeChat |
|---|---|---|
| Streaming | the reply is edited in place as it streams | sent in pieces when each paragraph is done (no editing) |
| Formatting | Markdown rendered as Telegram HTML | plain text |
| Message length | 4096 characters, split above | about 2000 characters, split above |
| Groups | `channels.telegram.groups = "mention"`: reply when mentioned or replied to | private chats only |
| Files | up to 50 MB | through the encrypted CDN |

## Configuration

All settings have defaults; add only what you want to change to
`~/.limbo/config.toml`:

```toml
[gateway]
workdir = "~/limbo-chat"     # working directory of chat sessions
posture = "daily"            # or "coding"
model = ""                   # empty = [llm] model
idle_unload_secs = 1800      # drop an idle chat's agent from memory (0 = never)
max_attachment_mb = 20
show_tools = true            # tool progress line (Telegram)

[channels.telegram]
enabled = true
owners = []                  # extra owner ids, added to those from login
proxy = ""                   # e.g. "socks5://127.0.0.1:7890" where Telegram is blocked
groups = "mention"           # "mention" or "off"

[channels.weixin]
enabled = true
owners = []
```

State lives under `~/.limbo/channels/`:

- `credentials.json` (mode 0600): bot tokens and owners from login;
- `sessions.json`: which session each chat is on, and its history;
- `cursor-*`: long-polling positions, so a restart neither loses nor
  repeats messages;
- `gateway.lock`: only one gateway runs per machine;
- `gateway.log`: service output.

`limbo channels logout <telegram|weixin>` removes a channel's credentials.

## Running as a service

`limbo gateway install` registers the gateway with the system's service
manager, using the current binary and any `--config` / `--session-dir`
given, and starts it. `limbo gateway uninstall` stops and removes it.

**macOS** — a launchd user agent, `~/Library/LaunchAgents/dev.limbo.gateway.plist`
(`RunAtLoad`, `KeepAlive`), loaded with `launchctl bootstrap gui/<uid>`
(falling back to `launchctl load -w` on older systems). It runs while you
are logged in. Check it with:

```bash
launchctl print gui/$(id -u)/dev.limbo.gateway
tail -f ~/.limbo/channels/gateway.log
```

**Linux** — a systemd user unit, `~/.config/systemd/user/limbo-gateway.service`,
enabled with `systemctl --user enable --now`. To keep it running after you
log out, run `loginctl enable-linger $USER` once.

```bash
systemctl --user status limbo-gateway
journalctl --user -u limbo-gateway -f
```

The service gets `PATH`, `LANG`, `LIMBO_HOME`, `LIMBO_AUTH_FILE`, the proxy
variables and `SSL_CERT_FILE` from the shell you install it from; reinstall
after changing them.

## Risks and limits

- **WeChat iLink is not a public API.** It is the protocol behind
  Tencent's own ClawBot plugin, documented only by that plugin's source.
  Tencent can change it or restrict the account at any time. Use an account
  you can afford to lose access to the bot feature on.
- WeChat replies need a recent message from you: the server hands out a
  per-chat context token that expires. When it expires, the gateway holds
  the pending replies and delivers them after your next message.
- When the WeChat login expires (`-14`), the gateway says so in the log;
  run `limbo channels login weixin` again.
- Anyone who controls an owner account controls limbo on this machine,
  including `bash` in coding posture. Limbo has no sandbox (see the
  README's Safety section); keep `posture = "daily"` unless you need more.
- Bot tokens are stored in plain text in `credentials.json` (0600).

## How it fits together

```
Telegram adapter ─┐                              ┌─ worker (chat A) ─ Conversation ─ Agent
                  ├─ inbox ─ router ─ dedup ─ owners ─┤
WeChat adapter  ──┘                              └─ worker (chat B) ─ …
                                                        │
                       Renderer (AgentEvent → OutItem) ─ outbox ─ Channel::send
```

See `AGENTS.md` (Gateway) and `CONTEXT.md` (Gateway glossary) for the
code map.
