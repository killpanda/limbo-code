# Bundled daily skills

The daily-mode starter pack. Seen only by daily sessions, at the lowest
priority: a skill of the same name in `~/.limbo/skills`, a project's
`.agents/skills` or `~/.limbo/daily/skills` replaces it, and `skill_manage`
copies a skill to `~/.limbo/daily/skills` before changing it.

Every skill here is ported from Nous Research's
[hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT; each
directory keeps its LICENSE). Porting changes: tool names mapped to limbo's
(`web_extract` → `web_fetch`, `read_file` → `read`, `terminal` → `bash`),
Hermes paths and env vars replaced (`~/.limbo/daily/...`,
`LIMBO_CITATION_LEDGER`), cron and browser references removed, upstream
tests dropped. Scripts are otherwise kept as upstream wrote them (excluded
from ruff and mypy).
