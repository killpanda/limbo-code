"""Generate a human-readable Markdown report for a pilot run.

Usage: python -m swebench_runner.report --run-dir runs/pilot-50

Reads results.jsonl + predictions.jsonl from the run dir, the official
per-instance eval reports under logs/run_evaluation/, and the agent's
stderr logs (for token counts). Writes <run-dir>/REPORT.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_LOGS = REPO_ROOT / "logs" / "run_evaluation"

TOKENS_RE = re.compile(r"exec end: total_tokens=(\d+)")

# Failure classes in precedence order; "resolved" is the non-failure case.
CLASSES = [
    "infra_error",
    "timeout",
    "agent_error",
    "empty_patch",
    "patch_not_applied",
    "eval_error",
    "tests_failed",
    "resolved",
]

CLASS_LABELS = {
    "infra_error": "基础设施失败（装环境/起容器炸了）",
    "timeout": "Agent 超时被杀",
    "agent_error": "Agent 报错退出",
    "empty_patch": "空 patch（没改任何代码）",
    "patch_not_applied": "patch apply 不上",
    "eval_error": "评测本身失败/超时",
    "tests_failed": "patch 应用成功但测试没过",
    "resolved": "✅ 已解决",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _tokens_for(run_dir: Path, instance_id: str) -> int | None:
    log_file = run_dir / "logs" / instance_id / "stderr.log"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text().splitlines()):
        if m := TOKENS_RE.search(line):
            return int(m.group(1))
    return None


def _eval_report(run_dir: Path, instance_id: str) -> dict | None:
    pattern = str(
        EVAL_LOGS / f"{run_dir.name}-{instance_id}" / "*" / instance_id
        / "report.json"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    return json.loads(Path(matches[0]).read_text())[instance_id]


def classify(record: dict, report: dict | None) -> str:
    """Assign one failure class per instance (see CLASSES precedence)."""
    status = record["exit_status"]
    if status == "infra_error":
        return "infra_error"
    if status == "timeout":
        return "timeout"
    if status == "agent_error":
        return "agent_error"
    if not record["patch_chars"]:
        return "empty_patch"
    if record["resolved"] is None or report is None:
        return "eval_error"
    if not report.get("patch_successfully_applied", False):
        return "patch_not_applied"
    return "resolved" if record["resolved"] else "tests_failed"


def build_report(run_dir: Path) -> str:
    records = _load_jsonl(run_dir / "results.jsonl")
    if not records:
        raise SystemExit(f"no results in {run_dir}/results.jsonl")

    rows = []
    for r in records:
        iid = r["instance_id"]
        report = _eval_report(run_dir, iid)
        rows.append({
            **r,
            "class": classify(r, report),
            "report": report,
            "tokens": _tokens_for(run_dir, iid),
        })

    resolved = [r for r in rows if r["class"] == "resolved"]
    by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for r in rows:
        by_class[r["class"]].append(r)

    total_tokens = sum(r["tokens"] or 0 for r in rows)
    lines = [
        f"# SWE-bench Pilot Report — {run_dir.name}",
        "",
        f"- **Resolve rate: {len(resolved)}/{len(rows)}"
        f" ({len(resolved) / len(rows):.1%})**",
        f"- Total tokens: {total_tokens:,}"
        f" (avg {total_tokens // len(rows):,}/instance)",
        f"- Agent 阶段总耗时: {sum(r['duration_s'] for r in rows) / 3600:.1f} h",
        "",
        "## 失败分类",
        "",
        "| 类别 | 数量 | 实例 |",
        "|---|---|---|",
    ]
    for c in CLASSES:
        group = by_class[c]
        if not group:
            continue
        ids = ", ".join(f"`{r['instance_id']}`" for r in group[:10])
        if len(group) > 10:
            ids += f" …（共 {len(group)} 个）"
        lines.append(f"| {CLASS_LABELS[c]} | {len(group)} | {ids} |")

    # Detail sections for the actionable failure classes.
    for c in ("tests_failed", "patch_not_applied", "timeout", "empty_patch"):
        group = by_class[c]
        if not group:
            continue
        lines += ["", f"## 明细：{CLASS_LABELS[c]}", ""]
        for r in group:
            lines.append(f"### `{r['instance_id']}`")
            lines.append(
                f"- exit={r['exit_status']}, patch={r['patch_chars']} chars,"
                f" tokens={r['tokens'] or '?'}, agent 耗时={r['duration_s']:.0f}s"
            )
            report = r["report"]
            if c == "tests_failed" and report:
                failed = report.get("tests_status", {}).get(
                    "FAIL_TO_PASS", {}
                ).get("failure", [])
                p2p_failed = report.get("tests_status", {}).get(
                    "PASS_TO_PASS", {}
                ).get("failure", [])
                if failed:
                    lines.append("- FAIL_TO_PASS 未通过：")
                    lines += [f"  - `{t}`" for t in failed[:5]]
                if p2p_failed:
                    lines.append(
                        f"- 破坏了 {len(p2p_failed)} 个 PASS_TO_PASS 测试"
                        f"（回归）"
                    )
            lines.append("")

    lines += [
        "## Top 10 耗时实例",
        "",
        "| 实例 | 结果 | agent 耗时 | tokens |",
        "|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["duration_s"])[:10]:
        lines.append(
            f"| `{r['instance_id']}` | {r['class']} | {r['duration_s']:.0f}s"
            f" | {r['tokens'] or '?'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="swebench_runner.report",
        description="Render REPORT.md for a pilot run dir.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <run-dir>/REPORT.md")
    args = parser.parse_args()

    markdown = build_report(args.run_dir)
    output = args.output or args.run_dir / "REPORT.md"
    output.write_text(markdown)
    print(f"report written: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
