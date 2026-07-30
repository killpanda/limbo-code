#!/usr/bin/env python3
"""对比度校验 CLI（RFC P2-3）：全枚举前景 × 三级背景矩阵。

用法::

    python scripts/check_contrast.py           # 全量报告，违规时退出码 1
    python scripts/check_contrast.py --failures-only
"""

from __future__ import annotations

import sys

from limbo.ui.contrast import check_palettes, format_report


def main() -> int:
    failures_only = "--failures-only" in sys.argv
    results, failures = check_palettes()
    shown = [r for r in results if r.status != "OK"] if failures_only else results
    print(format_report(shown))
    print(f"\n{len(results)} pairs checked, {failures} violation(s) "
          f"(text pairs must be >= 4.5:1, decorative >= 3.0:1)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
