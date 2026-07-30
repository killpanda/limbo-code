"""Regenerate the startup banner pixel map from a source image.

Usage:

    python scripts/gen_banner.py /path/to/mascot.jpg [--width 40]

Prints a ``STARTUP_ART`` block to paste into ``src/limbo/ui/banner.py``.

Pipeline (tuned for flat-color cartoon art on a solid background):

1. BOX downscale to ``width`` columns (height keeps the source aspect,
   rounded to an even number for half-block pairing).
2. Classify each pixel to the nearest palette color in RGB space; the
   sampled background blue maps to transparent.
3. Despeckle: flip only fully isolated pixels (all 8 neighbors differ),
   which removes JPEG noise but keeps 1px outlines intact.
4. Crop fully transparent rows/columns.

Requires Pillow (dev-only dependency, not needed at runtime).
"""

from __future__ import annotations

import argparse
from collections import Counter

from PIL import Image

# Palette sampled from the source image; keys match banner.py.
PALETTE = {"R": (237, 88, 82), "Y": (254, 231, 89), "D": (21, 24, 40)}
BACKGROUND = (48, 105, 169)  # rendered as transparent
ALL = list(PALETTE.items()) + [(" ", BACKGROUND)]


def classify(pixel: tuple[int, int, int]) -> str:
    return min(ALL, key=lambda kv: sum((a - b) ** 2 for a, b in zip(pixel, kv[1])))[0]


def to_grid(img: Image.Image, width: int) -> list[list[str]]:
    w0, h0 = img.size
    height = round(width * h0 / w0 / 2) * 2
    small = img.resize((width, height), Image.Resampling.BOX)
    return [
        [classify(small.getpixel((x, y))) for x in range(width)]
        for y in range(height)
    ]


def despeckle_isolated(grid: list[list[str]]) -> list[list[str]]:
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for y in range(h):
        for x in range(w):
            neighbors = [
                grid[yy][xx]
                for yy in range(max(0, y - 1), min(h, y + 2))
                for xx in range(max(0, x - 1), min(w, x + 2))
                if (yy, xx) != (y, x)
            ]
            if not neighbors:
                continue
            top, count = Counter(neighbors).most_common(1)[0]
            # Flip fully isolated pixels (JPEG speckle) and small holes
            # where 7+ of 8 neighbors agree — 1px outlines only have 6
            # agreeing neighbors along their sides, so lines survive.
            if top != grid[y][x] and (count >= 7 or all(c != grid[y][x] for c in neighbors)):
                out[y][x] = top
    return out


def crop(grid: list[list[str]]) -> list[list[str]]:
    h, w = len(grid), len(grid[0])
    cols = [x for x in range(w) if any(grid[y][x] != " " for y in range(h))]
    rows = [y for y in range(h) if any(grid[y][x] != " " for x in range(w))]
    out = [
        [grid[y][x] for x in range(cols[0], cols[-1] + 1)]
        for y in range(rows[0], rows[-1] + 1)
    ]
    if len(out) % 2:  # even row count for half-block pairing
        out.append([" "] * len(out[0]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="source mascot image")
    parser.add_argument("--width", type=int, default=40, help="target columns before cropping")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    grid = crop(despeckle_isolated(to_grid(img, args.width)))
    lines = ["".join(row).rstrip() for row in grid]

    print(f"# {max(len(line) for line in lines)} cols x {len(lines)} px rows"
          f" -> {len(lines) // 2} terminal rows")
    print('STARTUP_ART = """\\')
    for line in lines:
        print(line)
    print('"""')


if __name__ == "__main__":
    main()
