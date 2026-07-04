#!/usr/bin/env python3
"""Interactive card classifier: step through images, save report-style results.

Produces the same output style as the project report: the original scene with
the current card tinted, next to a panel showing the detected ``Suit`` and
``Rank``. Handles both single-card and multi-card images
for a multi-card
image it steps through one card at a time. Every rendered frame is saved to a
``results/`` folder (created if missing).

Usage
-----
    # one image, a folder of images, or several of each (files and/or folders):
    python scripts/run_pipeline.py ../proj3_data/CardImages
    python scripts/run_pipeline.py ../proj3_data/CardImages ../proj3_data/Multi_CardImages
    python scripts/run_pipeline.py ../proj3_data/Multi_CardImages/Cards_1.tif

Options
-------
    --rank-model PATH   trained rank .pkl (default: models/rank.pkl if it exists)
    --results DIR       output folder (default: results)
    --no-wait           don't pause for Enter between cards (just save them all)

Press <Enter> in the terminal to advance to the next card
type 'quit' then Enter
to quit
closing the figure window does NOT stop progress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Make `import cardclassifier` work no matter where this is run from: the
# project root (this file's parent's parent) is added to sys.path. No install
# needed, and the data can live anywhere outside this folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardclassifier import dataset, io_utils, pipeline  # noqa: E402
from cardclassifier.suit_classifier import ThresholdSuitClassifier  # noqa: E402


_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_TINT = np.array([90, 90, 235], dtype=np.float64)   # blue overlay, like the report


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for pth in paths:
        p = Path(pth)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in _EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"  (skipping, not found: {p})")
    return files


def _render(image: np.ndarray, mask, suit: str, rank: str) -> Image.Image:
    """Original scene (current card tinted) + a Suit/Rank text panel."""
    rgb = np.stack([image] * 3, axis=-1).astype(np.float64)
    if mask is not None:
        rgb[mask] = 0.45 * rgb[mask] + 0.55 * _TINT
    left = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    h = left.height
    panel = Image.new("RGB", (max(260, left.width // 2), h), (242, 242, 242))
    d = ImageDraw.Draw(panel)
    big = _font(max(28, h // 12))
    lines = ["Suit:", suit, "Rank:", rank]
    total = len(lines) * (big.size + 10)
    y = (h - total) // 2
    for ln in lines:
        w = d.textbbox((0, 0), ln, font=big)[2]
        d.text(((panel.width - w) // 2, y), ln, fill=(20, 20, 20), font=big)
        y += big.size + 10

    out = Image.new("RGB", (left.width + panel.width, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(panel, (left.width, 0))
    return out


def _show(pil_img: Image.Image):
    try:
        import matplotlib.pyplot as plt
        plt.figure("card")
        plt.clf()
        plt.imshow(pil_img)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.1)
        return plt
    except Exception:
        return None


def _auto_discover() -> list[str]:
    """Find the card-image folders automatically (data lives outside proj3)."""
    return [str(p) for p in dataset.image_dirs()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive card classifier.")
    ap.add_argument("paths", nargs="*", help="image files and/or folders "
                    "(optional: auto-discovers CardImages/Multi_CardImages if omitted)")
    ap.add_argument("--rank-model", default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    paths = args.paths or _auto_discover()
    if not paths:
        print("No image folders given and none auto-discovered.\n"
              "Pass a path, e.g.:  python scripts/run_pipeline.py ../proj3_data/CardImages")
        return 2

    import pickle

    def _load(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    rank_path = args.rank_model or (ROOT / "models" / "rank.pkl")
    rank_clf = _load(rank_path) if Path(rank_path).exists() else None
    if rank_clf is None:
        print("[note] no models/rank.pkl -> rank shown as '?'.")

    thr_path = ROOT / "models" / "suit_thresholds.json"
    if thr_path.exists():
        suit_clf = ThresholdSuitClassifier.load(thr_path)
    else:
        suit_clf = None
        print("[note] no models/suit_thresholds.json -> suit shown as '?'.")
    print()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    files = _collect(paths)
    if not files:
        print("No images found.")
        return 2

    plt = None
    for f in files:
        res = pipeline.classify_image(io_utils.load_gray(f),
                                      rank_clf=rank_clf, suit_clf=suit_clf)
        print(f"{f.name}: {res.n_cards} card(s) detected")
        for c in res.cards:
            suit = c.suit or "?"
            rank = c.rank if c.rank is not None else "?"
            print(f"   card {c.index}:  Suit: {suit}   Rank: {rank}")
            frame = _render(io_utils.load_gray(f), c.mask, suit, rank)
            out = results_dir / f"{f.stem}_card{c.index}.png"
            frame.save(out)
            if not args.no_wait:
                plt = _show(frame) or plt
                ans = input("   <Enter> = next card,  type 'quit' = stop: ").strip().lower()
                if ans == "quit":
                    if plt is not None:
                        plt.close("all")
                    print(f"\nStopped. Results saved to {results_dir}/")
                    return 0
    if plt is not None:
        plt.close("all")
    print(f"\nDone. Results saved to {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
