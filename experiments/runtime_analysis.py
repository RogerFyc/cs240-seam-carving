"""Runtime analysis for seam carving."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.seam_carving import carve_width
from src.utils import load_image, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--table-dir", type=str, default="results/tables")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--methods", nargs="+", default=["dp", "greedy"])
    parser.add_argument("--max-size", type=int, default=512)
    return parser.parse_args()


def resize_for_test(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / w
    height = max(1, int(round(h * scale)))
    pil = Image.fromarray(image.astype(np.uint8))
    return np.asarray(pil.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    image = load_image(args.input)

    table_dir = ensure_dir(ROOT / args.table_dir)
    fig_dir = ensure_dir(ROOT / args.fig_dir)

    widths = [128, 192, 256, 320, 384, 448, 512]
    widths = [w for w in widths if w <= args.max_size]

    records = []
    for width in widths:
        test_img = resize_for_test(image, width)
        h, w = test_img.shape[:2]
        target_width = int(round(0.9 * w))

        for method in args.methods:
            start = time.perf_counter()
            result = carve_width(test_img, target_width, method=method, show_progress=False)
            elapsed = time.perf_counter() - start

            records.append({
                "method": method,
                "height": h,
                "width": w,
                "pixels": h * w,
                "target_width": target_width,
                "removed_seams": w - target_width,
                "elapsed_seconds": elapsed,
            })
            print(f"width={w}, method={method}, time={elapsed:.3f}s")

    df = pd.DataFrame(records)
    table_path = table_dir / "runtime_analysis.csv"
    df.to_csv(table_path, index=False)

    plt.figure(figsize=(6, 4))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        plt.plot(sub["pixels"], sub["elapsed_seconds"], marker="o", label=method)
    plt.xlabel("Number of pixels")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime scalability")
    plt.legend()
    plt.tight_layout()

    fig_path = fig_dir / "runtime_analysis.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"Saved runtime table to {table_path}")
    print(f"Saved runtime figure to {fig_path}")


if __name__ == "__main__":
    main()
