"""Run seam carving methods on one image."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.resize_baseline import standard_resize
from src.seam_carving import carve_to_size
from src.utils import load_image, save_image, save_comparison_grid, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["all"], choices=["all", "standard", "dp", "greedy"])
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--table-dir", type=str, default="results/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    methods = args.methods
    if "all" in methods:
        methods = ["standard", "dp", "greedy"]

    image_path = Path(args.input)
    image = load_image(image_path)
    h, w = image.shape[:2]

    target_width = args.target_width
    target_height = args.target_height
    if target_width is None:
        target_width = max(1, int(round(w * 0.75)))
    if target_height is None:
        target_height = h

    output_dir = ensure_dir(ROOT / args.output_dir)
    fig_dir = ensure_dir(ROOT / args.fig_dir)
    table_dir = ensure_dir(ROOT / args.table_dir)

    base_name = image_path.stem
    records = []
    comparison_images = [image]
    comparison_titles = [f"original\n{w}x{h}"]

    for method in methods:
        if method == "standard":
            start = time.perf_counter()
            out = standard_resize(image, target_width=target_width, target_height=target_height)
            elapsed = time.perf_counter() - start
        else:
            result = carve_to_size(
                image,
                target_width=target_width,
                target_height=target_height,
                method=method,
                show_progress=True,
            )
            out = result.image
            elapsed = result.elapsed_seconds

        out_path = output_dir / f"{base_name}_{method}_{target_width}x{target_height}.png"
        save_image(out, out_path)

        oh, ow = out.shape[:2]
        records.append({
            "image": image_path.name,
            "method": method,
            "original_width": w,
            "original_height": h,
            "target_width": target_width,
            "target_height": target_height,
            "output_width": ow,
            "output_height": oh,
            "elapsed_seconds": elapsed,
        })

        comparison_images.append(out)
        comparison_titles.append(f"{method}\n{ow}x{oh}\n{elapsed:.2f}s")
        print(f"[{method}] saved to {out_path}, time={elapsed:.3f}s")

    fig_path = fig_dir / f"{base_name}_comparison.png"
    save_comparison_grid(comparison_images, comparison_titles, fig_path)

    table_path = table_dir / f"{base_name}_single_image_results.csv"
    pd.DataFrame(records).to_csv(table_path, index=False)

    print(f"Saved comparison figure to {fig_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
