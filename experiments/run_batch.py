"""Run methods on all images in data/input."""

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
from src.utils import load_image, save_image, save_comparison_grid, ensure_dir, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/input")
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--table-dir", type=str, default="results/tables")
    parser.add_argument("--target-width-ratio", type=float, default=0.75)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["standard", "dp", "forward", "greedy"],
        choices=["standard", "dp", "forward", "greedy"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ensure_dir(ROOT / args.output_dir)
    fig_dir = ensure_dir(ROOT / args.fig_dir)
    table_dir = ensure_dir(ROOT / args.table_dir)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir}. Run experiments/make_sample_data.py first."
        )

    all_records = []

    for image_path in image_paths:
        image = load_image(image_path)
        h, w = image.shape[:2]
        target_width = max(1, int(round(w * args.target_width_ratio)))
        target_height = h

        comparison_images = [image]
        comparison_titles = [f"original\n{w}x{h}"]

        for method in args.methods:
            if method == "standard":
                start = time.perf_counter()
                out = standard_resize(image, target_width=target_width, target_height=target_height)
                elapsed = time.perf_counter() - start
                removed_seams = 0
            else:
                result = carve_to_size(
                    image,
                    target_width=target_width,
                    target_height=target_height,
                    method=method,
                    show_progress=False,
                )
                out = result.image
                elapsed = result.elapsed_seconds
                removed_seams = result.removed_seams

            out_path = output_dir / f"{image_path.stem}_{method}_{target_width}x{target_height}.png"
            save_image(out, out_path)

            oh, ow = out.shape[:2]
            all_records.append({
                "image": image_path.name,
                "method": method,
                "original_width": w,
                "original_height": h,
                "target_width": target_width,
                "target_height": target_height,
                "output_width": ow,
                "output_height": oh,
                "removed_seams": removed_seams,
                "elapsed_seconds": elapsed,
            })

            comparison_images.append(out)
            comparison_titles.append(f"{method}\n{ow}x{oh}\n{elapsed:.2f}s")
            print(f"{image_path.name} [{method}] time={elapsed:.3f}s")

        fig_path = fig_dir / f"{image_path.stem}_batch_comparison.png"
        save_comparison_grid(comparison_images, comparison_titles, fig_path)

    table_path = table_dir / "batch_results.csv"
    pd.DataFrame(all_records).to_csv(table_path, index=False)
    print(f"Saved batch results to {table_path}")


if __name__ == "__main__":
    main()