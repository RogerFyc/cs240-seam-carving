"""Run object protection or object removal demo with a rectangular mask.

Examples:

Protect a region:
    python experiments/run_mask_demo.py \
        --input data/input/astronaut.png \
        --target-width 360 \
        --mode protect \
        --rect 170 40 340 330 \
        --method forward

Remove a region:
    python experiments/run_mask_demo.py \
        --input data/input/astronaut.png \
        --target-width 360 \
        --mode remove \
        --rect 250 250 340 360 \
        --method dp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.masks import create_rect_mask, overlay_mask
from src.seam_carving import carve_to_size
from src.utils import load_image, save_image, save_comparison_grid, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--mode", choices=["protect", "remove"], required=True)
    parser.add_argument("--rect", nargs=4, type=int, required=True, metavar=("x0", "y0", "x1", "y1"))
    parser.add_argument("--method", choices=["dp", "forward", "greedy"], default="forward")
    parser.add_argument("--protect-weight", type=float, default=1e6)
    parser.add_argument("--remove-weight", type=float, default=1e6)
    parser.add_argument("--output-dir", type=str, default="data/output")
    parser.add_argument("--fig-dir", type=str, default="results/figures")
    parser.add_argument("--table-dir", type=str, default="results/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.input)
    image = load_image(image_path)
    h, w = image.shape[:2]

    if args.target_width >= w:
        raise ValueError(f"target_width must be smaller than original width {w}.")

    x0, y0, x1, y1 = args.rect
    mask = create_rect_mask(image.shape, x0, y0, x1, y1)

    output_dir = ensure_dir(ROOT / args.output_dir)
    fig_dir = ensure_dir(ROOT / args.fig_dir)
    table_dir = ensure_dir(ROOT / args.table_dir)

    base_name = image_path.stem
    overlay = overlay_mask(
        image,
        mask,
        color=(0, 255, 0) if args.mode == "protect" else (255, 0, 0),
        alpha=0.45,
    )

    # Run the same method without mask as a fair comparison.
    no_mask_result = carve_to_size(
        image,
        target_width=args.target_width,
        method=args.method,
        show_progress=True,
    )

    if args.mode == "protect":
        masked_result = carve_to_size(
            image,
            target_width=args.target_width,
            method=args.method,
            show_progress=True,
            protect_mask=mask,
            protect_weight=args.protect_weight,
        )
    else:
        masked_result = carve_to_size(
            image,
            target_width=args.target_width,
            method=args.method,
            show_progress=True,
            remove_mask=mask,
            remove_weight=args.remove_weight,
        )

    no_mask_path = output_dir / f"{base_name}_{args.method}_no_mask_{args.target_width}.png"
    masked_path = output_dir / f"{base_name}_{args.method}_{args.mode}_mask_{args.target_width}.png"
    overlay_path = output_dir / f"{base_name}_{args.mode}_mask_overlay.png"

    save_image(no_mask_result.image, no_mask_path)
    save_image(masked_result.image, masked_path)
    save_image(overlay, overlay_path)

    fig_path = fig_dir / f"{base_name}_{args.method}_{args.mode}_mask_demo.png"
    save_comparison_grid(
        [image, overlay, no_mask_result.image, masked_result.image],
        [
            f"original\n{w}x{h}",
            f"{args.mode} mask",
            f"{args.method} no mask\n{no_mask_result.elapsed_seconds:.2f}s",
            f"{args.method} with {args.mode} mask\n{masked_result.elapsed_seconds:.2f}s",
        ],
        fig_path,
    )

    table_path = table_dir / f"{base_name}_{args.method}_{args.mode}_mask_demo.csv"
    pd.DataFrame([
        {
            "image": image_path.name,
            "method": args.method,
            "mode": "no_mask",
            "target_width": args.target_width,
            "elapsed_seconds": no_mask_result.elapsed_seconds,
            "removed_seams": no_mask_result.removed_seams,
        },
        {
            "image": image_path.name,
            "method": args.method,
            "mode": args.mode,
            "target_width": args.target_width,
            "elapsed_seconds": masked_result.elapsed_seconds,
            "removed_seams": masked_result.removed_seams,
        },
    ]).to_csv(table_path, index=False)

    print(f"Saved mask overlay to {overlay_path}")
    print(f"Saved no-mask result to {no_mask_path}")
    print(f"Saved masked result to {masked_path}")
    print(f"Saved comparison figure to {fig_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()