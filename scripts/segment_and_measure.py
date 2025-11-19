# scripts/segment_and_measure.py
import argparse
import os
import time

import numpy as np
import imageio.v2 as imageio
import pandas as pd

from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage import measure, morphology


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("preproc_tif", help="Preprocessed TIFF (background-corrected)")
    p.add_argument("sample_id", help="Sample ID (must match metadata table)")
    p.add_argument("metadata_tsv", help="Metadata TSV with sample_id column")
    p.add_argument("rois_dir", help="Output directory for ROIs")
    p.add_argument("out_table", help="Output TSV table for EcoTaxa")

    # smoothing + area filter
    p.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for smoothing (segment step)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=30,
        help="Min object area (pixels)",
    )
    p.add_argument(
        "--max-area",
        type=int,
        default=1000000,
        help="Max object area (pixels)",
    )

    # morphology
    p.add_argument(
        "--dilate-radius",
        type=int,
        default=0,
        help="Radius for binary dilation (0 = no dilation)",
    )
    p.add_argument(
        "--open-radius",
        type=int,
        default=0,
        help="Radius for binary opening to break thin connections (0 = off).",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Base pixel margin around each ROI (constant part).",
    )
    p.add_argument(
        "--margin-factor",
        type=float,
        default=0.0,
        help=(
            "Extra margin proportional to sqrt(area). "
            "Effective margin = margin + margin_factor * sqrt(area)."
        ),
    )

    # thresholding
    p.add_argument(
        "--thresh-mode",
        choices=["global_otsu", "local_mean", "hybrid"],
        default="global_otsu",
        help=(
            "global_otsu = classic Otsu on whole image; "
            "local_mean = purely adaptive; "
            "hybrid = local AND global (safer, fewer false positives)."
        ),
    )
    p.add_argument(
        "--local-window",
        type=int,
        default=151,
        help="Window size (odd) for local threshold (pixels).",
    )
    p.add_argument(
        "--local-offset",
        type=float,
        default=0.0,
        help=(
            "Offset for local threshold (in intensity units). "
            "Positive -> more conservative (fewer objects)."
        ),
    )

    # simple contrast filter on each ROI
    p.add_argument(
        "--min-contrast",
        type=float,
        default=0.0,
        help=(
            "Minimum (p95 - p5) contrast inside ROI. "
            "ROIs with lower contrast are discarded."
        ),
    )

    return p.parse_args()


def make_binary_mask(img_smooth, args):
    """
    Create a binary mask of 'objects darker than background'
    using global / local / hybrid threshold.
    """
    # Global Otsu
    t_global = threshold_otsu(img_smooth)
    mask_global = img_smooth < t_global

    if args.thresh_mode == "global_otsu":
        return mask_global

    # Local mean threshold
    block_size = max(3, int(args.local_window))
    if block_size % 2 == 0:
        block_size += 1

    local_thr = threshold_local(
        img_smooth,
        block_size=block_size,
        offset=args.local_offset,
        method="mean",
    )
    mask_local = img_smooth < local_thr

    if args.thresh_mode == "local_mean":
        return mask_local

    # Hybrid: AND to reduce crazy speckles while still handling gradients
    mask_hybrid = mask_local & mask_global
    return mask_hybrid


def main():
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.rois_dir, exist_ok=True)
    out_dir = os.path.dirname(args.out_table)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ---- load image ----
    img = imageio.imread(args.preproc_tif)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)

    # ---- smooth ----
    if args.sigma > 0:
        img_smooth = gaussian(img, sigma=args.sigma, preserve_range=True)
    else:
        img_smooth = img

    # ---- thresholding ----
    binary = make_binary_mask(img_smooth, args)

    # remove small noise specks before labeling
    binary = morphology.remove_small_objects(binary, args.min_area)

    # optional opening to break thin bridges between particles
    if args.open_radius > 0:
        from skimage.morphology import opening, disk

        binary = opening(binary, disk(args.open_radius))

    # optional dilation to “inflate” objects slightly
    if args.dilate_radius > 0:
        from skimage.morphology import dilation, disk

        binary = dilation(binary, disk(args.dilate_radius))

    # ---- connected components ----
    labeled = measure.label(binary)
    props = measure.regionprops(labeled, intensity_image=img)

    print(f"Found {len(props)} raw objects (before area & contrast filter)")

    # ---- metadata ----
    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    meta["sample_id"] = meta["sample_id"].astype(str)
    row = meta.loc[meta["sample_id"] == str(args.sample_id)]

    if row.empty:
        print(f"WARNING: no metadata row found for sample_id={args.sample_id}")
        meta_info = {}
    else:
        row = row.iloc[0]
        meta_info = {col: row[col] for col in row.index if col != "sample_id"}

    # ---- collect ROIs ----
    records = []
    obj_idx = 0

    for region in props:
        area = region.area
        if area < args.min_area or area > args.max_area:
            continue

        # size-dependent margin
        extra = int(args.margin_factor * np.sqrt(float(area)))
        eff_margin = max(0, int(args.margin) + extra)

        minr, minc, maxr, maxc = region.bbox

        minr = max(minr - eff_margin, 0)
        minc = max(minc - eff_margin, 0)
        maxr = min(maxr + eff_margin, img.shape[0])
        maxc = min(maxc + eff_margin, img.shape[1])

        roi = img[minr:maxr, minc:maxc]
        if roi.size == 0:
            continue

        # optional contrast filter on this ROI
        if args.min_contrast > 0:
            p95 = np.percentile(roi, 95)
            p5 = np.percentile(roi, 5)
            if p95 - p5 < args.min_contrast:
                continue

        obj_idx += 1
        obj_id = f"{args.sample_id}_obj{obj_idx:05d}"
        roi_name = f"{obj_id}.png"
        roi_path = os.path.join(args.rois_dir, roi_name)
        imageio.imwrite(roi_path, roi.astype(np.uint16))

        cy, cx = region.centroid

        rec = {
            "object_id": obj_id,
            "img_file_name": roi_name,
            "sample_id": args.sample_id,
            "x": cx,
            "y": cy,
            "area": area,
            "perimeter": region.perimeter,
            "major_axis_length": region.major_axis_length,
            "minor_axis_length": region.minor_axis_length,
            "eccentricity": region.eccentricity,
        }
        for k, v in meta_info.items():
            rec[k] = v

        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_table, sep="\t", index=False)
    print(f"Saved {len(df)} objects to {args.out_table}")
    print(f"ROIs stored in {args.rois_dir}")

    elapsed = time.time() - start_time
    print(
        f"Total time for segment_and_measure on sample {args.sample_id}: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()

