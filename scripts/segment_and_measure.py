# scripts/segment_and_measure.py

import argparse
import os
import time
import math

import numpy as np
import imageio.v2 as imageio
import pandas as pd

from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage import measure, morphology


def parse_args():
    p = argparse.ArgumentParser(
        description="Segment full ZooScan TIFF, extract ROIs, and write EcoTaxa-style table."
    )
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
        help="Radius for binary dilation (0 = no dilation; keeps thin parts connected)",
    )
    p.add_argument(
        "--open-radius",
        type=int,
        default=0,
        help="Radius for binary opening (0 = no opening; breaks tiny bridges)",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Base pixel margin around each ROI",
    )
    p.add_argument(
        "--margin-factor",
        type=float,
        default=0.0,
        help="Extra margin ~= margin_factor * sqrt(area); "
             "useful to give big animals proportionally more padding.",
    )

    # thresholding
    p.add_argument(
        "--thresh-mode",
        choices=["global_otsu", "local_mean", "hybrid"],
        default="global_otsu",
        help=(
            "global_otsu = classic Otsu on whole image; "
            "local_mean = purely adaptive; "
            "hybrid = local AND global (reduces background junk)."
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
            "Positive -> more conservative (fewer objects); "
            "negative -> more permissive."
        ),
    )

    # simple contrast filter on each ROI
    p.add_argument(
        "--min-contrast",
        type=float,
        default=0.0,
        help="If >0: drop ROIs with (p95 - p5) < min_contrast (in intensity units).",
    )

    # optional removal of border fragments
    p.add_argument(
        "--drop-border-frac",
        type=float,
        default=0.0,
        help=(
            "If >0: drop components that touch the image border AND have area "
            "< drop-border-frac * full-image-area. "
            "Useful to get rid of tiny cut-off edges."
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

    # Hybrid = local AND global → reduces crazy background speckles
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

    H, W = img.shape
    full_area = float(H * W)

    # ---- smooth ----
    if args.sigma > 0:
        img_smooth = gaussian(img, sigma=args.sigma, preserve_range=True)
    else:
        img_smooth = img

    # ---- thresholding ----
    binary = make_binary_mask(img_smooth, args)

    # optional opening to break tiny bridges / specks
    if args.open_radius > 0:
        selem_open = morphology.disk(args.open_radius)
        binary = morphology.opening(binary, selem_open)

    # remove small noise specks before labeling
    binary = morphology.remove_small_objects(binary, args.min_area)

    # optional dilation to “inflate” objects slightly
    if args.dilate_radius > 0:
        selem_dil = morphology.disk(args.dilate_radius)
        binary = morphology.dilation(binary, selem_dil)

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
        area = float(region.area)

        # basic area filter
        if area < args.min_area or area > args.max_area:
            continue

        minr, minc, maxr, maxc = region.bbox

        # optional drop of small border fragments
        if args.drop_border_frac > 0.0:
            touches_border = (
                minr == 0 or minc == 0 or maxr == H or maxc == W
            )
            if touches_border:
                if area < args.drop_border_frac * full_area:
                    # looks like a tiny cut-off piece at image edge → skip
                    continue

        # dynamic margin: base + margin_factor * sqrt(area)
        margin = args.margin
        if args.margin_factor > 0.0:
            extra = int(round(args.margin_factor * math.sqrt(area)))
            margin += max(0, extra)

        minr = max(minr - margin, 0)
        minc = max(minc - margin, 0)
        maxr = min(maxr + margin, H)
        maxc = min(maxc + margin, W)

        roi = img[minr:maxr, minc:maxc]
        if roi.size == 0:
            continue

        # simple contrast filter on the ROI if requested
        if args.min_contrast > 0.0:
            p5 = np.percentile(roi, 5.0)
            p95 = np.percentile(roi, 95.0)
            if (p95 - p5) < args.min_contrast:
                # very flat → likely junk
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

