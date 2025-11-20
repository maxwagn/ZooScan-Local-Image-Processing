# scripts/clean_rois_batch.py

import argparse
import os

import numpy as np
import imageio.v2 as imageio
import pandas as pd

from skimage.filters import gaussian, threshold_otsu
from skimage import measure, morphology
from scipy.ndimage import gaussian_filter


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Post-process ROIs for a sample: "
            "keep a single main object, optionally mask background and adjust tone, "
            "and write a clean EcoTaxa table (1:1 with input ROIs)."
        )
    )
    p.add_argument("rois_dir", help="Directory with original ROI PNGs for one sample")
    p.add_argument("orig_table", help="Original EcoTaxa-style TSV table for that sample")
    p.add_argument("clean_rois_dir", help="Directory to write cleaned ROI PNGs")
    p.add_argument("clean_table", help="Output TSV with cleaned objects")

    # segmentation inside ROI (for mask only)
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma (px) for *mask* smoothing (object detection only).",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum component area (pixels) within each ROI.",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Unused when keeping full ROI size (kept for compatibility).",
    )

    # background masking parameters
    p.add_argument(
        "--mask-background",
        dest="mask_background",
        action="store_true",
        help="Replace everything outside the main object mask by a flat background.",
    )
    p.add_argument(
        "--no-mask-background",
        dest="mask_background",
        action="store_false",
        help="Do NOT mask background; keep original ROI as-is.",
    )
    p.set_defaults(mask_background=False)

    p.add_argument(
        "--mask-dilate-radius",
        type=int,
        default=5,
        help="Radius (pixels) to dilate the object mask BEFORE masking "
             "(keeps hairs / antenna tips).",
    )
    p.add_argument(
        "--mask-feather-sigma",
        type=float,
        default=2.0,
        help="Sigma (pixels) for Gaussian feathering of the mask edge; 0 = hard edge.",
    )
    p.add_argument(
        "--bg-percentile",
        type=float,
        default=99.0,
        help="Percentile of intensities inside the ROI used as background level "
             "when masking (e.g. 99–99.5 ≈ very bright background).",
    )

    # tone / brightness adjustment AFTER masking
    p.add_argument(
        "--tone-mode",
        choices=["none", "gamma", "stretch"],
        default="none",
        help=(
            "How to adjust brightness/contrast AFTER masking: "
            "none = keep as is; "
            "gamma = apply gamma correction; "
            "stretch = percentile-based contrast stretch."
        ),
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma for --tone-mode gamma (gamma < 1 brightens, > 1 darkens).",
    )
    p.add_argument(
        "--stretch-low",
        type=float,
        default=1.0,
        help="Low percentile for --tone-mode stretch (e.g. 1).",
    )
    p.add_argument(
        "--stretch-high",
        type=float,
        default=99.0,
        help="High percentile for --tone-mode stretch (e.g. 98–99).",
    )

    return p.parse_args()


def apply_tone(crop_masked, tone_mode, gamma, stretch_low, stretch_high):
    """Apply optional brightness/contrast adjustment AFTER masking."""
    if tone_mode == "none":
        return crop_masked

    img = crop_masked.astype(np.float32)

    if tone_mode == "gamma":
        # normalise to 0–1, apply gamma, scale back to 0–65535
        vmin = img.min()
        vmax = img.max()
        if vmax > vmin:
            x = (img - vmin) / (vmax - vmin)
            x = np.clip(x, 0.0, 1.0)
            x = np.power(x, gamma)  # gamma < 1 → brighter
            img_out = x * 65535.0
        else:
            img_out = img
        return img_out

    if tone_mode == "stretch":
        # gentle contrast stretch between given percentiles
        lo = np.percentile(img, stretch_low)
        hi = np.percentile(img, stretch_high)
        if hi <= lo:
            return img
        x = np.clip(img, lo, hi)
        x = (x - lo) / (hi - lo)
        img_out = x * 65535.0
        return img_out

    # fallback
    return img


def choose_main_component(props, img_shape, min_area):
    """
    Choose a single component using a heuristic:

    1. ignore components with area < min_area
    2. prefer components that do NOT touch any image edge
    3. among those, prefer larger area and more central position
    4. if all touch edges, fall back to the best we can find
    """
    if not props:
        return None

    h, w = img_shape
    cy_img = h / 2.0
    cx_img = w / 2.0
    max_dist = np.hypot(cy_img, cx_img) + 1e-6

    candidates = []
    for r in props:
        if r.area < min_area:
            continue

        minr, minc, maxr, maxc = r.bbox
        touches_edge = (
            minr == 0 or minc == 0 or maxr == h or maxc == w
        )

        cy, cx = r.centroid
        dist_center = np.hypot(cy - cy_img, cx - cx_img)

        # sort key: (touches_edge, -area, dist_to_center)
        candidates.append(
            (touches_edge, -float(r.area), float(dist_center), r)
        )

    if not candidates:
        return None

    # best = minimal key → prefers non-edge, large, central
    candidates.sort()
    return candidates[0][3]


def process_single_roi(
    img,
    sigma,
    min_area,
    margin,  # kept for API compatibility, not used when keeping full ROI
    mask_background,
    mask_dilate_radius,
    mask_feather_sigma,
    bg_percentile,
    tone_mode,
    gamma,
    stretch_low,
    stretch_high,
):
    """
    Take one ROI image and return:
      - crop_out: ROI-sized, optionally background-masked & tone-adjusted image (uint16)
      - rloc: regionprops for the main object (for updated morphometrics)

    We keep the **full ROI size**: same width/height in and out.
    """

    # ----- prepare intensities -----
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    h, w = img.shape

    # Work on a smoothed copy to build a robust mask
    if sigma > 0:
        img_smooth = gaussian(img, sigma=sigma, preserve_range=True)
    else:
        img_smooth = img

    # objects darker than background
    t = threshold_otsu(img_smooth)
    binary = img_smooth < t

    # remove tiny specks
    binary = morphology.remove_small_objects(binary, min_size=min_area)

    # label components
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)

    if not props:
        # nothing found -> skip this ROI entirely
        return None

    # ---- choose a single main component (size + edge + centrality heuristic) ----
    main_region = choose_main_component(props, img.shape, min_area=min_area)
    if main_region is None:
        return None

    mask_full = (labeled == main_region.label)

    # optional dilation: keep hairs / antennae
    if mask_dilate_radius > 0:
        selem = morphology.disk(mask_dilate_radius)
        mask_full = morphology.dilation(mask_full, selem)

    # recompute regionprops on the *binary* main-object mask only
    labeled_local = measure.label(mask_full)
    props_local = measure.regionprops(labeled_local, intensity_image=img)
    if not props_local:
        return None
    rloc = props_local[0]

    # ----- background masking (on full-size ROI) -----
    if mask_background:
        mask_float = mask_full.astype(np.float32)

        # feather edges for smoother transition
        if mask_feather_sigma > 0:
            mask_float = gaussian_filter(mask_float, sigma=mask_feather_sigma)
            mask_float = np.clip(mask_float, 0.0, 1.0)

        # background level from bright part of the ROI
        bg_level = np.percentile(img, bg_percentile)
        crop_masked = mask_float * img + (1.0 - mask_float) * bg_level
    else:
        crop_masked = img

    # ----- tone adjustment AFTER masking -----
    crop_toned = apply_tone(
        crop_masked,
        tone_mode=tone_mode,
        gamma=gamma,
        stretch_low=stretch_low,
        stretch_high=stretch_high,
    )

    crop_out = np.clip(crop_toned, 0, 65535).astype(np.uint16)
    return crop_out, rloc


def main():
    args = parse_args()

    os.makedirs(args.clean_rois_dir, exist_ok=True)
    out_dir = os.path.dirname(args.clean_table)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.orig_table, sep="\t")

    clean_records = []
    num_new = 0
    num_orig = len(df)

    for idx, row in df.iterrows():
        img_name = row["img_file_name"]
        orig_obj_id = str(row["object_id"])

        roi_path = os.path.join(args.rois_dir, img_name)
        if not os.path.exists(roi_path):
            print(f"WARNING: ROI file not found: {roi_path}")
            continue

        img = imageio.imread(roi_path)

        result = process_single_roi(
            img,
            sigma=args.sigma,
            min_area=args.min_area,
            margin=args.margin,
            mask_background=args.mask_background,
            mask_dilate_radius=args.mask_dilate_radius,
            mask_feather_sigma=args.mask_feather_sigma,
            bg_percentile=args.bg_percentile,
            tone_mode=args.tone_mode,
            gamma=args.gamma,
            stretch_low=args.stretch_low,
            stretch_high=args.stretch_high,
        )

        if result is None:
            # whole ROI considered junk or no main object found
            continue

        crop_out, rloc = result
        num_new += 1

        new_obj_id = orig_obj_id  # no splitting → keep ID
        new_img_name = f"{new_obj_id}.png"
        out_path = os.path.join(args.clean_rois_dir, new_img_name)

        imageio.imwrite(out_path, crop_out)

        rec = row.to_dict()
        rec["object_id"] = new_obj_id
        rec["img_file_name"] = new_img_name

        # update morphometrics from cleaned main component
        rec["area"] = rloc.area
        rec["perimeter"] = rloc.perimeter
        rec["major_axis_length"] = rloc.major_axis_length
        rec["minor_axis_length"] = rloc.minor_axis_length
        rec["eccentricity"] = rloc.eccentricity

        clean_records.append(rec)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{num_orig} ROIs...")

    df_clean = pd.DataFrame.from_records(clean_records)
    df_clean.to_csv(args.clean_table, sep="\t", index=False)

    print(f"Processed {num_orig} original ROIs.")
    print(f"Kept {num_new} cleaned ROIs.")
    print(f"Clean table written to: {args.clean_table}")
    print(f"Clean ROI images in: {args.clean_rois_dir}")


if __name__ == "__main__":
    main()

