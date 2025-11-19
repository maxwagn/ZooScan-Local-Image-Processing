# scripts/preprocess.py
import argparse
import os

import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize


def parse_args():
    p = argparse.ArgumentParser(
        description="Background removal for ZooScan images (no cropping)."
    )
    p.add_argument("raw_tif", help="Raw ZooScan TIFF")
    p.add_argument("background_tif", help="Background TIFF")
    p.add_argument("out_tif", help="Output preprocessed TIFF")
    return p.parse_args()


def main():
    args = parse_args()

    # ensure output dir exists
    out_dir = os.path.dirname(args.out_tif)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    raw = imageio.imread(args.raw_tif)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    raw = raw.astype(np.float32)

    back = imageio.imread(args.background_tif)
    if back.ndim == 3:
        back = back.mean(axis=2)
    back = back.astype(np.float32)

    if back.shape != raw.shape:
        print(f"Resizing background from {back.shape} to {raw.shape}")
        back = resize(back, raw.shape, preserve_range=True, anti_aliasing=True)
        back = back.astype(np.float32)

    diff = raw - back
    diff -= diff.min()
    diff = np.clip(diff, 0, None)

    # scale to uint16
    diff_max = diff.max() if diff.max() > 0 else 1.0
    diff_uint16 = (diff / diff_max * 65535).astype(np.uint16)

    imageio.imwrite(args.out_tif, diff_uint16)
    print(f"Saved preprocessed image to {args.out_tif}")


if __name__ == "__main__":
    main()

