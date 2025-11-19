# ZooScan Local Processing Pipeline (Work in Progress)

This repository contains a lightweight, **still-in-development** workflow for processing **ZooScan** TIFF images using **Python** and **Snakemake**.  
It is *not* a complete or official replacement for ZooProcess/ImageJ — just a practical experimental pipeline intended for local research use.

---

## What This Workflow Does (in simple terms)

The pipeline runs in **three main steps**, mirroring the Snakefile:

### **1. Preprocessing**
- Load raw ZooScan `.tif`
- Subtract background (large master TIFF)
- Apply optional smoothing / clipping
- Output: `data/work/<sample>_preprocessed.tif`

### **2. Segmentation & ROI Extraction**
- Threshold using hybrid Otsu + local window
- Extract connected components
- Crop ROI PNGs
- Measure morphometrics
- Output:  
  - `data/raw_rois/<sample>/*.png`  
  - `data/tables/<sample>_objects.tsv`

### **3. ROI Cleanup (Optional but recommended)**
- Re-segment inside each ROI  
- Keep **only the largest component**  
- Dilate to keep fine structures (antennae, setae, etc.)  
- Mask background to pure white  
- Apply tone adjustment (gamma)  
- Output:  
  - `data/cleaned_rois/<sample>/*.png`  
  - `data/tables/<sample>_objects_clean.tsv`

---

## Example Comparisons

Below are three examples (from `comparison_examples/`) showing:

**Left:** Original ZooScan (ImageJ workflow)  
**Middle:** Raw ROI (this pipeline)  
**Right:** Masked/Cleaned ROI (this pipeline)

### **Example 1**
- `1_istria02_feb_6_d0_1_2950_zooscan.jpg`
- `1_ISTRIA02_FEB_6_obj01322_raw.png`
- `1_ISTRIA02_FEB_6_obj01322_edit.png`

### **Example 2**
- `2_istria02_feb_6_d0_1_5159_zooscan.jpg`
- `2_ISTRIA02_FEB_6_obj01752_raw.png`
- `2_ISTRIA02_FEB_6_obj01752_edit.png`

### **Example 3**
- `3_istria02_feb_6_d0_1_1232_zooscan.jpg`
- `3_ISTRIA02_FEB_6_obj06175_raw.png`
- `3_ISTRIA02_FEB_6_obj06175_edit.png`

You can place them in a Markdown viewer for side‑by‑side comparison.

---

## How To Run

Create the environment:

```
conda env create -f env.yaml
conda activate zooprocess_py
```

Dry run:

```
snakemake -np
```

Run full pipeline:

```
snakemake -c 8
```

---

## Limitations

- Not a full replica of ZooProcess  
- Background subtraction less sophisticated than ImageJ macros  
- Segmentation may require tuning per sample  
- Cleanup still experimental  
- Manual curation recommended before EcoTaxa upload  

---

## License

MIT License

---

## Disclaimer

This software is provided **“as-is”** with no guarantee of correctness or suitability.  
Use at your own risk.
