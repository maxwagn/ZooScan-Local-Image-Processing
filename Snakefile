import pandas as pd

configfile: "config.yaml"

project   = config["project"]
seg       = config["segmentation"]
cleanup   = config["cleanup"]

# Read sample IDs from metadata TSV
meta = pd.read_csv(project["metadata_tsv"], sep="\t")
SAMPLES = list(meta["sample_id"].astype(str))


rule all:
    input:
        # raw per-sample tables (from segment_and_measure.py)
        expand("{tables_dir}/{sample}_objects.tsv",
               tables_dir=project["tables_dir"],
               sample=SAMPLES),
        # cleaned per-sample tables (after ROI post-processing)
        expand("{tables_dir}/{sample}_objects_clean.tsv",
               tables_dir=project["tables_dir"],
               sample=SAMPLES)


rule preprocess:
    """
    Background correction (using preprocess.py).
    Input:  raw TIFF + background_tif from config
    Output: one preprocessed TIFF per sample
    """
    input:
        raw  = lambda wc: f'{project["raw_dir"]}/{wc.sample}.tif',
        back = project["background_tif"]
    output:
        preproc = f'{project["work_dir"]}/{{sample}}_preprocessed.tif'
    shell:
        """
        mkdir -p {project[work_dir]}
        python scripts/preprocess.py {input.raw} {input.back} {output.preproc}
        """


rule segment_and_measure:
    input:
        preproc  = rules.preprocess.output.preproc,
        metadata = project["metadata_tsv"]
    output:
        table = f'{project["tables_dir"]}/{{sample}}_objects.tsv'
    params:
        rois_dir       = lambda wc: f'{project["raw_rois_dir"]}/{wc.sample}',
        sigma          = seg["gaussian_sigma"],
        min_area       = seg["min_area"],
        max_area       = seg["max_area"],
        dilate_radius  = seg.get("dilate_radius", 0),
        open_radius    = seg.get("open_radius", 0),
        thresh_mode    = seg.get("thresh_mode", "global_otsu"),
        local_window   = seg.get("local_window", 151),
        local_offset   = seg.get("local_offset", 0.0),
        margin         = seg.get("margin", 20),
        margin_factor  = seg.get("margin_factor", 0.0),
        min_contrast   = seg.get("min_contrast", 0.0),
        border_frac    = seg.get("drop_border_frac", 0.005)
    shell:
        """
        mkdir -p {params.rois_dir}
        mkdir -p {project[tables_dir]}
        python scripts/segment_and_measure.py \
            {input.preproc} \
            {wildcards.sample} \
            {input.metadata} \
            {params.rois_dir} \
            {output.table} \
            --sigma {params.sigma} \
            --min-area {params.min_area} \
            --max-area {params.max_area} \
            --dilate-radius {params.dilate_radius} \
            --open-radius {params.open_radius} \
            --thresh-mode {params.thresh_mode} \
            --local-window {params.local_window} \
            --local-offset {params.local_offset} \
            --margin {params.margin} \
            --margin-factor {params.margin_factor} \
            --min-contrast {params.min_contrast} --drop-border-frac {params.border_frac}
        """

rule cleanup_rois:
    """
    Post-processing of ROIs:
    - read original ROI PNGs for a sample (data/raw_rois)
    - build a mask inside each ROI
    - mask background, apply gamma tone curve
    - write cleaned ROI PNGs (data/cleaned_rois)
    - write a clean EcoTaxa-style table with one row per cleaned ROI
    """
    input:
        table = rules.segment_and_measure.output.table
    output:
        clean_table = f'{project["tables_dir"]}/{{sample}}_objects_clean.tsv'
    params:
        rois_dir       = lambda wc: f'{project["raw_rois_dir"]}/{wc.sample}',
        clean_rois_dir = lambda wc: f'{project["clean_rois_dir"]}/{wc.sample}',
        sigma          = cleanup["sigma"],
        min_area       = cleanup["min_area"],
        margin         = cleanup["margin"],
        mask_flag      = "--mask-background" if cleanup.get("mask_background", False) else "--no-mask-background",
        mask_dilate_radius = cleanup["mask_dilate_radius"],
        mask_feather_sigma = cleanup["mask_feather_sigma"],
        bg_percentile      = cleanup["bg_percentile"],
        tone_mode          = cleanup.get("tone_mode", "none"),
        gamma              = cleanup.get("gamma", 1.0),
    shell:
        """
        mkdir -p {params.clean_rois_dir}
        mkdir -p {project[tables_dir]}
        python scripts/clean_rois_batch.py \
            --sigma {params.sigma} \
            --min-area {params.min_area} \
            --margin {params.margin} \
            {params.mask_flag} \
            --mask-dilate-radius {params.mask_dilate_radius} \
            --mask-feather-sigma {params.mask_feather_sigma} \
            --bg-percentile {params.bg_percentile} \
            --tone-mode {params.tone_mode} \
            --gamma {params.gamma} \
            {params.rois_dir} \
            {input.table} \
            {params.clean_rois_dir} \
            {output.clean_table}
        """

