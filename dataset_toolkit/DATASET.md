# Terrain Dataset Toolkit

Build a resumable terrain dataset with Step0–5. Each stage skips valid outputs
and updates `metadata.csv` through journals.

## 1. Create the Python environment

```bash
conda env create -f dataset_toolkit/dataset_requirements.yaml
conda activate dataset_gen
```

The environment file follows the tested GDP stack: `Python 3.10`, `PyTorch 2.5`,
`CUDA 12.4`, and Sonata dependencies. Choose matching PyTorch/CUDA packages if
the target GPU driver differs.

## 2. Install the Blender add-on

```bash
<BLENDER_BIN> -b --python dataset_toolkit/tools/install_range_scanner_addon.py \
  -- \
  --install-pyyaml
```
`<BLENDER_BIN>` is the Blender executable. The command installs and enables
the bundled range-scanner without opening the Blender GUI.

## 3. Step0: preprocess auxiliary models

Creates normalized auxiliary GLB files.

```bash
python dataset_toolkit/0_aux_preprocess.py \
  --dataset-root <DATASET_ROOT> \
  --aux-input <AUX_ROOT> \
  --blender <BLENDER_BIN>
```
- `<DATASET_ROOT>`: destination for generated files and `metadata.csv`.
- `<AUX_ROOT>`: directory tree containing auxiliary `.glb` models.
- `<BLENDER_BIN>`: Blender executable.

## 4. Step1: split terrain patches

Samples terrain assets and writes fixed-size valid patches.

```bash
python dataset_toolkit/1_split.py \
  --dataset-root <DATASET_ROOT> \
  --terrain-input <TERRAIN_ROOT> \
  --patch_num <PATCH_COUNT> \
  --blender <BLENDER_BIN>
```
- `<DATASET_ROOT>`: destination created in Step0.
- `<TERRAIN_ROOT>`: directory containing `<TERRAIN_TAG>/model.glb` assets.
- `<PATCH_COUNT>`: requested total number of valid patches.
- `<BLENDER_BIN>`: Blender executable.


## 5. Step2: generate LiDAR scans

Writes scans, JSON sidecars under `<DATASET_ROOT>/logs/step2/`, and scan
metadata rows.

```bash
python dataset_toolkit/2_scan.py \
  --dataset-root <DATASET_ROOT> \
  --target-samples <SCAN_COUNT> \
  --blender <BLENDER_BIN> \
  --workers <WORKERS>
```
- `<DATASET_ROOT>`: destination created by earlier steps.
- `<SCAN_COUNT>`: requested number of successful scans.
- `<BLENDER_BIN>`: Blender executable.
- `<WORKERS>`: concurrent Blender jobs.


## 6. Step3: extract Sonata features

Loads Sonata once and writes one feature archive per patch.

```bash
python dataset_toolkit/3_full_feature.py \
  --dataset-root <DATASET_ROOT> \
  --sonata-root <SONATA_ROOT> \
  --sonata-checkpoint <SONATA_CHECKPOINT> \
  --batch-size <BATCH_SIZE>
```
- `<DATASET_ROOT>`: destination created by earlier steps.
- `<SONATA_ROOT>`: local source derived from
  [facebookresearch/sonata](https://github.com/facebookresearch/sonata).
- `<SONATA_CHECKPOINT>`: Sonata weight file.
- `<BATCH_SIZE>`: patches per GPU forward pass.


## 7. Step4: render depth and normals

Writes multilayer EXR frames with depth and world-space geometric normals.

```bash
python dataset_toolkit/4_render.py \
  --dataset-root <DATASET_ROOT> \
  --blender <BLENDER_BIN> \
  --workers <WORKERS> \
  --num-views <VIEW_COUNT>
```
- `<DATASET_ROOT>`: destination created by earlier steps.
- `<BLENDER_BIN>`: Blender executable.
- `<WORKERS>`: concurrent Blender jobs.
- `<VIEW_COUNT>`: rendered views per patch.


## 8. Step5: encode latents feature

Writes one latent archive per ready patch and a dataset-wide `abstract.json`.

```bash
python dataset_toolkit/5_encode_latent.py \
  --dataset-root <DATASET_ROOT> \
  --encoder-config <ENCODER_CONFIG> \
  --encoder-checkpoint <ENCODER_CHECKPOINT> \
  --surfel-adaptor-root <PROJECT_ROOT> \
  --batch-size <BATCH_SIZE>
```
- `<DATASET_ROOT>`: destination created by earlier steps.
- `<ENCODER_CONFIG>`: encoder JSON configuration.
- `<ENCODER_CHECKPOINT>`: encoder weight file.
- `<PROJECT_ROOT>`: directory containing the `surfel_adaptor` package.
- `<BATCH_SIZE>`: patches per GPU forward pass.


## Recovery

Each stage reports processed, succeeded, failed, skipped, and failed IDs,
followed by total patches, scans, features, renders, and latents. Re-run the
same command after an interruption. Use `--dry-run` to inspect pending work,
or `--overwrite` to rebuild existing outputs. Temporary files are created
under `<DATASET_ROOT>/temp/` and are removed after each job.
