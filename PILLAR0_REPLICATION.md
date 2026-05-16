# Pillar-0 Replication Notes

This workspace mirrors the public YalaLab Pillar-0 stack and organizes the paths needed to reproduce the paper results as far as the public artifacts allow.

## What Is Here

- `rate-evals/`: main evaluation harness for paper-style frozen embedding extraction plus linear probing.
- `rave/`: Radiology Vision Engine, used to convert DICOM or NIfTI studies into model-ready cached volumes.
- `rate/`: Radiology Text Engine, used to extract structured finding labels from reports with Qwen + sglang.
- `pillar-finetune/`: downstream finetuning code, including the Sybil-1.5 / NLST-style lung cancer risk workflow.
- `pillar-pretrain/`: pretraining code for Pillar-0 style contrastive vision-language training.

Pinned clone commits at setup time:

| Repo | Commit |
| --- | --- |
| `rate-evals` | `0faf6362203228ff97caaa8d28cd578e29233a02` |
| `rave` | `20adeb873021c864e9410fae8898b0c8da309769` |
| `rate` | `79b23df905bd4252b3944b04b39f4aafa9371a12` |
| `pillar-finetune` | `47506d5267cb448867fe415d416910ce6aad9422` |
| `pillar-pretrain` | `ab06083ac0d3706017780fb3e3facd23c4fbd0cd` |

## Replication Targets

The paper reports several result families. They are not equally reproducible from public assets:

| Target | Publicly Reproducible? | Local Entry Point | Main Blockers |
| --- | --- | --- | --- |
| Merlin external abdomen CT RATE-Eval: Pillar-0 around 82.2 AUROC | Partially, if you have Merlin data and gated Hugging Face access | `rate-evals/` | Merlin dataset files, Pillar-0 HF model gate, GPU |
| Internal UCSF RATE-Evals across 366 findings | No, not exactly | `rate-evals/` + `rate/` | UCSF clinical imaging and reports are private |
| Sybil-1.5 lung cancer risk results on NLST | Partially, if you have NLST data and model access | `pillar-finetune/` | NLST data access, HF gate, GPU |
| MGH and CGMH external lung cancer validation | No, not from public data | `pillar-finetune/` | Private external cohorts |
| Brain hemorrhage data-efficiency experiment | Partially for public RSNA data, but recipe needs adaptation | `rate-evals/` or custom finetune | RSNA preprocessing and label setup |
| Full Pillar-0 pretraining | Conceptually reproducible, not practically exact | `pillar-pretrain/` | Private pretraining corpus, Qwen embeddings, large compute |

## Environment Baseline

Use Linux with NVIDIA CUDA for real runs. This local machine has `uv`, but no `nvidia-smi` command was available during setup, so treat it as a code-prep workspace rather than the final training/evaluation host.

Recommended runtime:

- Python 3.10 for `pillar-finetune`.
- Python 3.9-3.11 for `rave`, `rate`, and `rate-evals`.
- CUDA-capable PyTorch for Pillar-0 extraction and finetuning.
- Hugging Face login with access accepted for the gated YalaLab Pillar-0 checkpoints.
- `WANDB_MODE=offline` or `evaluation.use_wandb=false` unless you want to log to YalaLab's configured W&B project.

## 1. Install RAVE

```bash
cd rave
uv sync
source .venv/bin/activate
python - <<'PY'
import rve
print(rve.__version__)
print(rve.get_available_windows("CT"))
PY
```

RAVE outputs cached volume folders or archives that downstream tools load with `rve.load_sample()`.

## 2. Prepare Volumes

Create a CSV of DICOM series directories or NIfTI files:

```csv
series_path
/path/to/patient1/series1/
/path/to/scan2.nii.gz
```

For abdomen CT, run:

```bash
cd rave
uv run vision-engine process \
  --config configs/ct_abdomen.yaml \
  --input-series-csv /path/to/series_paths.csv \
  --output /path/to/rve_cache \
  --workers 4 \
  --debug \
  --debug-limit 10
```

After the debug pass succeeds, remove `--debug --debug-limit 10` for the full conversion.

## 3. Reproduce The Public-Facing Merlin Abdomen CT Evaluation

This is the most direct paper-result path exposed by the public code.

Expected inputs:

- Merlin abdomen CT split JSON files at:
  - `rate-evals/data/merlin/merlin_abd_ct/train.json`
  - `rate-evals/data/merlin/merlin_abd_ct/valid.json`
  - `rate-evals/data/merlin/merlin_abd_ct/test.json`
- RAVE cache manifest at:
  - `rate-evals/data/merlin/merlin_cache_1.5mm/manifest.csv`
- Label JSON at:
  - `rate-evals/data/merlin/final_results.json`

Install and extract Pillar-0 embeddings:

```bash
cd rate-evals
uv sync
uv add flash-attn --no-build-isolation
source .venv/bin/activate
huggingface-cli login

uv run rate-extract \
  --model pillar0 \
  --dataset abd_ct_merlin \
  --all-splits \
  --batch-size 4 \
  --output-dir cache/pillar0_abd_ct_merlin \
  --model-repo-id YalaLab/Pillar0-AbdomenCT \
  --ct-window-type all \
  --modality abdomen_ct
```

Evaluate cached embeddings:

```bash
cd rate-evals
uv run rate-evaluate \
  --checkpoint-dir cache/pillar0_abd_ct_merlin \
  --dataset-name abd_ct_merlin \
  --labels-json data/merlin/final_results.json \
  --output-dir results/pillar0_abd_ct_merlin \
  evaluation.use_wandb=false
```

Compare the mean AUROC against the paper/model-card external abdomen CT number of roughly `82.2` for Pillar-0.

## 4. Run A Tiny Custom RATE-Evals Smoke Test

`rate-evals/data/rve_example/` contains example metadata, but not the actual volume cache. Once you have at least one RAVE-processed sample, point these overrides at your local files:

```bash
cd rate-evals
uv run rate-extract \
  --model pillar0 \
  --dataset rve_abd_ct \
  --split train \
  --batch-size 1 \
  --model-repo-id YalaLab/Pillar0-AbdomenCT \
  --ct-window-type all \
  --output-dir cache/smoke_pillar0_abd_ct \
  data.train_json=/path/to/train.json \
  data.cache_manifest=/path/to/manifest.csv
```

The expected JSON shape is:

```json
[
  {
    "sample_name": "EXAMPLE_ACCESSION",
    "nii_path": null,
    "report_metadata": "FINDINGS: ..."
  }
]
```

The manifest needs at least:

```csv
sample_name,image_cache_path
EXAMPLE_ACCESSION,/path/to/rve_cache/EXAMPLE_ACCESSION.1.0
```

## 5. Recreate RATE Labels From Reports

Use this only if you need to generate `final_results.json` labels from raw radiology reports.

Start the Qwen server on a multi-GPU machine:

```bash
cd rate
uv sync
source .venv/bin/activate
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --port 8000 \
  --host 127.0.0.1 \
  --dp 8 \
  --schedule-conservativeness 0.1
```

Process reports:

```bash
cd rate
uv run python src/cli.py \
  --input-files /path/to/reports.csv \
  --modality-config config/modalities/abdomen_ct.yaml \
  --save-dir output \
  --batch-size 1024
```

Input report CSV columns default to `Accession` and `Report Text`. Output includes `final_results.json`.

## 6. Reproduce Sybil-1.5 / NLST Finetuning

Use `pillar-finetune` on a CUDA host with NLST data prepared as RAVE paths.

The built-in setup check downloads the public finetuned checkpoint and evaluates an example row:

```bash
cd pillar-finetune
uv sync
uv add flash-attn==2.8.3 --no-build-isolation
source .venv/bin/activate

OMP_NUM_THREADS=2 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=2300 \
  bash scripts/test_setup.sh
```

Expected smoke-test behavior from the repo README: test loss around `0.5797` and a CSV at `logs/csv/seed0/checkpoints/3/test.csv`.

For full NLST training:

```bash
cd pillar-finetune
bash scripts/run_nlst.sh
```

For your own CSV, update `pillar-finetune/configs/csv_dataset.yaml`:

```yaml
dataset:
  shared_dataset_kwargs:
    csv_path: /path/to/nlst_or_custom.csv
```

The CSV should include `accession`, `image_paths`, `split`, `y`, and `time_at_event`.

## 7. Pretraining Reference

`pillar-pretrain/` contains the Merlin abdomen CT pretraining recipe. Exact Pillar-0 pretraining cannot be reproduced without the original private training set:

- 42,990 abdomen-pelvis CTs
- 86,411 chest CTs
- 14,348 head CTs
- 11,543 breast MRIs

The public Merlin-oriented recipe is:

```bash
cd pillar-pretrain
uv sync
uv add flash-attn --no-build-isolation
source .venv/bin/activate

uv run vision-engine process \
  --config ../rave/configs/ct_abdomen.yaml \
  --input-series-csv data/merlin/accessions.csv \
  --output data/merlin/merlin_cache_1.5mm \
  --workers 128

mkdir -p data/text_cache_qwen3_embedding_8b
uv run python scripts/cache_merlin_embeddings.py \
  --model-name Qwen/Qwen3-Embedding-8B \
  --json-file data/merlin/merlin_abd_ct/train.json \
  --cache-dir data/text_cache_qwen3_embedding_8b/train \
  --batch-size 32 \
  --num-processes 8

bash train_unimodal_pillar0_merlin_abd_ct_1node.sh
```

Update `pillar-pretrain/src/miniclip/data_configs/pillar0_merlin_abd_ct_384.yaml` once your Merlin manifest, RAVE cache, and text cache locations are known.

## Practical Next Steps

1. Accept the Hugging Face gate for the relevant checkpoint:
   - `YalaLab/Pillar0-AbdomenCT`
   - `YalaLab/Pillar0-ChestCT`
   - `YalaLab/Pillar0-HeadCT`
   - `YalaLab/Pillar0-BreastMRI`
   - `YalaLab/Pillar0-Sybil-1.5`
2. Decide which result to reproduce first. Start with Merlin abdomen CT if you want the closest public paper number.
3. Put raw DICOM/NIfTI paths under a local data directory outside git.
4. Run RAVE debug preprocessing on 10 studies.
5. Run `rate-extract` with `--max-samples` first, then scale to all splits.
6. Run `rate-evaluate` with `evaluation.use_wandb=false`.
7. Record the exact repo commits, model revisions, data split checksums, GPU model, CUDA version, and batch size in each run log.

