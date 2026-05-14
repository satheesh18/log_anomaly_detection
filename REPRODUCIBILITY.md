# Reproducibility Checklist

This file lists the commands needed to reproduce the experiments and figures used by the final report.

## 1. Environment

Tested locally with:

```text
macOS
Python 3.11
PyTorch using MPS when available, otherwise CPU
```

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

The code automatically downloads these datasets into `data/` if they are missing:

- Full OpenStack archive from Loghub/Zenodo
- OpenStack2k structured CSV from Loghub
- HDFS2k structured CSV from Loghub
- HDFS anomaly labels from LogPAI/loglizer

GAIA MicroSS traces must be downloaded manually from the [GAIA dataset repository](https://github.com/CloudWise-OpenSource/GAIA-DataSet). Place the two July 2021 webservice trace files here:

```text
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice1_2021-07.csv
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice2_2021-07.csv
```

The `data/` directory is ignored by Git because the full local dataset folder is large.

## 3. Smoke Test

Run a small test before launching the full grid:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset hdfs \
  --epochs 1 \
  --max-events 500 \
  --output-dir outputs/smoke \
  --results-file outputs/smoke/results.csv
```

Expected output:

```text
outputs/smoke/results.csv
```

## 4. Main Focused Grid

Run OpenStack, OpenStack2k, and HDFS2k:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets openstack openstack2k hdfs \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/experiments
```

Expected output:

```text
outputs/experiments/raw_results.csv
```

This grid contains:

```text
3 datasets x 5 configurations x 3 models = 45 runs
```

## 5. GAIA Runs

Run GAIA without status code in the state token:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --output-dir outputs/experiments_gaia_no_status
```

Expected output:

```text
outputs/experiments_gaia_no_status/raw_results.csv
```

Run GAIA with status code included for the ablation:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --gaia-include-status \
  --output-dir outputs/experiments_gaia
```

Expected output:

```text
outputs/experiments_gaia/raw_results.csv
```

The no-status GAIA run is used in the main dataset/model comparison. The status-included run is only used to show why the feature choice can leak label information.

## 6. Summaries

Generate summary CSVs for the main grid:

```bash
PYTHONPATH=src python -m log_anomaly.summarize_results \
  --input outputs/experiments/raw_results.csv \
  --output-dir outputs/experiments/summaries
```

Expected outputs:

```text
outputs/experiments/summaries/best_by_dataset_model.csv
outputs/experiments/summaries/top3_by_dataset.csv
outputs/experiments/summaries/sequence_sweep.csv
outputs/experiments/summaries/capacity_sweep.csv
```

## 7. Final Report Artifacts

Build the combined final-report CSV and figures:

```bash
PYTHONPATH=src python -m log_anomaly.build_report_artifacts
```

Expected outputs:

```text
outputs/final_report/data/combined_results.csv
outputs/final_report/figures/best_f1_by_dataset_model.png
outputs/final_report/figures/accuracy_vs_f1.png
outputs/final_report/figures/precision_recall_f1_panel.png
outputs/final_report/figures/sequence_length_sweep.png
outputs/final_report/figures/gaia_status_ablation.png
```

## 8. Rebuild the Paper

Compile `main.tex` with LaTeX:

```bash
latexmk -pdf main.tex
```

If `latexmk` is not installed:

```bash
pdflatex main.tex
pdflatex main.tex
```

## Notes

- Random seed defaults to `42`.
- Results can vary slightly across PyTorch versions and hardware.
- OpenStack anomaly labels are sparse, so high next-event accuracy does not imply high anomaly F1.
- GAIA status-code inclusion is intentionally separated with `--gaia-include-status` because the report treats it as an ablation, not the main GAIA setup.
