# Learning State Transitions in Infrastructure Logs for Anomaly Detection

This repository contains the code, experiment outputs, and LaTeX report for a CS 271 machine learning final project. The project studies whether sequence models can learn normal state transitions in infrastructure logs and traces, then use prediction surprise as an anomaly score.

The main research gap is practical: log-anomaly papers often report results on one dataset or one preprocessing setup, which makes it hard to tell whether the model learned useful system behavior or benefited from the way the data was prepared. This project compares the same next-state prediction idea across multiple datasets, model architectures, state representations, and evaluation choices.

## What Is Included

Models:

- RNN
- LSTM
- Transformer encoder

Datasets:

- `openstack`: full Loghub OpenStack archive, grouped by VM instance, with real sparse anomaly labels.
- `openstack2k`: structured Loghub OpenStack sample with controlled synthetic transition anomalies when real labels are absent.
- `hdfs`: structured HDFS2k sample with block-derived anomaly labels.
- `gaia`: GAIA MicroSS web-service traces, using service, endpoint, duration bucket, and message as the main state fields.

Main report:

- `main.tex`: final report source.
- `proposal.tex`: original proposal/report draft used for writing style and background.
- `outputs/final_report/figures/`: final figures referenced by `main.tex`.
- `outputs/final_report/data/combined_results.csv`: combined results used for the final report.

## Method Summary

Each log or trace row is converted into a discrete state token:

```text
raw row -> stable fields -> normalized state token -> integer id
```

The model sees a fixed-length window of previous states and predicts the next state:

```text
previous states -> RNN/LSTM/Transformer -> predicted next state
```

If the true next state has high cross-entropy loss, the transition is treated as surprising. That loss becomes the anomaly score.

```text
low loss  -> expected transition   -> likely normal
high loss -> surprising transition -> possible anomaly
```

## Repository Structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── main.tex
├── proposal.tex
├── src/log_anomaly/
│   ├── data.py                    # dataset loading, parsing, state tokens, windows
│   ├── models.py                  # RNN, LSTM, Transformer
│   ├── train.py                   # training, scoring, thresholding, metrics
│   ├── experiments.py             # one experiment run
│   ├── run_grid.py                # focused grid runner
│   ├── summarize_results.py       # summary CSVs
│   ├── plot_results.py            # general plots
│   └── build_report_artifacts.py  # final-report CSVs and figures
└── outputs/final_report/
    ├── data/
    └── figures/
```

The `data/` folder is intentionally ignored by Git because the downloaded datasets are large.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11 if possible. The code uses PyTorch and will use MPS on Apple Silicon when available, otherwise CPU.

## Data

OpenStack, OpenStack2k, and HDFS2k are downloaded automatically by the code when missing.

GAIA is not downloaded automatically because the trace archive is large. For GAIA runs, download the MicroSS trace data from the [GAIA dataset repository](https://github.com/CloudWise-OpenSource/GAIA-DataSet), extract the July 2021 webservice trace CSVs, and place them here:

```text
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice1_2021-07.csv
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice2_2021-07.csv
```

## Quick Smoke Test

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset hdfs \
  --epochs 1 \
  --max-events 500 \
  --output-dir outputs/smoke \
  --results-file outputs/smoke/results.csv
```

This should train RNN, LSTM, and Transformer on a tiny HDFS run and write `outputs/smoke/results.csv`.

## Reproduce the Main Experiments

Run the non-GAIA focused grid:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets openstack openstack2k hdfs \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/experiments
```

Run GAIA without status code in the state token. This is the GAIA setting used in the main comparison table:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --output-dir outputs/experiments_gaia_no_status
```

Run GAIA with status code included. This is used only for the status-code ablation:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --gaia-include-status \
  --output-dir outputs/experiments_gaia
```

## Rebuild Final Report Figures

After the experiment CSVs exist, rebuild the final report artifacts:

```bash
PYTHONPATH=src python -m log_anomaly.build_report_artifacts
```

This writes:

```text
outputs/final_report/data/combined_results.csv
outputs/final_report/figures/best_f1_by_dataset_model.png
outputs/final_report/figures/accuracy_vs_f1.png
outputs/final_report/figures/precision_recall_f1_panel.png
outputs/final_report/figures/sequence_length_sweep.png
outputs/final_report/figures/gaia_status_ablation.png
```

## Compile the Report

With a LaTeX installation:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
pdflatex main.tex
```

The report source is kept in Git. LaTeX build artifacts such as `.aux`, `.log`, `.out`, `.fls`, and generated PDFs are ignored.

## Current Result Summary

The final report keeps both strong and weak results because that is the point of the project. OpenStack gets high next-event accuracy but weak anomaly F1 because the real anomaly labels are sparse. OpenStack2k gives a clearer controlled-anomaly result. GAIA shows why feature choices matter: including status code makes the task too easy because the label is based on status code, while removing it gives a more honest product-infrastructure trace experiment.

For exact numbers, see `outputs/final_report/data/combined_results.csv` and the Results section of `main.tex`.
