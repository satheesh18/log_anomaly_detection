# Log Anomaly Detection with Sequence Models

This codebase trains simple sequence models for infrastructure log anomaly detection. The idea is to convert each log or trace row into a state token, train a model to predict the next state, and use prediction loss as the anomaly score.

Implemented models:

- RNN
- LSTM
- Transformer encoder

Supported datasets:

- `openstack`: full Loghub OpenStack archive grouped by VM instance.
- `openstack2k`: small structured OpenStack sample.
- `hdfs`: HDFS2k structured log sample.
- `gaia`: GAIA MicroSS web-service trace sample.

## Codebase Layout

```text
src/log_anomaly/
├── data.py                    # downloads/loads data, builds state tokens, creates windows
├── models.py                  # RNN, LSTM, and Transformer model definitions
├── train.py                   # training loop, scoring, thresholding, metrics
├── experiments.py             # run one experiment configuration
├── run_grid.py                # run multiple datasets/configurations/models
├── summarize_results.py       # create summary CSVs from raw grid results
├── plot_results.py            # create result graphs from CSV files
└── build_report_artifacts.py  # rebuild final-report combined CSV and figures
```

Other files:

```text
main.tex              # final report source
proposal.tex          # proposal source
requirements.txt      # Python dependencies
REPRODUCIBILITY.md    # exact final-report reproduction checklist
```

Generated folders:

```text
data/                 # downloaded datasets, ignored by Git
outputs/              # checkpoints/results/figures, mostly ignored by Git
```

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repo root with `PYTHONPATH=src`.

## Data Setup

OpenStack, OpenStack2k, and HDFS2k are downloaded automatically when needed.

GAIA is large, so it is not downloaded automatically. For GAIA experiments, download the MicroSS trace data from:

```text
https://github.com/CloudWise-OpenSource/GAIA-DataSet
```

Place these files here:

```text
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice1_2021-07.csv
data/gaia/MicroSS/trace/usable/trace/trace_table_webservice2_2021-07.csv
```

## How the Pipeline Runs

1. `data.py` loads a dataset.
2. Each row becomes a state token.
3. State tokens are mapped to integer ids.
4. Sliding windows are created.
5. `models.py` builds an RNN, LSTM, or Transformer.
6. `train.py` trains on normal windows.
7. Validation scores choose an anomaly threshold.
8. Test scores produce accuracy, precision, recall, F1, AUC, and loss.

## Run a Quick Smoke Test

This checks that the environment and code work:

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
outputs/smoke/rnn.pt
outputs/smoke/lstm.pt
outputs/smoke/transformer.pt
```

## Run One Experiment

`experiments.py` runs one dataset/configuration. By default it trains all three models.

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset openstack2k \
  --epochs 3 \
  --seq-len 10 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --num-layers 2 \
  --batch-size 128 \
  --output-dir outputs/openstack2k_run \
  --results-file outputs/openstack2k_run/results.csv
```

Run only one model:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset hdfs \
  --models lstm \
  --epochs 3 \
  --output-dir outputs/hdfs_lstm \
  --results-file outputs/hdfs_lstm/results.csv
```

Useful options:

```text
--dataset                    openstack, openstack2k, hdfs, or gaia
--models                     rnn, lstm, transformer
--seq-len                    number of previous states used as input
--embed-dim                  embedding size
--hidden-dim                 RNN/LSTM hidden size or Transformer feedforward size
--num-layers                 recurrent/Transformer depth
--batch-size                 training batch size
--epochs                     training epochs
--max-events                 limit rows for a faster run
--gaia-include-status        include GAIA status code in the state token
```

## Run Each Dataset

OpenStack full archive:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset openstack \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/openstack \
  --results-file outputs/openstack/results.csv
```

OpenStack2k sample:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset openstack2k \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/openstack2k \
  --results-file outputs/openstack2k/results.csv
```

HDFS2k sample:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset hdfs \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/hdfs \
  --results-file outputs/hdfs/results.csv
```

GAIA without status code:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --output-dir outputs/gaia_no_status \
  --results-file outputs/gaia_no_status/results.csv
```

GAIA with status code, for the ablation:

```bash
PYTHONPATH=src python -m log_anomaly.experiments \
  --dataset gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --gaia-include-status \
  --output-dir outputs/gaia_with_status \
  --results-file outputs/gaia_with_status/results.csv
```

## Run the Experiment Grid

`run_grid.py` runs 5 configurations for each selected dataset, and trains RNN, LSTM, and Transformer for every configuration.

Main non-GAIA grid:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets openstack openstack2k hdfs \
  --epochs 3 \
  --batch-size 128 \
  --output-dir outputs/experiments
```

GAIA no-status grid:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --output-dir outputs/experiments_gaia_no_status
```

GAIA status-code grid:

```bash
PYTHONPATH=src python -m log_anomaly.run_grid \
  --datasets gaia \
  --max-events 20000 \
  --epochs 2 \
  --batch-size 128 \
  --gaia-include-status \
  --output-dir outputs/experiments_gaia
```

Grid output:

```text
outputs/<run_name>/raw_results.csv
outputs/<run_name>/checkpoints/rnn.pt
outputs/<run_name>/checkpoints/lstm.pt
outputs/<run_name>/checkpoints/transformer.pt
```

## Summarize Results

Use `summarize_results.py` on any grid CSV:

```bash
PYTHONPATH=src python -m log_anomaly.summarize_results \
  --input outputs/experiments/raw_results.csv \
  --output-dir outputs/experiments/summaries
```

It creates:

```text
best_by_dataset_model.csv
top3_by_dataset.csv
sequence_sweep.csv
capacity_sweep.csv
```

## Plot Results

For a grid-style result file:

```bash
PYTHONPATH=src python -m log_anomaly.plot_results \
  --results outputs/experiments/raw_results.csv \
  --figures-dir outputs/experiments/figures
```

It creates:

```text
best_f1_by_dataset_model.png
sequence_length_sweep.png
accuracy_vs_f1.png
precision_recall_f1_panel.png
```

For a single `experiments.py` result file:

```bash
PYTHONPATH=src python -m log_anomaly.plot_results \
  --results outputs/hdfs/results.csv \
  --output outputs/hdfs/model_comparison.png
```

## Build Final Report Artifacts

This combines the main grid and GAIA runs into the exact CSV/figures used by `main.tex`:

```bash
PYTHONPATH=src python -m log_anomaly.build_report_artifacts
```

It reads:

```text
outputs/experiments/raw_results.csv
outputs/experiments_gaia_no_status/raw_results.csv
outputs/experiments_gaia/raw_results.csv
```

It writes:

```text
outputs/final_report/data/combined_results.csv
outputs/final_report/figures/*.png
```

## Output Columns

Result CSVs include:

```text
dataset
model
seq_len
embed_dim
hidden_dim
num_layers
epochs
batch_size
train_windows
val_windows
test_windows
vocab_size
loss
next_event_accuracy
precision
recall
f1
auc
threshold
runtime_seconds
```

Metric meaning:

- `next_event_accuracy`: exact next-state prediction accuracy.
- `precision`: among predicted anomalies, how many were real anomalies.
- `recall`: among real anomalies, how many were caught.
- `f1`: balance between precision and recall.
- `auc`: how well anomaly scores rank anomalies above normal windows.
- `threshold`: validation-selected cutoff for anomaly scores.

## Notes for GitHub

Large downloaded data, checkpoints, and most generated outputs are ignored by Git. The final report figures and combined final-report CSV are kept so the report can render without rerunning the full grid.

For the exact final-paper reproduction checklist, see `REPRODUCIBILITY.md`.
