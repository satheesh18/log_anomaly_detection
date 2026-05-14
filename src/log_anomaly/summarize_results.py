from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment grid results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/experiments/raw_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/summaries"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.input)

    sort_cols = ["dataset", "model", "f1", "auc", "next_event_accuracy"]
    best_by_dataset_model = (
        results.sort_values(sort_cols, ascending=[True, True, False, False, False])
        .groupby(["dataset", "model"], as_index=False)
        .head(1)
        .sort_values(["dataset", "f1"], ascending=[True, False])
    )

    best_by_dataset = (
        results.sort_values(["dataset", "f1", "auc", "next_event_accuracy"], ascending=[True, False, False, False])
        .groupby("dataset", as_index=False)
        .head(3)
        .sort_values(["dataset", "f1"], ascending=[True, False])
    )

    sequence_sweep = results[
        (results["embed_dim"] == 32)
        & (results["hidden_dim"] == 64)
        & (results["num_layers"] == 1)
    ].sort_values(["dataset", "model", "seq_len"])

    capacity_sweep = results[
        (results["seq_len"] == 10)
        & (results["embed_dim"].isin([32, 64]))
        & (results["hidden_dim"].isin([64, 128]))
    ].sort_values(["dataset", "model", "embed_dim", "hidden_dim", "num_layers"])

    outputs = {
        "best_by_dataset_model.csv": best_by_dataset_model,
        "top3_by_dataset.csv": best_by_dataset,
        "sequence_sweep.csv": sequence_sweep,
        "capacity_sweep.csv": capacity_sweep,
    }

    for filename, frame in outputs.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved {path}")

    print("\nBest result per dataset/model:")
    cols = ["dataset", "model", "config_name", "seq_len", "embed_dim", "hidden_dim", "num_layers", "f1", "auc", "next_event_accuracy"]
    print(best_by_dataset_model[cols].to_string(index=False))


if __name__ == "__main__":
    main()

