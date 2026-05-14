from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from log_anomaly.plot_results import (
    plot_accuracy_vs_f1,
    plot_best_f1,
    plot_metric_panel,
    plot_sequence_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report CSVs and figures from experiment outputs.")
    parser.add_argument("--base-results", type=Path, default=Path("outputs/experiments/raw_results.csv"))
    parser.add_argument(
        "--gaia-no-status-results",
        type=Path,
        default=Path("outputs/experiments_gaia_no_status/raw_results.csv"),
    )
    parser.add_argument(
        "--gaia-with-status-results",
        type=Path,
        default=Path("outputs/experiments_gaia/raw_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_report"))
    return parser.parse_args()


def _load_results(path: Path, dataset_name: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    frame = pd.read_csv(path)
    if dataset_name is not None:
        frame = frame.copy()
        frame["dataset"] = dataset_name
    return frame


def plot_gaia_status_ablation(with_status: pd.DataFrame, no_status: pd.DataFrame, output: Path) -> None:
    values = pd.DataFrame(
        [
            {"setting": "With status", "best_f1": with_status["f1"].max()},
            {"setting": "No status", "best_f1": no_status["f1"].max()},
        ]
    )

    ax = values.plot(x="setting", y="best_f1", kind="bar", legend=False, figsize=(5, 4), ylim=(0, 1.05))
    ax.set_title("GAIA Status-Code Ablation")
    ax.set_xlabel("")
    ax.set_ylabel("Best F1")
    ax.set_xticklabels(values["setting"], rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    data_dir = args.output_dir / "data"
    figures_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    base = _load_results(args.base_results)
    gaia_no_status = _load_results(args.gaia_no_status_results, dataset_name="GAIA no-status")
    combined = pd.concat([base, gaia_no_status], ignore_index=True)

    combined_path = data_dir / "combined_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved {combined_path}")

    plot_best_f1(combined, figures_dir / "best_f1_by_dataset_model.png")
    plot_accuracy_vs_f1(combined, figures_dir / "accuracy_vs_f1.png")
    plot_metric_panel(combined, figures_dir / "precision_recall_f1_panel.png")
    plot_sequence_sweep(combined, figures_dir / "sequence_length_sweep.png")

    gaia_with_status = _load_results(args.gaia_with_status_results, dataset_name="GAIA with status")
    plot_gaia_status_ablation(gaia_with_status, gaia_no_status, figures_dir / "gaia_status_ablation.png")


if __name__ == "__main__":
    main()
