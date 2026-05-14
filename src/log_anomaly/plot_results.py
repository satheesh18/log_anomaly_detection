from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model comparison metrics.")
    parser.add_argument("--results", type=Path, default=Path("outputs/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_comparison.png"))
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args()


def _ordered_models(frame: pd.DataFrame) -> pd.DataFrame:
    order = [model for model in ["rnn", "lstm", "transformer"] if model in set(frame["model"])]
    return frame.set_index("model").loc[order].reset_index()


def plot_single_results(results: pd.DataFrame, output: Path, title: str) -> None:
    results = _ordered_models(results)
    metrics = ["next_event_accuracy", "precision", "recall", "f1"]
    display_names = {
        "next_event_accuracy": "Next-event accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }

    ax = results.plot(
        x="model",
        y=metrics,
        kind="bar",
        figsize=(10, 5.5),
        ylim=(0, 1.05),
        width=0.75,
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticklabels([model.upper() if model != "lstm" else "LSTM" for model in results["model"]], rotation=0)
    ax.legend([display_names[metric] for metric in metrics], loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def _best_by_dataset_model(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["dataset", "model", "f1", "auc", "next_event_accuracy"], ascending=[True, True, False, False, False])
        .groupby(["dataset", "model"], as_index=False)
        .head(1)
    )


def plot_best_f1(results: pd.DataFrame, output: Path) -> None:
    best = _best_by_dataset_model(results)
    pivot = best.pivot(index="dataset", columns="model", values="f1")
    pivot = pivot[[col for col in ["rnn", "lstm", "transformer"] if col in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0, max(0.1, min(1.0, pivot.max().max() * 1.25))))
    ax.set_title("Best Anomaly F1 by Dataset and Model")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best F1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def plot_sequence_sweep(results: pd.DataFrame, output: Path) -> None:
    sweep = results[
        (results["embed_dim"] == 32)
        & (results["hidden_dim"] == 64)
        & (results["num_layers"] == 1)
    ]
    datasets = sorted(sweep["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        subset = sweep[sweep["dataset"] == dataset]
        for model in ["rnn", "lstm", "transformer"]:
            model_rows = subset[subset["model"] == model].sort_values("seq_len")
            if model_rows.empty:
                continue
            axis.plot(model_rows["seq_len"], model_rows["f1"], marker="o", label=model)
        axis.set_title(dataset)
        axis.set_xlabel("Sequence length")
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Anomaly F1")
    axes[-1].legend(title="Model")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def plot_accuracy_vs_f1(results: pd.DataFrame, output: Path) -> None:
    best = _best_by_dataset_model(results)
    fig, ax = plt.subplots(figsize=(8, 5.3))
    markers = {"rnn": "o", "lstm": "s", "transformer": "^"}
    colors = {
        "hdfs": "#4C78A8",
        "openstack": "#F58518",
        "openstack2k": "#54A24B",
        "GAIA no-status": "#B279A2",
    }

    for _, row in best.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        ax.scatter(
            row["next_event_accuracy"],
            row["f1"],
            color=colors.get(dataset, "#777777"),
            marker=markers.get(model, "o"),
            s=80,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )

    ax.set_title("Next-event Accuracy vs Anomaly F1")
    ax.set_xlabel("Next-event accuracy")
    ax.set_ylabel("Anomaly F1")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)

    if "openstack" in set(best["dataset"]):
        openstack = best[best["dataset"] == "openstack"].sort_values("next_event_accuracy").iloc[-1]
        ax.annotate(
            "OpenStack: high accuracy,\nbut low anomaly F1",
            xy=(openstack["next_event_accuracy"], openstack["f1"]),
            xytext=(0.72, 0.14),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=8,
        )

    if "openstack2k" in set(best["dataset"]):
        openstack2k = best[best["dataset"] == "openstack2k"].sort_values("f1").iloc[-1]
        ax.annotate(
            "OpenStack2k catches\nanomalies better",
            xy=(openstack2k["next_event_accuracy"], openstack2k["f1"]),
            xytext=(0.54, 0.52),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=8,
        )

    dataset_handles = [
        Line2D([0], [0], marker="o", color="w", label=dataset, markerfacecolor=color, markersize=8)
        for dataset, color in colors.items()
        if dataset in set(best["dataset"])
    ]
    model_handles = [
        Line2D([0], [0], marker=marker, color="black", label=model.upper() if model != "lstm" else "LSTM", linestyle="None", markersize=7)
        for model, marker in markers.items()
        if model in set(best["model"])
    ]
    first_legend = ax.legend(handles=dataset_handles, title="Dataset", loc="upper right", fontsize=8)
    ax.add_artist(first_legend)
    ax.legend(handles=model_handles, title="Model", loc="center right", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def plot_metric_panel(results: pd.DataFrame, output: Path) -> None:
    best = _best_by_dataset_model(results)
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4), sharey=True)
    for axis, metric in zip(axes, metrics):
        pivot = best.pivot(index="dataset", columns="model", values=metric)
        pivot = pivot[[col for col in ["rnn", "lstm", "transformer"] if col in pivot.columns]]
        pivot.plot(kind="bar", ax=axis, ylim=(0, 1.0), legend=False)
        axis.set_title(metric.capitalize())
        axis.set_xlabel("Dataset")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=0)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Score")
    axes[-1].legend(title="Model", loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    if {"dataset", "config_name", "seq_len"}.issubset(results.columns):
        figures_dir = args.figures_dir or args.output.parent
        plot_best_f1(results, figures_dir / "best_f1_by_dataset_model.png")
        plot_sequence_sweep(results, figures_dir / "sequence_length_sweep.png")
        plot_accuracy_vs_f1(results, figures_dir / "accuracy_vs_f1.png")
        plot_metric_panel(results, figures_dir / "precision_recall_f1_panel.png")
    else:
        plot_single_results(results, args.output, "OpenStack Test Performance by Model")


if __name__ == "__main__":
    main()
