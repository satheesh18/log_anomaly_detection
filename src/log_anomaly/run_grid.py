from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from log_anomaly.experiments import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused experiment grid for the final paper.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--datasets", nargs="+", choices=["openstack", "openstack2k", "hdfs", "gaia"], default=["openstack", "openstack2k", "hdfs"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--gaia-include-status",
        action="store_true",
        help="Include HTTP status code in GAIA state tokens. Use only for the status-code ablation.",
    )
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []

    for dataset in args.datasets:
        configs.extend(
            [
                {
                    "config_name": "short_context_small_model",
                    "dataset": dataset,
                    "seq_len": 5,
                    "embed_dim": 32,
                    "hidden_dim": 64,
                    "num_layers": 1,
                },
                {
                    "config_name": "medium_context_small_model",
                    "dataset": dataset,
                    "seq_len": 10,
                    "embed_dim": 32,
                    "hidden_dim": 64,
                    "num_layers": 1,
                },
                {
                    "config_name": "long_context_small_model",
                    "dataset": dataset,
                    "seq_len": 15,
                    "embed_dim": 32,
                    "hidden_dim": 64,
                    "num_layers": 1,
                },
                {
                    "config_name": "medium_context_larger_model",
                    "dataset": dataset,
                    "seq_len": 10,
                    "embed_dim": 64,
                    "hidden_dim": 128,
                    "num_layers": 1,
                },
                {
                    "config_name": "medium_context_deeper_transformer",
                    "dataset": dataset,
                    "seq_len": 10,
                    "embed_dim": 64,
                    "hidden_dim": 128,
                    "num_layers": 2,
                },
            ]
        )

    return configs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[pd.DataFrame] = []

    for run_id, config in enumerate(build_grid(args), start=1):
        print("\n" + "=" * 80)
        print(f"Run {run_id}: {config['dataset']} / {config['config_name']}")
        print("=" * 80)

        experiment_args = SimpleNamespace(
            dataset=config["dataset"],
            data_dir=args.data_dir,
            output_dir=args.output_dir / "checkpoints",
            models=["rnn", "lstm", "transformer"],
            seq_len=config["seq_len"],
            max_events=args.max_events,
            max_sequences=None,
            synthetic_anomaly_fraction=0.08,
            gaia_include_status=args.gaia_include_status,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=0.1,
            lr=0.001,
            num_heads=4,
            num_layers=config["num_layers"],
            seed=args.seed,
        )
        experiment_args.output_dir.mkdir(parents=True, exist_ok=True)

        results = run_experiment(experiment_args)
        results.insert(0, "run_id", run_id)
        results.insert(1, "config_name", config["config_name"])
        all_results.append(results)

        partial = pd.concat(all_results, ignore_index=True)
        partial.to_csv(args.output_dir / "raw_results.csv", index=False)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output_dir / "raw_results.csv", index=False)
    print(f"\nSaved grid results to {args.output_dir / 'raw_results.csv'}")


if __name__ == "__main__":
    main()
