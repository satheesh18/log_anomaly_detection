from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from log_anomaly.data import prepare_data
from log_anomaly.models import build_model
from log_anomaly.train import choose_device, choose_threshold, evaluate_model, score_windows, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNN, LSTM, and Transformer log anomaly experiments.")
    parser.add_argument("--dataset", choices=["openstack", "openstack2k", "hdfs", "gaia"], default="openstack")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=["rnn", "lstm", "transformer"], default=["rnn", "lstm", "transformer"])
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-anomaly-fraction", type=float, default=0.08)
    parser.add_argument(
        "--gaia-include-status",
        action="store_true",
        help="Include HTTP status code in GAIA state tokens. Use only for the status-code ablation.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(args: argparse.Namespace, model_names: list[str] | None = None) -> pd.DataFrame:
    model_names = model_names or args.models
    max_events = args.max_events if getattr(args, "max_events", None) is not None else getattr(args, "max_sequences", None)
    set_seed(args.seed)

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Preparing {args.dataset} data...")
    data = prepare_data(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        max_sequences=max_events,
        seed=args.seed,
        dataset=args.dataset,
        synthetic_anomaly_fraction=args.synthetic_anomaly_fraction,
        gaia_include_status=getattr(args, "gaia_include_status", False),
    )
    print(
        "Windows: "
        f"train={len(data.train.x)} val={len(data.val.x)} test={len(data.test.x)} "
        f"vocab={data.vocab_size}"
    )

    results = []
    for model_name in model_names:
        set_seed(args.seed)
        print(f"\nTraining {model_name.upper()}...")
        started_at = time.perf_counter()
        model = build_model(
            model_name=model_name,
            vocab_size=data.vocab_size,
            seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        model = train_model(
            model=model,
            train_data=data.train,
            val_data=data.val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        val_scores, _, _ = score_windows(model, data.val, args.batch_size, device)
        threshold = choose_threshold(val_scores, data.val.labels)
        test_metrics = evaluate_model(model, data.test, args.batch_size, device, threshold)
        runtime_seconds = time.perf_counter() - started_at

        torch.save(model.state_dict(), args.output_dir / f"{model_name}.pt")
        result = {
            "dataset": args.dataset,
            "model": model_name,
            "seq_len": args.seq_len,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "gaia_include_status": bool(getattr(args, "gaia_include_status", False)),
            "device": str(device),
            "train_windows": len(data.train.x),
            "val_windows": len(data.val.x),
            "test_windows": len(data.test.x),
            "train_anomalies": int(data.train.labels.sum()),
            "val_anomalies": int(data.val.labels.sum()),
            "test_anomalies": int(data.test.labels.sum()),
            "vocab_size": data.vocab_size,
            "runtime_seconds": runtime_seconds,
            "loss": test_metrics.loss,
            "next_event_accuracy": test_metrics.next_event_accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1,
            "auc": test_metrics.auc,
            "threshold": test_metrics.threshold,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    return results_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_df = run_experiment(args)
    results_path = args.output_dir / "results.csv"
    if args.results_file is not None:
        results_path = args.results_file
        results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
