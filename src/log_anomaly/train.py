from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from log_anomaly.data import WindowDataset


@dataclass
class Metrics:
    loss: float
    next_event_accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    threshold: float


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(dataset.x),
        torch.from_numpy(dataset.y),
        torch.from_numpy(dataset.labels),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_data: WindowDataset,
    val_data: WindowDataset,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_data, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        val_scores, _, _ = score_windows(model, val_data, batch_size, device)
        val_loss = float(np.mean(val_scores))
        train_loss = running_loss / len(train_data.x)
        print(f"    epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_windows(
    model: nn.Module,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_scores: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            scores = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            all_scores.append(scores.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
    )


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float(np.percentile(scores, 95))

    best_threshold = float(np.percentile(scores, 95))
    best_f1 = -1.0
    for threshold in np.unique(np.percentile(scores, np.linspace(50, 99, 100))):
        predictions = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def evaluate_model(
    model: nn.Module,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> Metrics:
    scores, next_event_predictions, next_event_targets = score_windows(model, dataset, batch_size, device)
    anomaly_predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        dataset.labels,
        anomaly_predictions,
        average="binary",
        zero_division=0,
    )

    auc = None
    if len(np.unique(dataset.labels)) > 1:
        auc = float(roc_auc_score(dataset.labels, scores))

    return Metrics(
        loss=float(np.mean(scores)),
        next_event_accuracy=float(accuracy_score(next_event_targets, next_event_predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=auc,
        threshold=float(threshold),
    )

