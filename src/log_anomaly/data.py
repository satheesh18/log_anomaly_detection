from __future__ import annotations

import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


HDFS_STRUCTURED_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/"
    "HDFS_2k.log_structured.csv"
)
HDFS_LABEL_URL = (
    "https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/"
    "anomaly_label.csv"
)
OPENSTACK_STRUCTURED_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/OpenStack/"
    "OpenStack_2k.log_structured.csv"
)
OPENSTACK_FULL_URL = "https://zenodo.org/records/3227177/files/OpenStack.tar.gz?download=1"
OPENSTACK_ANOMALY_IDS = {
    "544fd51c-4edc-4780-baae-ba1d80a0acfc",
    "ae651dff-c7ad-43d6-ac96-bbcd820ccca8",
    "a445709b-6ad0-40ec-8860-bec60b6ca0c2",
    "1643649d-2f42-4303-bfcd-7798baec19f9",
}
GAIA_TRACE_FILES = (
    "trace_table_webservice1_2021-07.csv",
    "trace_table_webservice2_2021-07.csv",
)


@dataclass
class WindowDataset:
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray


@dataclass
class PreparedData:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    vocab_size: int
    event_to_id: dict[str, int]


def download_hdfs(data_dir: Path) -> None:
    """Download the small HDFS Loghub files used by this project."""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "HDFS_2k.log_structured.csv": HDFS_STRUCTURED_URL,
        "anomaly_label.csv": HDFS_LABEL_URL,
    }

    for filename, url in files.items():
        target = data_dir / filename
        if target.exists():
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, target)


def download_openstack(data_dir: Path) -> None:
    """Download the OpenStack Loghub sample used by this project."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "OpenStack_2k.log_structured.csv"
    if target.exists():
        return
    print("Downloading OpenStack_2k.log_structured.csv...")
    urllib.request.urlretrieve(OPENSTACK_STRUCTURED_URL, target)


def download_openstack_full(data_dir: Path) -> None:
    """Download the full OpenStack archive from Loghub/Zenodo."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "OpenStack.tar.gz"
    if target.exists():
        return
    print("Downloading full OpenStack.tar.gz...")
    urllib.request.urlretrieve(OPENSTACK_FULL_URL, target)


def load_hdfs_events(data_dir: Path, max_events: int | None = None) -> tuple[list[str], np.ndarray]:
    """Load HDFS logs as one chronological event stream with line labels.

    The small HDFS_2k sample has very short block-level sequences, so this
    project models the chronological event stream instead. A line is marked
    anomalous if any block id in that line is labeled Anomaly.
    """
    structured_path = data_dir / "HDFS_2k.log_structured.csv"
    label_path = data_dir / "anomaly_label.csv"
    if not structured_path.exists() or not label_path.exists():
        download_hdfs(data_dir)

    logs = pd.read_csv(structured_path)
    labels_df = pd.read_csv(label_path)
    label_lookup = dict(zip(labels_df["BlockId"], labels_df["Label"]))

    if max_events is not None:
        logs = logs.head(max_events)

    block_pattern = re.compile(r"blk_-?\d+")
    events: list[str] = []
    labels: list[int] = []

    for _, row in logs.iterrows():
        content = str(row["Content"])
        event_id = str(row["EventId"])
        block_ids = block_pattern.findall(content)
        is_anomaly = any(label_lookup.get(block_id) == "Anomaly" for block_id in block_ids)
        events.append(event_id)
        labels.append(1 if is_anomaly else 0)

    return events, np.asarray(labels, dtype=np.int64)


def load_openstack_sample_events(data_dir: Path, max_events: int | None = None) -> tuple[list[str], np.ndarray]:
    """Load OpenStack logs as a chronological event stream.

    The public OpenStack_2k sample does not include separate label files. We
    mark real anomalies when known injected-failure instance ids appear. If none
    appear in this small sample, later code injects controlled synthetic
    anomalies into validation/test windows for measurable evaluation.
    """
    structured_path = data_dir / "OpenStack_2k.log_structured.csv"
    if not structured_path.exists():
        download_openstack(data_dir)

    logs = pd.read_csv(structured_path)
    if max_events is not None:
        logs = logs.head(max_events)

    events: list[str] = []
    labels: list[int] = []
    for _, row in logs.iterrows():
        event = f"{row['Component']}|{row['Level']}|{row['EventId']}"
        row_text = " ".join(str(value) for value in row.values)
        is_anomaly = any(anomaly_id in row_text for anomaly_id in OPENSTACK_ANOMALY_IDS)
        events.append(event)
        labels.append(1 if is_anomaly else 0)

    return events, np.asarray(labels, dtype=np.int64)


def _duration_bucket(milliseconds: float) -> str:
    if milliseconds < 100:
        return "<100ms"
    if milliseconds < 500:
        return "<500ms"
    if milliseconds < 1000:
        return "<1s"
    if milliseconds < 5000:
        return "<5s"
    return ">=5s"


def _gaia_endpoint(url: str) -> str:
    parsed = urlparse(str(url))
    return parsed.path or "<unknown_path>"


def load_gaia_trace_events(
    data_dir: Path,
    max_events: int | None = None,
    include_status: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Load GAIA MicroSS web-service traces as chronological state tokens.

    The usable local subset contains product-facing webservice traces with
    service, URL, timing, status code, and message fields. A row is labeled
    anomalous when the trace status is not HTTP 200.
    """
    trace_dir = data_dir / "gaia" / "MicroSS" / "trace" / "usable" / "trace"
    missing = [filename for filename in GAIA_TRACE_FILES if not (trace_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "GAIA trace files are missing. Expected files under "
            f"{trace_dir}: {', '.join(missing)}"
        )

    usecols = ["timestamp", "service_name", "start_time", "end_time", "url", "status_code", "message"]
    per_file_limit = None if max_events is None else max(1, int(np.ceil(max_events / len(GAIA_TRACE_FILES))))
    frames = [
        pd.read_csv(trace_dir / filename, usecols=usecols, nrows=per_file_limit)
        for filename in GAIA_TRACE_FILES
    ]
    traces = pd.concat(frames, ignore_index=True)
    traces = traces.sort_values(["timestamp", "service_name"], kind="stable")
    if max_events is not None:
        traces = traces.head(max_events)

    start_time = pd.to_datetime(traces["start_time"], errors="coerce")
    end_time = pd.to_datetime(traces["end_time"], errors="coerce")
    duration_ms = (end_time - start_time).dt.total_seconds().mul(1000).fillna(0)

    events: list[str] = []
    labels: list[int] = []
    for row, duration in zip(traces.itertuples(index=False), duration_ms):
        status = str(row.status_code)
        event_parts = [
            str(row.service_name),
            _gaia_endpoint(str(row.url)),
            f"duration={_duration_bucket(float(duration))}",
            str(row.message),
        ]
        if include_status:
            event_parts.insert(2, f"status={status}")
        event = "|".join(event_parts)
        events.append(event)
        labels.append(0 if status == "200" else 1)

    print(f"GAIA trace rows: total={len(events)} anomalous={sum(labels)}")
    return events, np.asarray(labels, dtype=np.int64)


def _normalize_openstack_content(content: str) -> str:
    content = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", content)
    content = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", content)
    content = re.sub(r"\b[0-9a-f]{32}\b", "<hex>", content)
    content = re.sub(r"\b\d+\.\d+\b", "<float>", content)
    content = re.sub(r"\b\d+\b", "<num>", content)
    return re.sub(r"\s+", " ", content).strip()


def _parse_openstack_raw_line(line: str) -> tuple[str, str, str] | None:
    match = re.match(
        r"^(?P<logrecord>\S+)\s+"
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+"
        r"(?P<pid>\d+)\s+"
        r"(?P<level>\S+)\s+"
        r"(?P<component>\S+)\s+"
        r"(?P<content>.*)$",
        line,
    )
    if match is None:
        return None

    component = match.group("component")
    level = match.group("level")
    content = _normalize_openstack_content(match.group("content"))
    return component, level, content


def _read_openstack_archive_file(
    archive: tarfile.TarFile,
    filename: str,
    anomaly_ids: set[str],
    max_events: int | None,
) -> tuple[list[str], np.ndarray]:
    events: list[str] = []
    labels: list[int] = []
    member = archive.extractfile(filename)
    if member is None:
        raise FileNotFoundError(f"{filename} not found inside OpenStack archive")

    for raw_line in member:
        line = raw_line.decode("utf-8", errors="replace").strip()
        parsed = _parse_openstack_raw_line(line)
        if parsed is None:
            continue
        component, level, content = parsed
        events.append(f"{component}|{level}|{content}")
        labels.append(1 if any(anomaly_id in line for anomaly_id in anomaly_ids) else 0)
        if max_events is not None and len(events) >= max_events:
            break

    return events, np.asarray(labels, dtype=np.int64)


def _load_openstack_anomaly_ids(archive: tarfile.TarFile) -> set[str]:
    member = archive.extractfile("anomaly_labels.txt")
    if member is None:
        return set(OPENSTACK_ANOMALY_IDS)

    text = member.read().decode("utf-8", errors="replace")
    ids = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text))
    return ids or set(OPENSTACK_ANOMALY_IDS)


def load_openstack_full_streams(
    data_dir: Path,
    max_events: int | None = None,
) -> tuple[list[int], np.ndarray, list[int], np.ndarray, dict[str, int]]:
    """Load full OpenStack as normal training stream and abnormal evaluation stream."""
    archive_path = data_dir / "OpenStack.tar.gz"
    if not archive_path.exists():
        download_openstack_full(data_dir)

    with tarfile.open(archive_path, "r:gz") as archive:
        anomaly_ids = _load_openstack_anomaly_ids(archive)
        normal1_events, normal1_labels = _read_openstack_archive_file(
            archive, "openstack_normal1.log", anomaly_ids, max_events
        )
        normal2_events, normal2_labels = _read_openstack_archive_file(
            archive, "openstack_normal2.log", anomaly_ids, max_events
        )
        abnormal_events, abnormal_labels = _read_openstack_archive_file(
            archive, "openstack_abnormal.log", anomaly_ids, max_events
        )

    train_events_raw = normal1_events + normal2_events
    train_labels = np.concatenate([normal1_labels, normal2_labels])
    all_events_raw = train_events_raw + abnormal_events
    event_to_id = {event: idx for idx, event in enumerate(sorted(set(all_events_raw)))}
    train_events = [event_to_id[event] for event in train_events_raw]
    abnormal_events_encoded = [event_to_id[event] for event in abnormal_events]

    print(
        "OpenStack full lines: "
        f"normal={len(train_events)} abnormal={len(abnormal_events_encoded)} "
        f"abnormal_labeled={int(abnormal_labels.sum())}"
    )

    return train_events, train_labels, abnormal_events_encoded, abnormal_labels, event_to_id


def _read_openstack_instance_sequences(
    archive: tarfile.TarFile,
    filename: str,
    anomaly_ids: set[str],
    max_events: int | None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    instance_pattern = re.compile(r"instance:\s*([0-9a-f-]{36})")
    sequences: dict[str, list[str]] = {}
    labels: dict[str, int] = {}
    member = archive.extractfile(filename)
    if member is None:
        raise FileNotFoundError(f"{filename} not found inside OpenStack archive")

    count = 0
    for raw_line in member:
        line = raw_line.decode("utf-8", errors="replace").strip()
        instance_match = instance_pattern.search(line)
        parsed = _parse_openstack_raw_line(line)
        if instance_match is None or parsed is None:
            continue

        instance_id = instance_match.group(1)
        component, level, content = parsed
        sequences.setdefault(instance_id, []).append(f"{component}|{level}|{content}")
        labels[instance_id] = 1 if instance_id in anomaly_ids else 0

        count += 1
        if max_events is not None and count >= max_events:
            break

    return sequences, labels


def load_openstack_full_instance_sequences(
    data_dir: Path,
    max_events: int | None = None,
) -> tuple[list[list[int]], np.ndarray, list[list[int]], np.ndarray, dict[str, int]]:
    """Load full OpenStack as VM-instance sequences with real anomaly labels."""
    archive_path = data_dir / "OpenStack.tar.gz"
    if not archive_path.exists():
        download_openstack_full(data_dir)

    with tarfile.open(archive_path, "r:gz") as archive:
        anomaly_ids = _load_openstack_anomaly_ids(archive)
        normal1_sequences, normal1_labels = _read_openstack_instance_sequences(
            archive, "openstack_normal1.log", anomaly_ids, max_events
        )
        normal2_sequences, normal2_labels = _read_openstack_instance_sequences(
            archive, "openstack_normal2.log", anomaly_ids, max_events
        )
        abnormal_sequences, abnormal_labels = _read_openstack_instance_sequences(
            archive, "openstack_abnormal.log", anomaly_ids, max_events
        )

    train_sequences_raw = list(normal1_sequences.values()) + list(normal2_sequences.values())
    train_labels = np.asarray(
        list(normal1_labels.values()) + list(normal2_labels.values()),
        dtype=np.int64,
    )
    eval_sequences_raw = list(abnormal_sequences.values())
    eval_labels = np.asarray(list(abnormal_labels.values()), dtype=np.int64)

    all_events = {event for sequence in train_sequences_raw + eval_sequences_raw for event in sequence}
    event_to_id = {event: idx for idx, event in enumerate(sorted(all_events))}
    train_sequences = [[event_to_id[event] for event in sequence] for sequence in train_sequences_raw]
    eval_sequences = [[event_to_id[event] for event in sequence] for sequence in eval_sequences_raw]

    print(
        "OpenStack full VM sequences: "
        f"normal={len(train_sequences)} abnormal={len(eval_sequences)} "
        f"anomalous_instances={int(eval_labels.sum())}"
    )

    return train_sequences, train_labels, eval_sequences, eval_labels, event_to_id


def encode_events(events: list[str]) -> tuple[list[int], dict[str, int]]:
    unique_events = sorted(set(events))
    event_to_id = {event: idx for idx, event in enumerate(unique_events)}
    encoded = [event_to_id[event] for event in events]
    return encoded, event_to_id


def make_windows(events: list[int], labels: np.ndarray, seq_len: int) -> WindowDataset:
    x, y, window_labels = [], [], []
    for start in range(len(events) - seq_len):
        target_index = start + seq_len
        x.append(events[start:target_index])
        y.append(events[target_index])
        window_labels.append(labels[target_index])

    if not x:
        raise ValueError(
            "No windows were created. Try lowering --seq-len or increasing "
            "--max-sequences."
        )

    return WindowDataset(
        x=np.asarray(x, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        labels=np.asarray(window_labels, dtype=np.int64),
    )


def make_sequence_windows(sequences: list[list[int]], labels: np.ndarray, seq_len: int) -> WindowDataset:
    x, y, window_labels = [], [], []
    for sequence, label in zip(sequences, labels):
        if len(sequence) <= seq_len:
            continue
        for start in range(len(sequence) - seq_len):
            target_index = start + seq_len
            x.append(sequence[start:target_index])
            y.append(sequence[target_index])
            window_labels.append(label)

    if not x:
        raise ValueError(
            "No sequence windows were created. Try lowering --seq-len or using more data."
        )

    return WindowDataset(
        x=np.asarray(x, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        labels=np.asarray(window_labels, dtype=np.int64),
    )


def _stratify_or_none(labels: np.ndarray) -> np.ndarray | None:
    return labels if len(np.unique(labels)) > 1 else None


def _subset(dataset: WindowDataset, indices: np.ndarray) -> WindowDataset:
    return WindowDataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        labels=dataset.labels[indices],
    )


def _concat_datasets(datasets: list[WindowDataset]) -> WindowDataset:
    return WindowDataset(
        x=np.concatenate([dataset.x for dataset in datasets]),
        y=np.concatenate([dataset.y for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
    )


def inject_synthetic_anomalies(
    dataset: WindowDataset,
    vocab_size: int,
    fraction: float,
    seed: int,
) -> WindowDataset:
    """Create controlled unexpected next-event anomalies in an evaluation split."""
    if fraction <= 0:
        return dataset

    rng = np.random.default_rng(seed)
    x = dataset.x.copy()
    y = dataset.y.copy()
    labels = dataset.labels.copy()
    candidate_count = max(1, int(len(y) * fraction))
    anomaly_indices = rng.choice(len(y), size=candidate_count, replace=False)

    for index in anomaly_indices:
        original = y[index]
        replacement = int(rng.integers(0, vocab_size))
        while replacement == original or replacement in x[index]:
            replacement = int(rng.integers(0, vocab_size))
        y[index] = replacement
        labels[index] = 1

    return WindowDataset(x=x, y=y, labels=labels)


def prepare_data(
    data_dir: Path,
    seq_len: int,
    max_sequences: int | None,
    seed: int,
    dataset: str = "openstack",
    synthetic_anomaly_fraction: float = 0.08,
    gaia_include_status: bool = False,
) -> PreparedData:
    if dataset == "openstack":
        train_sequences, train_labels, abnormal_sequences, abnormal_labels, event_to_id = load_openstack_full_instance_sequences(
            data_dir, max_sequences
        )
        normal_windows = make_sequence_windows(train_sequences, train_labels, seq_len)
        abnormal_windows = make_sequence_windows(abnormal_sequences, abnormal_labels, seq_len)

        normal_indices = np.arange(len(normal_windows.x))
        train_idx, val_normal_idx = train_test_split(
            normal_indices,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
        )
        abnormal_indices = np.arange(len(abnormal_windows.x))
        val_abnormal_idx, test_idx = train_test_split(
            abnormal_indices,
            test_size=0.5,
            random_state=seed,
            stratify=_stratify_or_none(abnormal_windows.labels),
        )

        return PreparedData(
            train=_subset(normal_windows, train_idx),
            val=_concat_datasets(
                [_subset(normal_windows, val_normal_idx), _subset(abnormal_windows, val_abnormal_idx)]
            ),
            test=_subset(abnormal_windows, test_idx),
            vocab_size=len(event_to_id),
            event_to_id=event_to_id,
        )

    if dataset == "openstack2k":
        raw_events, labels = load_openstack_sample_events(data_dir, max_sequences)
    elif dataset == "gaia":
        raw_events, labels = load_gaia_trace_events(data_dir, max_sequences, include_status=gaia_include_status)
    elif dataset == "hdfs":
        raw_events, labels = load_hdfs_events(data_dir, max_sequences)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    events, event_to_id = encode_events(raw_events)
    all_windows = make_windows(events, labels, seq_len)
    indices = np.arange(len(all_windows.x))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.4,
        random_state=seed,
        stratify=_stratify_or_none(all_windows.labels),
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=_stratify_or_none(all_windows.labels[temp_idx]),
    )

    # Train only on normal behavior so anomalies are treated as surprising.
    train = _subset(all_windows, train_idx)
    train_normal_idx = np.where(train.labels == 0)[0]

    val = _subset(all_windows, val_idx)
    test = _subset(all_windows, test_idx)
    if dataset == "openstack2k" and labels.sum() == 0:
        print("No real OpenStack anomaly labels found in sample; injecting synthetic anomalies.")
        val = inject_synthetic_anomalies(val, len(event_to_id), synthetic_anomaly_fraction, seed)
        test = inject_synthetic_anomalies(test, len(event_to_id), synthetic_anomaly_fraction, seed + 1)

    return PreparedData(
        train=_subset(train, train_normal_idx),
        val=val,
        test=test,
        vocab_size=len(event_to_id),
        event_to_id=event_to_id,
    )
