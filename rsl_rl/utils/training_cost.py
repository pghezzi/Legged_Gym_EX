"""Small, dependency-light helpers for post-specialist cost accounting."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


PROVENANCE_STATES = ("measured", "reconstructed", "unavailable")


def cuda_device_info(device: Any) -> dict[str, Any]:
    device = torch.device(device)
    available = device.type == "cuda" and torch.cuda.is_available()
    if not available:
        return {
            "gpu_model": None,
            "gpu_count": 0,
            "cuda_device": None,
        }
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_model": torch.cuda.get_device_name(index),
        "gpu_count": 1,
        "cuda_device": index,
    }


def synchronize(device: Any) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device: Any) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: Any) -> float | None:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)


def artifact_size_mb(path: os.PathLike[str] | str) -> float | None:
    path = Path(path)
    return float(path.stat().st_size) / (1024.0 ** 2) if path.is_file() else None


def directory_size_mb(path: os.PathLike[str] | str) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return artifact_size_mb(path)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024.0 ** 2)


def provenance_map(record: dict[str, Any], default: str = "measured") -> dict[str, str]:
    """Return explicit provenance for every scalar accounting field."""
    statuses = {}
    for key, value in record.items():
        if key in {"metric_status", "notes"} or isinstance(value, (dict, list, tuple)):
            continue
        statuses[key] = default if value is not None else "unavailable"
    return statuses


def write_cost_record(path: os.PathLike[str] | str, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n")
    os.replace(temporary, path)


class WallTimer:
    def __init__(self, device: Any = "cpu"):
        self.device = torch.device(device)
        self.started = 0.0
        self.elapsed_s = 0.0

    def start(self) -> "WallTimer":
        synchronize(self.device)
        self.started = time.perf_counter()
        return self

    def stop(self) -> float:
        synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self.started
        return self.elapsed_s
