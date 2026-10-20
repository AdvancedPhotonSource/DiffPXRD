from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import nullcontext
from typing import Any

import torch


class PerfTimer:
    def __init__(
        self,
        *,
        enabled: bool = False,
        device: str | torch.device | None = None,
        profile_scopes: bool = False,
        scope_prefix: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device) if device is not None else None
        self.profile_scopes = bool(profile_scopes)
        self.scope_prefix = "" if scope_prefix is None else str(scope_prefix).strip("/")
        self.records: dict[str, list[float]] = {}

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _scope_name(self, name: str) -> str:
        name = str(name)
        if not self.scope_prefix:
            return name
        return f"{self.scope_prefix}/{name}"

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        scope = (
            torch.profiler.record_function(self._scope_name(str(name)))
            if self.profile_scopes
            else nullcontext()
        )
        with scope:
            if not self.enabled:
                yield
                return
            self._sync()
            start = time.perf_counter()
            try:
                yield
            finally:
                self._sync()
                self.records.setdefault(str(name), []).append(time.perf_counter() - start)

    def summary(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for name, values in sorted(self.records.items()):
            if not values:
                continue
            total = float(sum(values))
            summary[name] = {
                "count": int(len(values)),
                "total_sec": total,
                "mean_sec": float(total / len(values)),
                "median_sec": float(statistics.median(values)),
                "min_sec": float(min(values)),
                "max_sec": float(max(values)),
            }
        return summary
