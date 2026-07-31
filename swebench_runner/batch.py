"""Batch driver: iterate instances with concurrency + resume."""

from __future__ import annotations

import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from .instance import InstanceResult, RunConfig, run_instance

log = logging.getLogger(__name__)


def load_instances(
    dataset: str,
    instance_ids: list[str] | None = None,
    count: int | None = None,
    seed: int = 42,
) -> list[dict]:
    ds = load_dataset(dataset, split="test")
    instances = list(ds)
    if instance_ids:
        wanted = set(instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            raise ValueError(f"instance ids not in dataset: {sorted(missing)}")
    elif count is not None and count < len(instances):
        instances = random.Random(seed).sample(instances, count)
    return instances


def load_completed_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    ids = set()
    for line in predictions_path.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["instance_id"])
    return ids


class PredictionWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, result: InstanceResult) -> None:
        record = result.prediction()
        with self._lock, open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")


def run_batch(
    instances: list[dict],
    config: RunConfig,
    run_dir: Path,
    concurrency: int = 1,
) -> None:
    predictions_path = run_dir / "predictions.jsonl"
    completed = load_completed_ids(predictions_path)
    todo = [i for i in instances if i["instance_id"] not in completed]
    log.info(
        "batch: %d instances, %d already done, %d to run",
        len(instances), len(completed), len(todo),
    )
    writer = PredictionWriter(predictions_path)

    def worker(instance: dict) -> InstanceResult:
        iid = instance["instance_id"]
        log_dir = run_dir / "logs" / iid
        log.info("start %s", iid)
        result = run_instance(instance, config, log_dir)
        log.info(
            "done %s: %s in %.0fs, patch %d chars",
            iid, result.exit_status, result.duration_s,
            len(result.model_patch),
        )
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, i): i for i in todo}
        for future in as_completed(futures):
            instance = futures[future]
            try:
                writer.write(future.result())
            except Exception:
                log.exception("instance failed: %s", instance["instance_id"])
