"""M3 pilot: per-instance pipeline pull -> agent -> evaluate -> rmi.

Unlike `batch.py` (bulk agent runs), the pilot interleaves the official
evaluation per instance so the instance image can be deleted right after
both phases are done with it. Peak disk usage is therefore roughly
`concurrency` images (~4 GB each) instead of the whole pilot's worth.

Resume: instances already present in <run-dir>/results.jsonl are skipped
entirely (prediction + evaluation both done).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .batch import PredictionWriter, load_instances
from .instance import RunConfig, _docker, _remote_image_key, run_instance

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_LOGS = REPO_ROOT / "logs" / "run_evaluation"

_disk_condition = threading.Condition()


def _free_gb() -> float:
    # Docker Desktop stores its VM disk on the home volume, so home free
    # space is the right proxy for "room for one more image".
    return shutil.disk_usage(Path.home()).free / 1e9


def _wait_for_disk(min_free_gb: float) -> None:
    with _disk_condition:
        while (free := _free_gb()) < min_free_gb:
            log.warning(
                "low disk: %.1f GB free (< %.1f GB); waiting for other "
                "workers to finish and clean up",
                free, min_free_gb,
            )
            _disk_condition.wait(timeout=60)


def _notify_disk_maybe_freed() -> None:
    with _disk_condition:
        _disk_condition.notify_all()


def evaluate_instance(
    instance_id: str,
    prediction: dict,
    dataset: str,
    run_dir: Path,
    eval_timeout: int,
) -> bool | None:
    """Run the official harness for one instance. Returns resolved, or
    None if evaluation itself failed."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_file = eval_dir / f"{instance_id}.jsonl"
    pred_file.write_text(json.dumps(prediction) + "\n")
    run_id = f"{run_dir.name}-{instance_id}"

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--instance_ids", instance_id,
        "--predictions_path", str(pred_file),
        "--max_workers", "1",
        "--run_id", run_id,
        "--cache_level", "none",
    ]
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=(eval_dir / f"{instance_id}.eval.log").open("w"),
            timeout=eval_timeout,
        )
    except subprocess.TimeoutExpired:
        log.error("eval timed out for %s", instance_id)
        # Kill leftover harness containers for this instance.
        _docker(
            "rm", "-f", f"sweb.eval.{instance_id.lower()}.{run_id}",
            check=False,
        )
        return None

    reports = glob.glob(str(EVAL_LOGS / run_id / "*" / instance_id / "report.json"))
    if not reports:
        log.error("no eval report for %s", instance_id)
        return None
    report = json.loads(Path(reports[0]).read_text())
    return bool(report[instance_id].get("resolved"))


def _load_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    return {
        json.loads(line)["instance_id"]
        for line in results_path.read_text().splitlines()
        if line.strip()
    }


def pilot_instance(
    instance: dict,
    config: RunConfig,
    dataset: str,
    run_dir: Path,
    eval_timeout: int,
    min_free_gb: float,
    keep_images: bool,
    pred_writer: PredictionWriter,
) -> dict:
    iid = instance["instance_id"]
    image = _remote_image_key(instance)
    record = {
        "instance_id": iid,
        "exit_status": "infra_error",
        "resolved": None,
        "patch_chars": 0,
        "duration_s": 0.0,
    }
    try:
        _wait_for_disk(min_free_gb)
        log.info("start %s (%.1f GB free)", iid, _free_gb())
        result = run_instance(instance, config, run_dir / "logs" / iid)
        record.update(
            exit_status=result.exit_status,
            patch_chars=len(result.model_patch),
            duration_s=round(result.duration_s, 1),
        )
        pred_writer.write(result)
        if result.model_patch.strip():
            record["resolved"] = evaluate_instance(
                iid, result.prediction(), dataset, run_dir, eval_timeout
            )
        else:
            log.warning("%s produced an empty patch; skipping eval", iid)
            record["resolved"] = False
    except Exception:
        log.exception("pilot failed for %s", iid)
    finally:
        if not keep_images:
            _docker("rmi", image, check=False)
            log.info("removed image %s (%.1f GB free)", image, _free_gb())
        _notify_disk_maybe_freed()
    log.info(
        "done %s: %s, resolved=%s", iid, record["exit_status"], record["resolved"]
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="swebench_runner.pilot",
        description="M3 pilot: agent + evaluation per instance, deleting "
        "images as it goes to bound disk usage.",
    )
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="agent wall-clock seconds per instance")
    parser.add_argument("--eval-timeout", type=int, default=1800,
                        help="evaluation wall-clock seconds per instance")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=15.0,
                        help="pause new instances below this much free disk")
    parser.add_argument("--keep-images", action="store_true",
                        help="do not delete instance images afterwards")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.run_dir / "results.jsonl"
    done = _load_done(results_path)

    instances = load_instances(
        args.dataset,
        instance_ids=args.instance_ids,
        count=args.count,
        seed=args.seed,
    )
    todo = [i for i in instances if i["instance_id"] not in done]
    log.info(
        "pilot: %d instances, %d done, %d to run", len(instances), len(done), len(todo)
    )

    config = RunConfig(limbo_repo=REPO_ROOT, model=args.model, timeout=args.timeout)
    pred_writer = PredictionWriter(args.run_dir / "predictions.jsonl")
    results_lock = threading.Lock()

    def worker(instance: dict) -> dict:
        record = pilot_instance(
            instance, config, args.dataset, args.run_dir,
            args.eval_timeout, args.min_free_gb, args.keep_images, pred_writer,
        )
        with results_lock, open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(worker, i): i for i in todo}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception:
                log.exception("worker crashed: %s", futures[future]["instance_id"])

    # Merge with prior records for the final summary (resume case).
    prior = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    resolved = sum(1 for r in prior if r["resolved"])
    by_status: dict[str, int] = {}
    for r in prior:
        by_status[r["exit_status"]] = by_status.get(r["exit_status"], 0) + 1
    summary = {
        "total": len(prior),
        "resolved": resolved,
        "resolve_rate": round(resolved / len(prior), 4) if prior else 0.0,
        "by_exit_status": by_status,
        "empty_patches": sum(1 for r in prior if not r["patch_chars"]),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
