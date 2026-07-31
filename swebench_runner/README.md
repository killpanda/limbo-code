# SWE-bench Runner for limbo-code

Runs limbo-code (`limbo exec`, headless mode) on SWE-bench instances and
emits predictions in the official format, consumable by
`swebench.harness.run_evaluation`.

## How it works

Per instance (all inside the official per-instance Docker image, where
`/testbed` is the repo at `base_commit` with dependencies pre-installed):

1. Start a throwaway container, with the host's limbo-code checkout
   mounted read-only at `/limbo-src`
2. `pip install /limbo-src` inside the container
   (**requires container Python >= 3.11**)
3. Write a minimal `~/.limbo/config.toml` (model/key/base_url resolved
   from the host config) and the prompt file into the container
4. `limbo exec --workdir /testbed --prompt-file ...` with a host-side
   wall-clock timeout; stdout/stderr land in `runs/<run>/logs/<id>/`
5. Collect the patch via `git -C /testbed diff`, then destroy the container

Timeout mid-turn still collects whatever diff exists (SWE-bench convention:
a partial patch is scored like any other).

## Usage

```bash
# 5 random instances, sequentially
python -m swebench_runner --count 5 --concurrency 1

# specific instances
python -m swebench_runner --instance-ids astropy__astropy-12907

# full Verified set, 8-way parallel, 30 min per instance
python -m swebench_runner --concurrency 8 --timeout 1800 \
    --run-dir runs/full-glm4.7 --model glm-4.7
```

Re-running with the same `--run-dir` skips instances already present in
`predictions.jsonl` (resume).

## Evaluate

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path runs/<run>/predictions.jsonl \
    --max_workers 8 --run_id <run>
```

## Notes

- Images pull from Docker Hub (`swebench/sweb.eval.x86_64.*`, ~1-3 GB
  each); on Apple Silicon they run under x86_64 emulation — expect slower
  test execution, and prefer modest `--concurrency`.
- Instance images with Python < 3.11 fail at the install step
  (`exit_status=infra_error`). A uv-based fallback (install a standalone
  3.12 in the container) is a possible future fix.
