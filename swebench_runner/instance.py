"""Single-instance lifecycle: container -> limbo exec -> git diff."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec

from .prompts import build_prompt

log = logging.getLogger(__name__)

TESTBED = "/testbed"
LIMBO_SRC = "/limbo-src"  # host limbo-code repo, mounted read-only
PROMPT_FILE = "/tmp/limbo-prompt.md"
CONTAINER_CONFIG = Path("/root/.limbo/config.toml")


@dataclass
class RunConfig:
    limbo_repo: Path  # host path to the limbo-code checkout
    model: str | None  # None -> use the host config's model
    timeout: int = 1800  # wall-clock seconds per instance


@dataclass
class InstanceResult:
    instance_id: str
    model_patch: str
    model_name_or_path: str
    exit_status: str  # "ok" | "agent_error" | "timeout" | "infra_error"
    duration_s: float
    log_dir: Path

    def prediction(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model_patch": self.model_patch,
            "model_name_or_path": self.model_name_or_path,
        }


def _docker(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
        **kwargs,
    )


def _remote_image_key(instance: dict) -> str:
    spec = make_test_spec(instance, namespace="swebench")
    return spec.instance_image_key


def _ensure_image(image: str) -> None:
    if _docker("image", "inspect", image, check=False).returncode == 0:
        return
    log.info("pulling image %s ...", image)
    subprocess.run(["docker", "pull", image], check=True)


def _container_config_toml(model: str | None) -> str:
    """Minimal limbo config for inside the container.

    Model/api_key/base_url are resolved from the *host* config so the
    container only needs this one file.
    """
    from limbo.config import load_config
    from limbo.llm.catalog import resolve_api_key, resolve_base_url, resolve_model

    config = load_config()
    if model:
        config.llm.model = model
    spec = resolve_model(config.llm.model)
    api_key = resolve_api_key(spec, config)
    base_url = resolve_base_url(spec, config)
    lines = ["[llm]", f'model = "{config.llm.model}"']
    if api_key:
        lines.append(f'api_key = "{api_key}"')
    if base_url:
        lines.append(f'base_url = "{base_url}"')
    return "\n".join(lines) + "\n"


def _exec(container: str, cmd: str, **kwargs) -> subprocess.CompletedProcess:
    return _docker("exec", container, "bash", "-lc", cmd, **kwargs)


def _container_python(container: str) -> tuple[int, int]:
    probe = _exec(
        container,
        "python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'",
    )
    return tuple(int(p) for p in probe.stdout.strip().split("."))  # type: ignore[return-value]


def _install_limbo(container: str) -> str:
    """Install limbo inside the container; return the python to run it with.

    Instance images frequently ship Python < 3.11 (the repo's historical
    test env), too old for limbo. Fallback: install a standalone 3.12 via
    uv into /opt/limbo-venv and use that interpreter instead.

    Raises RuntimeError on failure.
    """
    version = _container_python(container)
    log.info("container python: %s.%s", *version)
    # hatch-vcs derives the version from .git; the mount is read-only and
    # some build frontends copy the tree without it, so pin explicitly.
    env = "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0"
    if version >= (3, 11):
        install = (
            f"{env} python3 -m pip install --quiet {shlex.quote(LIMBO_SRC)}"
            " && python3 -c 'import limbo'"
        )
        python = "python3"
    else:
        log.info("python < 3.11; installing standalone 3.12 via uv")
        install = " && ".join([
            "python3 -m pip install --quiet uv",
            "python3 -m uv python install --quiet 3.12",
            "python3 -m uv venv --quiet --python 3.12 /opt/limbo-venv",
            f"{env} python3 -m uv pip install --quiet"
            f" --python /opt/limbo-venv/bin/python {shlex.quote(LIMBO_SRC)}",
            "/opt/limbo-venv/bin/python -c 'import limbo'",
        ])
        python = "/opt/limbo-venv/bin/python"
    result = _exec(container, install, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "limbo install failed:\n" + result.stderr[-2000:]
        )
    return python


def run_instance(
    instance: dict, config: RunConfig, log_dir: Path
) -> InstanceResult:
    """Run limbo on one SWE-bench instance and collect the patch.

    Never raises for agent-level failures; infra problems surface as
    exit_status="infra_error" with details in the log dir.
    """
    instance_id = instance["instance_id"]
    image = _remote_image_key(instance)
    container = f"limbo-swe-{instance_id.lower().replace('_', '-').replace('.', '-')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    status = "infra_error"
    patch = ""

    try:
        _ensure_image(image)
        _docker(
            "run", "-d", "--rm",
            "--name", container,
            "-v", f"{config.limbo_repo.resolve()}:{LIMBO_SRC}:ro",
            image, "tail", "-f", "/dev/null",
        )
        try:
            limbo_python = _install_limbo(container)
            _exec(
                container,
                f"mkdir -p {CONTAINER_CONFIG.parent}",
            )
            config_toml = _container_config_toml(config.model)
            subprocess.run(
                ["docker", "exec", "-i", container, "bash", "-c",
                 f"cat > {CONTAINER_CONFIG}"],
                input=config_toml, text=True, check=True,
            )
            prompt = build_prompt(instance)
            subprocess.run(
                ["docker", "exec", "-i", container, "bash", "-c",
                 f"cat > {PROMPT_FILE}"],
                input=prompt, text=True, check=True,
            )

            model = config.model or _host_model()
            agent_args = [
                "docker", "exec", container,
                limbo_python, "-m", "limbo", "exec",
                "--workdir", TESTBED,
                "--prompt-file", PROMPT_FILE,
                "--session-dir", "/tmp/limbo-sessions",
                "--model", model,
            ]
            with (
                open(log_dir / "stdout.log", "w") as out,
                open(log_dir / "stderr.log", "w") as err,
            ):
                try:
                    proc = subprocess.run(
                        agent_args, stdout=out, stderr=err,
                        timeout=config.timeout,
                    )
                    status = "ok" if proc.returncode == 0 else "agent_error"
                except subprocess.TimeoutExpired:
                    # docker exec is dead; the in-container process is
                    # cleaned up when the container is removed below.
                    status = "timeout"

            diff = _exec(
                container,
                f"git -C {TESTBED} -c safe.directory={TESTBED} diff",
                check=False,
            )
            patch = diff.stdout
        finally:
            _docker("rm", "-f", container, check=False)
    except Exception:
        log.exception("infra failure on %s", instance_id)
        (log_dir / "infra_error.log").write_text(
            json.dumps({"instance_id": instance_id}, indent=2)
        )
        raise

    return InstanceResult(
        instance_id=instance_id,
        model_patch=patch,
        model_name_or_path=config.model or _host_model(),
        exit_status=status,
        duration_s=time.monotonic() - start,
        log_dir=log_dir,
    )


def _host_model() -> str:
    from limbo.config import load_config

    return load_config().llm.model
