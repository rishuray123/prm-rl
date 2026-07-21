"""Best-effort evaluation sandbox.

Prevents a policy from tampering with reward / eval code while we run
generations against it. Every eval invocation runs in a subprocess with a
scrubbed environment and a tmp working directory. This is *not* a security
boundary — it's a guardrail for research on reward hacking.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_KEEP_ENV = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
    "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE", "TORCH_HOME", "WANDB_MODE", "WANDB_API_KEY",
    "PYTHONPATH",
}


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _KEEP_ENV}
    if extra:
        env.update(extra)
    return env


@contextmanager
def sandbox_cwd(prefix: str = "prmrl-eval-") -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_sandboxed(cmd: str | list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    with sandbox_cwd() as cwd:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=scrubbed_env(),
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
