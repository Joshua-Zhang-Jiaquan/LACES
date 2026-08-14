"""Hardened subprocess sandbox for the code capability eval.

A faithful, self-contained vendor of ``RL/reward/sandbox.run_tests`` (the
"Task 2" sandbox that replaced the unsafe-fallback executor in
``RL/eval/run_eval.py``). Kept inside ``eval/capability`` so the capability
harness has zero cross-tree import on the ``RL/`` package — which uses a
bare ``eval.`` / ``reward.`` namespace incompatible with the scale/
``eval.capability.`` layout.

Isolation stack (per test case):
* process: ``python -I`` (isolated mode: no site user dir, no cwd on
  ``sys.path``, ignores ``PYTHON*`` env vars);
* environment: fully scrubbed (``env={}``), ``stdin=DEVNULL``;
* network: ``unshare -n`` when the container permits it, else a ``socket``
  monkeypatch stub (best-effort — see ``ExecResult.isolation_mode``);
* resources (POSIX): ``RLIMIT_AS`` (mem_mb), ``RLIMIT_CPU`` (ceil timeout),
  ``RLIMIT_FSIZE`` (1 MiB), ``RLIMIT_NPROC`` (0,0);
* timeout: wall-clock via ``subprocess.run(timeout=)`` plus the CPU rlimit.

Each test case runs in its own fresh subprocess; a crash or timeout is a
fail, never the scorer. ``failed``/``timed_out``/``isolation_mode`` are
reported back for the artifact, because socket-stub mode is bypassable by a
determined program and should be logged.
"""

from __future__ import annotations

import ast
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import resource
except ImportError:  # non-POSIX
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    per_test: list[bool]
    frac_passed: float
    error: str
    timed_out: bool
    wall_ms: int
    isolation_mode: str
    # coarse failure taxonomy from the AST pre-check ("" if not run / not applicable)
    syntax_error: str
    entry_point_missing: bool


_SOCKET_STUB = "\n".join(
    (
        "import socket as _sandbox_socket",
        "",
        "class _SandboxBlockedSocket:",
        "    def __init__(self, *args, **kwargs):",
        "        raise OSError('network disabled by sandbox')",
        "",
        "def _sandbox_block_network(*args, **kwargs):",
        "    raise OSError('network disabled by sandbox')",
        "",
        "_sandbox_socket.socket = _SandboxBlockedSocket",
        "_sandbox_socket.create_connection = _sandbox_block_network",
        "_sandbox_socket.create_server = _sandbox_block_network",
        "_sandbox_socket.fromfd = _sandbox_block_network",
        "_sandbox_socket.socketpair = _sandbox_block_network",
    )
)


@lru_cache(maxsize=1)
def _unshare_path() -> str | None:
    return shutil.which("unshare")


@lru_cache(maxsize=1)
def _unshare_network_available() -> bool:
    unshare = _unshare_path()
    if not unshare:
        return False
    try:
        probe = subprocess.run(
            [unshare, "-n", sys.executable, "-I", "-c", "print('ok')"],
            env={},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _active_isolation_mode() -> str:
    return "unshare" if _unshare_network_available() else "socket_stub"


def _resource_limiter(timeout_s: float, mem_mb: int):
    def limit_resources() -> None:
        if resource is None:
            return
        mem_bytes = max(16, int(mem_mb)) * 1024 * 1024
        cpu_seconds = max(1, int(math.ceil(timeout_s)))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return limit_resources


def _runner_source(code: str, test: str, isolation_mode: str) -> str:
    preamble = _SOCKET_STUB if isolation_mode == "socket_stub" else ""
    return "\n".join((preamble, code, "", test, ""))


def _run_single_test(code: str, test: str, timeout_s: float, mem_mb: int, isolation_mode: str) -> tuple[bool, str, bool]:
    with tempfile.TemporaryDirectory(prefix="cap_sandbox_") as tmp:
        tmp_path = Path(tmp)
        runner = tmp_path / "runner.py"
        runner.write_text(_runner_source(code, test, isolation_mode), encoding="utf-8")

        command = [sys.executable, "-I", str(runner)]
        if isolation_mode == "unshare":
            unshare = _unshare_path()
            if unshare:
                command = [unshare, "-n", *command]

        try:
            completed = subprocess.run(
                command,
                cwd=tmp,
                env={},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                preexec_fn=_resource_limiter(timeout_s, mem_mb) if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return False, stderr or f"timed out after {timeout_s:.3f}s", True
        except OSError as exc:
            return False, f"sandbox launch failed: {exc}", False

        if completed.returncode == 0:
            return True, "", False

        error = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return False, textwrap.shorten(error, width=4000, placeholder="..."), False


def _ast_precheck(code: str, entry_point: str | None) -> tuple[str, bool]:
    """Cheap static checks before paying for execution.

    Returns ``(syntax_error, entry_point_missing)``. ``syntax_error`` is the
    message line ("" if the code parses); ``entry_point_missing`` is True when
    ``entry_point`` is given but not a top-level ``def``/``async def`` name.
    Ported from ``RL/eval/v2_adapter_code.py`` (AST parse + entry-point scan).
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError as exc:
        return f"syntax_error line {exc.lineno}: {exc.msg}", False
    if not entry_point:
        return "", False
    top_level = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return ("", entry_point not in top_level)


def run_tests(
    code: str,
    tests: Sequence[str],
    *,
    timeout_s: float = 5.0,
    mem_mb: int = 256,
    entry_point: str | None = None,
) -> ExecResult:
    """Execute ``code`` against each test in isolation; return pass@1 + taxonomy.

    ``code`` is the full candidate program (completion already assembled with
    its prompt per suite rules); each ``test`` is an executable assertion
    block (HumanEval's ``def check(...)`` body or an MBPP ``assert`` + imports).
    The candidate's ``entry_point`` (if known) is verified statically first so
    a missing function short-circuits to a named failure rather than a generic
    traceback.
    """
    test_list = list(tests)
    syntax_error, entry_missing = _ast_precheck(code, entry_point)
    # short-circuit: no point running tests against a program that won't parse
    # or that lacks the required entry function.
    if syntax_error or entry_missing or not test_list:
        return ExecResult(
            passed=False,
            per_test=[False] * len(test_list),
            frac_passed=0.0,
            error=syntax_error or ("entry_point_missing" if entry_missing else "no_tests"),
            timed_out=False,
            wall_ms=0,
            isolation_mode=_active_isolation_mode(),
            syntax_error=syntax_error,
            entry_point_missing=entry_missing,
        )

    start = time.monotonic()
    isolation_mode = _active_isolation_mode()
    per_test: list[bool] = []
    errors: list[str] = []
    timed_out = False
    for index, test in enumerate(test_list):
        ok, error, did_timeout = _run_single_test(code, test, timeout_s, mem_mb, isolation_mode)
        per_test.append(ok)
        if error:
            errors.append(f"test {index}: {error}")
        timed_out = timed_out or did_timeout
    passed_count = sum(per_test)
    frac = (passed_count / len(test_list)) if test_list else 0.0
    wall_ms = int((time.monotonic() - start) * 1000)
    return ExecResult(
        passed=bool(test_list) and all(per_test),
        per_test=per_test,
        frac_passed=frac,
        error="\n".join(errors),
        timed_out=timed_out,
        wall_ms=wall_ms,
        isolation_mode=isolation_mode,
        syntax_error="",
        entry_point_missing=False,
    )
