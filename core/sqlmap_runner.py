"""core.sqlmap_runner

Runs the bundled sqlmap (./sqlmap/sqlmap.py) as a subprocess.

One RL step = one sqlmap execution.

By default we do NOT use sqlmap's --output-dir to avoid writing lots of files.
We only capture stdout/stderr for parsing and reporting.

If you later need artifacts, you can re-add output_dir plumbing.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SqlmapTarget:
    url: str
    method: str = "GET"  # GET/POST


@dataclass(frozen=True)
class SqlmapRunConfig:
    python_exe: str = "python"
    sqlmap_script: str = "sqlmap/sqlmap.py"
    timeout_sec: int = 120
    threads: int = 1
    batch: bool = True
    flush_session: bool = True
    verbosity: int = 1
    # If provided, will be passed as extra args (e.g. proxy, headers)
    extra_args: Optional[List[str]] = None


@dataclass(frozen=True)
class SqlmapRunResult:
    cmd: List[str]
    cmd_str: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool


class SqlmapRunner:
    def __init__(self, run_cfg: SqlmapRunConfig):
        self.cfg = run_cfg

    def build_cmd(
        self,
        target: SqlmapTarget,
        argv_options: Sequence[str],
    ) -> List[str]:
        cmd: List[str] = [
            self.cfg.python_exe,
            self.cfg.sqlmap_script,
            "-u",
            target.url,
            "--method",
            target.method,
            "--threads",
            str(self.cfg.threads),
            "-v",
            str(self.cfg.verbosity),
        ]
        if self.cfg.batch:
            cmd.append("--batch")
        if self.cfg.flush_session:
            cmd.append("--flush-session")

        if self.cfg.extra_args:
            cmd.extend(self.cfg.extra_args)

        cmd.extend(list(argv_options))
        return cmd

    def run(
        self,
        *,
        target: SqlmapTarget,
        argv_options: Sequence[str],
    ) -> SqlmapRunResult:
        cmd = self.build_cmd(target=target, argv_options=argv_options)

        t0 = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.cfg.timeout_sec,
                cwd=str(Path(__file__).resolve().parents[1]),  # repo root
            )
            rc = int(proc.returncode)
            out = proc.stdout or ""
            err = proc.stderr or ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = -1
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""

        dt = time.perf_counter() - t0

        cmd_str = " ".join(_shell_quote(x) for x in cmd)
        return SqlmapRunResult(
            cmd=list(cmd),
            cmd_str=cmd_str,
            returncode=rc,
            stdout=out,
            stderr=err,
            duration_sec=float(dt),
            timed_out=bool(timed_out),
        )


def _shell_quote(s: str) -> str:
    # Simple quote for report readability (not a security boundary)
    if not s:
        return "''"
    if any(c.isspace() for c in s) or '"' in s or "'" in s:
        return "'" + s.replace("'", "'\\''") + "'"
    return s
