"""core.sqlmap_runner

Runs the bundled sqlmap (./sqlmap/sqlmap.py) as a subprocess.

One RL step = one sqlmap execution.

To count HTTP requests accurately, this runner supports sqlmap's traffic log
option (-t). It will parse the generated log to count requests.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SqlmapTarget:
    url: str


@dataclass(frozen=True)
class SqlmapRunConfig:
    python_exe: str = "python"
    sqlmap_script: str = "sqlmap/sqlmap.py"
    timeout_sec: int = 120
    threads: int = 1
    batch: bool = True
    flush_session: bool = True
    verbosity: int = 1
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
    requests_count: int


# Regex to find the start of a request in sqlmap's traffic log file.
# Format is: "HTTP request #[number]:"
_RE_REQUEST_START = re.compile(r"^HTTP request #\[\d+\]:", re.MULTILINE)


class SqlmapRunner:
    def __init__(self, run_cfg: SqlmapRunConfig):
        self.cfg = run_cfg

    def build_cmd(
        self, target: SqlmapTarget, argv_options: Sequence[str], traffic_log_path: Optional[str]
    ) -> List[str]:
        cmd: List[str] = [
            self.cfg.python_exe,
            self.cfg.sqlmap_script,
            "-u",
            target.url,
            "--threads",
            str(self.cfg.threads),
            "-v",
            str(self.cfg.verbosity),
        ]
        if self.cfg.batch:
            cmd.append("--batch")
        if self.cfg.flush_session:
            cmd.append("--flush-session")

        if traffic_log_path:
            cmd.extend(["-t", traffic_log_path])

        if self.cfg.extra_args:
            cmd.extend(self.cfg.extra_args)

        cmd.extend(list(argv_options))
        return cmd

    def run(self, *, target: SqlmapTarget, argv_options: Sequence[str], traffic_log_path: Optional[str] = None) -> SqlmapRunResult:
        cmd = self.build_cmd(target=target, argv_options=argv_options, traffic_log_path=traffic_log_path)

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

        req_count = 0
        if traffic_log_path and os.path.exists(traffic_log_path):
            try:
                with open(traffic_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    req_count = len(_RE_REQUEST_START.findall(content))
            except Exception:
                pass  # best-effort

        cmd_str = " ".join(_shell_quote(x) for x in cmd)
        return SqlmapRunResult(
            cmd=list(cmd),
            cmd_str=cmd_str,
            returncode=rc,
            stdout=out,
            stderr=err,
            duration_sec=float(dt),
            timed_out=bool(timed_out),
            requests_count=int(req_count),
        )


def _shell_quote(s: str) -> str:
    if not s:
        return "''"
    if any(c.isspace() for c in s) or '"' in s or "'" in s:
        return "'" + s.replace("'", "'\\''") + "'"
    return s
