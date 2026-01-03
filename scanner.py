"""scanner.py (sqlmap-driven RL)

This project uses sqlmap as the execution backend.

One RL step = one sqlmap execution.
Episode ends when:
- sqlmap confirms an injectable parameter, OR
- max_steps reached (default 15)

Crawler integration:
- We extract RequestTemplate(s): (url, method, default params)
- Then we test ONE parameter at a time (-p <param>) while keeping other
  parameters fixed at their default values.

Important practical note:
- If the currently tested parameter has an empty default value, sqlmap will
  warn and can behave poorly. We therefore provide a baseline non-empty value
  for the injected parameter using a simple heuristic:
    - numeric-ish param names -> "1"
    - else -> "a"

Output/report includes:
- url
- vulnerable parameter
- exploited payload (best-effort parsed from sqlmap output)
- RL-generated payload: the sqlmap command that led to the result
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Set

import requests

from core.crawler import Crawler, RequestTemplate, encode_get_url
from core.sqlmap_evaluator import RewardConfig, compute_reward
from core.sqlmap_parser import SqlmapObservation, parse_sqlmap_output
from core.sqlmap_runner import SqlmapRunConfig, SqlmapRunner, SqlmapTarget

from rl.agent import Agent, AgentConfig
from rl.sqlmap_config import SqlmapConfig, apply_option, to_sqlmap_args
from rl.sqlmap_option_space import OptionAction, default_option_actions


def _one_hot(idx: int, dim: int) -> List[float]:
    v = [0.0] * dim
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _default_injected_value(param_name: str) -> str:
    """Heuristic baseline value for the injected parameter when default is empty."""
    p = (param_name or "").lower()
    numeric_hints = [
        "id",
        "uid",
        "pid",
        "gid",
        "user",
        "page",
        "num",
        "no",
        "index",
        "item",
        "cat",
        "category",
    ]
    if any(h in p for h in numeric_hints) or p.endswith("id"):
        return "1"
    return "a"


def build_state(
    obs: Optional[SqlmapObservation],
    cfg: SqlmapConfig,
    *,
    action_space: List[OptionAction],
    steps_left: int,
    last_duration_sec: float,
    episode_index: int,
    episodes_per_input: int,
    tried_techniques: Set[str],
) -> List[float]:
    if obs is None:
        injectable = 0.0
        blocked = 0.0
        timeout = 0.0
    else:
        injectable = 1.0 if obs.injectable else 0.0
        blocked = 1.0 if obs.blocked else 0.0
        timeout = 1.0 if obs.timeout else 0.0

    steps_left_norm = _clamp(steps_left / 15.0, 0.0, 1.0)
    last_dur_norm = _clamp(float(last_duration_sec) / 120.0, 0.0, 1.0)

    denom = max(1, int(episodes_per_input) - 1)
    ep_norm = _clamp(float(episode_index) / float(denom), 0.0, 1.0)

    tried_list = ["B", "E", "T", "U", "BE", "BT", "ET", "BET"]
    tried_mask = [1.0 if t in tried_techniques else 0.0 for t in tried_list]

    technique_list = ["", "B", "E", "T", "U", "BE", "BT", "ET", "BET"]
    tech_idx = technique_list.index(cfg.technique) if cfg.technique in technique_list else 0
    tech_oh = _one_hot(tech_idx, len(technique_list))

    level_list = [1, 3, 5]
    lvl_idx = level_list.index(cfg.level) if cfg.level in level_list else 0
    lvl_oh = _one_hot(lvl_idx, len(level_list))

    risk_list = [1, 2, 3]
    risk_idx = risk_list.index(cfg.risk) if cfg.risk in risk_list else 0
    risk_oh = _one_hot(risk_idx, len(risk_list))

    delay_bins = [None, 0.0, 0.5, 1.0]
    d_idx = delay_bins.index(cfg.delay) if cfg.delay in delay_bins else 0
    delay_oh = _one_hot(d_idx, len(delay_bins))

    time_bins = [None, 3, 5, 8, 10]
    ts_idx = time_bins.index(cfg.time_sec) if cfg.time_sec in time_bins else 0
    time_oh = _one_hot(ts_idx, len(time_bins))

    tamper_names = sorted({a.value for a in action_space if a.kind == "add_tamper" and a.value})
    tamper_flags = [1.0 if t in cfg.tampers else 0.0 for t in tamper_names]

    ra = [1.0 if cfg.random_agent else 0.0]

    timeout_bins = [None, 10, 20, 30, 60]
    to_idx = timeout_bins.index(cfg.timeout) if cfg.timeout in timeout_bins else 0
    timeout_oh = _one_hot(to_idx, len(timeout_bins))

    retries_bins = [None, 0, 1, 2, 3]
    r_idx = retries_bins.index(cfg.retries) if cfg.retries in retries_bins else 0
    retries_oh = _one_hot(r_idx, len(retries_bins))

    return (
        [injectable, blocked, timeout, steps_left_norm, last_dur_norm, ep_norm]
        + tried_mask
        + tech_oh
        + lvl_oh
        + risk_oh
        + delay_oh
        + time_oh
        + ra
        + timeout_oh
        + retries_oh
        + tamper_flags
    )


def scan(
    target_url: str,
    *,
    max_steps: int = 15,
    episodes_per_input: int = 1,
    timeout_sec: int = 120,
    epsilon: float = 0.3,
    seed: int = 1337,
    model_in: Optional[str] = None,
    model_out: Optional[str] = None,
    json_output: bool = False,
    debug_log: Optional[str] = None,
) -> List[dict]:
    session = requests.Session()
    session.headers.setdefault("User-Agent", "scanner-sqli-sqlmap-rl/1.0")

    crawler = Crawler(timeout=15, session=session)

    allowed_tampers = [
        "randomcase",
        "space2comment",
        "between",
        "charencode",
        "equaltolike",
        "space2plus",
    ]

    action_space = default_option_actions(allowed_tampers)
    action_ids = list(range(len(action_space)))

    dummy_cfg = SqlmapConfig()
    dummy_state = build_state(
        None,
        dummy_cfg,
        action_space=action_space,
        steps_left=max_steps,
        last_duration_sec=0.0,
        episode_index=0,
        episodes_per_input=episodes_per_input,
        tried_techniques=set(),
    )
    state_dim = len(dummy_state)

    agent_cfg = AgentConfig(
        state_dim=state_dim,
        epsilon=epsilon,
        seed=seed,
        lr=0.05,
        gamma=0.95,
        batch_size=64,
        use_rnd=True,
        intrinsic_scale=0.05,
    )

    agent = Agent(action_ids=action_ids, config=agent_cfg)
    if model_in and os.path.exists(model_in):
        agent = Agent.load_from_file(model_in, action_ids=action_ids, config=agent_cfg)

    runner = SqlmapRunner(
        SqlmapRunConfig(
            timeout_sec=timeout_sec,
            threads=1,
            batch=True,
            flush_session=True,
            verbosity=1,
            extra_args=["--smart", "--skip-static"],
        )
    )

    reward_cfg = RewardConfig()

    templates: List[RequestTemplate] = crawler.crawl(target_url)

    results: List[dict] = []

    debug_f = None
    if debug_log:
        os.makedirs("debug", exist_ok=True)
        path = debug_log
        if path.lower() in {"1", "true", "yes", "on"}:
            path = os.path.join("debug", f"debug_{int(time.time())}.jsonl")
        elif not os.path.isabs(path):
            path = os.path.join("debug", path)
        debug_f = open(path, "a", encoding="utf-8")

    for tpl in templates:
        base_params = dict(tpl.params)
        for inject_param in list(base_params.keys()):
            ip_found = False
            last_obs_for_ip: Optional[SqlmapObservation] = None

            for ep in range(max(1, int(episodes_per_input))):
                episode_id = str(uuid.uuid4())
                cfg = SqlmapConfig()
                last_obs: Optional[SqlmapObservation] = None
                last_duration = 0.0
                total_duration = 0.0
                tried_techniques: Set[str] = set()

                for step in range(max_steps):
                    params = dict(base_params)
                    if str(params.get(inject_param, "")) == "":
                        params[inject_param] = _default_injected_value(inject_param)

                    if cfg.technique:
                        tried_techniques.add(cfg.technique)

                    state = build_state(
                        last_obs,
                        cfg,
                        action_space=action_space,
                        steps_left=(max_steps - step),
                        last_duration_sec=last_duration,
                        episode_index=ep,
                        episodes_per_input=episodes_per_input,
                        tried_techniques=tried_techniques,
                    )

                    action_id = agent.select_action(action_ids, state)
                    act = action_space[action_id]
                    apply_option(cfg, act, max_tampers=15)

                    if cfg.technique:
                        tried_techniques.add(cfg.technique)

                    argv_opts = to_sqlmap_args(cfg)

                    if tpl.method.upper() == "GET":
                        url_with_qs = encode_get_url(tpl.url, params)
                        target = SqlmapTarget(url=url_with_qs)
                    else:
                        target = SqlmapTarget(url=tpl.url)
                        argv_opts = list(argv_opts) + ["--data", _encode_body(params)]

                    argv_opts = list(argv_opts) + ["-p", inject_param]

                    run_res = runner.run(target=target, argv_options=argv_opts)

                    last_duration = float(run_res.duration_sec)
                    total_duration += float(run_res.duration_sec)

                    obs = parse_sqlmap_output(run_res.stdout, run_res.stderr, timed_out=run_res.timed_out)
                    reward = compute_reward(obs, duration_sec=run_res.duration_sec, cfg=reward_cfg)

                    next_state = build_state(
                        obs,
                        cfg,
                        action_space=action_space,
                        steps_left=(max_steps - step - 1),
                        last_duration_sec=last_duration,
                        episode_index=ep,
                        episodes_per_input=episodes_per_input,
                        tried_techniques=tried_techniques,
                    )
                    done = bool(obs.injectable or (step == max_steps - 1))

                    total_reward = agent.observe(
                        state=state,
                        action=action_id,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )

                    if debug_f is not None:
                        dbg = {
                            "ts": time.time(),
                            "target": target_url,
                            "injection_url": target.url,
                            "method": tpl.method.upper(),
                            "injection_param": inject_param,
                            "episode_id": episode_id,
                            "episode_index": ep,
                            "step": step,
                            "action_id": int(action_id),
                            "action": {"kind": act.kind, "value": act.value},
                            "argv_opts": argv_opts,
                            "cmd": run_res.cmd_str,
                            "run": {
                                "duration_sec": run_res.duration_sec,
                                "timed_out": run_res.timed_out,
                                "returncode": run_res.returncode,
                            },
                            "obs": {
                                "injectable": obs.injectable,
                                "blocked": obs.blocked,
                                "timeout": obs.timeout,
                                "vulnerable_param": obs.vulnerable_param,
                                "payload": obs.exploited_payload,
                                "technique": obs.technique,
                                "dbms": obs.dbms,
                                "raw_snippet": obs.raw_snippet,
                            },
                            "reward": {"extrinsic": reward, "total": total_reward},
                            "config": asdict(cfg),
                            "defaults": {"method": tpl.method, "url": tpl.url, "params": base_params},
                            "effective_params": params,
                        }
                        # if blocked, include a small stdout head for diagnosis
                        if obs.blocked:
                            dbg["stdout_head"] = (run_res.stdout or "")[:1000]
                            dbg["stderr_head"] = (run_res.stderr or "")[:1000]

                        debug_f.write(json.dumps(dbg, ensure_ascii=False) + "\n")

                    last_obs = obs
                    last_obs_for_ip = obs

                    if obs.injectable:
                        finding = {
                            "url": target.url,
                            "method": tpl.method.upper(),
                            "vulnerable_param": inject_param,
                            "sqlmap_payload": obs.exploited_payload,
                            "technique": obs.technique,
                            "dbms": obs.dbms,
                            "blocked": obs.blocked,
                            "timeout": obs.timeout,
                            "episode_id": episode_id,
                            "episode_index": ep,
                            "steps_used": step + 1,
                            "total_duration_sec": total_duration,
                            "rl_sqlmap_cmd": run_res.cmd_str,
                            "raw_snippet": obs.raw_snippet,
                            "final_config": asdict(cfg),
                        }
                        results.append(finding)

                        try:
                            rec = {
                                "ts": time.time(),
                                "url": finding["url"],
                                "method": finding.get("method"),
                                "param": finding.get("vulnerable_param"),
                                "payload": finding.get("sqlmap_payload"),
                                "cmd_sqlmap": finding.get("rl_sqlmap_cmd"),
                                "episode_id": finding.get("episode_id"),
                                "episode_index": finding.get("episode_index"),
                                "steps_used": finding.get("steps_used"),
                                "total_duration_sec": finding.get("total_duration_sec"),
                                "technique": finding.get("technique"),
                                "dbms": finding.get("dbms"),
                            }
                            with open("success.jsonl", "a", encoding="utf-8") as sf:
                                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                        ip_found = True
                        break

                if ip_found:
                    break

            if not ip_found:
                results.append(
                    {
                        "url": encode_get_url(tpl.url, base_params) if tpl.method.upper() == "GET" else tpl.url,
                        "method": tpl.method.upper(),
                        "vulnerable_param": inject_param,
                        "sqlmap_payload": None,
                        "technique": last_obs_for_ip.technique if last_obs_for_ip else None,
                        "dbms": last_obs_for_ip.dbms if last_obs_for_ip else None,
                        "blocked": last_obs_for_ip.blocked if last_obs_for_ip else False,
                        "timeout": last_obs_for_ip.timeout if last_obs_for_ip else False,
                        "episode_id": None,
                        "episodes_used": int(episodes_per_input),
                        "steps_used": max_steps,
                        "total_duration_sec": None,
                        "rl_sqlmap_cmd": None,
                        "raw_snippet": last_obs_for_ip.raw_snippet if last_obs_for_ip else "",
                        "final_config": None,
                    }
                )

    if model_out:
        agent.save(model_out)

    if debug_f is not None:
        debug_f.flush()
        debug_f.close()

    if json_output:
        print(json.dumps({"target": target_url, "results": results}, indent=2, ensure_ascii=False))

    return results


def _encode_body(params: Dict[str, str]) -> str:
    from urllib.parse import urlencode

    return urlencode(params, doseq=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="scanner-sqli (sqlmap-driven RL)")
    ap.add_argument("-u", "--url", required=True, help="Target URL")
    ap.add_argument("--max-steps", type=int, default=15, help="Max steps (atomic options) per episode")
    ap.add_argument("--episodes-per-input", type=int, default=1, help="Training episodes to run per injection point")
    ap.add_argument("--timeout-sec", type=int, default=120, help="Timeout per sqlmap run")
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model-in", default=None)
    ap.add_argument("--model-out", default=None)
    ap.add_argument(
        "--debug-log",
        default=None,
        help='Write per-step debug events as JSONL into ./debug (use "1" to auto-name)',
    )

    args = ap.parse_args()

    scan(
        target_url=args.url,
        max_steps=args.max_steps,
        episodes_per_input=args.episodes_per_input,
        timeout_sec=args.timeout_sec,
        epsilon=args.epsilon,
        seed=args.seed,
        model_in=args.model_in,
        model_out=args.model_out,
        json_output=args.json,
        debug_log=args.debug_log,
    )


if __name__ == "__main__":
    main()
