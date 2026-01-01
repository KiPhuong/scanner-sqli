"""scanner.py (context-aware)

Main orchestrator for the context-aware SQLi scanner.

New RL-driven fuzzing loop:
1) Pick a seed payload from the pool.
2) Tokenize payload.
3) Build valid actions (token_idx, mutation_id) based on token context.
4) Agent selects among valid actions (epsilon-greedy).
5) Apply mutation -> new payload.
6) Send request, evaluate, update agent + detector.
7) Continue mutation chain until detection / block / max_steps.

Debugging
- Use --debug to print per-step loop details.
- Use --debug-steps N to limit debug prints to first N steps per injection point.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import requests

from core.crawler import Crawler
from core.baseline import BaselineResponse
from core.requester import Requester
from core.evaluator import Evaluator
from core.detector import SQLiDetector

from payload.payload_pool import PayloadPool
from payload.mutations import apply_mutation, get_available_mutations
from payload.mutations import registry as mutation_registry
from payload.tokenizer import Token, default_tokenizer

from rl.agent import Agent, AgentConfig

from utils.logger import PayloadEvent, PayloadLogger, now_ts


InjectionPoint = Dict[str, object]
Baseline = Dict[str, object]
Action = Tuple[int, str]  # (token_idx, mutation_id)


def _to_injection_point_obj(ip) -> InjectionPoint:
    return {"url": ip.url, "method": ip.method, "param": ip.param, "base_value": ip.value}


def _baseline_to_obj(b: BaselineResponse) -> Baseline:
    return {
        "status": b.status_code,
        "time": b.elapsed,
        "length": b.length,
        "embedding": b.embedding,
        "params": b.params,
    }


def _detection_method(evidence_summary: Dict[str, int]) -> str:
    if evidence_summary.get("time_based", 0) > 0:
        return "time-based"
    if evidence_summary.get("sql_error", 0) > 0:
        return "error"
    if evidence_summary.get("semantic_dev", 0) > 0:
        return "semantic"
    return "unknown"


def _print_finding(f: dict) -> None:
    ip = f["injection_point"]
    print("[+] Vulnerable parameter found")
    print(f"    URL: {ip['url']}")
    print(f"    Parameter: {ip['param']}")
    print(f"    Method: {ip['method']}")
    print(f"    Payload: {f.get('payload', '')}")
    print(f"    Detection: {f.get('detection', 'unknown')}")
    print(f"    Requests used: {f.get('requests_used', 0)}")


def _debug_print(
    enabled: bool,
    step: int,
    max_steps_to_print: int,
    msg: str,
) -> None:
    if not enabled:
        return
    if max_steps_to_print >= 0 and step >= max_steps_to_print:
        return
    print(msg)


def _build_valid_actions(valid_actions_map: Dict[int, List[str]]) -> List[Action]:
    actions: List[Action] = []
    for idx, mids in valid_actions_map.items():
        for mid in mids:
            actions.append((idx, mid))
    return actions


def scan(
    target_url: str,
    payload_csv: str = "data/payloads.csv",
    max_steps: int = 50,
    timeout: int = 15,
    retries: int = 1,
    epsilon: float = 0.3,
    url_encode_payload: bool = False,
    seed: int = 1337,
    log_path: str = "logs/payload_events.jsonl",
    model_in: Optional[str] = None,
    model_out: Optional[str] = None,
    debug: bool = False,
    debug_steps: int = 10,
) -> List[dict]:
    rng = random.Random(seed)

    session = requests.Session()
    session.headers.setdefault("User-Agent", "scanner-sqli/2.0")

    crawler = Crawler(timeout=timeout, session=session)
    requester = Requester(timeout=timeout, retries=retries, session=session)
    evaluator = Evaluator()
    payload_logger = PayloadLogger(out_path=log_path)

    pool = PayloadPool(payload_csv, seed=seed)

    # RL init
    muts = mutation_registry.ids()

    # state_dim = one_hot(cur) + one_hot(prev) + one_hot(next) + [td, ld, ss, status]
    # TokenType count is derived from tokenizer
    token_type_dim = len(list(__import__("payload.tokenizer", fromlist=["TokenType"]).TokenType))
    state_dim = (token_type_dim * 3) + 4

    agent_cfg = AgentConfig(
        state_dim=state_dim,
        epsilon=epsilon,
        seed=seed,
        lr=0.03,
        gamma=0.9,
        batch_size=64,
        use_rnd=True,
        intrinsic_scale=0.1,
    )

    agent = Agent(mutation_ids=muts, config=agent_cfg)
    if model_in and os.path.exists(model_in):
        agent = Agent.load_from_file(model_in, mutation_ids=muts, config=agent_cfg)

    raw_ips = crawler.crawl(target_url)
    injection_points = [_to_injection_point_obj(ip) for ip in raw_ips]

    results: List[dict] = []

    for ip in injection_points:
        # baseline
        try:
            baseline_resp = BaselineResponse.from_injection_point(
                type("IP", (), {
                    "url": ip["url"],
                    "method": ip["method"],
                    "param": ip["param"],
                    "value": ip["base_value"],
                })(),
                timeout=timeout,
                session=session,
            )
        except Exception:
            results.append({"injection_point": ip, "skipped": True, "reason": "baseline_failed"})
            continue

        baseline_obj = _baseline_to_obj(baseline_resp)

        detector = SQLiDetector()
        if agent.rnd is not None:
            agent.rnd._stats = (0.0, 0.0, 0.0)  # reset per injection point

        # seed payload
        current_payload = pool.sample_payload()

        last_metrics = {
            "time_delta": 0.0,
            "length_delta": 0.0,
            "semantic_similarity": 1.0,
            "status_code": int(baseline_obj["status"]),
        }

        requests_used = 0

        _debug_print(debug, 0, debug_steps, f"\n[DBG] Injection point: {ip['method']} {ip['url']} param={ip['param']}")
        _debug_print(debug, 0, debug_steps, f"[DBG] Seed payload: {current_payload!r}")

        for step in range(max_steps):
            tokens: List[Token] = default_tokenizer.tokenize(current_payload)

            valid_actions_map = get_available_mutations(current_payload)
            valid_actions = _build_valid_actions(valid_actions_map)

            _debug_print(debug, step, debug_steps, f"[DBG] step={step} tokens={len(tokens)} valid_actions={len(valid_actions)}")

            if not valid_actions:
                _debug_print(debug, step, debug_steps, "[DBG] No valid actions; stop.")
                break

            def state_builder(token_idx: int) -> List[float]:
                return Agent.build_state(
                    tokens=tokens,
                    token_idx=token_idx,
                    time_delta=last_metrics["time_delta"],
                    length_delta=last_metrics["length_delta"],
                    semantic_similarity=last_metrics["semantic_similarity"],
                    status_code=last_metrics["status_code"],
                )

            token_idx, mutation_id = agent.select_action(valid_actions, state_builder)

            tok_text = tokens[token_idx].text if 0 <= token_idx < len(tokens) else "?"
            _debug_print(
                debug,
                step,
                debug_steps,
                f"[DBG] chosen: token_idx={token_idx} token={tok_text!r} mutation={mutation_id}",
            )

            new_payload = apply_mutation(current_payload, mutation_id, token_idx, rng=rng)
            if not new_payload:
                _debug_print(debug, step, debug_steps, "[DBG] mutation returned None; continue.")
                continue

            _debug_print(debug, step, debug_steps, f"[DBG] payload: {current_payload!r} -> {new_payload!r}")

            resp = requester.send(
                url=str(ip["url"]),
                method=str(ip["method"]),
                params=dict(baseline_obj["params"]),
                inject_param=str(ip["param"]),
                payload=new_payload,
                url_encode_payload=url_encode_payload,
            )
            requests_used += 1

            eval_res = evaluator.evaluate(baseline_resp, resp)
            detector.record_attempt(new_payload, eval_res)

            # next state/action list
            next_tokens = default_tokenizer.tokenize(new_payload)
            next_valid_actions = _build_valid_actions(get_available_mutations(new_payload))

            def next_state_builder(next_token_idx: int) -> List[float]:
                return Agent.build_state(
                    tokens=next_tokens,
                    token_idx=next_token_idx,
                    time_delta=eval_res.time_delta,
                    length_delta=float(eval_res.length_delta),
                    semantic_similarity=eval_res.semantic_similarity,
                    status_code=resp.status_code,
                )

            done = bool(detector.verdict().vulnerable or eval_res.blocked)
            total_reward = agent.observe(
                state=state_builder(token_idx),
                action=(token_idx, mutation_id),
                extrinsic_reward=eval_res.reward,
                next_valid_actions=next_valid_actions,
                next_state_builder=next_state_builder,
                done=done,
            )

            _debug_print(
                debug,
                step,
                debug_steps,
                f"[DBG] resp: status={resp.status_code} tΔ={eval_res.time_delta:.3f}s lenΔ={eval_res.length_delta} sim={eval_res.semantic_similarity:.3f} sqlerr={eval_res.sql_error} blocked={eval_res.blocked} reward={eval_res.reward:.3f} total={total_reward:.3f}",
            )

            payload_logger.log(
                PayloadEvent(
                    ts=now_ts(),
                    target_url=target_url,
                    injection_url=str(ip["url"]),
                    method=str(ip["method"]),
                    param=str(ip["param"]),
                    payload_id=-1,
                    mutation_id=mutation_id,
                    base_payload=current_payload,
                    mutated_payload=new_payload,
                    reward=float(eval_res.reward),
                    time_delta=float(eval_res.time_delta),
                    length_delta=float(eval_res.length_delta),
                    semantic_similarity=float(eval_res.semantic_similarity),
                    sql_error=bool(eval_res.sql_error),
                    blocked=bool(eval_res.blocked),
                    vulnerable=bool(detector.verdict().vulnerable),
                )
            )

            current_payload = new_payload
            last_metrics = {
                "time_delta": float(eval_res.time_delta),
                "length_delta": float(eval_res.length_delta),
                "semantic_similarity": float(eval_res.semantic_similarity),
                "status_code": int(resp.status_code),
            }

            if eval_res.blocked:
                _debug_print(debug, step, debug_steps, "[DBG] blocked -> stop.")
                break

            if detector.verdict().vulnerable:
                _debug_print(debug, step, debug_steps, "[DBG] vulnerable -> stop.")
                break

        det = detector.verdict()
        results.append(
            {
                "injection_point": ip,
                "vulnerable": bool(det.vulnerable),
                "payload": det.trigger_payloads[0] if det.trigger_payloads else None,
                "detection": _detection_method(det.evidence_summary),
                "requests_used": int(requests_used),
                "trigger_payloads": det.trigger_payloads,
                "evidence_summary": det.evidence_summary,
            }
        )

    payload_logger.flush()
    if model_out:
        agent.save(model_out)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="scanner-sqli v2 (context-aware)")
    ap.add_argument("-u", "--url", required=True, help="Target URL")
    ap.add_argument("--payload-csv", default="data/payloads.csv", help="CSV with seed payloads (query,label)")
    ap.add_argument("--max-steps", type=int, default=50, help="Max mutation steps per injection point")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--url-encode-payload", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model-in", default=None, help="Load context-aware agent model")
    ap.add_argument("--model-out", default=None, help="Save context-aware agent model")
    ap.add_argument("--debug", action="store_true", help="Print debug info for fuzzing loop")
    ap.add_argument("--debug-steps", type=int, default=10, help="Max debug steps printed per injection point (-1 = all)")

    args = ap.parse_args()

    results = scan(
        target_url=args.url,
        payload_csv=args.payload_csv,
        max_steps=args.max_steps,
        timeout=args.timeout,
        retries=args.retries,
        epsilon=args.epsilon,
        url_encode_payload=args.url_encode_payload,
        seed=args.seed,
        model_in=args.model_in,
        model_out=args.model_out,
        debug=args.debug,
        debug_steps=args.debug_steps,
    )

    if args.json:
        print(json.dumps({"target": args.url, "results": results}, indent=2, ensure_ascii=False))
        return

    vulns = [r for r in results if r.get("vulnerable")]
    if not vulns:
        print("[-] No SQLi findings detected.")
        return

    for f in vulns:
        _print_finding(f)


if __name__ == "__main__":
    main()
