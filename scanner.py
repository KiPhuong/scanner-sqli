"""scanner.py

scanner-sqli main orchestrator.

This file is the ONLY orchestrator.
- It wires together: crawler, baseline, requester, evaluator, RL agent, detector,
  payload pool, payload encoder, logging.
- It should not contain SQL logic, mutation logic, or ML model definitions.

CLI
    python scanner.py -u http://target/item.php?id=1

Output (pentest-friendly)
[+] Vulnerable parameter found
    URL: ...
    Parameter: ...
    Method: GET
    Payload: ...
    Mutation: ...
    Detection: time-based|semantic|error
    Requests used: N

Note: Only test targets you own or have explicit permission to test.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

import requests

from core.crawler import Crawler
from core.baseline import BaselineResponse
from core.requester import Requester
from core.evaluator import Evaluator
from core.detector import SQLiDetector

from payload.payload_pool import PayloadPool
from payload.encoder import PayloadEncoder
from payload.mutations import apply_mutation_by_id, mutation_ids

from rl.agent import Agent, AgentConfig

from utils.logger import PayloadEvent, PayloadLogger, now_ts


InjectionPoint = Dict[str, object]
Baseline = Dict[str, object]


def _to_injection_point_obj(ip) -> InjectionPoint:
    """Convert core.crawler.InjectionPoint dataclass to a plain dict."""
    return {
        "url": ip.url,
        "method": ip.method,
        "param": ip.param,
        "base_value": ip.value,
    }


def _baseline_to_obj(b: BaselineResponse) -> Baseline:
    return {
        "status": b.status_code,
        "time": b.elapsed,
        "length": b.length,
        "embedding": b.embedding,
        "params": b.params,  # keep for requester
    }


def _detection_method(evidence_summary: Dict[str, int]) -> str:
    # Prefer strongest/most interpretable in this order
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
    print(f"    Mutation: {f.get('mutation', '')}")
    print(f"    Detection: {f.get('detection', 'unknown')}")
    print(f"    Requests used: {f.get('requests_used', 0)}")


def scan(
    target_url: str,
    payload_csv: str = "data/sqli.csv",
    max_steps: int = 30,
    timeout: int = 15,
    retries: int = 1,
    epsilon: float = 0.25,
    url_encode_payload: bool = False,
    seed: int = 1337,
    log_path: str = "logs/payload_events.jsonl",
    model_in: Optional[str] = None,
    model_out: Optional[str] = None,
) -> List[dict]:
    """Run scan(target_url) and return per-injection-point results."""

    rng = random.Random(seed)

    # Shared HTTP session for stability (cookies/keep-alive)
    session = requests.Session()
    session.headers.setdefault("User-Agent", "scanner-sqli/1.0")

    # Shared components
    crawler = Crawler(timeout=timeout, session=session)
    requester = Requester(timeout=timeout, retries=retries, session=session)
    evaluator = Evaluator()
    payload_logger = PayloadLogger(out_path=log_path)

    pool = PayloadPool(payload_csv, seed=seed)
    encoder = PayloadEncoder(dim=64)

    muts = mutation_ids()  # mutation IDs are strings in this codebase
    mut_index = {mid: i for i, mid in enumerate(muts)}

    # Discrete action space: (payload_id, mutation_id)
    actions: List[Tuple[int, str]] = [(pid, mid) for pid in range(pool.size()) for mid in muts]

    # RL state: payload_embedding + one_hot(mutation) + deltas/similarity/status/error
    payload_emb_dim = encoder.dim
    state_dim = payload_emb_dim + len(muts) + 5

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

    agent = Agent(action_space=actions, config=agent_cfg)

    # Optional: load persisted model
    if model_in:
        agent = Agent.load_from_file(model_in, action_space=actions, config=agent_cfg, strict=True)

    # --- Step 1: crawling ---
    raw_ips = crawler.crawl(target_url)
    injection_points: List[InjectionPoint] = []
    for ip in raw_ips:
        ip_obj = _to_injection_point_obj(ip)
        injection_points.append(ip_obj)

    results: List[dict] = []

    for ip in injection_points:
        # --- Step 2: baseline ---
        try:
            baseline_resp = BaselineResponse.from_injection_point(
                injection_point=type("IP", (), {
                    "url": ip["url"],
                    "method": ip["method"],
                    "param": ip["param"],
                    "value": ip["base_value"],
                })(),
                timeout=timeout,
                session=session,
            )
        except Exception:
            # Skip if baseline fails
            results.append(
                {
                    "injection_point": ip,
                    "skipped": True,
                    "reason": "baseline_failed",
                }
            )
            continue

        baseline_obj = _baseline_to_obj(baseline_resp)

        # --- Step 3: RL context init (per injection point) ---
        detector = SQLiDetector()
        # Reset agent hidden state if any (none in current agent)
        if agent.rnd is not None:
            # Re-init RND stats for isolation
            agent.rnd._stats = (0.0, 0.0, 0.0)  # type: ignore[attr-defined]

        last_state_metrics = {
            "delta_time": 0.0,
            "delta_length": 0.0,
            "semantic_similarity": 1.0,
            "status_code": float(baseline_obj["status"]),
            "error_flag": 0.0,
        }

        requests_used = 0
        trigger_payload: Optional[str] = None
        trigger_mutation: Optional[str] = None

        # Track last chosen action for deterministic state construction
        last_action_mut_id: str = muts[0]
        last_action_payload_emb: List[float] = [0.0] * payload_emb_dim

        # --- Step 4: RL-driven payload loop ---
        for _step in range(max_steps):
            # Deterministic control flow: pick action from a stable state vector
            # representing the last observed outcome for this injection point.
            # We must pick a mutation for one-hot encoding; use the *last* mutation
            # (or the first in registry for step 0). This keeps the control flow
            # deterministic and avoids random placeholders.
            if _step == 0:
                cur_mut_for_state = muts[0]
                cur_payload_emb_for_state = [0.0] * payload_emb_dim
            else:
                cur_mut_for_state = last_action_mut_id  # type: ignore[name-defined]
                cur_payload_emb_for_state = last_action_payload_emb  # type: ignore[name-defined]

            state_vec = Agent.build_state(
                payload_embedding=cur_payload_emb_for_state,
                mutation_id=cur_mut_for_state,
                mutation_id_to_index=mut_index,
                time_delta=last_state_metrics["delta_time"],
                length_delta=last_state_metrics["delta_length"],
                semantic_similarity=last_state_metrics["semantic_similarity"],
            ) + [
                float(last_state_metrics["status_code"]),
                float(last_state_metrics["error_flag"]),
            ]

            payload_id, mut_id = agent.select_action(state_vec)

            base_payload = pool.get_payload_by_id(payload_id)
            mutated_payload = apply_mutation_by_id(base_payload, mut_id, rng=rng)

            # send request
            resp = requester.send(
                url=str(ip["url"]),
                method=str(ip["method"]),
                params=dict(baseline_obj["params"]),
                inject_param=str(ip["param"]),
                payload=mutated_payload,
                url_encode_payload=url_encode_payload,
            )
            requests_used += 1

            eval_res = evaluator.evaluate(baseline_resp, resp)
            detector.record_attempt(mutated_payload, eval_res)

            # Build state/next_state for RL
            payload_emb = encoder.encode(base_payload)
            state = Agent.build_state(
                payload_embedding=payload_emb,
                mutation_id=mut_id,
                mutation_id_to_index=mut_index,
                time_delta=last_state_metrics["delta_time"],
                length_delta=last_state_metrics["delta_length"],
                semantic_similarity=last_state_metrics["semantic_similarity"],
            ) + [
                float(last_state_metrics["status_code"]),
                float(last_state_metrics["error_flag"]),
            ]

            next_state = Agent.build_state(
                payload_embedding=payload_emb,
                mutation_id=mut_id,
                mutation_id_to_index=mut_index,
                time_delta=eval_res.time_delta,
                length_delta=float(eval_res.length_delta),
                semantic_similarity=eval_res.semantic_similarity,
            ) + [
                float(resp.status_code),
                1.0 if eval_res.sql_error else 0.0,
            ]
            
            # Update last action for next step's state construction
            last_action_mut_id = mut_id
            last_action_payload_emb = payload_emb

            # update agent (extrinsic + intrinsic handled inside agent.observe)
            agent.observe(
                state=state,
                action=(payload_id, mut_id),
                extrinsic_reward=eval_res.reward,
                next_state=next_state,
                done=False,
            )

            # logging hook
            payload_logger.log(
                PayloadEvent(
                    ts=now_ts(),
                    target_url=str(target_url),
                    injection_url=str(ip["url"]),
                    method=str(ip["method"]),
                    param=str(ip["param"]),
                    payload_id=int(payload_id),
                    mutation_id=str(mut_id),
                    base_payload=base_payload,
                    mutated_payload=mutated_payload,
                    reward=float(eval_res.reward),
                    time_delta=float(eval_res.time_delta),
                    length_delta=float(eval_res.length_delta),
                    semantic_similarity=float(eval_res.semantic_similarity),
                    sql_error=bool(eval_res.sql_error),
                    blocked=bool(eval_res.blocked),
                    vulnerable=False,
                )
            )

            # termination conditions
            if eval_res.blocked:
                # Stop early if target blocks
                break

            last_state_metrics = {
                "delta_time": float(eval_res.time_delta),
                "delta_length": float(eval_res.length_delta),
                "semantic_similarity": float(eval_res.semantic_similarity),
                "status_code": float(resp.status_code),
                "error_flag": 1.0 if eval_res.sql_error else 0.0,
            }

            det = detector.verdict()
            if det.vulnerable:
                trigger_payload = det.trigger_payloads[0] if det.trigger_payloads else mutated_payload
                trigger_mutation = mut_id
                break

        det = detector.verdict()
        evidence_summary = det.evidence_summary

        results.append(
            {
                "injection_point": ip,
                "baseline": {
                    "status": baseline_obj["status"],
                    "time": baseline_obj["time"],
                    "length": baseline_obj["length"],
                },
                "vulnerable": bool(det.vulnerable),
                "payload": trigger_payload,
                "mutation": trigger_mutation,
                "detection": _detection_method(evidence_summary),
                "requests_used": int(requests_used),
                "evidence_summary": evidence_summary,
                "trigger_payloads": det.trigger_payloads,
            }
        )

    payload_logger.flush()

    # Optional: save model
    if model_out:
        agent.save(model_out)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="scanner-sqli (black-box SQLi scanner)")
    ap.add_argument("-u", "--url", required=True, help="Target URL")
    ap.add_argument("--payload-csv", default="data/payloads.csv", help="CSV with query,label")
    ap.add_argument("--max-steps", type=int, default=30, help="Max attempts per parameter")
    ap.add_argument("--timeout", type=int, default=15, help="Request timeout (s)")
    ap.add_argument("--retries", type=int, default=1, help="Retries per request")
    ap.add_argument("--epsilon", type=float, default=0.25, help="Epsilon-greedy exploration")
    ap.add_argument("--url-encode-payload", action="store_true", help="URL-encode payload before sending")
    ap.add_argument("--seed", type=int, default=1337, help="PRNG seed")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--model-in", default=None, help="Load agent model from JSON")
    ap.add_argument("--model-out", default=None, help="Save agent model to JSON")

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
    )

    if args.json:
        print(json.dumps({"target": args.url, "results": results}, indent=2, ensure_ascii=False))
        return

    # Pentest-friendly summary
    vulns = [r for r in results if r.get("vulnerable")]
    if not vulns:
        print("[-] No SQLi findings detected.")
        return

    for f in vulns:
        _print_finding(f)


if __name__ == "__main__":
    main()
