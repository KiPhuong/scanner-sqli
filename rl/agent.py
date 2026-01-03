"""rl.agent (sqlmap-driven RL)

Discrete-action RL agent for sequential sqlmap option selection.

Action Space
- Fixed discrete action ids: 0..N-1
- Each action corresponds to an atomic OptionAction (e.g. set technique, add tamper)

State Representation
- A fixed-length float vector produced by the orchestrator (scanner.py)

Q-Function
- Linear Q per action: Q(s, a) = w_a · s

Stability / exploration
- Epsilon-greedy policy
- Optional RND intrinsic reward to encourage exploration
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from rl.policy import EpsilonGreedyPolicy
from rl.replay_buffer import ReplayBuffer, Transition
from rl.rnd import RND


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rand_small(rng: random.Random, n: int, scale: float = 1e-3) -> List[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


@dataclass
class AgentConfig:
    state_dim: int
    gamma: float = 0.95
    lr: float = 0.05
    epsilon: float = 0.3
    replay_capacity: int = 50_000
    batch_size: int = 64
    use_rnd: bool = True
    intrinsic_scale: float = 0.1
    seed: Optional[int] = None


class Agent:
    def __init__(self, action_ids: List[int], config: AgentConfig):
        if not action_ids:
            raise ValueError("action_ids is empty")

        self.action_ids = sorted(action_ids)
        self.action_index: Dict[int, int] = {aid: i for i, aid in enumerate(self.action_ids)}

        self.cfg = config
        self._rng = random.Random(config.seed)

        self.policy = EpsilonGreedyPolicy(epsilon=config.epsilon, seed=config.seed)
        self.replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)

        # One weight vector per ACTION id. Small random init reduces greedy tie-bias.
        self.weights: List[List[float]] = [
            _rand_small(self._rng, self.cfg.state_dim) for _ in range(len(self.action_ids))
        ]

        self.rnd: Optional[RND] = None
        if self.cfg.use_rnd:
            self.rnd = RND(input_dim=self.cfg.state_dim, seed=config.seed)

    # -------------------------
    # Persistence
    # -------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 3,
            "state_dim": self.cfg.state_dim,
            "action_ids": self.action_ids,
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_from_file(cls, path: str, action_ids: List[int], config: AgentConfig) -> "Agent":
        agent = cls(action_ids=action_ids, config=config)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if int(data.get("version", 0)) != 3:
            raise ValueError(f"Incompatible model version: {data.get('version')}")

        if int(data["state_dim"]) != config.state_dim:
            raise ValueError("state_dim mismatch")

        saved_ids = data["action_ids"]
        if sorted(saved_ids) != sorted(action_ids):
            raise ValueError("action_ids mismatch")

        agent.weights = data["weights"]
        return agent

    # -------------------------
    # Action selection
    # -------------------------

    def select_action(self, valid_action_ids: List[int], state: List[float]) -> int:
        if not valid_action_ids:
            raise ValueError("valid_action_ids is empty")

        q_values: List[float] = []
        for aid in valid_action_ids:
            a_idx = self.action_index[aid]
            q_values.append(_dot(self.weights[a_idx], state))

        chosen_idx = self.policy.select(q_values)
        return valid_action_ids[chosen_idx]

    # -------------------------
    # Learning / Replay
    # -------------------------

    def observe(
        self,
        *,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> float:
        intrinsic = 0.0
        if self.rnd is not None:
            intrinsic = self.rnd.update(state)

        total_reward = float(reward + self.cfg.intrinsic_scale * intrinsic)

        # For this linear-Q agent we store the whole next_state and compute maxQ during learn.
        self.replay.push(
            Transition(
                state=list(state),
                action=int(action),
                reward=total_reward,
                next_state=list(next_state),
                done=bool(done),
            )
        )

        self.learn()
        return total_reward

    def learn(self) -> None:
        if len(self.replay) < max(1, self.cfg.batch_size):
            return

        batch = self.replay.sample(self.cfg.batch_size)
        for tr in batch:
            a_idx = self.action_index.get(tr.action)
            if a_idx is None:
                continue

            q_sa = _dot(self.weights[a_idx], tr.state)

            if tr.done:
                max_next_q = 0.0
            else:
                # Over fixed action space
                next_qs = [_dot(self.weights[i], tr.next_state) for i in range(len(self.action_ids))]
                max_next_q = max(next_qs) if next_qs else 0.0

            target = tr.reward + self.cfg.gamma * max_next_q
            td_err = target - q_sa

            w = self.weights[a_idx]
            lr = self.cfg.lr
            for i in range(self.cfg.state_dim):
                w[i] += lr * td_err * tr.state[i]
