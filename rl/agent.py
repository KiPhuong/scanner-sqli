"""rl.agent

A lightweight reinforcement learning agent for selecting (payload_id, mutation_id)
actions.

Requirements satisfied:
- Observes state vectors composed of:
  - payload embedding
  - mutation ID (encoded)
  - response deltas
  - semantic similarity
- Selects actions as: (payload_id, mutation_id)
- Learns which payload-mutation combinations are effective
- Supports:
  - epsilon-greedy exploration
  - experience replay
  - integration with intrinsic reward (RND)

Implementation notes
- To keep dependencies minimal, we implement a linear Q-function:
    Q(s, a) = w_a · s
  i.e., one weight vector per discrete action.

- Learning is Q-learning with replay:
    target = r + gamma * max_a' Q(s', a')
    w_a <- w_a + lr * (target - Q(s,a)) * s

Model persistence
- save()/load_from_file() store only the policy parameters (weights) and the
  action space mapping required to interpret them.
- Replay buffer is not persisted (by design, to keep it simple and portable).
- RND state is not persisted (recommended for per-injection-point resets).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from rl.policy import EpsilonGreedyPolicy
from rl.replay_buffer import ReplayBuffer, Transition
from rl.rnd import RND


Action = Tuple[int, str]  # (payload_id, mutation_id)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _zeros(n: int) -> List[float]:
    return [0.0] * n


@dataclass
class AgentConfig:
    state_dim: int
    gamma: float = 0.95
    lr: float = 0.05

    epsilon: float = 0.2

    replay_capacity: int = 50_000
    batch_size: int = 64

    use_rnd: bool = True
    intrinsic_scale: float = 0.1

    seed: Optional[int] = None


class Agent:
    def __init__(self, action_space: List[Action], config: AgentConfig):
        if not action_space:
            raise ValueError("action_space is empty")

        self.action_space = list(action_space)
        self.action_index: Dict[Action, int] = {a: i for i, a in enumerate(self.action_space)}

        self.cfg = config
        self._rng = random.Random(config.seed)

        self.policy = EpsilonGreedyPolicy(epsilon=config.epsilon, seed=config.seed)
        self.replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)

        # One weight vector per action
        self.weights: List[List[float]] = [_zeros(self.cfg.state_dim) for _ in range(len(self.action_space))]

        self.rnd: Optional[RND] = None
        if self.cfg.use_rnd:
            self.rnd = RND(input_dim=self.cfg.state_dim, seed=config.seed)

    # -------------------------
    # Persistence
    # -------------------------

    def save(self, path: str) -> None:
        """Save model weights + action space mapping to a JSON file."""

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "version": 1,
            "state_dim": self.cfg.state_dim,
            "action_space": [[pid, mid] for (pid, mid) in self.action_space],
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_from_file(
        cls,
        path: str,
        action_space: List[Action],
        config: AgentConfig,
        strict: bool = True,
    ) -> "Agent":
        """Load weights from disk into a new Agent.

        strict=True will require that:
        - saved state_dim matches config.state_dim
        - saved action_space matches the provided action_space exactly

        If strict=False, will try best-effort load when possible.
        """

        agent = cls(action_space=action_space, config=config)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if int(data.get("version", 0)) != 1:
            raise ValueError(f"Unsupported model version: {data.get('version')}")

        saved_state_dim = int(data.get("state_dim"))
        if saved_state_dim != config.state_dim:
            msg = f"state_dim mismatch: saved={saved_state_dim} current={config.state_dim}"
            if strict:
                raise ValueError(msg)

        saved_actions = [(int(pid), str(mid)) for pid, mid in data.get("action_space", [])]
        if strict and saved_actions != agent.action_space:
            raise ValueError("action_space mismatch between saved model and current run")

        saved_weights = data.get("weights")
        if not isinstance(saved_weights, list) or not saved_weights:
            raise ValueError("Invalid weights in saved model")

        # Best-effort assignment
        if strict:
            agent.weights = saved_weights
        else:
            # Align by action when possible
            saved_map = {a: i for i, a in enumerate(saved_actions)}
            for i, a in enumerate(agent.action_space):
                j = saved_map.get(a)
                if j is None:
                    continue
                agent.weights[i] = saved_weights[j]

        return agent

    # -------------------------
    # State construction
    # -------------------------

    @staticmethod
    def build_state(
        payload_embedding: List[float],
        mutation_id: str,
        mutation_id_to_index: Dict[str, int],
        # response/evaluator signals
        time_delta: float,
        length_delta: float,
        semantic_similarity: float,
    ) -> List[float]:
        """Create a flat state vector.

        State layout:
        [payload_embedding..., one_hot(mutation_id)..., time_delta, length_delta, semantic_similarity]

        You must provide a stable mutation_id_to_index mapping.
        """

        if mutation_id not in mutation_id_to_index:
            raise KeyError(f"Unknown mutation_id: {mutation_id}")

        mut_dim = len(mutation_id_to_index)
        one_hot = [0.0] * mut_dim
        one_hot[mutation_id_to_index[mutation_id]] = 1.0

        # Simple scaling/clamping to reduce instability
        td = max(-30.0, min(30.0, float(time_delta)))
        ld = max(-1_000_000.0, min(1_000_000.0, float(length_delta)))
        ss = max(0.0, min(1.0, float(semantic_similarity)))

        return list(payload_embedding) + one_hot + [td, ld, ss]

    # -------------------------
    # Action selection
    # -------------------------

    def q_values(self, state: List[float]) -> List[float]:
        if len(state) != self.cfg.state_dim:
            raise ValueError(f"state_dim mismatch: expected {self.cfg.state_dim}, got {len(state)}")
        return [_dot(w, state) for w in self.weights]

    def select_action(self, state: List[float]) -> Action:
        qs = self.q_values(state)
        idx = self.policy.select(qs)
        return self.action_space[idx]

    # -------------------------
    # Learning / Replay
    # -------------------------

    def observe(
        self,
        state: List[float],
        action: Action,
        extrinsic_reward: float,
        next_state: List[float],
        done: bool,
    ) -> float:
        """Record a transition and optionally update. Returns total reward used.

        Integrates intrinsic reward (RND) if enabled.
        """

        intrinsic = 0.0
        if self.rnd is not None:
            intrinsic = self.rnd.update(next_state)

        total_reward = float(extrinsic_reward + self.cfg.intrinsic_scale * intrinsic)

        self.replay.push(
            Transition(
                state=list(state),
                action=action,
                reward=total_reward,
                next_state=list(next_state),
                done=bool(done),
            )
        )

        # Learn from replay
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

            # Q-learning target
            q_sa = _dot(self.weights[a_idx], tr.state)
            if tr.done:
                target = tr.reward
            else:
                next_qs = self.q_values(tr.next_state)
                target = tr.reward + self.cfg.gamma * max(next_qs)

            td_err = target - q_sa

            # SGD update for linear function approximator
            w = self.weights[a_idx]
            lr = self.cfg.lr
            for i in range(self.cfg.state_dim):
                w[i] += lr * td_err * tr.state[i]

    # -------------------------
    # Utilities
    # -------------------------

    def action_to_index(self, action: Action) -> int:
        return self.action_index[action]

    def index_to_action(self, idx: int) -> Action:
        return self.action_space[idx]
