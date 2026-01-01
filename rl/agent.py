"""rl.agent (context-aware)

This agent operates on a tokenized payload representation.

Action Space (dynamic)
- The agent selects from a list of valid (token_idx, mutation_id) pairs provided
  by the orchestrator for the current payload state.

State Representation
- The state represents the local context of a specific token within the payload,
  plus global response metrics.

Q-Function
- We use a linear Q-function per mutation type: Q(s, m) = w_m · s
- This means the agent learns a weight vector for each mutation ID.
- To select an action, it computes Q-values for all valid (token, mutation)
  pairs and picks the best one via epsilon-greedy.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from payload.tokenizer import Token, TokenType
from rl.policy import EpsilonGreedyPolicy
from rl.replay_buffer import ReplayBuffer, Transition
from rl.rnd import RND


# Action is now context-aware: (token_index, mutation_id)
Action = Tuple[int, str]


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
    def __init__(self, mutation_ids: List[str], config: AgentConfig):
        if not mutation_ids:
            raise ValueError("mutation_ids is empty")

        self.mutation_ids = sorted(mutation_ids)
        self.mutation_index: Dict[str, int] = {mid: i for i, mid in enumerate(self.mutation_ids)}

        self.cfg = config
        self._rng = random.Random(config.seed)

        self.policy = EpsilonGreedyPolicy(epsilon=config.epsilon, seed=config.seed)
        self.replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)

        # One weight vector per MUTATION type
        self.weights: List[List[float]] = [
            _zeros(self.cfg.state_dim) for _ in range(len(self.mutation_ids))
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
            "version": 2,  # Context-aware agent
            "state_dim": self.cfg.state_dim,
            "mutation_ids": self.mutation_ids,
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load_from_file(cls, path: str, mutation_ids: List[str], config: AgentConfig) -> "Agent":
        agent = cls(mutation_ids=mutation_ids, config=config)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if int(data.get("version", 0)) != 2:
            raise ValueError(f"Incompatible model version: {data.get('version')}")

        if int(data["state_dim"]) != config.state_dim:
            raise ValueError("state_dim mismatch")

        saved_muts = data["mutation_ids"]
        if sorted(saved_muts) != sorted(mutation_ids):
            raise ValueError("mutation_ids mismatch")

        agent.weights = data["weights"]
        return agent

    # -------------------------
    # State construction
    # -------------------------

    @staticmethod
    def build_state(
        tokens: List[Token],
        token_idx: int,
        # response/evaluator signals
        time_delta: float,
        length_delta: float,
        semantic_similarity: float,
        status_code: int,
    ) -> List[float]:
        """Create a flat state vector for a specific token's context."""
        tok = tokens[token_idx]
        prev_tok = tokens[token_idx - 1] if token_idx > 0 else None
        next_tok = tokens[token_idx + 1] if token_idx + 1 < len(tokens) else None

        # Token type embeddings (one-hot)
        tok_type_emb = [0.0] * len(TokenType)
        tok_type_emb[list(TokenType).index(tok.type)] = 1.0

        prev_type_emb = [0.0] * len(TokenType)
        if prev_tok:
            prev_type_emb[list(TokenType).index(prev_tok.type)] = 1.0

        next_type_emb = [0.0] * len(TokenType)
        if next_tok:
            next_type_emb[list(TokenType).index(next_tok.type)] = 1.0

        # Simple scaling/clamping
        td = max(-30.0, min(30.0, float(time_delta)))
        ld = max(-1e6, min(1e6, float(length_delta)))
        ss = max(0.0, min(1.0, float(semantic_similarity)))
        sc = float(status_code) / 1000.0  # Normalize status

        return tok_type_emb + prev_type_emb + next_type_emb + [td, ld, ss, sc]

    # -------------------------
    # Action selection
    # -------------------------

    def select_action(self, valid_actions: List[Action], state_builder) -> Action:
        """Select the best (token_idx, mutation_id) from a valid list."""

        q_values = []
        for token_idx, mutation_id in valid_actions:
            state = state_builder(token_idx)
            mut_idx = self.mutation_index[mutation_id]
            q = _dot(self.weights[mut_idx], state)
            q_values.append(q)

        # Epsilon-greedy selection on the indices of valid_actions
        chosen_idx = self.policy.select(q_values)
        return valid_actions[chosen_idx]

    # -------------------------
    # Learning / Replay
    # -------------------------

    def observe(
        self,
        state: List[float],
        action: Action,
        extrinsic_reward: float,
        next_valid_actions: List[Action],
        next_state_builder,
        done: bool,
    ) -> float:
        intrinsic = 0.0
        if self.rnd is not None:
            # RND novelty is based on the state that was acted upon
            intrinsic = self.rnd.update(state)

        total_reward = float(extrinsic_reward + self.cfg.intrinsic_scale * intrinsic)

        # For replay buffer, we need a stable next_state vector.
        # We compute max_a' Q(s', a') now and store it as part of the transition.
        if done or not next_valid_actions:
            max_next_q = 0.0
        else:
            next_qs = []
            for token_idx, mutation_id in next_valid_actions:
                next_s = next_state_builder(token_idx)
                mut_idx = self.mutation_index[mutation_id]
                next_qs.append(_dot(self.weights[mut_idx], next_s))
            max_next_q = max(next_qs)

        # We store the max_next_q directly, simplifying the learning step.
        # The `next_state` in the buffer is just the state that was acted upon.
        self.replay.push(
            Transition(state=state, action=action, reward=total_reward, next_state=[max_next_q], done=done)
        )

        self.learn()
        return total_reward

    def learn(self) -> None:
        if len(self.replay) < self.cfg.batch_size:
            return

        batch = self.replay.sample(self.cfg.batch_size)
        for tr in batch:
            _token_idx, mutation_id = tr.action
            mut_idx = self.mutation_index.get(mutation_id)
            if mut_idx is None:
                continue

            q_sa = _dot(self.weights[mut_idx], tr.state)
            max_next_q = tr.next_state[0]

            target = tr.reward + self.cfg.gamma * max_next_q
            td_err = target - q_sa

            # SGD update
            w = self.weights[mut_idx]
            for i in range(self.cfg.state_dim):
                w[i] += self.cfg.lr * td_err * tr.state[i]
