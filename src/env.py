"""
env.py -- an RLHF-style reward-optimization environment.

Abstraction: a set of S contexts ("prompts"), each with A discrete actions
("responses"). Every (context, action) has a hidden GOLD reward -- the thing we
actually care about (true human preference). A reference policy `piref` plays the role
of the base/SFT model from which preference data is collected and which the optimized
policy is regularized toward (exactly as in RLHF's KL-to-reference objective).

This is a controllable stand-in for the full RLHF pipeline that isolates the
reward-model-overoptimization phenomenon and runs in milliseconds on CPU.
"""
from __future__ import annotations

import numpy as np

SEED = 20260617


def make_env(n_contexts: int = 600, n_actions: int = 48,
             base_scale: float = 1.2, seed: int = SEED):
    rng = np.random.RandomState(seed)
    gold = rng.randn(n_contexts, n_actions)
    gold = (gold - gold.mean()) / gold.std()        # true reward ~ N(0,1)
    # reference policy: a non-uniform base model (some responses common, many rare)
    base = rng.randn(n_contexts, n_actions) * base_scale
    z = base - base.max(axis=1, keepdims=True)
    piref = np.exp(z)
    piref /= piref.sum(axis=1, keepdims=True)
    return {"gold": gold, "piref": piref,
            "n_contexts": n_contexts, "n_actions": n_actions}
