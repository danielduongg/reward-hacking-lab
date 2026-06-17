"""
reward_model.py -- a model of a learned reward model (RM) and its error structure.

A real RM is trained on a finite set of preference comparisons sampled from the
reference policy. Two empirical facts about such RMs drive overoptimization:

  1. They are ACCURATE where they have data (responses the reference visits often).
  2. They are UNRELIABLE and OPTIMISTICALLY BIASED off-distribution (rare responses),
     where they must extrapolate.

We model this directly. With a preference budget `n_pref` per context, the expected
coverage of action a is count = piref(a) * n_pref. A confidence weight
conf = count / (count + k0) interpolates the RM between the true reward (in-distribution)
and an optimistic prior `opt` plus noise (out-of-distribution):

    rhat = conf * gold + (1 - conf) * opt + Normal(0, (sigma0 * (1 - conf))^2)

More preference data -> higher coverage -> conf -> 1 -> rhat -> gold (less error).
This is a transparent, controllable abstraction of preference-based RM error; it is not
a literal neural-net fit, by design, so the overoptimization dynamics can be studied in
isolation (cf. Gao et al., 2022).
"""
from __future__ import annotations

import numpy as np


def make_reward_model(env, n_pref: int = 1600, k0: float = 30.0, opt: float = 1.5,
                      sigma0: float = 0.8, seed: int = 0, normalize: bool = True):
    rng = np.random.RandomState(seed)
    gold, piref = env["gold"], env["piref"]
    count = piref * n_pref
    conf = count / (count + k0)
    rhat = conf * gold + (1.0 - conf) * opt + rng.randn(*gold.shape) * (sigma0 * (1.0 - conf))
    if normalize:
        rhat = (rhat - rhat.mean()) / rhat.std()
    return rhat


def make_ensemble(env, n_models: int = 8, n_pref: int = 1600, base_seed: int = 0,
                  **kw):
    """A reward-model ensemble: same data budget, different sampling/noise draws."""
    return [make_reward_model(env, n_pref=n_pref, seed=base_seed + 100 + i, **kw)
            for i in range(n_models)]


def ensemble_score(models, lam: float = 0.0):
    """Aggregate an ensemble. lam>0 gives a pessimistic lower-confidence-bound
    (mean - lam*std) that penalizes high-disagreement, off-distribution actions."""
    stack = np.stack(models, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    score = mean - lam * std
    return (score - score.mean()) / score.std()
