"""
policy.py -- KL-regularized policy optimization (the RLHF objective) + metrics.

We optimize, per context, the standard RLHF objective

    maximize_pi   E_{a~pi}[ rhat(s,a) ]  -  beta * KL( pi(.|s) || piref(.|s) )

Two solvers:
  * `kl_optimal_policy`  -- the exact closed-form optimum  pi ∝ piref * exp(rhat/beta).
  * `policy_gradient`    -- an iterative softmax policy-gradient optimizer that converges
                            to it (used to show training dynamics and to validate the
                            closed form).

`evaluate` returns the proxy reward (what we optimize), the GOLD reward (what we care
about), and KL(pi||piref) (the optimization "distance").
"""
from __future__ import annotations

import numpy as np


def _softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kl_optimal_policy(rhat, piref, beta: float):
    """Closed-form optimum of E[rhat] - beta*KL(pi||piref):  pi ∝ piref * exp(rhat/beta)."""
    logits = np.log(piref + 1e-12) + rhat / beta
    return _softmax_rows(logits)


def evaluate(pi, gold, rhat, piref) -> dict:
    proxy = float((pi * rhat).sum(axis=1).mean())
    gold_v = float((pi * gold).sum(axis=1).mean())
    kl = float((pi * (np.log(pi + 1e-12) - np.log(piref + 1e-12))).sum(axis=1).mean())
    return {"proxy": proxy, "gold": gold_v, "kl": kl}


def policy_gradient(rhat, piref, gold, beta: float = 0.0, lr: float = 0.05,
                    steps: int = 700, log_every: int = 1):
    """Natural policy gradient / mirror descent on the KL-regularized objective, starting
    from the reference policy:

        log pi_{t+1}  <-  log pi_t + lr * ( rhat - beta * (log pi_t - log piref) )

    This converges to the closed-form optimum pi_beta. With beta=0 it traces the exact
    optimization frontier (pi_t ∝ piref * exp(t*lr*rhat)), so "training step" and the
    KL-budget sweep agree. Returns the final policy and a per-step history."""
    logpiref = np.log(piref + 1e-12)
    logp = logpiref.copy()
    history = []
    for t in range(steps):
        pi = _softmax_rows(logp)
        logratio = np.log(pi + 1e-12) - logpiref
        logp = logp + lr * (rhat - beta * logratio)
        logp -= logp.max(axis=1, keepdims=True)     # stabilize (softmax is shift-invariant)
        if t % log_every == 0 or t == steps - 1:
            m = evaluate(pi, gold, rhat, piref)
            m["step"] = t
            history.append(m)
    return _softmax_rows(logp), history
