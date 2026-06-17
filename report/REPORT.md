# Optimizing the Proxy Until It Breaks: Reproducing RLHF Reward-Model Overoptimization (and Fixing It)

*A reproducible mini-study. Author: Daniel Duong. Code: this repository.*

## Abstract

Reinforcement learning from human feedback (RLHF) trains a policy against a **learned
reward model (RM)** — a *proxy* for what humans actually want. Optimize that proxy too
hard and you get **reward hacking**: the proxy reward keeps climbing while the true
objective silently degrades (Goodhart's law). This project builds a small, fully
reproducible RLHF-style environment and the actual RLHF optimization objective
(KL-regularized policy optimization) to reproduce the phenomenon end to end, then tests
two mitigations.

The results match the published picture. As optimization pressure (KL to the reference
policy) increases, **proxy reward rises monotonically (+1.71 → +2.21) while the gold
reward rises then turns over, peaking at +1.07 (KL ≈ 2.8) and falling to +0.78** — a
**0.30 drop you cannot see from the proxy alone.** The same overoptimization appears
*during* RL training: gold peaks early (step 60) then degrades as training continues,
which is exactly why an early-stop / KL budget is needed. My natural-policy-gradient
optimizer reproduces the closed-form KL-optimal policy to three decimals, confirming the
dynamics are real and not an optimizer artifact. Two mitigations work and quantify the
trade-off: **5× more preference data lifts the achievable gold from +0.51 to +1.73**,
and a **pessimistic reward-model ensemble raises peak gold from +1.05 (single RM) to
+1.80** by refusing to trust off-distribution, high-disagreement responses.

---

## 1. Motivation

RLHF is how models like Claude are aligned: collect human preferences, fit a reward
model, then RL-optimize a policy to maximize it [4, 6]. The catch is structural — the RM
is an imperfect proxy, and **optimizing a proxy past a point degrades the true goal**.
Gao et al. measured "scaling laws" for exactly this RM overoptimization [1]; Skalse et
al. formalized reward hacking [5]; and mitigating it (KL penalties, RM ensembles) is an
active safety problem [2]. This repo is a compact, transparent reconstruction of the
attack surface and its defenses that runs in seconds on CPU.

## 2. Setup and method

**Environment (`src/env.py`).** S = 600 contexts ("prompts"), each with A = 48 discrete
actions ("responses"). Each (context, action) has a hidden **gold reward** ~ N(0,1)
(the true objective). A non-uniform **reference policy** `piref` plays the role of the
base/SFT model: it is what preference data is drawn from and what we regularize toward.

**Reward model (`src/reward_model.py`).** A learned RM is accurate where it has data and
unreliable — and optimistically biased — off-distribution. With a preference budget
`n_pref`, coverage of action a is `count = piref(a)·n_pref`, and a confidence weight
`conf = count/(count+k0)` interpolates the RM between the truth and an optimistic prior:

> `rhat = conf·gold + (1−conf)·opt + Normal(0, (σ0·(1−conf))²)`

More data → higher coverage → `conf → 1` → `rhat → gold`. This is a controllable
abstraction of preference-based RM error (in the spirit of Gao et al.'s noise model),
chosen so the overoptimization dynamics can be studied in isolation.

**Objective and optimizer (`src/policy.py`).** We optimize the standard RLHF objective
per context,

> maximize  E_{a∼π}[ rhat(s,a) ] − β·KL(π(·|s) ‖ piref(·|s)),

whose exact optimum is the closed form `π_β ∝ piref·exp(rhat/β)`. We also implement a
**natural policy gradient / mirror-descent** optimizer that converges to it, used to show
training dynamics and to validate correctness. We report three quantities: **proxy**
reward (optimized), **gold** reward (true objective), and **KL** to the reference
(optimization "distance").

## 3. Results

### 3.1 The overoptimization curve: the proxy lies

Sweeping the KL budget (β from 5 down to 0.04) traces the canonical picture: proxy
reward rises the whole way, but gold rises then **turns over**.

| Operating point | KL | Proxy | Gold |
|---|---:|---:|---:|
| Optimal (gold peak, β ≈ 0.33) | 2.76 | +1.71 | **+1.07** |
| Over-optimized (β ≈ 0.04) | 5.39 | **+2.21** | +0.78 |

Pushing from the peak to maximum optimization **raises the proxy by +0.50 while lowering
the true reward by 0.30.** If you only watched the proxy — as you must in practice, since
gold is unobserved — you would conclude training was going *great*.

![Overoptimization curve](../results/figures/overopt_curve.png)

### 3.2 It happens during training — and the optimizer is correct

Running the RL optimizer on the proxy with no KL leash, the true reward **peaks at step
60 (gold +1.07, KL 2.7) and then degrades to +0.76 (KL 5.5)** while the proxy keeps
climbing (+1.69 → +2.21). This is the case for an early-stop / KL budget. As a correctness
check, at a fixed β = 0.3 the natural-policy-gradient optimizer matches the closed-form
KL-optimal policy to three decimals (proxy/gold/KL = 1.786 / 1.066 / 2.995 for both),
so the dynamics are a property of the objective, not a training bug.

![Training dynamics](../results/figures/training_dynamics.png)

### 3.3 Mitigation 1 — more preference data

The RM's error is what gets exploited, so better RMs overoptimize less. Increasing the
preference budget lifts the **peak achievable gold** monotonically and pushes the
overoptimization point further out:

| Preference budget `n_pref` | 400 | 800 | 1600 | 3200 | 6400 |
|---|---:|---:|---:|---:|---:|
| Peak gold | +0.51 | +0.76 | +1.07 | +1.41 | **+1.73** |

![Data scaling](../results/figures/data_scaling.png)

### 3.4 Mitigation 2 — pessimistic reward-model ensembles

Following Coste et al. [2], an ensemble of RMs and a **pessimistic** aggregate
(`mean − 1.5·std`, a lower-confidence bound that distrusts high-disagreement,
off-distribution responses) substantially raises peak gold:

| Reward signal | Single RM | Ensemble mean | Pessimistic ensemble |
|---|---:|---:|---:|
| Peak gold | +1.05 | +1.74 | **+1.80** |

An 8-model ensemble alone recovers most of the benefit (≈ matching a 6× larger preference
budget) by averaging away the noise that drives the winner's-curse exploitation;
pessimism adds a further margin by explicitly penalizing the uncertain regions the policy
would otherwise hack.

![Ensemble mitigation](../results/figures/mitigation_ensemble.png)

## 4. Findings

1. **Proxy and truth diverge under optimization pressure.** Past an interior optimum,
   harder optimization *increases* the proxy and *decreases* the gold — invisible if you
   only monitor the proxy.
2. **The KL budget is the key control.** There is an optimal KL; RLHF's KL penalty is not
   a regularization detail but the mechanism that stops reward hacking.
3. **Better reward models help, predictably.** Overoptimization severity scales with RM
   error, so more preference data and reward-model ensembles raise the peak true reward.
4. **Pessimism beats trust.** A lower-confidence-bound over an RM ensemble — declining to
   chase rewards the models disagree about — is a cheap, effective mitigation.

## 5. Limitations

- The RM is a **transparent parametric error model**, not a trained neural network; the
  absolute numbers depend on its parameters (`k0`, `opt`, `σ0`). The *shape* of the
  results — overoptimization, KL as the control, data/ensemble scaling — is the
  transferable part, and matches the empirical literature.
- A **contextual-bandit** abstraction of RLHF (one step per prompt), not a full
  token-level sequential MDP; this isolates the reward dynamics at the cost of sequence
  structure.
- Synthetic gold reward; no human data.

## 6. Future work

- Replace the parametric RM with a **trained** reward model (Bradley-Terry on sampled
  preferences) and a sequence-level environment; check the curves persist.
- Fit the **scaling law** `gold(KL)` functional form of Gao et al. [1] to the data here.
- Add more mitigations: **KL-adaptive** controllers, RM **uncertainty penalties**,
  iterated/online preference collection, and **constrained** RL.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py        # overopt curve -> training dynamics -> mitigations
python tests/test_smoke.py
```

Seeded (`SEED = 20260617`) and byte-reproducible across `PYTHONHASHSEED`. Numbers land in
`results/` (`overopt_summary.json`, `training_summary.json`, `mitigations_summary.json`,
plus CSVs); figures in `results/figures/`.

## 8. Responsible-research note

This is a **defensive** study of a known RLHF failure mode using entirely synthetic
rewards. It contains no model weights, no human data, and nothing that elicits harmful
behavior — only a measurement of how optimizing a proxy reward can quietly harm the true
objective, and how to mitigate it.

## References

1. Gao, Schulman, Hilton, "Scaling Laws for Reward Model Overoptimization" (2022). https://arxiv.org/abs/2210.10760
2. Coste, Anwar, Kirk, Krueger, "Reward Model Ensembles Help Mitigate Overoptimization" (2023). https://arxiv.org/abs/2310.02743
3. Skalse, Howe, Krasheninnikov, Krueger, "Defining and Characterizing Reward Hacking" (NeurIPS 2022). https://arxiv.org/abs/2209.13085
4. Stiennon et al., "Learning to summarize from human feedback" (2020). https://arxiv.org/abs/2009.01325
5. Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022). https://arxiv.org/abs/2203.02155
6. Bai et al., "Training a Helpful and Harmless Assistant with RLHF" (Anthropic, 2022). https://arxiv.org/abs/2204.05862
