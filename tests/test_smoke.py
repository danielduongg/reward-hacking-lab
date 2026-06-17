"""Fast smoke test: overoptimization occurs, the optimizer is correct, pessimism helps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from src.env import make_env  # noqa: E402
from src.policy import evaluate, kl_optimal_policy, policy_gradient  # noqa: E402
from src.reward_model import ensemble_score, make_ensemble, make_reward_model  # noqa: E402


def _curve(rhat, env, betas):
    kl, gold = [], []
    for b in betas:
        m = evaluate(kl_optimal_policy(rhat, env["piref"], b), env["gold"], rhat, env["piref"])
        kl.append(m["kl"]); gold.append(m["gold"])
    return np.array(kl), np.array(gold)


def test_overoptimization_and_mitigation():
    env = make_env(n_contexts=300, n_actions=32)
    betas = np.geomspace(5.0, 0.05, 16)
    rhat = make_reward_model(env, n_pref=1200)
    kl, gold = _curve(rhat, env, betas)
    # gold rises then falls: its peak is interior, and the most-optimized point is lower
    peak = int(np.argmax(gold))
    assert 0 < peak < len(gold) - 1, peak
    assert gold[peak] - gold[-1] > 0.03, (gold[peak], gold[-1])

    # optimizer correctness: PG converges to the closed form
    pi, _ = policy_gradient(rhat, env["piref"], env["gold"], beta=0.3, lr=0.5, steps=300)
    a = evaluate(pi, env["gold"], rhat, env["piref"])
    b = evaluate(kl_optimal_policy(rhat, env["piref"], 0.3), env["gold"], rhat, env["piref"])
    assert abs(a["gold"] - b["gold"]) < 0.03, (a, b)

    # pessimistic ensemble raises peak gold vs a single model
    models = make_ensemble(env, n_models=6, n_pref=1200)
    _, g_single = _curve(models[0], env, betas)
    _, g_pess = _curve(ensemble_score(models, lam=1.5), env, betas)
    assert g_pess.max() >= g_single.max() - 1e-6, (g_pess.max(), g_single.max())


if __name__ == "__main__":
    test_overoptimization_and_mitigation()
    print("smoke test passed")
