"""Step 3: mitigations -- more preference data, and pessimistic reward-model ensembles."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from src.env import make_env  # noqa: E402
from src.plots import plot_curve_family  # noqa: E402
from src.policy import evaluate, kl_optimal_policy  # noqa: E402
from src.reward_model import ensemble_score, make_ensemble, make_reward_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)
BETAS = np.geomspace(5.0, 0.04, 22)


def curve(rhat, env):
    kl, gold = [], []
    for b in BETAS:
        pi = kl_optimal_policy(rhat, env["piref"], b)
        m = evaluate(pi, env["gold"], rhat, env["piref"])
        kl.append(m["kl"]); gold.append(m["gold"])
    return np.array(kl), np.array(gold)


env = make_env()

# (a) preference-data scaling
N_list = [400, 800, 1600, 3200, 6400]
colors_n = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#16a085"]
scaling, fam = {}, []
for N, c in zip(N_list, colors_n):
    kl, gold = curve(make_reward_model(env, n_pref=N), env)
    scaling[N] = round(float(gold.max()), 3)
    fam.append((f"N={N}", kl, gold, c))
plot_curve_family(fam, RES / "figures" / "data_scaling.png",
                  "More preference data -> higher peak gold, later overoptimization")

# (b) pessimistic reward-model ensemble
models = make_ensemble(env, n_models=8, n_pref=1600)
single = make_reward_model(env, n_pref=1600, seed=100)   # one ensemble member
mean_rm = ensemble_score(models, lam=0.0)
pess_rm = ensemble_score(models, lam=1.5)
fam2, peaks = [], {}
for label, rm, col in [("Single RM", single, "#c0392b"),
                       ("Ensemble mean", mean_rm, "#e67e22"),
                       ("Pessimistic ensemble (mean - 1.5 std)", pess_rm, "#27ae60")]:
    kl, gold = curve(rm, env)
    peaks[label] = round(float(gold.max()), 3)
    fam2.append((label, kl, gold, col))
plot_curve_family(fam2, RES / "figures" / "mitigation_ensemble.png",
                  "Pessimistic reward-model ensembles mitigate overoptimization")

json.dump({"peak_gold_vs_preference_data": scaling, "peak_gold_by_defense": peaks},
          open(RES / "mitigations_summary.json", "w"), indent=2)
print("Preference-data scaling (peak gold):", scaling)
print("Mitigation peak gold:", peaks)
