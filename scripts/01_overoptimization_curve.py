"""Step 1: sweep the KL budget and trace proxy vs. gold -- the overoptimization curve."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from src.env import make_env  # noqa: E402
from src.plots import plot_overopt_curve  # noqa: E402
from src.policy import evaluate, kl_optimal_policy  # noqa: E402
from src.reward_model import make_reward_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

env = make_env()
rhat = make_reward_model(env, n_pref=1600)
betas = np.geomspace(5.0, 0.04, 24)

kl, proxy, gold = [], [], []
for b in betas:
    pi = kl_optimal_policy(rhat, env["piref"], b)
    m = evaluate(pi, env["gold"], rhat, env["piref"])
    kl.append(m["kl"]); proxy.append(m["proxy"]); gold.append(m["gold"])

peak = int(np.argmax(gold))
with open(RES / "overopt_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["beta", "kl", "proxy", "gold"])
    w.writerows(zip(betas, kl, proxy, gold))
plot_overopt_curve(kl, proxy, gold, RES / "figures" / "overopt_curve.png", peak_idx=peak)

summary = dict(
    peak_gold=round(gold[peak], 3), kl_at_peak=round(kl[peak], 2),
    beta_at_peak=round(float(betas[peak]), 3),
    gold_overoptimized=round(gold[-1], 3), kl_max=round(kl[-1], 2),
    proxy_at_peak=round(proxy[peak], 3), proxy_overoptimized=round(proxy[-1], 3),
    overoptimization_drop=round(gold[peak] - gold[-1], 3))
json.dump(summary, open(RES / "overopt_summary.json", "w"), indent=2)
print("Overoptimization curve:")
print(f"  peak GOLD {summary['peak_gold']} at KL={summary['kl_at_peak']} (beta={summary['beta_at_peak']})")
print(f"  push to KL={summary['kl_max']}: proxy {summary['proxy_at_peak']}->{summary['proxy_overoptimized']} (UP) "
      f"but GOLD {summary['peak_gold']}->{summary['gold_overoptimized']} (DOWN, drop {summary['overoptimization_drop']})")
