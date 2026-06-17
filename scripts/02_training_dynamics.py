"""Step 2: run the RL optimizer and watch gold peak then degrade as training proceeds.

We optimize the PROXY reward directly (beta=0, no KL leash) -- the failure mode RLHF's
KL penalty exists to prevent -- and log the true (gold) reward each step. Separately we
validate the optimizer against the closed-form KL-optimum at a moderate beta.
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from src.env import make_env  # noqa: E402
from src.plots import plot_training_dynamics  # noqa: E402
from src.policy import evaluate, kl_optimal_policy, policy_gradient  # noqa: E402
from src.reward_model import make_reward_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

env = make_env()
rhat = make_reward_model(env, n_pref=1600)

# --- dynamics: optimize the proxy with NO KL penalty -> overoptimization over steps ---
_, hist = policy_gradient(rhat, env["piref"], env["gold"], beta=0.0, lr=0.05,
                          steps=700, log_every=5)
steps = [h["step"] for h in hist]
proxy = [h["proxy"] for h in hist]
gold = [h["gold"] for h in hist]
kl = [h["kl"] for h in hist]
peak_i = int(np.argmax(gold))
with open(RES / "training_dynamics.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "proxy", "gold", "kl"])
    w.writerows(zip(steps, proxy, gold, kl))
plot_training_dynamics(steps, proxy, gold, kl, RES / "figures" / "training_dynamics.png",
                       peak_step=steps[peak_i])

# --- validation: at a moderate beta the optimizer should match the closed form ---
BETA_VAL = 0.3
pi_pg, _ = policy_gradient(rhat, env["piref"], env["gold"], beta=BETA_VAL, lr=0.2, steps=400)
m_pg = evaluate(pi_pg, env["gold"], rhat, env["piref"])
m_cf = evaluate(kl_optimal_policy(rhat, env["piref"], BETA_VAL), env["gold"], rhat, env["piref"])
val = {k: [round(m_pg[k], 3), round(m_cf[k], 3)] for k in ["proxy", "gold", "kl"]}

json.dump(dict(peak_gold=round(gold[peak_i], 3), peak_step=steps[peak_i],
               final_gold=round(gold[-1], 3), final_kl=round(kl[-1], 2),
               overoptimization_drop=round(gold[peak_i] - gold[-1], 3),
               proxy_peak=round(proxy[peak_i], 3), proxy_final=round(proxy[-1], 3),
               validation_pg_vs_closedform=val),
          open(RES / "training_summary.json", "w"), indent=2)
print(f"Proxy-only training: GOLD peaks {round(gold[peak_i],3)} at step {steps[peak_i]} "
      f"(KL={round(kl[peak_i],2)}), then degrades to {round(gold[-1],3)} at KL={round(kl[-1],2)} "
      f"(drop {round(gold[peak_i]-gold[-1],3)}); proxy climbs {round(proxy[peak_i],3)}->{round(proxy[-1],3)}")
print(f"Optimizer validation @beta={BETA_VAL} (PG vs closed-form) proxy/gold/kl: {val}  <- match => correct")
