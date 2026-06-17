"""plots.py -- figures for the report (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_overopt_curve(kl, proxy, gold, path, peak_idx=None):
    plt.figure(figsize=(7.2, 4.7))
    plt.plot(kl, proxy, "o-", color="#c0392b", lw=2, label="Proxy reward (optimized)")
    plt.plot(kl, gold, "s-", color="#2980b9", lw=2, label="Gold reward (true objective)")
    if peak_idx is not None:
        plt.axvline(kl[peak_idx], ls="--", color="#27ae60", lw=1.5,
                    label="Optimal KL (gold peak)")
        plt.scatter([kl[peak_idx]], [gold[peak_idx]], color="#27ae60", zorder=5, s=60)
    plt.xlabel("KL(policy || reference)  =  optimization pressure")
    plt.ylabel("Reward")
    plt.title("Reward-model overoptimization: the proxy rises, the truth turns over")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_dynamics(steps, proxy, gold, kl, path, peak_step=None):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.7))
    ax1.plot(steps, gold, color="#2980b9", lw=2, label="Gold reward")
    ax1.plot(steps, proxy, color="#c0392b", lw=2, label="Proxy reward")
    ax1.set_xlabel("Policy-gradient training step")
    ax1.set_ylabel("Reward")
    if peak_step is not None:
        ax1.axvline(peak_step, ls="--", color="#27ae60", lw=1.5, label="Early-stop (gold peak)")
    ax2 = ax1.twinx()
    ax2.plot(steps, kl, color="#7f8c8d", lw=1.2, ls=":", label="KL to reference")
    ax2.set_ylabel("KL to reference", color="#7f8c8d")
    ax1.set_title("Overoptimization during training: gold peaks, then degrades")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curve_family(curves, path, title, xlabel="KL(policy || reference)"):
    """curves: list of (label, kl_array, gold_array, color)."""
    plt.figure(figsize=(7.2, 4.7))
    for label, kl, gold, color in curves:
        plt.plot(kl, gold, "-o", lw=2, ms=3, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Gold reward (true objective)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
