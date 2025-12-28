"""Plotting utilities for steering experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sweep_results(results: dict, system_prompt: str, output_path: str):
    """Plot cosine similarity and L2 distance vs strength."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(
        results["strength"],
        results["cosine_mean"],
        yerr=results["cosine_std"],
        marker="o",
        capsize=3,
        color="blue",
    )
    axes[0].set_xlabel("Strength")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Cosine Similarity vs Strength")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        results["strength"],
        results["l2_mean"],
        yerr=results["l2_std"],
        marker="o",
        capsize=3,
        color="red",
    )
    axes[1].set_xlabel("Strength")
    axes[1].set_ylabel("L2 Distance")
    axes[1].set_title("L2 Distance vs Strength")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"System Prompt vs Steered Output\n'{system_prompt[:50]}...'")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
