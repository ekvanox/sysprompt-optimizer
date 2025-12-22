import time

import matplotlib
import torch

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from steering.generate import (
    _extract_response,
    compare_outputs,
)
from utils.loading import load_config, load_hooked_model


def get_embedding(text: str, model, sae) -> torch.Tensor:
    """Get the mean residual stream embedding for text."""
    _, cache = model.run_with_cache([text])
    resid = cache[sae.cfg.metadata.hook_name][:, 1:, :]
    return resid.mean(dim=1).squeeze(0)


def compute_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute distance metrics between embeddings."""
    cosine_sim = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()
    l2_dist = torch.norm(emb1 - emb2).item()
    return {"cosine_similarity": cosine_sim, "l2_distance": l2_dist}


def run_experiment(
    model, sae, system_prompt, user_prompt, strength_ratio, num_features
):
    """Run one comparison and return distances."""
    result = compare_outputs(
        model,
        sae,
        system_prompt,
        user_prompt,
        strength_ratio=strength_ratio,
        max_new_tokens=40,
        num_features=num_features,
    )

    sys_response = _extract_response(result["system_output"])
    steer_response = _extract_response(result["steered_output"])

    sys_emb = get_embedding(sys_response, model, sae)
    steer_emb = get_embedding(steer_response, model, sae)

    return compute_distance(sys_emb, steer_emb)


if __name__ == "__main__":
    config = load_config("config.yaml")

    t0 = time.time()
    model, sae = load_hooked_model(config, device="cpu")
    print(f"Model load time: {time.time() - t0:.1f}s")

    system_prompt = "You are a pirate. Speak like a pirate using words like 'arrr', 'matey', 'ye', and 'treasure'."

    # 5 different user prompts for stability
    user_prompts = [
        "What is the capital of France?",
        "Explain how computers work.",
        "What should I eat for dinner?",
        "Tell me about the ocean.",
        "How do I learn to code?",
    ]

    # 10 strength ratios from 0 to 0.5, fixed num_features=10
    strength_ratios = np.linspace(0, 0.5, 10).tolist()
    num_features = 10

    total = len(strength_ratios) * len(user_prompts)
    count = 0

    print("\n" + "=" * 80)
    print(
        f"Running sweep: {len(strength_ratios)} strengths x {len(user_prompts)} prompts"
    )
    print(f"Fixed: num_features={num_features}")
    print("=" * 80)

    results = {
        "strength": [],
        "cosine_mean": [],
        "cosine_std": [],
        "l2_mean": [],
        "l2_std": [],
    }

    for strength in strength_ratios:
        cosines = []
        l2s = []
        for user_prompt in user_prompts:
            count += 1
            print(
                f"[{count}/{total}] strength={strength:.3f}, prompt='{user_prompt[:30]}...'"
            )

            distances = run_experiment(
                model, sae, system_prompt, user_prompt, strength, num_features
            )
            cosines.append(distances["cosine_similarity"])
            l2s.append(distances["l2_distance"])

        results["strength"].append(strength)
        results["cosine_mean"].append(np.mean(cosines))
        results["cosine_std"].append(np.std(cosines))
        results["l2_mean"].append(np.mean(l2s))
        results["l2_std"].append(np.std(l2s))
        print(f"  -> Cosine: {np.mean(cosines):.3f} ± {np.std(cosines):.3f}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(
        results["strength"],
        results["cosine_mean"],
        yerr=results["cosine_std"],
        marker="o",
        capsize=3,
        color="blue",
    )
    axes[0].fill_between(
        results["strength"],
        np.array(results["cosine_mean"]) - np.array(results["cosine_std"]),
        np.array(results["cosine_mean"]) + np.array(results["cosine_std"]),
        alpha=0.2,
    )
    axes[0].set_xlabel("Strength Ratio")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Cosine Similarity vs Strength Ratio")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        results["strength"],
        results["l2_mean"],
        yerr=results["l2_std"],
        marker="o",
        capsize=3,
        color="red",
    )
    axes[1].fill_between(
        results["strength"],
        np.array(results["l2_mean"]) - np.array(results["l2_std"]),
        np.array(results["l2_mean"]) + np.array(results["l2_std"]),
        alpha=0.2,
        color="red",
    )
    axes[1].set_xlabel("Strength Ratio")
    axes[1].set_ylabel("L2 Distance")
    axes[1].set_title("L2 Distance vs Strength Ratio")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"System Prompt vs Steered Output (num_features={num_features})\n'{system_prompt[:60]}...'"
    )
    plt.tight_layout()
    plt.savefig("steering_comparison.png", dpi=150)
    plt.close()
    print("\nPlot saved to steering_comparison.png")
