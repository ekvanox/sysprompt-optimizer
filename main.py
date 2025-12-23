import json
import os
import time
from datetime import datetime

import matplotlib
import torch

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from steering.generate import (
    _extract_response,
    compare_outputs,
    generate_with_system_prompt,
)
from utils.loading import load_config, load_hooked_model
from utils.neuronpedia import format_feature_list, lookup_features

# Ensure logs directory exists
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


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


def save_sweep_results(
    results: dict,
    system_prompt: str,
    user_prompts: list[str],
    num_features: int,
    run_dir: str,
    top_feature_indices: list[int],
    top_feature_strengths: list[float],
    model_id: str,
    layer: str,
) -> str:
    # Build features with descriptions
    features_with_desc = []
    if top_feature_indices:
        for i, f in enumerate(lookup_features(model_id, layer, top_feature_indices)):
            features_with_desc.append(
                {
                    "index": f["index"],
                    "strength": top_feature_strengths[i]
                    if i < len(top_feature_strengths)
                    else 0.0,
                    "description": f["description"],
                }
            )

    output = {
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "user_prompts": user_prompts,
        "num_features": num_features,
        "top_features": features_with_desc,
        "results": results,
    }

    filepath = os.path.join(run_dir, "sweep_results.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def save_run_output(
    run_dir: str,
    system_prompt: str,
    user_prompt: str,
    strength_ratio: float,
    num_features: int,
    top_features: list[dict],
    system_output: str,
    steered_output: str,
    distances: dict,
    prompt_idx: int,
    strength_idx: int,
) -> None:
    """Save individual run output to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "strength_ratio": strength_ratio,
        "num_features": num_features,
        "top_features": top_features,
        "system_output": system_output,
        "steered_output": steered_output,
        "distances": distances,
    }

    filename = f"run_s{strength_idx:02d}_p{prompt_idx:02d}.json"
    filepath = os.path.join(run_dir, "runs", filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


def run_experiment(
    model,
    sae,
    user_prompt,
    strength_ratio,
    num_features,
    system_output: str,
    sys_emb: torch.Tensor,
    steering_data: torch.Tensor,
):
    """Run one steered generation and compare against precomputed system output."""
    from steering.generate import (
        format_gemma_user_only,
        generate_steered,
    )

    formatted_user_prompt = format_gemma_user_only(user_prompt)
    steered_raw, top_feature_indices, top_feature_strengths = generate_steered(
        model,
        sae,
        steering_data,
        formatted_user_prompt,
        strength_ratio=strength_ratio,
        max_new_tokens=40,
        num_features=num_features,
    )

    steer_response = _extract_response(steered_raw)
    steer_emb = get_embedding(steer_response, model, sae)
    distances = compute_distance(sys_emb, steer_emb)

    return {
        "distances": distances,
        "system_output": system_output,
        "steered_output": steer_response,
        "top_feature_indices": top_feature_indices,
        "top_feature_strengths": top_feature_strengths,
    }


if __name__ == "__main__":
    config = load_config("config.yaml")

    t0 = time.time()
    model, sae = load_hooked_model(config, device="cpu")
    print(f"Model load time: {time.time() - t0:.1f}s")

    system_prompt = (
        "You are angry and frustrated. All your responses should reflect your anger."
    )

    user_prompts = [
        "What is the capital of France?",
        "Explain how computers work.",
        "What should I eat for dinner?",
        "Tell me about the ocean.",
        "How do I learn to code?",
        "What is the meaning of life?",
        "Describe a sunset.",
        "How does gravity work?",
        "What makes a good friend?",
        "Explain photosynthesis.",
        "What is your favorite color?",
        "How do airplanes fly?",
        "What should I read next?",
        "Describe the moon.",
        "How do I make coffee?",
        "What is democracy?",
        "Tell me a fun fact.",
        "How do plants grow?",
        "What is music?",
        "Explain the internet.",
    ]

    # 25 strength ratios from 0 to 1
    strength_ratios = np.linspace(0, 1.0, 25).tolist()
    num_features = 10

    total = len(strength_ratios) * len(user_prompts)
    count = 0

    # Create run folder early so we can save outputs during sweep
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOGS_DIR, f"run_{run_timestamp}")
    os.makedirs(os.path.join(run_dir, "runs"), exist_ok=True)

    # Get model/layer info for feature lookups
    model_id = config["model"]["name"]
    sae_id = config["model"]["sae"]["sae_id"]
    layer_num = sae_id.split(".")[1]
    layer = f"{layer_num}-res-jb"

    # Compute averaged steering data over all user prompts
    from steering.generate import get_averaged_activations, generate_steered, format_gemma_user_only
    
    print("\nComputing averaged feature activations over all user prompts...")
    steering_data = get_averaged_activations(system_prompt, user_prompts, model, sae)
    
    # Get top features from the averaged activations
    topk = torch.topk(steering_data.squeeze(0), k=num_features)
    top_strengths = topk.values.tolist()
    top_indices = topk.indices.tolist()

    # Build features with descriptions once
    features_with_desc = []
    if top_indices:
        for i, f in enumerate(lookup_features(model_id, layer, top_indices)):
            features_with_desc.append(
                {
                    "index": f["index"],
                    "strength": top_strengths[i] if i < len(top_strengths) else 0.0,
                    "description": f["description"],
                }
            )

    print("\n" + "=" * 80)
    print(
        f"Running sweep: {len(strength_ratios)} strengths x {len(user_prompts)} prompts"
    )
    print(f"Fixed: num_features={num_features}")
    print(f"Run folder: {run_dir}")
    print("=" * 80)

    results = {
        "strength": [],
        "cosine_mean": [],
        "cosine_std": [],
        "l2_mean": [],
        "l2_std": [],
    }

    # Pre-generate system outputs once per prompt (they don't change with strength)
    print("\nPre-generating system prompt outputs...")
    system_outputs = {}
    system_embeddings = {}
    for prompt_idx, user_prompt in enumerate(user_prompts):
        print(f"  [{prompt_idx + 1}/{len(user_prompts)}] '{user_prompt[:40]}...'")
        raw_output = generate_with_system_prompt(
            model, system_prompt, user_prompt, max_new_tokens=40
        )
        sys_response = _extract_response(raw_output)
        system_outputs[prompt_idx] = sys_response
        system_embeddings[prompt_idx] = get_embedding(sys_response, model, sae)

    print("\nRunning strength sweep...")
    for strength_idx, strength in enumerate(strength_ratios):
        cosines = []
        l2s = []
        for prompt_idx, user_prompt in enumerate(user_prompts):
            count += 1
            print(
                f"[{count}/{total}] strength={strength:.3f}, prompt='{user_prompt[:30]}...'"
            )

            result = run_experiment(
                model,
                sae,
                user_prompt,
                strength,
                num_features,
                system_output=system_outputs[prompt_idx],
                sys_emb=system_embeddings[prompt_idx],
                steering_data=steering_data,
            )
            cosines.append(result["distances"]["cosine_similarity"])
            l2s.append(result["distances"]["l2_distance"])

            # Save individual run output
            save_run_output(
                run_dir=run_dir,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strength_ratio=strength,
                num_features=num_features,
                top_features=features_with_desc,
                system_output=result["system_output"],
                steered_output=result["steered_output"],
                distances=result["distances"],
                prompt_idx=prompt_idx,
                strength_idx=strength_idx,
            )

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

    # Save plot
    plot_path = os.path.join(run_dir, "steering_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Save sweep summary with features
    results_path = save_sweep_results(
        results,
        system_prompt,
        user_prompts,
        num_features,
        run_dir,
        top_indices,
        top_strengths,
        model_id,
        layer,
    )
    print(f"Results saved to {results_path}")

    # Print top features
    print("\n" + "=" * 80)
    print("Top Features Used (from Neuronpedia):")
    print("=" * 80)

    if top_indices:
        try:
            feature_info = format_feature_list(model_id, layer, top_indices[:5])
            print(feature_info)
        except Exception as e:
            print(f"Could not fetch feature descriptions: {e}")
            print(f"Feature indices: {top_indices[:5]}")
