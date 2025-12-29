"""Main script for steering vector extraction and evaluation."""

import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from evaluator import Evaluator
from prompts.user_prompts import USER_PROMPTS
from steering.generate import (
    extract_response,
    format_gemma_chat,
    format_gemma_user_only,
    generate_and_capture_activations,
    generate_steered,
)
from utils.loading import load_config, load_model
from utils.plotting import plot_sweep_results

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def compute_steering_vector(
    model,
    system_prompt: str,
    user_prompts: list[str],
    hook_name: str,
    max_new_tokens: int,
    reduction_method: str = "mean",
) -> torch.Tensor:
    """Compute steering vector by averaging activations with/without sysprompt."""
    with_sys_acts, without_sys_acts = [], []

    for i, user_prompt in enumerate(user_prompts):
        print(f"  [{i + 1}/{len(user_prompts)}] '{user_prompt[:40]}...'")
        prompt_with = format_gemma_chat(system_prompt, user_prompt)
        _, act_with = generate_and_capture_activations(
            model,
            prompt_with,
            max_new_tokens=max_new_tokens,
            hook_name=hook_name,
            reduction_method=reduction_method,
        )
        with_sys_acts.append(act_with)

        prompt_without = format_gemma_user_only(user_prompt)
        _, act_without = generate_and_capture_activations(
            model,
            prompt_without,
            max_new_tokens=max_new_tokens,
            hook_name=hook_name,
            reduction_method=reduction_method,
        )
        without_sys_acts.append(act_without)

    return (torch.stack(with_sys_acts) - torch.stack(without_sys_acts)).mean(0)


def save_json(filepath: str, data: dict):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_statistics(metrics: list[dict]) -> dict:
    """Compute statistics from a list of metrics."""
    return {
        "cosine_mean": np.mean([m["cosine_similarity"] for m in metrics]),
        "cosine_std": np.std([m["cosine_similarity"] for m in metrics]),
        "l2_mean": np.mean([m["l2_distance"] for m in metrics]),
        "l2_std": np.std([m["l2_distance"] for m in metrics]),
        "bert_f1_mean": np.mean([m["bert_f1"] for m in metrics]),
        "bert_f1_std": np.std([m["bert_f1"] for m in metrics]),
    }


def run_sweep(
    model,
    evaluator,
    system_prompt,
    user_prompts,
    strength_ratios,
    max_new_tokens,
    hook_name,
    steering_vec,
    run_dir,
):
    """Run the sweep."""
    total, count = len(strength_ratios) * len(user_prompts), 0
    print(
        f"Running sweep for {total} samples ({len(strength_ratios)} strengths x {len(user_prompts)} prompts)"
    )
    metrics = []
    for s_idx, strength in enumerate(strength_ratios):
        metrics_by_strength = []
        for p_idx, up in enumerate(user_prompts):
            count += 1
            print(f"[{count}/{total}] strength={strength:.2f}, prompt='{up[:30]}...'")

            steered_out = generate_steered(
                model,
                format_gemma_user_only(up),
                steering_vec,
                strength,
                max_new_tokens,
                hook_name,
            )
            steered_resp = extract_response(steered_out)

            sample_metrics = evaluator.compare_outputs(sys_outputs[p_idx], steered_resp)

            save_json(
                os.path.join(run_dir, "runs", f"run_s{s_idx:02d}_p{p_idx:02d}.json"),
                {
                    "timestamp": datetime.now().isoformat(),
                    "system_prompt": system_prompt,
                    "user_prompt": up,
                    "strength": strength,
                    "system_output": sys_outputs[p_idx],
                    "steered_output": steered_resp,
                    "metrics": sample_metrics,
                },
            )

            metrics_by_strength.append(sample_metrics)

        print(
            f"-> Cosine similarity: {compute_statistics(metrics_by_strength)['cosine_mean']:.3f} ± {compute_statistics(metrics_by_strength)['cosine_std']:.3f} for strength: {strength:.2f}"
        )
        metrics.extend(metrics_by_strength)
    results = compute_statistics(metrics)

    return results


if __name__ == "__main__":
    config = load_config("config.yaml")
    hook_name = config["model"]["hook_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = load_model(config, device=device)
    evaluator = Evaluator(device=device)
    print(f"Model load time: {time.time() - t0:.1f}s")

    system_prompt = "You should speak like a pirate. Only speak like a pirate, no matter what the user says."
    user_prompts = USER_PROMPTS

    strength_ratios = np.linspace(0, 1.0, 11).tolist()
    max_new_tokens = 40

    run_dir = os.path.join(LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(run_dir, "runs"), exist_ok=True)

    print("\n" + "=" * 60 + "\nComputing steering vector...\n" + "=" * 60)
    steering_vec = compute_steering_vector(
        model,
        system_prompt,
        user_prompts,
        hook_name,
        max_new_tokens,
        reduction_method="mean",
    )
    print(f"Steering vector norm: {steering_vec.norm():.4f}")
    torch.save(steering_vec, os.path.join(run_dir, "steering_vector.pt"))

    print("\nPre-generating system prompt outputs...")
    sys_outputs, sys_embeddings = {}, {}
    for i, up in enumerate(user_prompts):
        out, _ = generate_and_capture_activations(
            model, format_gemma_chat(system_prompt, up), max_new_tokens, hook_name
        )
        sys_outputs[i] = extract_response(out)

    results = run_sweep(
        model,
        evaluator,
        system_prompt,
        user_prompts,
        strength_ratios,
        max_new_tokens,
        hook_name,
        steering_vec,
        run_dir,
    )

    plot_sweep_results(
        strength_ratios,
        results,
        system_prompt,
        os.path.join(run_dir, "steering_comparison.png"),
    )
    save_json(
        os.path.join(run_dir, "sweep_results.json"),
        {
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompts": user_prompts,
            "strength_ratios": strength_ratios,
            "results": results,
        },
    )
    print(f"\nResults saved to {run_dir}")
