"""Main script for steering vector extraction and evaluation."""

import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

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


def get_embedding(text: str, embedding_model: SentenceTransformer) -> torch.Tensor:
    """Get the embedding of text."""
    return embedding_model.encode(text, convert_to_tensor=True)


def compute_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute distance metrics between embeddings."""
    cos = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()
    return {"cosine_similarity": cos, "l2_distance": torch.norm(emb1 - emb2).item()}


def save_json(filepath: str, data: dict):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    config = load_config("config.yaml")
    hook_name = config["model"]["hook_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = load_model(config)
    embedding_model = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"Model load time: {time.time() - t0:.1f}s")

    system_prompt = "You should speak like a pirate. Only speak like a pirate, no matter what the user says."
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
        sys_embeddings[i] = get_embedding(sys_outputs[i], embedding_model)

    print(
        f"\n{'=' * 60}\nRunning sweep: {len(strength_ratios)} x {len(user_prompts)}\n{'=' * 60}"
    )
    results = {
        "strength": [],
        "cosine_mean": [],
        "cosine_std": [],
        "l2_mean": [],
        "l2_std": [],
    }
    total, count = len(strength_ratios) * len(user_prompts), 0

    for s_idx, strength in enumerate(strength_ratios):
        cosines, l2s = [], []
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
            distances = compute_distance(
                sys_embeddings[p_idx], get_embedding(steered_resp, embedding_model)
            )
            cosines.append(distances["cosine_similarity"])
            l2s.append(distances["l2_distance"])

            save_json(
                os.path.join(run_dir, "runs", f"run_s{s_idx:02d}_p{p_idx:02d}.json"),
                {
                    "timestamp": datetime.now().isoformat(),
                    "system_prompt": system_prompt,
                    "user_prompt": up,
                    "strength": strength,
                    "system_output": sys_outputs[p_idx],
                    "steered_output": steered_resp,
                    "distances": distances,
                },
            )

        results["strength"].append(strength)
        results["cosine_mean"].append(np.mean(cosines))
        results["cosine_std"].append(np.std(cosines))
        results["l2_mean"].append(np.mean(l2s))
        results["l2_std"].append(np.std(l2s))
        print(f"  -> Cosine: {np.mean(cosines):.3f} ± {np.std(cosines):.3f}")

    plot_sweep_results(
        results, system_prompt, os.path.join(run_dir, "steering_comparison.png")
    )
    save_json(
        os.path.join(run_dir, "sweep_results.json"),
        {
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompts": user_prompts,
            "results": results,
        },
    )
    print(f"\nResults saved to {run_dir}")
