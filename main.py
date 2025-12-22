import time

import torch

from steering.generate import (
    compare_outputs,
    get_ground_truth_activations,
    print_comparison,
)
from utils.loading import load_config, load_hooked_model

if __name__ == "__main__":
    config = load_config("config.yaml")

    t0 = time.time()
    model, sae = load_hooked_model(config, device="cpu")
    print(f"Model load time: {time.time() - t0:.1f}s")

    system_prompt = "You are a pirate. Speak like a pirate using words like 'arrr', 'matey', 'ye', and 'treasure'."
    user_prompt = "What is the capital of France?"

    print("\n" + "=" * 80)
    print("COMPARISON: System Prompt vs Feature Vector Steering")
    print("=" * 80)

    t0 = time.time()
    steering_data = get_ground_truth_activations([system_prompt], model, sae)
    print(f"Feature extraction time: {time.time() - t0:.1f}s")

    topk = torch.topk(steering_data, k=10)
    print(f"\nTop 10 feature activations: {topk.values}")

    t0 = time.time()
    result = compare_outputs(
        model,
        sae,
        system_prompt,
        user_prompt,
        strength_ratio=0.3,
        max_new_tokens=50,
        num_features=5,
    )
    print(f"Generation time: {time.time() - t0:.1f}s")
    print_comparison(result)
