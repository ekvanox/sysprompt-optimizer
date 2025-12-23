import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformer


def format_gemma_chat(system_prompt: str, user_prompt: str) -> str:
    """Format prompts for Gemma instruction-tuned model."""
    return f"<start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def format_gemma_user_only(user_prompt: str) -> str:
    """Format user prompt only (no system prompt) for Gemma."""
    return f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def get_ground_truth_activations(
    text_batch: list[str], model: HookedTransformer, sae: SAE
) -> torch.Tensor:
    _, cache = model.run_with_cache(text_batch)
    resid = cache[sae.cfg.metadata.hook_name][:, 1:, :]

    base_activations = torch.load("data/base_vec.pt", map_location=resid.device)

    feature_acts = sae.encode(resid)
    differential_acts = feature_acts - base_activations
    activations, _ = differential_acts.max(dim=1)

    return F.relu(activations)


def generate_steered(
    model: HookedTransformer,
    sae: SAE,
    steering_data: torch.Tensor,
    prompt: str,
    strength_ratio: float = 0.1,
    max_new_tokens: int = 50,
    num_features: int = 5,
):
    steering_data = steering_data.squeeze(0)

    topk = torch.topk(steering_data, k=num_features)
    values, indices = topk.values, topk.indices

    feature_directions = sae.W_dec[indices]
    weighted_direction = feature_directions * values.unsqueeze(-1)
    concept_vector = weighted_direction.sum(dim=0)
    concept_vector = concept_vector / concept_vector.norm()

    def adaptive_hook(resid, hook):
        current_norm = resid.norm(dim=-1).mean()
        steering_vec = concept_vector * current_norm * strength_ratio
        return resid + steering_vec

    with model.hooks(fwd_hooks=[(sae.cfg.metadata.hook_name, adaptive_hook)]):
        output = model.generate(
            prompt, max_new_tokens=max_new_tokens, verbose=False, temperature=0
        )

    return output, indices.tolist(), values.tolist()


def generate_with_system_prompt(
    model: HookedTransformer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate response using traditional system prompt injection."""
    formatted_prompt = format_gemma_chat(system_prompt, user_prompt)
    return model.generate(
        formatted_prompt, max_new_tokens=max_new_tokens, verbose=False, temperature=0
    )


def compare_outputs(
    model: HookedTransformer,
    sae: SAE,
    system_prompt: str,
    user_prompt: str,
    strength_ratio: float = 0.1,
    max_new_tokens: int = 50,
    num_features: int = 5,
) -> dict:
    system_output = generate_with_system_prompt(
        model, system_prompt, user_prompt, max_new_tokens
    )

    steering_data = get_ground_truth_activations([system_prompt], model, sae)
    formatted_user_prompt = format_gemma_user_only(user_prompt)
    steered_output, top_feature_indices, top_feature_strengths = generate_steered(
        model,
        sae,
        steering_data,
        formatted_user_prompt,
        strength_ratio=strength_ratio,
        max_new_tokens=max_new_tokens,
        num_features=num_features,
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_output": system_output,
        "steered_output": steered_output,
        "top_feature_indices": top_feature_indices,
        "top_feature_strengths": top_feature_strengths,
    }


def _extract_response(text: str) -> str:
    """Extract just the model response, removing prompt formatting."""
    if "model" in text:
        parts = text.split("model")
        if len(parts) > 1:
            return parts[-1].strip()
    return text


def print_comparison(result: dict, width: int = 80) -> None:
    sep = "=" * width
    half_width = (width - 3) // 2

    print(f"\n{sep}")
    print(f"SYSTEM PROMPT: {result['system_prompt']}")
    print(f"USER PROMPT: {result['user_prompt']}")
    print(sep)

    print(
        f"\n{'WITH SYSTEM PROMPT':<{half_width}} | {'WITH FEATURE STEERING':<{half_width}}"
    )
    print("-" * width)

    sys_response = _extract_response(result["system_output"])
    steer_response = _extract_response(result["steered_output"])

    sys_lines = _wrap_text(sys_response, half_width)
    steer_lines = _wrap_text(steer_response, half_width)

    max_lines = max(len(sys_lines), len(steer_lines))

    for i in range(max_lines):
        left = sys_lines[i] if i < len(sys_lines) else ""
        right = steer_lines[i] if i < len(steer_lines) else ""
        print(f"{left:<{half_width}} | {right:<{half_width}}")

    print(sep + "\n")


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line = f"{current_line} {word}".strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word[:width]  # truncate very long words

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def run_comparison_suite(
    model: HookedTransformer,
    sae: SAE,
    system_prompt: str,
    user_prompts: list[str],
    strength_ratio: float = 0.1,
    max_new_tokens: int = 50,
    num_features: int = 5,
) -> list[dict]:
    """
    Run comparisons across multiple user prompts with the same system prompt.
    Returns list of comparison results.
    """
    results = []
    for user_prompt in user_prompts:
        print(f"\n🔄 Processing: '{user_prompt[:50]}...'")
        result = compare_outputs(
            model,
            sae,
            system_prompt,
            user_prompt,
            strength_ratio=strength_ratio,
            max_new_tokens=max_new_tokens,
            num_features=num_features,
        )
        results.append(result)
        print_comparison(result)

    return results
