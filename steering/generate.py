"""Generation functions with residual stream activation capture and steering."""

import torch
from transformer_lens import HookedTransformer

DEFAULT_HOOK_NAME = "blocks.15.hook_resid_post"


def format_gemma_chat(system_prompt: str, user_prompt: str) -> str:
    """Format prompts for Gemma instruction-tuned model with system prompt."""
    return f"<start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def format_gemma_user_only(user_prompt: str) -> str:
    """Format user prompt only (no system prompt) for Gemma."""
    return f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def generate_and_capture_activations(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 50,
    hook_name: str = DEFAULT_HOOK_NAME,
) -> tuple[str, torch.Tensor]:
    """Generate tokens and capture mean residual stream activation."""
    all_activations = []

    def capture_hook(resid, hook):
        all_activations.append(resid[:, -1, :].detach().clone())
        return resid

    with model.hooks(fwd_hooks=[(hook_name, capture_hook)]):
        output = model.generate(
            prompt, max_new_tokens=max_new_tokens, verbose=False, temperature=0
        )

    if all_activations:
        stacked = torch.stack(all_activations, dim=0).squeeze(1)
        mean_activation = stacked.mean(dim=0)
    else:
        mean_activation = torch.zeros(model.cfg.d_model)

    return output, mean_activation


def generate_steered(
    model: HookedTransformer,
    prompt: str,
    steering_vec: torch.Tensor,
    strength: float = 0.5,
    max_new_tokens: int = 50,
    hook_name: str = DEFAULT_HOOK_NAME,
) -> str:
    """Generate with steering vector added to residual stream."""
    steering_vec_normalized = steering_vec / steering_vec.norm()

    def steering_hook(resid, hook):
        current_norm = resid.norm(dim=-1).mean()
        return resid + steering_vec_normalized * current_norm * strength

    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        output = model.generate(
            prompt, max_new_tokens=max_new_tokens, verbose=False, temperature=0
        )
    return output


def extract_response(text: str) -> str:
    """Extract just the model response, removing prompt formatting."""
    if "model" in text:
        parts = text.split("model")
        if len(parts) > 1:
            return parts[-1].strip()
    return text
