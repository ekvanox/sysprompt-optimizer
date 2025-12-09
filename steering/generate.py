import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformer


def get_ground_truth_activations(
    text_batch: list[str], model: HookedTransformer, sae: SAE
) -> torch.Tensor:
    """
    Returns a sparse matrix of feature activations for the text.
    """
    _, cache = model.run_with_cache(text_batch)
    resid = cache[sae.cfg.metadata.hook_name][:, 1:, :]  # remove BOS token

    base_activations = torch.load("data/base_vec.pt")

    feature_acts = sae.encode(resid)  # shape: [batch, seq_len, d_sae]

    differential_acts = feature_acts - base_activations
    activations, _ = differential_acts.max(dim=1)  # [batch, d_sae]

    return F.relu(activations)


def generate_steered(
    model: HookedTransformer,
    sae: SAE,
    steering_data: tuple[torch.Tensor, torch.Tensor],
    prompt: str,
    strength_ratio: float = 0.1,
    max_new_tokens: int = 50,
    num_features: int = 5,
):
    """Injects the predicted features into the forward pass with strength relative to the norm of the residual stream."""

    topk_activations = torch.topk(steering_data, k=num_features)
    values, indices = topk_activations

    feature_directions = sae.W_dec[indices]  # [batch_size, num_features, d_model]
    weighted_direction = feature_directions * values.unsqueeze(-1)

    concept_vector = weighted_direction.sum(dim=1)  # [batch_size, d_model]

    concept_vector = concept_vector / concept_vector.norm(dim=-1, keepdim=True)

    def adaptive_hook(resid, hook):
        """Adapts the strength of injection based on the norm of the residual stream."""

        # TODO: If this is inefficient, pre-compute some "average norm" to normalize by
        current_norm = resid.norm(dim=-1).mean(dim=1)

        adaptive_strength = current_norm * strength_ratio

        steering_vec = concept_vector * adaptive_strength

        return resid + steering_vec

    with model.hooks(fwd_hooks=[(sae.cfg.metadata.hook_name, adaptive_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)

    return output
