"""Utility functions for interacting with the Neuronpedia API."""

from typing import Optional

import requests

NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"


def get_feature_info(
    model_id: str,
    layer: str,
    index: int,
    api_key: Optional[str] = None,
) -> dict:
    """
    Get feature information from Neuronpedia API.

    Args:
        model_id: The model ID (e.g., "gemma-2b-it")
        layer: The SAE layer/source (e.g., "12-res-jb")
        index: The feature index
        api_key: Optional API key for authenticated requests

    Returns:
        Dictionary containing feature information including explanations
    """
    url = f"{NEURONPEDIA_BASE_URL}/api/feature/{model_id}/{layer}/{index}"

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


def get_feature_description(
    model_id: str,
    layer: str,
    index: int,
    api_key: Optional[str] = None,
) -> str:
    """
    Get the description/explanation for a feature.

    Args:
        model_id: The model ID (e.g., "gemma-2b-it")
        layer: The SAE layer/source (e.g., "12-res-jb")
        index: The feature index
        api_key: Optional API key for authenticated requests

    Returns:
        The feature description string, or a default message if not found
    """
    try:
        info = get_feature_info(model_id, layer, index, api_key)

        # Try to get explanation from the response
        explanations = info.get("explanations", [])
        if explanations:
            # Return the first (usually best) explanation
            return explanations[0].get("description", f"Feature {index}")

        return f"Feature {index} (no description available)"

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Feature {index} (not found on Neuronpedia)"
        raise


def get_feature_url(model_id: str, layer: str, index: int) -> str:
    """
    Get the Neuronpedia URL for a feature.

    Args:
        model_id: The model ID (e.g., "gemma-2b-it")
        layer: The SAE layer/source (e.g., "12-res-jb")
        index: The feature index

    Returns:
        The URL to view the feature on Neuronpedia
    """
    return f"{NEURONPEDIA_BASE_URL}/{model_id}/{layer}/{index}"


def lookup_features(
    model_id: str,
    layer: str,
    indices: list[int],
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Look up multiple features and return their info.

    Args:
        model_id: The model ID (e.g., "gemma-2b-it")
        layer: The SAE layer/source (e.g., "12-res-jb")
        indices: List of feature indices to look up
        api_key: Optional API key for authenticated requests

    Returns:
        List of dictionaries with index, description, and url for each feature
    """
    results = []
    for idx in indices:
        try:
            desc = get_feature_description(model_id, layer, idx, api_key)
            url = get_feature_url(model_id, layer, idx)
            results.append(
                {
                    "index": idx,
                    "description": desc,
                    "url": url,
                }
            )
        except Exception as e:
            results.append(
                {
                    "index": idx,
                    "description": f"Error: {e}",
                    "url": get_feature_url(model_id, layer, idx),
                }
            )
    return results


def format_feature_list(
    model_id: str,
    layer: str,
    indices: list[int],
    api_key: Optional[str] = None,
) -> str:
    """
    Format a list of features with their descriptions for display.

    Args:
        model_id: The model ID (e.g., "gemma-2b-it")
        layer: The SAE layer/source (e.g., "12-res-jb")
        indices: List of feature indices to look up
        api_key: Optional API key for authenticated requests

    Returns:
        Formatted string with feature information
    """
    features = lookup_features(model_id, layer, indices, api_key)

    lines = []
    for f in features:
        lines.append(f"  [{f['index']}] {f['description']}")
        lines.append(f"       {f['url']}")

    return "\n".join(lines)
