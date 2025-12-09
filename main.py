from steering.generate import generate_steered, get_ground_truth_activations
from utils.loading import load_config, load_hooked_model

if __name__ == "__main__":
    config = load_config("config.yaml")
    model, sae = load_hooked_model(config)

    prompt = ["Talk about classical music."]
    steering_data = get_ground_truth_activations(prompt, model, sae)

    input_prompt = "What's important in life?"
    output = generate_steered(
        model,
        sae,
        steering_data,
        input_prompt,
        strength_ratio=0.4,
        max_new_tokens=50,
        num_features=5,
    )
    print(output)
