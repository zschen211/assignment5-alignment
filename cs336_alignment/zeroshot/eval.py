import os
from vllm import LLM, SamplingParams
from typing import Callable, List

def load_prompt_template(prompt_template_name: str) -> str:
    prompt_template_filepath = os.path.join(
        os.path.dirname(__file__), 
        os.path.pardir, 
        "prompts", 
        f"{prompt_template_name}.prompt")

    print(f"Loading prompt template from {prompt_template_filepath}")

    with open(prompt_template_filepath, "r") as f:
        prompt_template = f.read()

    return prompt_template


def evaluate_vllm(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    


if __name__ == "__main__":
    p = load_prompt_template("r1_zero")
    print(p)
    