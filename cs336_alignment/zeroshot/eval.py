import os
# 设置环境变量来解决CUDA兼容性问题
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from cs336_alignment.common import evaluate_vllm, load_prompt_template, load_test_set

import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Tuple
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn



def evaluate_zero_shot_math():
    """
    Evaluate a language model on math problems using zero-shot learning approach.
    
    This function performs the complete evaluation pipeline:
    1. Loads a prompt template for math problem solving
    2. Loads the GSM8K test dataset
    3. Generates prompts by formatting problems with the template
    4. Sets up sampling parameters for text generation
    5. Initializes a vLLM model (Qwen2.5-Math-1.5B)
    6. Runs evaluation using the reward function
    
    Configuration:
        - Model: Qwen/Qwen2.5-Math-1.5B
        - Temperature: 1.0 (high creativity)
        - Top-p: 1.0 (no nucleus sampling)
        - Max tokens: 1024
        - Stop sequence: ["</answer>"]
    
    Returns:
        None: Results are handled by evaluate_vllm function
    """
    prompt_template_name = "r1_zero"
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    temperature = 1.0
    top_p = 1.0
    max_tokens = 1024
    stop = ["</answer>"]

    # load prompt template
    prompt_template = load_prompt_template(prompt_template_name)

    # load test set
    test_set = load_test_set()
    qa_pairs = [(testcase["question"], testcase["answer"].split("####")[-1].strip()) for testcase in test_set]

    # generate prompts
    prompts = [prompt_template.format(question=testcase["question"]) for testcase in test_set]
    
    # load sampling params
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop
    ) 
    # initialize vllm model
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        max_model_len=2048,
        enforce_eager=True
    )
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, qa_pairs, sampling_params, "zeroshot_math")


if __name__ == "__main__":
    evaluate_zero_shot_math()