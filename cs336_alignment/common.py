import json
import os
from pathlib import Path
from typing import Callable, List, Tuple
from vllm import LLM, SamplingParams


# project root directory
current_file = Path(__file__)
project_root = current_file.parent.parent

# data directory
data_dir = project_root / "data" / "gsm8k"

# prompts directory
prompts_dir = project_root / "cs336_alignment" / "prompts"

# evaluation results directory
evaluation_results_dir = project_root / "evaluation_results"


def evaluate_vllm(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    qa_pairs: List[Tuple[str, str]],
    eval_sampling_params: SamplingParams,
    evalres_filename: str) -> None:
    """
    Evaluate a language model on a list of prompts using vLLM,
    compute evaluation metrics, and serialize results to disk.
    
    Args:
        vllm_model (LLM): Initialized vLLM language model instance
        reward_fn (Callable[[str, str], dict[str, float]]): Reward function that takes 
            prompt and generated text, returns reward scores
        prompts (List[str]): List of prompts to evaluate
        eval_sampling_params (SamplingParams): Generation parameters including 
            temperature, top_p, max_tokens, etc.
    
    Returns:
        None: Function doesn't return value, results are printed to console
        
    Note:
        Current version only prints results. Evaluation metrics computation 
        and result saving functionality are TODO items.
    """

    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    # compute evaluation metrics
    evaluation_results = {}
    evaluation_results["details"] = []
    for output, (question, ground_truth) in zip(outputs, qa_pairs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward_body = reward_fn(generated_text, ground_truth)
        format_reward = reward_body["format_reward"]
        answer_reward = reward_body["answer_reward"]
        reward = reward_body["reward"]

        evaluation_results["details"].append({
            "question": question,
            "generated_answer": generated_text,
            "ground_truth": ground_truth,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "reward": reward
        })

    # calculate overall answer correctness rate and answer format correctness rate
    overall_answer_correctness = 0
    overall_answer_format_correctness = 0
    for detail in evaluation_results["details"]:
        if detail["answer_reward"] > 0:
            overall_answer_correctness += 1
        if detail["format_reward"] > 0:
            overall_answer_format_correctness += 1
    evaluation_results["answer_correctness_rate"] = overall_answer_correctness / len(evaluation_results["details"])
    evaluation_results["answer_format_correctness_rate"] = overall_answer_format_correctness / len(evaluation_results["details"])
    evaluation_results["total_reward"] = sum(detail["reward"] for detail in evaluation_results["details"])
    evaluation_results["total_answer_reward"] = sum(detail["answer_reward"] for detail in evaluation_results["details"])
    evaluation_results["total_format_reward"] = sum(detail["format_reward"] for detail in evaluation_results["details"])
    evaluation_results["question_count"] = len(qa_pairs)


    # save evaluation results to disk file
    os.makedirs(evaluation_results_dir, exist_ok=True)
    with open(evaluation_results_dir / f"{evalres_filename}.json", "w") as f:
        json.dump(evaluation_results, f)


def load_prompt_template(prompt_template_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        prompt_template_name (str): Name of the prompt template file 
            (without .prompt extension)
    
    Returns:
        str: Content of the prompt template file
    """
    prompt_template_filepath = prompts_dir / f"{prompt_template_name}.prompt"

    with open(prompt_template_filepath, "r") as f:
        prompt_template = f.read()

    return prompt_template


def load_test_set() -> List[dict]:
    """
    Load the GSM8K test dataset from JSONL format.
    
    This function reads the test set file located in the data/gsm8k directory
    and parses each line as a JSON object. The test set contains math problems
    for evaluation.
    
    Returns:
        List[dict]: List of test problems, where each problem is a dictionary
            containing problem text and potentially other metadata
    """
    test_set_path = data_dir / "test.jsonl"

    with open(test_set_path, "r") as f:
        test_set = [json.loads(line) for line in f]

    return test_set