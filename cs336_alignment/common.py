import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import json
import torch
from pathlib import Path
from typing import Callable, List, Tuple
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax


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


def load_train_set() -> List[dict]:
    """
    Load the GSM8K train dataset from JSONL format.
    """
    train_set_path = data_dir / "train.jsonl"

    with open(train_set_path, "r") as f:
        train_set = [json.loads(line) for line in f]

    return train_set


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


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    input_ids = []
    response_mask = []

    prompt_inputs = tokenizer(prompt_strs, add_special_tokens=False)
    output_inputs = tokenizer(output_strs, add_special_tokens=False)

    for i in range(len(prompt_strs)):
        input_ids.append(
            torch.tensor(
                prompt_inputs["input_ids"][i] + 
                output_inputs["input_ids"][i]
            )
        )
        response_mask.append(
            torch.tensor(
                [0] * len(prompt_inputs["input_ids"][i]) + 
                [1] * len(output_inputs["attention_mask"][i])
            )
        )
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    response_mask = pad_sequence(response_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "response_mask": response_mask[:, 1:] > 0,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""

    log_sum_exp = torch.logsumexp(logits, dim=2, keepdim=True)
    log_probs = logits - log_sum_exp
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=2)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """

    logits = model(input_ids).logits
    probs = softmax(logits, dim=2)
    unsqueezed_labels = labels.unsqueeze(-1)
    selected_probs = probs.gather(dim=-1, index=unsqueezed_labels)

    log_probs = torch.log(selected_probs).squeeze(-1)
    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy
    
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask.float()
    tensor_sum = torch.sum(masked_tensor, dim=dim)
    return tensor_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size, _ = policy_log_probs.shape
    microbatch_loss = -masked_normalize(
        policy_log_probs, 
        response_mask, dim=-1, 
        normalize_constant=normalize_constant).sum() / gradient_accumulation_steps / batch_size

    microbatch_loss.backward()

    return microbatch_loss, {}


def log_generation(
    input_prompt: str,
    resp_generated: str,
    resp_ground_truth: str,
    format_reward: float,
    answer_reward: float,
    avg_token_entropy: float,
    avg_resp_length: float,
    avg_resp_length_correct: float,
    avg_resp_length_incorrect: float
):
    pass