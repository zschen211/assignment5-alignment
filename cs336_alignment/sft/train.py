import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from cs336_alignment import common


class Gsm8kDataset(Dataset):
    def __init__(self, data_list: List[str]):
        self.data = []

        for data in data_list:
            answer = data["answer"]
            think_part, ans_part = answer.split("####")
            think_part = think_part.strip()
            ans_part = ans_part.strip()
            think_part = "<think>" + think_part + "</think>"
            ans_part = "<answer>" + ans_part + "</answer>"
            self.data.append((data["question"], think_part + " " + ans_part))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        return question, answer


device = "cuda" if torch.cuda.is_available() else "cpu"
gradient_accumulation_steps = 4

dataset = Gsm8kDataset(common.load_train_set())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model_name = "Qwen/Qwen2.5-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
) 
model = model.to(device)  # 将模型移动到指定设备
tokenizer = AutoTokenizer.from_pretrained(model_name)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for i, (questions, answers) in enumerate(dataloader):
    tokenization_output = common.tokenize_prompt_and_output(questions, answers, tokenizer)
    input_ids = tokenization_output["input_ids"]
    labels = tokenization_output["labels"]
    response_mask = tokenization_output["response_mask"]

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)

    response_log_probs_output = common.get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
    policy_log_probs = response_log_probs_output["log_probs"]
    policy_log_probs = policy_log_probs.to(device)

    microbatch_loss, _ = common.sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps)
    print(microbatch_loss)
    
    break









