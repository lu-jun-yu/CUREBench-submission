#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training Script for CUREBench

Full parameter training (no LoRA) with few-shot examples from valset.

Usage:
    python train_grpo.py
    python train_grpo.py --epochs 3 --batch_size 2
"""

import os
import re
import json
import argparse
import torch
import weave
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ================= Configuration =================

BASE_MODEL_PATH = "models/Qwen3-0.6B"
OUTPUT_DIR = "models/Qwen3-0.6B-GRPO-CUREBench"
DATASET_PATH = "curebench_valset_pharse1.jsonl"

# System prompt with few-shot examples from valset
SYSTEM_PROMPT = """You are a medical expert assistant specializing in pharmacology, clinical medicine, and drug safety.
Analyze the question carefully and reason step by step before providing your answer.

Format your response as:
<think>
Your step-by-step reasoning process here.
</think>
<answer>
The option letter (A, B, C, D, or E) here.
</answer>

Here are some examples:

Example 1:
Question: Which drug brand name is associated with the treatment of acne?
A: Salicylic Acid
B: Minoxidil
C: Ketoconazole
D: Fluocinonide

<think>
Let me analyze each option:
1. Salicylic Acid - This is a beta hydroxy acid commonly used in acne treatment. It exfoliates the skin and unclogs pores.
2. Minoxidil - Used for hair loss treatment, not acne.
3. Ketoconazole - An antifungal medication for fungal infections.
4. Fluocinonide - A corticosteroid for inflammatory skin conditions, not specifically for acne.
Salicylic Acid is the correct answer as it is specifically associated with acne treatment.
</think>
<answer>
A
</answer>

Example 2:
Question: What is the primary consideration for lactating mothers using Albuterol Sulfate HFA?
A: It is contraindicated during lactation.
B: Plasma levels of albuterol are low, and effects on breastfed children are likely minimal.
C: It significantly reduces milk production.
D: It should only be used in emergencies.

<think>
Analyzing Albuterol use in lactating mothers:
1. Albuterol is a short-acting beta-2 agonist bronchodilator
2. Studies show plasma levels after inhalation are relatively low
3. The amount transferred to breast milk is expected to be minimal
4. It is not contraindicated during lactation
5. No evidence it reduces milk production
The primary consideration is that plasma levels are low and effects on breastfed children are minimal.
</think>
<answer>
B
</answer>"""

# Few-shot example IDs to exclude from training (to avoid data leakage)
FEW_SHOT_IDS = ["U9PHZ83RKYV8", "wzkMQ7uHtlLs"]


# ================= Data Loading =================

def load_curebench_dataset(dataset_path: str, exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
    """Load CUREBench dataset from JSONL file, optionally excluding few-shot examples."""
    data = []
    exclude_ids = exclude_ids or []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item['id'] not in exclude_ids:
                    data.append(item)
    print(f"Loaded {len(data)} examples from {dataset_path} (excluded {len(exclude_ids)} few-shot examples)")
    return data


def format_question_from_dataset(item: Dict[str, Any]) -> str:
    """Format raw question from dataset (aligned with dataset_utils.py)."""
    question_type = item['question_type']
    question = item.get('question', '')
    options = item.get('options', {})

    if question_type == 'multi_choice':
        options_list = '\n'.join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
        return f"{question}\n{options_list}"
    elif question_type == 'open_ended_multi_choice':
        return question
    elif question_type == 'open_ended':
        return question
    else:
        options_list = '\n'.join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
        return f"{question}\n{options_list}"


def format_user_prompt(question: str, question_type: str) -> str:
    """Format user prompt (aligned with eval_framework.py)."""
    if question_type == "multi_choice":
        return f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:"
    elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
        return f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:"
    else:
        return f"Question: {question}\n\nAnswer:"


def prepare_dataset_for_grpo(data: List[Dict[str, Any]], tokenizer) -> Dataset:
    """Prepare dataset for GRPO training."""
    prompts = []
    ground_truths = []
    question_types = []

    for item in data:
        question_type = item['question_type']
        answer = item.get('correct_answer', item.get('answer', ''))

        formatted_question = format_question_from_dataset(item)
        user_prompt = format_user_prompt(formatted_question, question_type)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt += "<think>\n"

        prompts.append(prompt)
        ground_truths.append(answer)
        question_types.append(question_type)

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
        "question_type": question_types
    })

    return dataset


# ================= Reward Functions =================

def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response - only from <answer></answer> tags."""
    if not response:
        return ""

    response_text = response.strip()

    # Only match <answer>X</answer> format
    xml_match = re.search(r"<answer>\s*([A-E])\s*</answer>", response_text, re.IGNORECASE | re.DOTALL)
    if xml_match:
        return xml_match.group(1).upper()

    # If no valid <answer> tag found, return empty string
    return ""


def check_format_compliance(response: str) -> float:
    """Check format compliance with <think> and <answer> tags.

    Returns:
        float: Format compliance score (0-0.5)
            - 0.3 for valid <think>...</think> with content
            - 0.2 for valid <answer>X</answer> with A-E letter
    """
    score = 0.0

    # Check for valid <think> tag with meaningful content
    think_match = re.search(r"(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1).strip()
        if len(think_content) > 10:
            score += 0.3

    # Check for valid <answer> tag with A-E letter (strict format)
    answer_match = re.search(r"<answer>\s*([A-E])\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        score += 0.2

    return score


def reward_function(
    prompts: List[str],
    completions: List[str],
    ground_truths: List[str] = None,
    question_types: List[str] = None,
    **kwargs
) -> List[float]:
    """Reward function for GRPO training.

    Reward structure:
        - Format compliance (0-0.5): <think> and <answer> tags
        - Correctness (0-1.5): Only awarded if answer is extracted from <answer> tag
            - 1.5 for correct answer in <answer> tag
            - 0.0 for wrong answer or missing <answer> tag
            - -0.3 penalty for missing <answer> tag in multi_choice questions
        - Reasoning quality (0-0.3): Medical terminology and logical structure
        - Length penalty: -0.5 if output too short
    """
    rewards = []

    for i, completion in enumerate(completions):
        reward = 0.0

        gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        q_type = question_types[i] if question_types and i < len(question_types) else "multi_choice"

        # Format compliance (0-0.5)
        format_score = check_format_compliance(completion)
        reward += format_score

        # Correctness (only from <answer> tag)
        if gt and q_type in ['multi_choice', 'open_ended_multi_choice']:
            # Extract answer strictly from <answer> tag
            extracted_answer = extract_answer_from_response(completion)

            if extracted_answer:
                # Answer was properly formatted in <answer> tag
                if extracted_answer == gt.upper():
                    reward += 1.5  # Correct answer with proper format
                else:
                    reward += 0.0  # Wrong answer, no reward
            else:
                # No valid <answer> tag found - penalty for multi_choice questions
                reward -= 0.3

        elif q_type == 'open_ended':
            # For open-ended questions, check for meaningful content
            if len(completion.strip()) > 50:
                reward += 0.3

        # Reasoning quality (0-0.3)
        think_match = re.search(r"(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
        if think_match:
            reasoning = think_match.group(1).strip()

            # Check for medical terminology
            medical_terms = ['patient', 'treatment', 'drug', 'medication', 'symptom',
                           'diagnosis', 'dose', 'effect', 'clinical', 'adverse',
                           'contraindication', 'indication', 'allergy', 'reaction']
            term_count = sum(1 for term in medical_terms if term.lower() in reasoning.lower())
            if term_count >= 3:
                reward += 0.2
            elif term_count >= 1:
                reward += 0.1

            # Check for logical structure
            if any(phrase in reasoning.lower() for phrase in ['first', 'second', 'therefore', 'because', 'since', 'thus']):
                reward += 0.1

        # Length penalty for very short outputs
        if len(completion.strip()) < 20:
            reward -= 0.5

        # Clamp reward to [0, 2.5]
        reward = max(0.0, min(2.5, reward))
        rewards.append(reward)

    return rewards


class CUREBenchRewardWrapper:
    """Wrapper to provide reward function with ground truth access."""

    # TRL GRPOTrainer expects reward_funcs to have __name__ attribute
    __name__ = "curebench_reward"

    def __init__(self, dataset):
        self.prompt_to_info = {}
        for i in range(len(dataset)):
            prompt = dataset["prompt"][i]
            gt = dataset["ground_truth"][i]
            qt = dataset["question_type"][i]
            self.prompt_to_info[prompt] = (gt, qt)
        print(f"Initialized reward wrapper with {len(self.prompt_to_info)} prompts")

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        ground_truths = []
        question_types = []

        for prompt in prompts:
            if prompt in self.prompt_to_info:
                gt, qt = self.prompt_to_info[prompt]
            else:
                matched = False
                for stored_prompt, (gt, qt) in self.prompt_to_info.items():
                    if prompt[:100] == stored_prompt[:100]:
                        matched = True
                        break
                if not matched:
                    gt = ""
                    qt = "multi_choice"

            ground_truths.append(gt)
            question_types.append(qt)

        return reward_function(
            prompts=prompts,
            completions=completions,
            ground_truths=ground_truths,
            question_types=question_types,
            **kwargs
        )


# ================= Main Training Function =================

def main():
    parser = argparse.ArgumentParser(description="GRPO Full Parameter Training for CUREBench")
    parser.add_argument("--model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (24GB 3090)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--use_vllm", action="store_true", default=True, help="Use vLLM for inference acceleration")
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM (use HF generate)")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.35, help="vLLM GPU memory utilization (0-1)")
    args = parser.parse_args()

    # Handle vLLM flag
    if args.no_vllm:
        args.use_vllm = False

    print("="*60)
    print("GRPO Full Parameter Training for CUREBench")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: Full Parameter Training (optimized for 24GB 3090)")
    print(f"vLLM Acceleration: {'Enabled' if args.use_vllm else 'Disabled'}")
    print("="*60)

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # 清理显存
        torch.cuda.empty_cache()
    else:
        print("Warning: CUDA not available!")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for training
    # When using vLLM, we can pass model path directly to GRPOTrainer
    # vLLM will handle inference, HF model handles training
    if args.use_vllm:
        print("Using vLLM for inference - will load model in trainer...")
        model = args.model_path  # Pass path string, GRPOTrainer will handle loading
    else:
        print("Loading model for full parameter training...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset (exclude few-shot examples)
    print("\nLoading dataset...")
    raw_data = load_curebench_dataset(args.dataset_path, exclude_ids=FEW_SHOT_IDS)
    dataset = prepare_dataset_for_grpo(raw_data, tokenizer)
    print(f"Prepared {len(dataset)} examples for training")

    reward_wrapper = CUREBenchRewardWrapper(dataset)

    # Effective batch size = batch_size * gradient_accumulation * num_generations
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(f"\nEffective batch size: {effective_batch} (per step)")
    print(f"Num generations per prompt: {args.num_generations}")

    # GRPO Configuration
    print("\nConfiguring GRPO trainer...")
    if args.use_vllm:
        print("vLLM inference acceleration enabled (colocate mode for single GPU)")
        print(f"vLLM GPU memory utilization: {args.vllm_gpu_memory}")

    # Calculate max model length for vLLM
    vllm_max_model_len = args.max_prompt_length + args.max_completion_length

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        # vLLM acceleration settings
        use_vllm=args.use_vllm,
        vllm_mode="colocate",  # Share GPU between training and vLLM inference
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        vllm_max_model_length=vllm_max_model_len,
        # learning_rate, beta, temperature etc. use TRL defaults
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
    )

    # Create trainer
    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_wrapper,
    )

    # Train
    print("\n" + "="*60)
    print("Starting GRPO Full Parameter Training...")
    print("="*60)

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print("="*60)

    # Save config
    config_info = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "training_mode": "full_parameter",
        "use_vllm": args.use_vllm,
        "vllm_mode": "colocate" if args.use_vllm else None,
        "vllm_gpu_memory": args.vllm_gpu_memory if args.use_vllm else None,
        "few_shot_ids_excluded": FEW_SHOT_IDS,
        "note": "Using TRL default values for learning_rate, beta, temperature, etc."
    }

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {args.output_dir}/training_config.json")


if __name__ == "__main__":
    main()
