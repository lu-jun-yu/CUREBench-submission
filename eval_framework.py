"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass
    
    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model
        
        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""
    
    def load(self, **kwargs):
        """Load ChatGPT model"""


        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview" #"2025-03-01-preview"

        if not api_key:
            raise ValueError(f"API key not found in environment. Please set the appropriate environment variable.")
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        from openai import AzureOpenAI
        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """ChatGPT inference"""
        messages = [{"role": "user", "content": prompt}]
        
        responses = self.model_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=8192,
            )
        # print("\033[94m" + str(responses) + "\033[0m")
        response = responses.choices[0].message.content
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response}]
        
        return response, complete_messages


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""
    
    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kwargs
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        system_prompt = """You are a medical expert assistant specializing in pharmacology, clinical medicine, and drug safety.
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # print("messages:", messages) # Debug用
        
        # 2. 预处理 Prompt，并强制添加 <think> 引导
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        # 手动追加 <think> 标签，触发模型的思维链
        input_text += "<think>\n"
        
        input_ids = self.tokenizer(input_text, return_tensors='pt').to(self.model.device).input_ids
        
        outputs = self.model.generate(
            input_ids,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True # 建议开启采样以获得多样化的推理，或者 False 追求稳定
        )
        
        # 解码，注意这里要包含生成的全部内容
        response_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print("response: ", response_text)
        
        # 补全 response，因为我们手动加了 <think>，生成的内容不包含开头的 <think>
        # 为了后续解析方便，我们把它补回去
        full_response = "<think>\n" + response_text
        
        # print("response_text:", full_response)
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": full_response}]
        
        return full_response, complete_messages


class VLLMModel(BaseModel):
    """vLLM model wrapper for accelerated inference"""

    def load(self, **kwargs):
        """Load model using vLLM for accelerated inference"""
        try:
            import os
            import warnings

            # Suppress vLLM verbose output
            os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
            os.environ["VLLM_NO_USAGE_STATS"] = "1"

            # Suppress other verbose loggers
            import logging as std_logging
            std_logging.getLogger("vllm").setLevel(std_logging.ERROR)
            std_logging.getLogger("ray").setLevel(std_logging.ERROR)

            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            # Extract vLLM-specific parameters
            tensor_parallel_size = kwargs.pop('tensor_parallel_size', 1)
            gpu_memory_utilization = kwargs.pop('gpu_memory_utilization', 0.9)
            max_model_len = kwargs.pop('max_model_len', 4096)
            dtype = kwargs.pop('dtype', 'bfloat16')

            # Load tokenizer for chat template
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Store system prompt for batch inference
            self.system_prompt = """You are a medical expert assistant specializing in pharmacology, clinical medicine, and drug safety.
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

            logger.info(f"Loading vLLM model: {self.model_name}")

            # Initialize vLLM engine with suppressed output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = LLM(
                    model=self.model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    dtype=dtype,
                    trust_remote_code=True,
                    disable_log_stats=True,
                    **kwargs
                )

            logger.info(f"Loaded vLLM model: {self.model_name}")
            logger.info(f"  - tensor_parallel_size: {tensor_parallel_size}")
            logger.info(f"  - gpu_memory_utilization: {gpu_memory_utilization}")
            logger.info(f"  - max_model_len: {max_model_len}")
            logger.info(f"  - dtype: {dtype}")

        except ImportError as e:
            logger.error(f"Failed to import vLLM. Please install with: pip install vllm")
            raise ImportError("vLLM is required for VLLMModel. Install with: pip install vllm") from e

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """vLLM accelerated inference (single prompt)"""
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        # Manually append <think> tag to trigger chain-of-thought
        input_text += "<think>\n"

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
        )

        # Generate using vLLM
        outputs = self.model.generate([input_text], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text

        # Prepend <think> since we manually added it to the prompt
        full_response = "<think>\n" + response_text

        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": full_response}]

        return full_response, complete_messages

    def batch_inference(self, prompts: List[str], max_tokens: int = 1024) -> List[Tuple[str, List[Dict]]]:
        """
        Batch inference for multiple prompts - vLLM's key advantage

        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens to generate per response

        Returns:
            List of (response, messages) tuples
        """
        from vllm import SamplingParams

        # Prepare all inputs
        input_texts = []
        all_messages = []

        for prompt in prompts:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            all_messages.append(messages)

            input_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            input_text += "<think>\n"
            input_texts.append(input_text)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
        )

        # Batch generate using vLLM (disable internal tqdm)
        outputs = self.model.generate(input_texts, sampling_params, use_tqdm=False)

        # Process results
        results = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            full_response = "<think>\n" + response_text
            complete_messages = all_messages[i] + [{"role": "assistant", "content": full_response}]
            results.append((full_response, complete_messages))

        return results


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""

    def __init__(self, model_name: str, model_instance, inference_func):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func

    def load(self, **kwargs):
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]

            response = self._inference_func(self.model, prompt, max_tokens)

            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]

            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages


class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit
        
        Args:
            output_dir: Directory to save results and submissions
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None
        
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        
        self.output_dir = self.config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)
    
    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation

        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b-chat-hf")
            model_type: Type of model ("chatgpt", "local", "vllm", "custom", "auto" for auto-detection)
            **kwargs: Additional model configuration
                For vLLM models:
                    - tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
                    - gpu_memory_utilization: GPU memory utilization ratio (default: 0.9)
                    - max_model_len: Maximum model sequence length (default: 4096)
                    - dtype: Model dtype (default: "bfloat16")
        """
        self.model_name = model_name

        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)

        logger.info(f"Loading model: {model_name} (type: {model_type})")

        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "vllm":
            self.model = VLLMModel(model_name)
        elif model_type == "custom":
            # For custom models, user should provide model_instance and inference_func
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: chatgpt, local, vllm, custom, auto")

        # Load the model
        self.model.load(**kwargs)
    
    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults
        
        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("Not config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("Not config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        else:
            return "local"
    
    def evaluate(self, dataset_name: str, batch_size: int = 16) -> EvaluationResult:
        """
        Evaluate model on a dataset

        Args:
            dataset_name: Name of dataset to evaluate on
            batch_size: Batch size for vLLM batch inference (default: 16)

        Returns:
            EvaluationResult object with scores and predictions
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")

        # Load dataset
        dataset = self._load_dataset(dataset_config)

        # Store dataset examples for later use in save_submission
        self._last_dataset_examples = dataset

        total_count = len(dataset)
        logger.info(f"Running evaluation on {total_count} examples...")

        # Check if model supports batch inference (VLLMModel)
        if isinstance(self.model, VLLMModel):
            predictions, reasoning_traces = self._evaluate_batch(dataset, batch_size)
        else:
            predictions, reasoning_traces = self._evaluate_sequential(dataset)

        # Calculate accuracy (excluding open-ended questions)
        accuracy_correct_count = 0
        accuracy_total_count = 0

        for i, (prediction, example) in enumerate(zip(predictions, dataset)):
            question_type = example["question_type"]
            expected_answer = example.get("answer")

            if question_type == "multi_choice" or question_type == "open_ended_multi_choice":
                accuracy_total_count += 1
                if expected_answer != '' and prediction["choice"] == expected_answer:
                    accuracy_correct_count += 1

        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces
        )

        logger.info(f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions")
        logger.info(f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)")

        return result

    def _evaluate_batch(self, dataset: List[Dict], batch_size: int = 16) -> Tuple[List[Dict], List[str]]:
        """
        Batch evaluation using vLLM - significantly faster for large datasets

        Args:
            dataset: List of examples to evaluate
            batch_size: Number of examples to process in each batch

        Returns:
            Tuple of (predictions, reasoning_traces)
        """
        predictions = []
        reasoning_traces = []
        total_count = len(dataset)

        # Process in batches with progress bar
        for batch_start in tqdm(range(0, total_count, batch_size), desc="Evaluating (batch)"):
            batch_end = min(batch_start + batch_size, total_count)
            batch_examples = dataset[batch_start:batch_end]

            # Prepare prompts for batch
            prompts = []
            for example in batch_examples:
                question = example["question"]
                question_type = example["question_type"]

                if question_type == "multi_choice":
                    prompt = f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:"
                else:
                    prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:"

                prompts.append(prompt)

            # Batch inference
            try:
                batch_results = self.model.batch_inference(prompts)

                # Process batch results
                for j, (response, full_messages) in enumerate(batch_results):
                    example = batch_examples[j]
                    prediction, reasoning_trace = self._process_response(response, example)
                    predictions.append(prediction)
                    reasoning_traces.append(reasoning_trace)

            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Fall back to error predictions for this batch
                for example in batch_examples:
                    predictions.append({
                        "choice": "NOTAVALUE",
                        "open_ended_answer": "Error"
                    })
                    reasoning_traces.append("Error occurred during batch inference")

        return predictions, reasoning_traces

    def _evaluate_sequential(self, dataset: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Sequential evaluation for non-vLLM models

        Args:
            dataset: List of examples to evaluate

        Returns:
            Tuple of (predictions, reasoning_traces)
        """
        predictions = []
        reasoning_traces = []

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                predictions.append({
                    "choice": "NOTAVALUE",
                    "open_ended_answer": "Error"
                })
                reasoning_traces.append("Error occurred during inference")

        return predictions, reasoning_traces

    def _process_response(self, response: str, example: Dict) -> Tuple[Dict, str]:
        """
        Process model response to extract prediction and reasoning trace

        Args:
            response: Model response text
            example: Original example dict

        Returns:
            Tuple of (prediction_dict, reasoning_trace)
        """
        import re

        question_type = example["question_type"]

        # Extract reasoning trace
        reasoning_content = ""
        think_match = re.search(r"(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
        if think_match:
            reasoning_content = think_match.group(1).strip()
        else:
            if "<think>" in response:
                reasoning_content = response.split("<think>")[-1].strip()
            else:
                reasoning_content = response

        # Initialize prediction
        prediction = {
            "choice": "",
            "open_ended_answer": ""
        }

        # Extract answer based on question type
        if question_type == "multi_choice":
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice else ""
            prediction["open_ended_answer"] = response.strip()

        elif question_type == "open_ended_multi_choice":
            prediction["open_ended_answer"] = response.strip()
            if "meta_question" in example:
                # Need additional inference for meta question - fall back to direct extraction
                choice = self._extract_multiple_choice_answer(response)
                prediction["choice"] = choice if choice else ""
            else:
                choice = self._extract_multiple_choice_answer(response)
                prediction["choice"] = choice if choice else ""

        elif question_type == "open_ended":
            prediction["choice"] = "NOTAVALUE"
            prediction["open_ended_answer"] = response.strip()

        return prediction, reasoning_content
    
    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader
        
        # Build dataset
        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )
        
        # Convert to list of dictionaries for easier processing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []
        
        for batch in dataloader:
            question_type = batch[0][0]
            
            if question_type == "multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
            elif question_type == "open_ended_multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                    "meta_question": batch[4][0],
                })
            elif question_type == "open_ended":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
        
        return dataset_list

    
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]
        
        # Format prompt
        if question_type == "multi_choice":
            prompt = f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:"
        elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
            prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:"
        
        # Get model response and messages using the model's inference method
        response, full_messages = self.model.inference(prompt)

        import re
        reasoning_content = ""
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
        if think_match:
            reasoning_content = think_match.group(1).strip()
        else:
            # 如果没有闭合标签，可能生成中断了，取 <think> 之后的所有内容
            if "<think>" in response:
                reasoning_content = response.split("<think>")[-1].strip()
            else:
                reasoning_content = response

        reasoning_trace = reasoning_content
        
        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Then use meta question to get choice, if available
            if "meta_question" in example:
                meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        return prediction, reasoning_trace
    
    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response - only from <answer></answer> tags"""
        if not response or response is None:
            return ""

        response = response.strip()

        import re
        # Only match <answer>X</answer> format
        xml_match = re.search(r"<answer>\s*([A-E])\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        if xml_match:
            return xml_match.group(1).upper()

        # If no valid <answer> tag found, return empty string
        return ""
    
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        
        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile
        
        # Get metadata from various sources with priority order
        metadata = self.get_metadata(config_path, args, metadata)
        
        # Create submission data for CSV
        submission_data = []
        
        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                # if result.reasoning_traces and i < len(result.reasoning_traces):
                #     trace = result.reasoning_traces[i]
                #     if isinstance(trace, list) and len(trace) > 0:
                #         # Convert list of messages to a simple text format
                #         text_parts = []
                #         for msg in trace:
                #             if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                #                 role = msg['role']
                #                 content = msg['content'].replace('\n', ' ').replace('\r', '').replace('"', "'")
                #                 text_parts.append(f"{role}: {content}")
                #         reasoning_trace = " | ".join(text_parts)
                #     else:
                #         # Fallback to string representation
                #         reasoning_trace = str(trace).replace('\n', ' ').replace('\r', '').replace('"', "'")
                
                # Clean up text fields to avoid CSV formatting issues
                prediction_text = prediction.get("open_ended_answer", "") or ""  # Ensure not None
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                
                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
                else:
                    choice_clean = str(choice_raw).strip()
                
                # Ensure reasoning trace is not null
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                # Create CSV row - let pandas handle the escaping
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                # Debug: Log if choice is NULL-like
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE")
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        # Create DataFrame and save CSV with proper quoting and NaN handling
        df = pd.DataFrame(submission_data)
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Aggressive null value cleaning
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',  # Use NOTAVALUE for choice instead of empty string
            'reasoning': 'No reasoning available'
        }
        
        # Replace all possible null-like values
        for col in df.columns:
            # Replace pandas null values
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            
            # Replace string representations of null
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            
            # Special handling for choice column - ensure it's never empty or null-like
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')  # Keep NOTAVALUE as is for choice
                # Replace any null-like values with NOTAVALUE
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                # Replace empty strings with NOTAVALUE for choice column
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')  # Also replace whitespace-only
            
            # Replace empty strings (except for choice column which can be empty)
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])  # Also replace whitespace-only
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Validate DataFrame before saving
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Final validation - check for any remaining nulls
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")
        
        # Check for any problematic data
        for idx, row in df.head().iterrows():
            logger.debug(f"Sample row {idx}: id={row['id']}, choice='{row['choice']}', prediction_len={len(str(row['prediction']))}, reasoning_len={len(str(row['reasoning']))}")
        
        # Final safety check: ensure choice column has no NULL values or empty strings
        logger.info("Performing final NULL check on choice column...")
        null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
        for pattern in null_patterns:
            count_before = (df['choice'] == pattern).sum()
            if count_before > 0:
                logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')
        
        # Replace empty strings with NOTAVALUE to avoid NULL validation issues
        empty_count = (df['choice'] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
            df['choice'] = df['choice'].replace('', 'NOTAVALUE')
        
        # Also replace any remaining pandas nulls in choice column
        null_mask = df['choice'].isnull()
        if null_mask.sum() > 0:
            logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
            df.loc[null_mask, 'choice'] = 'NOTAVALUE'
        

        # Use proper CSV parameters for robust handling of complex data
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)  # index=False to avoid pandas index issues
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP file with CSV and metadata
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(csv_path, filename)
            # Add metadata JSON to zip
            zipf.write(metadata_path, metadata_filename)
        
        # Calculate and log overall accuracy
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")
        
        return zip_path
    
    def save_submission_with_metadata(self, results: List[EvaluationResult], 
                                     metadata: Dict = None, filename: str = "submission.csv",
                                     config_path: str = None, args: argparse.Namespace = None):
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package
        
        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used  
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # Use the stored dataset examples from the last evaluation
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Metadata dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
        
        # Extract metadata from config
        metadata = config.get('metadata', config.get('meta_data', {}))
        
        # Validate required fields
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")
        
        return metadata
    
    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Map argument names to metadata fields
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }
        
        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)
        
        return metadata
    
    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, 
                    fallback_metadata: Dict = None) -> Dict:
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)
        
        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary
            
        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }
        
        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)
        
        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")
        
        return metadata

def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata
    
    Returns:
        ArgumentParser with metadata-related arguments
    """
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')
    
    # Model information
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'], 
                       help='Type of base model (API or OpenWeighted)')
    
    # Track information
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')
    
    # Dataset and submission info
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='competition_results', 
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv', 
                       help='Output CSV filename for submission (will be packaged in zip)')
    
    # Evaluation settings
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')
    
    return parser


def load_config_file(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args
    
    config = load_config_file(args.config)
    
    # First, handle the metadata section specially - merge its contents directly
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)
    
    add_config_to_args(config)
    return args
