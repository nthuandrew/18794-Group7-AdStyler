#!/usr/bin/env python3
"""
Fine-tune a language model to generate text_layout JSON from ad copy.

This module provides functions to fine-tune Qwen2.5 models with LoRA.
Uses Qwen2.5-1.5B or Qwen2.5-3B as base model with LoRA for efficient training.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import set_seed
import torch


class InstructionDataCollator:
    """
    Custom data collator for instruction-following fine-tuning.
    Only computes loss on the assistant's response (output) tokens,
    ignoring instruction and input tokens.
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features):
        # First, pad all sequences to the same length
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            return_tensors="pt",
            padding=True,
        )
        
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        
        # Use pre-computed assistant start positions if available
        for i, feature in enumerate(features):
            assistant_start_idx = feature.get("assistant_start_positions", 0)
            
            # Mask everything before assistant response (set to -100 to ignore in loss)
            if assistant_start_idx > 0 and assistant_start_idx < labels.shape[1]:
                labels[i, :assistant_start_idx] = -100
        
        # Also mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask"),
            "labels": labels,
        }


def format_prompt(instruction: str, input_text: str, output: str = None) -> str:
    """Format instruction-following prompt in Qwen chat format."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    
    # Qwen2.5 uses apply_chat_template
    return messages


def load_and_preprocess_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
) -> List[Dict]:
    """Load JSONL dataset and preprocess for training."""
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(examples)} examples")
    
    def tokenize_function(examples):
        # Format as chat messages
        formatted = []
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            messages = format_prompt(inst, inp, out)
            formatted.append(messages)
        
        # Build raw texts - we need to tokenize instruction+input and output separately
        # to identify where assistant response starts
        texts = []
        instruction_input_texts = []
        output_texts = []
        
        for messages in formatted:
            # Full text for tokenization
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(full_text)
            
            # Separate instruction+input and output for boundary detection
            instruction_input_msgs = messages[:-1]  # All except assistant response
            output_msg = messages[-1]["content"] if len(messages) > 0 and messages[-1].get("role") == "assistant" else ""
            
            instruction_input_text = tokenizer.apply_chat_template(
                instruction_input_msgs,
                tokenize=False,
                add_generation_prompt=True,  # Add prompt to mark where assistant should start
            )
            instruction_input_texts.append(instruction_input_text)
            output_texts.append(output_msg)

        # Tokenize full sequences
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Tokenize instruction+input to find boundary
        instruction_input_tokenized = tokenizer(
            instruction_input_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Calculate assistant start positions
        assistant_start_positions = []
        for i in range(len(texts)):
            # Find where assistant response starts by comparing tokenized lengths
            inst_input_len = len(instruction_input_tokenized["input_ids"][i])
            full_len = len(tokenized["input_ids"][i])
            # Assistant starts after instruction+input
            # Ensure it's within bounds
            assistant_start = min(inst_input_len, full_len)
            assistant_start_positions.append(assistant_start)

        # Do NOT set labels here; let custom data collator handle it.
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
            "assistant_start_positions": assistant_start_positions,  # Store for collator
        }
    
    # Convert to dataset format
    dataset_dict = {
        "instruction": [ex["instruction"] for ex in examples],
        "input": [ex["input"] for ex in examples],
        "output": [ex["output"] for ex in examples],
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def train(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    dataset_path: str = "sft_dataset.jsonl",
    output_dir: str = "output_layout_llm",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    seed: int = 42,
):
    """Fine-tune the model."""
    set_seed(seed)

    # Optional: set Hugging Face mirror for faster downloads
    # You can override this by exporting HF_ENDPOINT in your shell before running.
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    if not hf_endpoint:
        # Default mirror (you can change this to your preferred mirror)
        hf_endpoint = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = hf_endpoint
    print(f"Using Hugging Face endpoint: {hf_endpoint}")

    # Check if model_name is a local path
    model_path = Path(model_name)
    is_local_path = model_path.exists() and model_path.is_dir()
    
    if is_local_path:
        # Convert to absolute path for local models
        model_name = str(model_path.resolve())
        print(f"Detected local model path: {model_name}")
        local_files_only = True
    else:
        local_files_only = False

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if use_4bit and not cuda_available:
        print("Warning: 4-bit quantization requires CUDA, but CUDA is not available.")
        print("Disabling 4-bit quantization and using full precision on CPU.")
        use_4bit = False
    
    # Configure quantization if needed
    model_kwargs = {}
    if use_4bit and cuda_available:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            print("Using 4-bit quantization (CUDA available)")
        except Exception as e:
            print(f"Warning: Failed to configure 4-bit quantization: {e}")
            print("Falling back to full precision")
            use_4bit = False
    else:
        print("Using full precision (CPU mode or 4-bit disabled)")
    
    if not use_4bit:
        # For CPU/MPS, use appropriate dtype
        if torch.backends.mps.is_available():
            # MPS (Apple Silicon) - use float32 for stability
            model_kwargs["torch_dtype"] = torch.float32
            print("Using MPS (Apple Silicon) backend")
        elif cuda_available:
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # CPU fallback
            model_kwargs["torch_dtype"] = torch.float32
            print("Using CPU backend")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        **model_kwargs
    )
    
    # Prepare for LoRA
    if use_4bit and cuda_available:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and preprocess dataset
    tokenized_dataset = load_and_preprocess_dataset(
        dataset_path,
        tokenizer,
        max_length=max_length,
    )
    
    # Split train/val
    if len(tokenized_dataset) > 100:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    # Disable mixed precision on CPU/MPS
    use_fp16 = False
    use_bf16 = False
    if cuda_available:
        use_fp16 = not use_4bit
        use_bf16 = use_4bit or torch.cuda.is_bf16_supported()
    
    print(f"Training config: fp16={use_fp16}, bf16={use_bf16}, 4bit={use_4bit}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=500,
        eval_steps=500 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",  # Changed from evaluation_strategy
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        warmup_steps=100,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Use custom instruction data collator (only compute loss on assistant response)
    data_collator = InstructionDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Training complete! Model saved to {output_dir}")


# Module functions can be imported and used directly
# Example usage:
#   from finetune_layout_llm import train
#   train(
#       model_name="Qwen/Qwen2.5-1.5B",
#       dataset_path="sft_dataset.jsonl",
#       output_dir="output_layout_llm",
#       num_epochs=3,
#       batch_size=4,
#       learning_rate=2e-4
#   )

