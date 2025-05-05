import os
import yaml
import json
import torch
import glob
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import load_dataset
import wandb
from pathlib import Path

# Disable MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
torch.backends.mps.enabled = False

def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load and validate training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_lora_config(config_path: str = "config/lora_config.json") -> dict:
    """Load and validate LoRA configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Extract checkpoint numbers and find the latest
    checkpoint_numbers = [int(cp.split("-")[-1]) for cp in checkpoints]
    latest_checkpoint_number = max(checkpoint_numbers)
    latest_checkpoint = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_number}")
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def prepare_model_and_tokenizer(config, checkpoint_path=None):
    """Initialize and prepare the base model and tokenizer."""
    print("Loading base model and tokenizer...")
    
    # Load base model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        device_map="cpu",  # Force CPU
        use_cache=False,  # Disable KV cache for training
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Add special tokens
    special_tokens = {
        "pad_token": " ",
        "eos_token": "",
        "bos_token": ""
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def prepare_lora_model(model, lora_config: dict, checkpoint_path=None):
    """Prepare model for LoRA fine-tuning."""
    print("Preparing model for LoRA training...")

    # Always initialize LoRA on the base model (do not attempt to load from Trainer checkpoint)
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        modules_to_save=lora_config["modules_to_save"]
    )
    model = get_peft_model(model, peft_config)

    model.train()
    for param in model.parameters():
        param.requires_grad = True
    model.print_trainable_parameters()
    return model

def format_chatml(example: dict) -> dict:
    """Format examples in ChatML format."""
    messages = example["messages"]
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_text += f"<|{role}|>\n{content}\n"
    formatted_text += " "
    return {"text": formatted_text}

def prepare_datasets(config: dict, tokenizer):
    """Prepare and process the datasets."""
    print("Preparing datasets...")
    
    # Define cache paths
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    train_cache_path = cache_dir / "train_cache"
    eval_cache_path = cache_dir / "eval_cache"
    
    # Check if cached datasets exist
    if train_cache_path.exists() and eval_cache_path.exists():
        print("Loading datasets from cache...")
        try:
            dataset = {
                "train": load_dataset("json", data_files=config["data"]["train_file"], cache_dir=str(train_cache_path)),
                "validation": load_dataset("json", data_files=config["data"]["eval_file"], cache_dir=str(eval_cache_path))
            }
            print("Successfully loaded from cache!")
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Falling back to processing datasets...")
            dataset = load_dataset("json", data_files={
                "train": config["data"]["train_file"],
                "validation": config["data"]["eval_file"]
            })
    else:
        # Load datasets
        dataset = load_dataset("json", data_files={
            "train": config["data"]["train_file"],
            "validation": config["data"]["eval_file"]
        })
    
    # Apply max_samples limit if specified
    if config["data"]["max_samples"] is not None:
        print(f"Limiting dataset to {config['data']['max_samples']} samples")
        dataset["train"] = dataset["train"].select(range(min(config["data"]["max_samples"], len(dataset["train"]))))
    
    # Format datasets
    dataset = dataset.map(format_chatml, remove_columns=["messages"])
    
    # Tokenize datasets with memory optimizations
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=config["model"]["padding"],
            truncation=config["model"]["truncation"],
            max_length=config["model"]["max_length"],
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        batch_size=8  # Process in smaller batches
    )
    
    return tokenized_dataset

def setup_training_args(config: dict, resume_from_checkpoint=None) -> TrainingArguments:
    """Setup training arguments with memory optimizations."""
    # Ensure output directory exists
    Path(config["output"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Get evaluation strategy from config
    eval_strategy = config["training"]["evaluation_strategy"]
    
    # Reduce batch sizes for memory efficiency
    train_batch_size = config["training"]["per_device_train_batch_size"]
    eval_batch_size = config["training"]["per_device_eval_batch_size"]
    
    return TrainingArguments(
        output_dir=config["output"]["output_dir"],
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy,
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        fp16=config["hardware"]["fp16"],
        bf16=config["hardware"]["bf16"],
        gradient_checkpointing=config["hardware"]["gradient_checkpointing"],
        optim=config["hardware"]["optim"],
        report_to="wandb",
        # Memory optimizations
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Additional memory optimizations
        max_grad_norm=0.5,
        optim_args="eps=1e-8",
        # Force CPU
        no_cuda=True,
        use_mps_device=False,
        # Resume from checkpoint
        resume_from_checkpoint=resume_from_checkpoint
    )

def get_latest_checkpoint(output_dir):
    """Helper function to find the latest checkpoint in the output directory."""
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(output_dir, checkpoints[-1])

def is_valid_checkpoint(checkpoint_path):
    """
    Checks if the checkpoint directory contains essential files for resuming training.
    """
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return False
    required_files = ["pytorch_model.bin", "trainer_state.json"]
    for fname in required_files:
        if not os.path.isfile(os.path.join(checkpoint_path, fname)):
            return False
    return True

def main():
    """Main training function."""
    try:
        # Load configurations
        config = load_config()
        lora_config = load_lora_config()
        
        # Initialize wandb
        run_name = f"lora-finetune-{config['model']['base_model'].split('/')[-1]}"
        wandb.init(
            project="raadhe-ai",
            config=config,
            name=run_name,
            resume="allow"
        )
        
        # Prepare model and tokenizer
        model, tokenizer = prepare_model_and_tokenizer(config)
        
        # Prepare LoRA model, loading from checkpoint if available
        model = prepare_lora_model(model, lora_config)
        
        # Move model to CPU
        device = torch.device("cpu")
        model = model.to(device)
        print("Using CPU device for training")
        
        # Prepare datasets
        tokenized_dataset = prepare_datasets(config, tokenizer)
        
        # Setup training arguments
        output_dir = config["output"]["output_dir"]
        latest_checkpoint = get_latest_checkpoint(output_dir)
        training_args = setup_training_args(config, latest_checkpoint)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator
        )
        
        # Only resume if checkpoint is valid
        if is_valid_checkpoint(latest_checkpoint):
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            if latest_checkpoint:
                print(f"Checkpoint {latest_checkpoint} is invalid or incomplete. Starting training from scratch.")
            else:
                print("No valid checkpoint found, starting from scratch.")
            trainer.train()
        
        # Save final model
        print("Saving model...")
        trainer.save_model()
        
        # Close wandb
        wandb.finish()
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.finish()
        raise e

if __name__ == "__main__":
    main()