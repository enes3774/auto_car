"""
Chess MAV Model - Training Script
Full fine-tuning Qwen3 1.7B with HL-Gauss loss for value prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import os
from hl_gauss_pytorch import HLGaussLoss

from data_preparation import (
    ChessTokenizer,
    ChessMAVDataset,
    ChessMAVIterableDataset,
    chess_collate_fn,
    NUM_VALUE_BUCKETS
)


# ============================================================================
# CUSTOM LOSS COMPUTATION
# ============================================================================

class ChessMAVLoss(nn.Module):
    """
    Combined loss for Chess MAV model:
    - Standard Cross-Entropy for FEN tokens and move tokens
    - HL-Gauss for value bucket tokens
    """

    def __init__(
            self,
            value_token_start: int,
            value_token_end: int,
            num_buckets: int = 64,
            sigma_to_bin_ratio: float = 2.0,
            value_loss_weight: float = 1.0,
    ):
        """
        Args:
            value_token_start: Start ID of value tokens in vocabulary
            value_token_end: End ID of value tokens (exclusive)
            num_buckets: Number of value buckets
            sigma: Gaussian sigma for HL-Gauss
            value_loss_weight: Weight for value loss relative to CE loss
        """
        super().__init__()
        self.value_token_start = value_token_start
        self.value_token_end = value_token_end
        self.num_buckets = num_buckets
        self.value_loss_weight = value_loss_weight

        self.hl_gauss = HLGaussLoss(min_value=0.0, max_value=1.0, num_bins=num_buckets,
                                    sigma_to_bin_ratio=sigma_to_bin_ratio)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            value_token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            labels: Target labels (batch, seq_len)
            value_token_mask: Boolean mask for value tokens (batch, seq_len)
        Returns:
            (total_loss, loss_dict) where loss_dict contains individual losses
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Create value token mask from labels if not provided
        if value_token_mask is None:
            value_token_mask = (
                    (shift_labels >= self.value_token_start) &
                    (shift_labels < self.value_token_end)
            )
        else:
            value_token_mask = value_token_mask[..., 1:].contiguous()

        # Valid positions (not padding, not ignored)
        valid_mask = shift_labels != -100

        # Separate masks
        value_positions = value_token_mask & valid_mask
        non_value_positions = ~value_token_mask & valid_mask

        # Flatten for loss computation
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_value_mask = value_positions.view(-1)
        flat_non_value_mask = non_value_positions.view(-1)

        loss_dict = {}

        # 1. Standard CE loss for non-value tokens (FEN, moves, format tokens)
        if flat_non_value_mask.sum() > 0:
            ce_logits = flat_logits[flat_non_value_mask]
            ce_labels = flat_labels[flat_non_value_mask]
            ce_loss = self.ce_loss(ce_logits, ce_labels)
            loss_dict['ce_loss'] = ce_loss.item()
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)
            loss_dict['ce_loss'] = 0.0

        # 2. HL-Gauss loss for value tokens
        if flat_value_mask.sum() > 0:
            # Extract only the value token logits (64 buckets)
            value_logits_full = flat_logits[flat_value_mask]
            value_logits = value_logits_full[:, self.value_token_start:self.value_token_end]

            # Convert labels to bucket indices (0-6)
            value_labels = flat_labels[flat_value_mask] - self.value_token_start

            # Compute HL-Gauss loss
            hl_loss = self.hl_gauss(value_logits, value_labels)
            loss_dict['hl_gauss_loss'] = hl_loss.item()
        else:
            hl_loss = torch.tensor(0.0, device=logits.device)
            loss_dict['hl_gauss_loss'] = 0.0

        # Combined loss
        total_loss = ce_loss + self.value_loss_weight * hl_loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


# ============================================================================
# CUSTOM TRAINER
# ============================================================================

class ChessMAVTrainer(Trainer):
    """
    Custom Trainer with HL-Gauss loss for value tokens
    """

    def __init__(
            self,
            *args,
            value_token_start: int,
            value_token_end: int,
            sigma_to_bin_ratio: float,
            value_loss_weight: float = 1.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.mav_loss = ChessMAVLoss(
            value_token_start=value_token_start,
            value_token_end=value_token_end,
            num_buckets=NUM_VALUE_BUCKETS,
            sigma_to_bin_ratio=sigma_to_bin_ratio,
            value_loss_weight=value_loss_weight
        )
        self.mav_loss = self.mav_loss.to(self.model.device)

        # Track running losses for logging
        self.running_losses = {'ce_loss': 0.0, 'hl_gauss_loss': 0.0, 'total_loss': 0.0}
        self.loss_count = 0

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: Optional[int] = None
    ):
        """
        Override compute_loss to use custom HL-Gauss loss
        """
        labels = inputs.pop("labels")
        value_token_mask = inputs.pop("value_token_mask", None)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute custom loss
        loss, loss_dict = self.mav_loss(logits, labels, value_token_mask)

        # Update running losses for logging
        for key, value in loss_dict.items():
            self.running_losses[key] += value
        self.loss_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """Override log to include component losses"""
        if self.loss_count > 0:
            logs['ce_loss'] = self.running_losses['ce_loss'] / self.loss_count
            logs['hl_gauss_loss'] = self.running_losses['hl_gauss_loss'] / self.loss_count

            # Reset running losses
            self.running_losses = {'ce_loss': 0.0, 'hl_gauss_loss': 0.0, 'total_loss': 0.0}
            self.loss_count = 0

        # Call parent with start_time if provided
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(
        model_name: str = "Qwen/Qwen-1.7B",
        uci_moves_file: str = None,
        use_flash_attention: bool = True
) -> Tuple[AutoModelForCausalLM, ChessTokenizer]:
    """
    Load model and tokenizer, add chess tokens, resize embeddings
    Args:
        model_name: HuggingFace model name
        uci_moves_file: Path to JSON file with UCI moves
        use_flash_attention: Whether to use Flash Attention 2
    """
    print(f"Loading model: {model_name}")

    # Initialize chess tokenizer (adds special tokens)
    chess_tokenizer = ChessTokenizer(
        model_name,
        uci_moves_file=uci_moves_file
    )

    # Load model with optimizations
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Resize token embeddings for new tokens
    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(chess_tokenizer)

    print(f"Resizing embeddings: {original_vocab_size} -> {new_vocab_size}")
    model.resize_token_embeddings(new_vocab_size)

    # Initialize new embeddings with mean of existing embeddings
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight

        # Compute mean and std of original embeddings
        orig_mean = embeddings[:original_vocab_size].mean(dim=0)
        orig_std = embeddings[:original_vocab_size].std(dim=0)

        # Initialize new tokens with similar distribution
        num_new_tokens = new_vocab_size - original_vocab_size
        new_embeds = torch.randn(num_new_tokens, embeddings.size(1)) * orig_std + orig_mean
        embeddings[original_vocab_size:] = new_embeds.to(embeddings.dtype).to(embeddings.device)

        # Also update lm_head if tied weights
        if hasattr(model, 'lm_head') and model.lm_head.weight is not embeddings:
            lm_head = model.lm_head.weight
            lm_head[original_vocab_size:] = new_embeds.to(lm_head.dtype).to(lm_head.device)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    print(f"Model ready with {model.num_parameters():,} parameters")

    return model, chess_tokenizer


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class ChessTrainingConfig:
    """Training configuration for Chess MAV model"""

    # Paths
    data_path: str = "chess_data.jsonl"
    uci_moves_file: str = "uci_moves.json"  # JSON array of UCI move strings
    output_dir: str = "./chess_mav_output"

    # Model
    model_name: str = "Qwen/Qwen-1.7B"
    use_flash_attention: bool = True

    # Data loading
    streaming: bool = True  # Use streaming for large datasets (recommended)
    max_samples: int = None  # Limit samples for testing (None = all)
    eval_data_path: str = None  # Path to eval data (optional)
    eval_split: float = 0.0  # If > 0, split train data for eval (e.g., 0.05 = 5%)

    # For streaming mode: must specify max_steps since dataset has no __len__
    max_steps: int = -1  # -1 means calculate from epochs (only works for non-streaming)
    estimated_samples: int = None  # Estimate total samples for streaming mode
    eval_max_samples: int = 5000  # Limit eval samples (None = all)

    # Training hyperparameters
    num_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_length: int = 2048

    # HL-Gauss parameters
    sigma_to_bin_ratio: float = 2.0
    value_loss_weight: float = 1.0

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # DeepSpeed
    deepspeed_config: Optional[str] = None


def create_training_args(config: ChessTrainingConfig, is_streaming: bool = False) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config"""

    # For streaming, we need max_steps instead of num_epochs
    if is_streaming:
        if config.max_steps <= 0 and config.estimated_samples:
            # Calculate max_steps from estimated samples
            effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
            steps_per_epoch = config.estimated_samples // effective_batch
            config.max_steps = steps_per_epoch * config.num_epochs
            print(
                f"Calculated max_steps: {config.max_steps} ({steps_per_epoch} steps/epoch × {config.num_epochs} epochs)")
        elif config.max_steps <= 0:
            raise ValueError(
                "Streaming mode requires --max_steps or --estimated_samples.\n"
                "Example: --max_steps 10000 or --estimated_samples 500000000"
            )

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs if not is_streaming else 1,
        max_steps=config.max_steps if is_streaming else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",

        # Precision
        fp16=config.fp16,
        bf16=config.bf16,

        # Checkpointing
        gradient_checkpointing=config.gradient_checkpointing,

        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],

        # Saving
        save_steps=config.save_steps,
        save_total_limit=1,

        # Evaluation - only enable if we have eval data
        eval_strategy="no",  # Will be overridden if eval_dataset provided
        eval_steps=config.eval_steps,

        # DeepSpeed
        deepspeed=config.deepspeed_config,

        # Other
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important! We need value_token_mask
        dataloader_pin_memory=True,
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_chess_mav(config: ChessTrainingConfig):
    """
    Main training function
    """
    print("=" * 60)
    print("Chess MAV Model Training")
    print("=" * 60)

    # Setup model and tokenizer
    model, chess_tokenizer = setup_model_and_tokenizer(
        config.model_name,
        uci_moves_file=config.uci_moves_file,
        use_flash_attention=config.use_flash_attention
    )

    # Create datasets
    print(f"\nLoading training data from: {config.data_path}")

    eval_dataset = None

    if config.streaming:
        # Streaming mode - loads one file at a time (for large datasets)
        print("Using STREAMING mode (memory efficient)")
        train_dataset = ChessMAVIterableDataset(
            data_path=config.data_path,
            tokenizer=chess_tokenizer,
            max_length=config.max_length,
            shuffle_files=True,
            shuffle_moves=True,
        )

        # For streaming, eval must be a separate path (can't split)
        if config.eval_data_path:
            print(f"Loading eval data from: {config.eval_data_path}")
            eval_dataset = ChessMAVDataset(
                data_path=config.eval_data_path,
                tokenizer=chess_tokenizer,
                max_length=config.max_length,
                max_samples=config.eval_max_samples,
            )
    else:
        # Load all into memory (for smaller datasets)
        print("Using IN-MEMORY mode")
        train_dataset = ChessMAVDataset(
            data_path=config.data_path,
            tokenizer=chess_tokenizer,
            max_length=config.max_length,
            max_samples=config.max_samples,
        )

        # Split for eval if requested
        if config.eval_split > 0:
            total = len(train_dataset)
            eval_size = int(total * config.eval_split)
            train_size = total - eval_size

            print(f"Splitting: {train_size} train, {eval_size} eval")

            from torch.utils.data import random_split
            train_dataset, eval_dataset = random_split(
                train_dataset,
                [train_size, eval_size]
            )
        elif config.eval_data_path:
            print(f"Loading eval data from: {config.eval_data_path}")
            eval_dataset = ChessMAVDataset(
                data_path=config.eval_data_path,
                tokenizer=chess_tokenizer,
                max_length=config.max_length,
                max_samples=config.eval_max_samples,
            )

    # Create data collator
    def collate_fn(batch):
        return chess_collate_fn(
            batch,
            pad_token_id=chess_tokenizer.tokenizer.pad_token_id or 0
        )

    # Create training arguments
    training_args = create_training_args(config, is_streaming=config.streaming)

    # Enable eval if we have eval dataset
    if eval_dataset is not None:
        training_args.eval_strategy = "steps"
        if not config.eval_steps:
            training_args.eval_steps = 500  # Default eval every 500 steps
        print(f"Evaluation enabled: every {training_args.eval_steps} steps")

    # Create trainer
    trainer = ChessMAVTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        sigma_to_bin_ratio=config.sigma_to_bin_ratio,
        value_token_start=chess_tokenizer.value_token_start,
        value_token_end=chess_tokenizer.value_token_end,
        value_loss_weight=config.value_loss_weight,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    chess_tokenizer.tokenizer.save_pretrained(
        os.path.join(config.output_dir, "final_model")
    )

    print("\nTraining complete!")
    return trainer


# ============================================================================
# DEEPSPEED CONFIG
# ============================================================================

DEEPSPEED_CONFIG = """
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
"""

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Chess MAV Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL data or folder")
    parser.add_argument("--uci_moves_file", type=str, required=True, help="Path to UCI moves JSON")
    parser.add_argument("--output_dir", type=str, default="./chess_mav_output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--sigma_to_bin_ratio", type=float, default=2.0, help="HL-Gauss sigma")
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--no-streaming", action="store_true", help="Load all data into memory")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to eval data folder/file")
    parser.add_argument("--eval_split", type=float, default=0.0, help="Split ratio for eval (e.g., 0.05)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--eval_max_samples", type=int, default=5000, help="Max samples for eval dataset")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (required for streaming)")
    parser.add_argument("--estimated_samples", type=int, default=None, help="Estimated total samples (for streaming)")

    args = parser.parse_args()

    # Create config
    config = ChessTrainingConfig(
        data_path=args.data_path,
        uci_moves_file=args.uci_moves_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        sigma_to_bin_ratio=args.sigma_to_bin_ratio,
        streaming=not getattr(args, 'no_streaming', False),
        max_samples=args.max_samples,
        eval_data_path=args.eval_data_path,
        eval_split=args.eval_split,
        eval_steps=args.eval_steps,
        eval_max_samples=args.eval_max_samples,
        max_steps=args.max_steps,
        estimated_samples=args.estimated_samples,
    )

    # Setup DeepSpeed if requested
    if args.deepspeed:
        ds_config_path = os.path.join(args.output_dir, "ds_config.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(ds_config_path, 'w') as f:
            f.write(DEEPSPEED_CONFIG)
        config.deepspeed_config = ds_config_path

    # Train
    train_chess_mav(config)
