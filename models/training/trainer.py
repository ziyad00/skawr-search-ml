"""
SKAWR Model Trainer

Handles training, validation, and evaluation of the custom transformer model.
Supports multi-task learning with proper loss weighting and optimization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from tqdm.auto import tqdm
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.architecture.skawr_transformer import SKAWRTransformer, SKAWRConfig, ContrastiveLoss

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Training basics
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Multi-task weights
    masked_lm_weight: float = 1.0
    domain_weight: float = 0.5
    specificity_weight: float = 0.5
    contrastive_weight: float = 1.0

    # Optimization
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8

    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Paths
    output_dir: str = "models/checkpoints"
    logging_dir: str = "logs"

    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "skawr-search-ml"

    def __post_init__(self):
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)

class SKAWRTrainer:
    """Trainer for SKAWR transformer model."""

    def __init__(
        self,
        model: SKAWRTransformer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config

        # Move model to device
        self.model.to(self.config.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss functions
        self.contrastive_loss = ContrastiveLoss()

        # Mixed precision training
        if self.config.mixed_precision and self.config.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_counter = 0

        # Initialize wandb if enabled
        if self.config.use_wandb:
            self._init_wandb()

        logger.info(f"Trainer initialized with device: {self.config.device}")
        logger.info(f"Model has {self._count_parameters():,} parameters")

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.config.num_epochs

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking."""
        try:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    "model_config": self.model.config.__dict__,
                    "training_config": self.config.__dict__
                }
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info("Starting training...")

        training_stats = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }

        self.model.train()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Training epoch
            train_loss = self._train_epoch()
            training_stats["train_loss"].append(train_loss)

            # Evaluation
            eval_loss = self._evaluate()
            training_stats["eval_loss"].append(eval_loss)

            # Learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            training_stats["learning_rate"].append(current_lr)

            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, LR: {current_lr:.2e}")

            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "learning_rate": current_lr
                })

            # Save checkpoint
            self._save_checkpoint(eval_loss)

            # Early stopping check
            if self._should_early_stop(eval_loss):
                logger.info("Early stopping triggered")
                break

        logger.info("Training completed!")
        return training_stats

    def _train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)

            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    if self.config.use_wandb:
                        wandb.log({
                            "train_step_loss": loss.item() * self.config.gradient_accumulation_steps,
                            "learning_rate": current_lr,
                            "global_step": self.global_step
                        })

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}"
            })

        return total_loss / num_batches

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            domain_labels=batch.get("domain_labels"),
            specificity_labels=batch.get("specificity_labels"),
            return_embeddings=True
        )

        total_loss = 0.0

        # Masked language modeling loss
        if "masked_lm_loss" in outputs:
            total_loss += self.config.masked_lm_weight * outputs["masked_lm_loss"]

        # Domain classification loss
        if "domain_loss" in outputs:
            total_loss += self.config.domain_weight * outputs["domain_loss"]

        # Specificity prediction loss
        if "specificity_loss" in outputs:
            total_loss += self.config.specificity_weight * outputs["specificity_loss"]

        # Contrastive loss (if we have query-product pairs)
        if "query_embeddings" in batch and "product_embeddings" in batch:
            query_embeddings = self.model.get_search_embeddings(
                batch["query_input_ids"],
                batch.get("query_attention_mask")
            )
            product_embeddings = self.model.get_search_embeddings(
                batch["product_input_ids"],
                batch.get("product_attention_mask")
            )
            contrastive_loss = self.contrastive_loss(query_embeddings, product_embeddings)
            total_loss += self.config.contrastive_weight * contrastive_loss

        return total_loss

    def _evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _save_checkpoint(self, eval_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model state
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "eval_loss": eval_loss,
            "model_config": self.model.config.__dict__,
            "training_config": self.config.__dict__
        }, checkpoint_dir / "pytorch_model.bin")

        # Save best model
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            best_model_dir = Path(self.config.output_dir) / "best_model"
            best_model_dir.mkdir(exist_ok=True)

            torch.save({
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config.__dict__,
                "eval_loss": eval_loss
            }, best_model_dir / "pytorch_model.bin")

            logger.info(f"New best model saved with eval loss: {eval_loss:.4f}")

    def _should_early_stop(self, eval_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.early_stopping_counter = 0
            return False

        self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.config.early_stopping_patience:
            return True

        return False

    def save_final_model(self, output_path: str):
        """Save final trained model."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config.__dict__
        }, output_path / "pytorch_model.bin")

        logger.info(f"Final model saved to {output_path}")

def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto") -> SKAWRTransformer:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model with saved config
    config = SKAWRConfig(**checkpoint["model_config"])
    model = SKAWRTransformer(config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model.to(device)
    model.eval()

    return model