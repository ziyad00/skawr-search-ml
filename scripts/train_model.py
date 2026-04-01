#!/usr/bin/env python3
"""
SKAWR Model Training Script

Main script to train the custom SKAWR transformer model.
Handles configuration loading, data preparation, and training orchestration.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.architecture.skawr_transformer import SKAWRTransformer, SKAWRConfig
from models.training.trainer import SKAWRTrainer, TrainingConfig
from models.training.dataset import DataConfig, create_dataloaders, create_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_config(config: Dict[str, Any], vocab_size: int) -> SKAWRConfig:
    """Setup model configuration."""
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size
    return SKAWRConfig(**model_config)

def setup_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Setup training configuration."""
    training_config = config['training'].copy()
    training_config.update(config.get('hardware', {}))
    training_config.update(config.get('logging', {}))
    return TrainingConfig(**training_config)

def setup_data_config(config: Dict[str, Any]) -> DataConfig:
    """Setup data configuration."""
    data_config = config['data'].copy()
    return DataConfig(
        max_sequence_length=data_config.get('max_sequence_length', 512),
        tokenizer_name=data_config.get('tokenizer_name', 'bert-base-multilingual-cased')
    )

def check_data_files(config: Dict[str, Any]) -> bool:
    """Check if required data files exist."""
    data_config = config['data']

    required_files = [
        data_config['train_file'],
        data_config['validation_file']
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error("Missing required data files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        logger.error("Please run data collection first:")
        logger.error("  python scripts/data_collection/run_data_collection.py")
        return False

    return True

def create_dummy_data(output_dir: str, num_samples: int = 1000):
    """Create dummy data for testing when real data is not available."""
    import pandas as pd
    import random

    logger.info(f"Creating dummy data with {num_samples} samples...")

    # Sample texts for each domain
    sample_texts = {
        "automotive": [
            "2020 Toyota Camry sedan with automatic transmission",
            "Used BMW 3 Series sports car for sale",
            "Honda Civic with low mileage and excellent condition",
            "Ford F-150 pickup truck with V8 engine",
            "Luxury Mercedes-Benz with leather interior"
        ],
        "electronics": [
            "iPhone 13 Pro Max with 256GB storage",
            "Samsung Galaxy S21 smartphone unlocked",
            "MacBook Pro laptop with M1 chip",
            "Gaming laptop with RTX 3060 graphics",
            "Wireless Bluetooth headphones noise cancelling"
        ],
        "fashion": [
            "Designer handbag leather black color",
            "Men's casual shirt cotton blue",
            "Women's dress summer style floral",
            "Running shoes Nike size 9",
            "Watch luxury brand stainless steel"
        ],
        "general": [
            "Kitchen appliance microwave oven",
            "Home furniture dining table wood",
            "Garden tools lawn mower electric",
            "Book collection vintage rare",
            "Art painting canvas abstract modern"
        ]
    }

    # Create training data
    train_data = []
    val_data = []

    for _ in range(num_samples):
        domain = random.choice(list(sample_texts.keys()))
        base_text = random.choice(sample_texts[domain])

        # Add some variation
        modifiers = ["excellent", "good", "fair", "new", "used", "vintage", "modern", "classic"]
        colors = ["black", "white", "blue", "red", "green", "silver", "gold"]

        text = base_text
        if random.random() < 0.5:
            text = f"{random.choice(modifiers)} {text}"
        if random.random() < 0.3:
            text = f"{text} {random.choice(colors)}"

        data_point = {
            'text': text,
            'domain': domain,
            'category': f"{domain}_category",
            'source': 'dummy_data',
            'description': 'Generated dummy data for testing'
        }

        # 80% train, 20% validation
        if random.random() < 0.8:
            train_data.append(data_point)
        else:
            val_data.append(data_point)

    # Save data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_file = output_dir / "train.csv"
    val_file = output_dir / "validation.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    logger.info(f"Created dummy data:")
    logger.info(f"  Train: {len(train_df)} samples -> {train_file}")
    logger.info(f"  Val: {len(val_df)} samples -> {val_file}")

    return str(train_file), str(val_file)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SKAWR transformer model")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to config file")
    parser.add_argument("--create-dummy-data", action="store_true", help="Create dummy data for testing")
    parser.add_argument("--dummy-samples", type=int, default=1000, help="Number of dummy samples")

    args = parser.parse_args()

    logger.info("Starting SKAWR model training...")

    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Check or create data files
    if args.create_dummy_data:
        train_file, val_file = create_dummy_data(
            output_dir="data/processed",
            num_samples=args.dummy_samples
        )
        # Update config with dummy data paths
        config['data']['train_file'] = train_file
        config['data']['validation_file'] = val_file

    if not check_data_files(config):
        logger.error("Use --create-dummy-data to create test data, or run data collection first")
        sys.exit(1)

    # Setup configurations
    data_config = setup_data_config(config)
    training_config = setup_training_config(config)

    logger.info(f"Training device: {training_config.device}")

    # Create tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)

    # Setup model config with correct vocab size
    model_config = setup_model_config(config, len(tokenizer))

    # Create model
    logger.info("Creating model...")
    model = SKAWRTransformer(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_file=config['data']['train_file'],
        val_file=config['data']['validation_file'],
        config=data_config,
        batch_size=training_config.batch_size,
        num_workers=config['data'].get('dataloader_num_workers', 4)
    )

    logger.info(f"Training batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = SKAWRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        config=training_config
    )

    # Start training
    logger.info("Starting training...")
    training_stats = trainer.train()

    # Save final model
    final_model_path = Path(training_config.output_dir) / "final_model"
    trainer.save_final_model(final_model_path)

    logger.info("Training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")

    # Print training summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Total epochs: {len(training_stats['train_loss'])}")
    logger.info(f"Final train loss: {training_stats['train_loss'][-1]:.4f}")
    logger.info(f"Final eval loss: {training_stats['eval_loss'][-1]:.4f}")
    logger.info(f"Best eval loss: {trainer.best_eval_loss:.4f}")

if __name__ == "__main__":
    main()