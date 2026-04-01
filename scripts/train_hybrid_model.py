#!/usr/bin/env python3
"""
Hybrid Transformer-Mamba Training Script

Train the 2026 state-of-the-art hybrid architecture for SKAWR search.
Supports multiple architecture patterns and optimizations.
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

from models.architecture.hybrid_transformer_mamba import (
    HybridTransformerMamba, HybridConfig,
    create_balanced_hybrid, create_efficiency_focused, create_semantic_focused
)
from models.training.trainer import SKAWRTrainer, TrainingConfig
from models.training.dataset import DataConfig, create_dataloaders

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

def setup_hybrid_config(config: Dict[str, Any], vocab_size: int) -> HybridConfig:
    """Setup hybrid model configuration."""
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size

    # Handle preset configurations
    if 'preset' in config and config['preset'] in config.get('presets', {}):
        preset = config['presets'][config['preset']]
        logger.info(f"Using preset: {config['preset']} - {preset['description']}")

        # Override with preset values
        for key, value in preset.items():
            if key not in ['description', 'use_case']:
                model_config[key] = value

    return HybridConfig(**model_config)

def setup_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Setup training configuration with hybrid optimizations."""
    training_config = config['training'].copy()
    training_config.update(config.get('hardware', {}))
    training_config.update(config.get('logging', {}))

    # Add hybrid-specific settings
    if 'optimizations' in config:
        training_config.update(config['optimizations'])

    return TrainingConfig(**training_config)

def setup_data_config(config: Dict[str, Any]) -> DataConfig:
    """Setup data configuration."""
    data_config = config['data'].copy()
    return DataConfig(
        max_sequence_length=data_config.get('max_sequence_length', 1024),  # Longer for Mamba
        tokenizer_name=data_config.get('tokenizer_name', 'bert-base-multilingual-cased')
    )

def create_hybrid_model_from_preset(
    preset: str,
    vocab_size: int,
    **kwargs
) -> HybridTransformerMamba:
    """Create hybrid model using predefined preset."""

    if preset == "balanced":
        return create_balanced_hybrid(vocab_size=vocab_size, **kwargs)
    elif preset == "efficiency":
        return create_efficiency_focused(vocab_size=vocab_size, **kwargs)
    elif preset == "semantic":
        return create_semantic_focused(vocab_size=vocab_size, **kwargs)
    else:
        # Create custom model from config
        config = HybridConfig(vocab_size=vocab_size, **kwargs)
        return HybridTransformerMamba(config)

def create_dummy_data(output_dir: str, num_samples: int = 1000):
    """Create dummy data optimized for hybrid model testing."""
    import pandas as pd
    import random

    logger.info(f"Creating dummy data with {num_samples} samples for hybrid model...")

    # Sample texts with longer sequences to test Mamba efficiency
    sample_texts = {
        "automotive": [
            "2023 BMW M3 Competition sports car with twin-turbo inline-6 engine producing 503 horsepower, 8-speed automatic transmission, rear-wheel drive, carbon fiber roof, M adaptive suspension, Brembo brakes, 19-inch wheels, premium leather interior with heated seats, advanced driver assistance systems, navigation, and excellent condition with only 5000 miles",
            "Used Mercedes-Benz C63 AMG sedan with hand-built 4.0L V8 biturbo engine, AMG RIDE CONTROL suspension, AMG performance exhaust system, premium interior with Nappa leather, COMAND infotainment system, advanced safety features including active brake assist and blind spot monitoring, pristine condition, full service history available",
            "Toyota Camry Hybrid 2024 with advanced Toyota Hybrid Synergy Drive system, excellent fuel efficiency rated at 52 MPG city, spacious interior with premium cloth seating, Toyota Safety Sense 2.0 including pre-collision system, lane departure alert, automatic high beams, and adaptive cruise control, factory warranty remaining",
            "Ford F-150 Lightning electric pickup truck with dual motor all-wheel drive, 452 horsepower, 775 lb-ft torque, Pro Power Onboard capability, 15.5-inch touchscreen with SYNC 4A, over-the-air software updates, advanced towing capabilities, and innovative front trunk storage compartment, perfect for work and recreation",
            "Luxury Porsche 911 Carrera sports coupe with naturally aspirated flat-six engine, precise manual transmission, sport suspension, leather interior with carbon fiber accents, advanced infotainment system, premium sound system, and exceptional build quality representing decades of engineering excellence"
        ],
        "electronics": [
            "Apple iPhone 15 Pro Max with A17 Pro chip featuring 3-nanometer technology, advanced camera system with 5x optical zoom telephoto lens, Action button for customizable controls, USB-C connectivity, titanium design with aerospace-grade construction, always-on display with ProMotion technology, exceptional battery life and wireless charging capabilities",
            "Samsung Galaxy S24 Ultra smartphone featuring advanced AI photography capabilities, S Pen stylus integration, 200MP main camera with improved low-light performance, 6.8-inch Dynamic AMOLED display with adaptive refresh rate, powerful Snapdragon processor, enhanced security features, and comprehensive connectivity options including 5G",
            "Gaming laptop with NVIDIA RTX 4080 graphics card, Intel Core i9 processor with advanced cooling system, 32GB DDR5 RAM, 1TB NVMe SSD storage, mechanical keyboard with RGB backlighting, high-refresh-rate display with G-SYNC technology, premium build quality and advanced thermal management for sustained performance",
            "MacBook Pro with Apple M3 Max chip featuring unified memory architecture, Liquid Retina XDR display with mini-LED technology, professional-grade camera and microphone system, advanced connectivity including Thunderbolt ports, exceptional battery life, and optimized performance for creative workflows and professional applications",
            "Wireless noise-cancelling headphones with adaptive audio technology, premium drivers for exceptional sound quality, advanced active noise cancellation, comfortable over-ear design with premium materials, intuitive touch controls, long battery life with fast charging, and seamless connectivity across multiple devices"
        ],
        "fashion": [
            "Designer leather handbag crafted from premium Italian leather with hand-stitched details, gold-tone hardware, multiple compartments for organization, adjustable shoulder strap, dust bag included, authentic brand certification, timeless design suitable for both professional and casual occasions, excellent craftsmanship and attention to detail",
            "Men's premium cotton dress shirt with classic fit, French cuffs, mother-of-pearl buttons, precise tailoring, wrinkle-resistant fabric treatment, available in multiple sizes and colors, machine washable for easy care, professional appearance suitable for business meetings and formal events",
            "Women's summer dress featuring flowing silhouette with floral print design, breathable fabric blend perfect for warm weather, comfortable fit with adjustable elements, versatile styling options, high-quality construction with attention to seam finishing, suitable for various occasions from casual to semi-formal",
            "Running shoes with advanced cushioning technology, breathable mesh upper construction, lightweight design optimized for performance, durable rubber outsole with strategic traction patterns, supportive midsole engineering, comfortable fit for extended wear, and modern aesthetic appeal",
            "Luxury wristwatch with Swiss automatic movement, sapphire crystal glass, premium stainless steel construction, water resistance to 100 meters, precise timekeeping mechanism, elegant design suitable for formal occasions, comprehensive warranty coverage, and prestigious brand heritage"
        ],
        "general": [
            "Professional kitchen appliance set including high-performance stand mixer with multiple attachments, food processor with various blades and discs, precision blender with variable speed controls, all featuring durable construction, easy cleaning design, comprehensive warranty coverage, and user-friendly operation for culinary enthusiasts",
            "Home furniture collection featuring solid wood construction with premium finishes, ergonomic design principles, modular components for versatile arrangement, environmentally sustainable materials, professional assembly service available, comprehensive care instructions, and timeless styling that complements various interior design themes",
            "Garden tools and equipment set with professional-grade construction, ergonomic handles for comfortable use, rust-resistant coatings, precision engineering for effective performance, comprehensive tool selection for various gardening tasks, durable storage solutions, and detailed usage guides for optimal results",
            "Rare book collection featuring first edition publications, pristine condition with careful preservation, historical significance, expert authentication, detailed provenance documentation, protective storage materials, and scholarly research value for collectors and academic institutions",
            "Art painting collection with original works by established artists, various mediums and styles represented, professional framing and conservation, detailed certificates of authenticity, investment potential, cultural significance, and aesthetic appeal for discerning collectors"
        ]
    }

    # Create training and validation data
    train_data = []
    val_data = []

    for _ in range(num_samples):
        domain = random.choice(list(sample_texts.keys()))
        base_text = random.choice(sample_texts[domain])

        # Add some variation to create more diverse training examples
        modifiers = ["excellent condition", "premium quality", "limited edition", "professional grade", "high performance"]
        if random.random() < 0.3:
            base_text = f"{random.choice(modifiers)} {base_text}"

        data_point = {
            'text': base_text,
            'domain': domain,
            'category': f"{domain}_category",
            'source': 'dummy_hybrid_data',
            'description': 'Generated dummy data for hybrid model testing with longer sequences'
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

    logger.info(f"Created hybrid dummy data:")
    logger.info(f"  Train: {len(train_df)} samples -> {train_file}")
    logger.info(f"  Val: {len(val_df)} samples -> {val_file}")
    logger.info(f"  Average text length: {train_df['text'].str.len().mean():.0f} characters")

    return str(train_file), str(val_file)

def main():
    """Main hybrid training function."""
    parser = argparse.ArgumentParser(description="Train SKAWR Hybrid Transformer-Mamba model")
    parser.add_argument("--config", default="config/hybrid_model_config.yaml", help="Path to config file")
    parser.add_argument("--preset", choices=["balanced", "efficiency", "semantic"], help="Use predefined architecture preset")
    parser.add_argument("--create-dummy-data", action="store_true", help="Create dummy data for testing")
    parser.add_argument("--dummy-samples", type=int, default=500, help="Number of dummy samples")
    parser.add_argument("--architecture", help="Custom layer pattern (e.g., TMTMTM)")

    args = parser.parse_args()

    logger.info("🚀 Starting SKAWR Hybrid Transformer-Mamba training...")

    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Override with command line arguments
    if args.preset:
        config['preset'] = args.preset
    if args.architecture:
        config['model']['layer_pattern'] = args.architecture

    # Create or check data files
    if args.create_dummy_data:
        train_file, val_file = create_dummy_data(
            output_dir="data/processed",
            num_samples=args.dummy_samples
        )
        config['data']['train_file'] = train_file
        config['data']['validation_file'] = val_file

    # Setup configurations
    data_config = setup_data_config(config)
    training_config = setup_training_config(config)

    logger.info(f"Training device: {training_config.device}")
    logger.info(f"Mixed precision: {training_config.mixed_precision}")

    # Create tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)

    # Setup hybrid model config
    hybrid_config = setup_hybrid_config(config, len(tokenizer))

    # Create hybrid model
    logger.info(f"Creating hybrid model with pattern: {hybrid_config.layer_pattern}")

    if args.preset:
        model = create_hybrid_model_from_preset(
            args.preset,
            vocab_size=len(tokenizer),
            **hybrid_config.__dict__
        )
    else:
        model = HybridTransformerMamba(hybrid_config)

    # Model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    transformer_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and 'attention' in name
    )
    mamba_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and 'mamba' in name
    )

    logger.info(f"Model created with {total_params:,} parameters")
    logger.info(f"  Transformer parameters: {transformer_params:,}")
    logger.info(f"  Mamba parameters: {mamba_params:,}")
    logger.info(f"  Layer pattern: {hybrid_config.layer_pattern}")

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
    logger.info("Initializing hybrid trainer...")
    trainer = SKAWRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        config=training_config
    )

    # Start training
    logger.info("🔥 Starting hybrid model training...")
    training_stats = trainer.train()

    # Save final model
    final_model_path = Path(training_config.output_dir) / "final_hybrid_model"
    trainer.save_final_model(final_model_path)

    logger.info("✅ Hybrid training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")

    # Print training summary
    logger.info("\n=== Hybrid Training Summary ===")
    logger.info(f"Architecture pattern: {hybrid_config.layer_pattern}")
    logger.info(f"Total epochs: {len(training_stats['train_loss'])}")
    logger.info(f"Final train loss: {training_stats['train_loss'][-1]:.4f}")
    logger.info(f"Final eval loss: {training_stats['eval_loss'][-1]:.4f}")
    logger.info(f"Best eval loss: {trainer.best_eval_loss:.4f}")

    # Performance benefits
    logger.info("\n=== Expected Performance Benefits ===")
    logger.info("🚀 5x faster inference than pure Transformers")
    logger.info("📏 Linear scaling with sequence length")
    logger.info("🧠 Maintained semantic understanding")
    logger.info("⚡ Optimized for 2026 ML infrastructure")

if __name__ == "__main__":
    main()