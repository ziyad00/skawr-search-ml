#!/usr/bin/env python3
"""
Quick test script for Hybrid Transformer-Mamba architecture
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.architecture.hybrid_transformer_mamba import (
    HybridTransformerMamba, HybridConfig,
    create_balanced_hybrid, create_efficiency_focused, create_semantic_focused
)
from transformers import AutoTokenizer

def test_model_architecture():
    """Test basic model forward pass and architecture."""
    print("🧪 Testing Hybrid Transformer-Mamba Architecture...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    vocab_size = len(tokenizer)

    print(f"Vocabulary size: {vocab_size:,}")

    # Test different architectures
    architectures = {
        "Balanced (TMTMTM)": lambda: create_balanced_hybrid(vocab_size=vocab_size),
        "Efficiency (TTMMMM)": lambda: create_efficiency_focused(vocab_size=vocab_size),
        "Semantic (TTTTMM)": lambda: create_semantic_focused(vocab_size=vocab_size)
    }

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {}

    for arch_name, create_func in architectures.items():
        print(f"\n🏗️ Testing {arch_name}...")

        try:
            # Create model
            model = create_func()
            model.to(device)
            model.eval()

            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")

            # Test forward pass
            batch_size = 4
            seq_length = 128
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                outputs = model(input_ids, return_embeddings=True)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000

            print(f"  Inference time: {inference_time:.2f} ms")
            print(f"  Output shapes:")
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"    {key}: {tuple(tensor.shape)}")

            # Test search embeddings
            search_embeddings = model.get_search_embeddings(input_ids)
            print(f"  Search embeddings shape: {tuple(search_embeddings.shape)}")

            results[arch_name] = {
                'parameters': total_params,
                'inference_time_ms': inference_time,
                'success': True
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[arch_name] = {
                'success': False,
                'error': str(e)
            }

    # Summary
    print("\n📊 Architecture Comparison:")
    for arch_name, result in results.items():
        if result['success']:
            print(f"  {arch_name}: {result['parameters']:,} params, {result['inference_time_ms']:.2f}ms")
        else:
            print(f"  {arch_name}: ❌ Failed - {result['error']}")

    # Find fastest
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]['inference_time_ms'])
        print(f"\n🏆 Fastest architecture: {fastest[0]} ({fastest[1]['inference_time_ms']:.2f}ms)")

    return results

def test_query_examples():
    """Test with real query examples."""
    print("\n🔍 Testing with real search queries...")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    vocab_size = len(tokenizer)

    # Create balanced hybrid model
    model = create_balanced_hybrid(vocab_size=vocab_size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Test queries
    queries = [
        "sports car Ferrari red",
        "gaming laptop with RTX graphics",
        "luxury handbag designer leather",
        "سيارة رياضية فيراري"  # Arabic: "Ferrari sports car"
    ]

    print("Query embeddings:")
    for query in queries:
        try:
            # Tokenize
            encoded = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = encoded['input_ids'].to(device)

            # Get embeddings
            with torch.no_grad():
                embeddings = model.get_search_embeddings(input_ids)

            print(f"  '{query}': shape {tuple(embeddings.shape)}, norm {embeddings.norm().item():.3f}")

        except Exception as e:
            print(f"  '{query}': ❌ Error - {e}")

if __name__ == "__main__":
    print("🚀 SKAWR Hybrid Transformer-Mamba Quick Test")
    print("=" * 50)

    try:
        # Test architectures
        results = test_model_architecture()

        # Test with examples if any architecture worked
        if any(r['success'] for r in results.values()):
            test_query_examples()

        print("\n✅ Testing completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)