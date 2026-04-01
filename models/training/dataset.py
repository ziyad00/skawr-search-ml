"""
SKAWR Dataset Classes

Custom PyTorch datasets for training the SKAWR transformer model.
Handles text preprocessing, tokenization, and multi-task data preparation.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import random
from transformers import AutoTokenizer
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

@dataclass
class DataConfig:
    """Configuration for data processing."""

    max_sequence_length: int = 512
    tokenizer_name: str = "bert-base-multilingual-cased"  # For Arabic support
    mask_probability: float = 0.15

    # Domain mappings
    domain_to_id: Dict[str, int] = None
    specificity_to_id: Dict[str, int] = None

    def __post_init__(self):
        if self.domain_to_id is None:
            self.domain_to_id = {
                "automotive": 0,
                "electronics": 1,
                "fashion": 2,
                "general": 3
            }

        if self.specificity_to_id is None:
            self.specificity_to_id = {
                "generic": 0,
                "moderate": 1,
                "specific": 2
            }

class SKAWRDataset(Dataset):
    """Dataset for SKAWR transformer training."""

    def __init__(
        self,
        data_file: str,
        config: DataConfig,
        tokenizer=None,
        mode: str = "train"
    ):
        self.data_file = data_file
        self.config = config
        self.mode = mode

        # Load data
        self.data = self._load_data()

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            self.tokenizer = tokenizer

        # Add special tokens if needed
        self._setup_special_tokens()

        print(f"Loaded {len(self.data)} samples from {data_file}")

    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(self.data_file)

        # Clean and filter data
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() > 10]  # Minimum length
        df = df[df['text'].str.len() < 2000]  # Maximum length

        return df.reset_index(drop=True)

    def _setup_special_tokens(self):
        """Setup special tokens in tokenizer."""
        special_tokens = {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]"
        }

        # Add tokens if they don't exist
        tokens_to_add = []
        for token_type, token in special_tokens.items():
            if not hasattr(self.tokenizer, token_type) or getattr(self.tokenizer, token_type) is None:
                tokens_to_add.append(token)

        if tokens_to_add:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": tokens_to_add
            })

    def _determine_specificity(self, text: str, domain: str) -> str:
        """Determine query specificity level based on text analysis."""
        text_lower = text.lower()
        word_count = len(text.split())

        # Domain-specific keywords for specificity
        specific_keywords = {
            "automotive": [
                "model", "year", "engine", "transmission", "mileage", "color",
                "sport", "luxury", "performance", "turbo", "v6", "v8", "manual",
                "automatic", "leather", "sunroof", "navigation"
            ],
            "electronics": [
                "model", "brand", "storage", "memory", "processor", "screen",
                "resolution", "battery", "camera", "wireless", "bluetooth",
                "pro", "plus", "max", "mini", "specifications"
            ],
            "fashion": [
                "size", "color", "material", "brand", "style", "design",
                "pattern", "fit", "length", "sleeve", "collar", "vintage",
                "designer", "cotton", "leather", "silk"
            ]
        }

        # Count specific keywords
        domain_keywords = specific_keywords.get(domain, [])
        specific_count = sum(1 for keyword in domain_keywords if keyword in text_lower)

        # Determine specificity based on multiple factors
        if specific_count >= 3 or word_count >= 10:
            return "specific"
        elif specific_count >= 1 or word_count >= 5:
            return "moderate"
        else:
            return "generic"

    def _create_masked_lm_labels(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Create masked language model labels."""
        labels = [-100] * len(input_ids)  # -100 is ignored in loss computation
        masked_input_ids = input_ids.copy()

        # Don't mask special tokens
        special_token_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        }

        for i, token_id in enumerate(input_ids):
            if token_id in special_token_ids:
                continue

            # Mask with probability
            if random.random() < self.config.mask_probability:
                labels[i] = token_id  # Original token for loss

                # 80% mask token, 10% random token, 10% keep original
                rand = random.random()
                if rand < 0.8:
                    masked_input_ids[i] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    masked_input_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                # else: keep original token

        return masked_input_ids, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = row['text']
        domain = row.get('domain', 'general')

        # Tokenize text
        tokens = self.tokenizer.tokenize(text)

        # Truncate if necessary
        max_tokens = self.config.max_sequence_length - 2  # Account for [CLS] and [SEP]
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Add special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create masked LM labels for training
        if self.mode == "train":
            masked_input_ids, mlm_labels = self._create_masked_lm_labels(input_ids)
        else:
            masked_input_ids = input_ids
            mlm_labels = [-100] * len(input_ids)

        # Pad to max length
        max_len = self.config.max_sequence_length
        attention_mask = [1] * len(masked_input_ids) + [0] * (max_len - len(masked_input_ids))
        masked_input_ids.extend([self.tokenizer.pad_token_id] * (max_len - len(masked_input_ids)))
        mlm_labels.extend([-100] * (max_len - len(mlm_labels)))

        # Domain label
        domain_label = self.config.domain_to_id.get(domain, 3)  # Default to 'general'

        # Specificity label
        specificity = self._determine_specificity(text, domain)
        specificity_label = self.config.specificity_to_id[specificity]

        return {
            "input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(mlm_labels, dtype=torch.long),
            "domain_labels": torch.tensor(domain_label, dtype=torch.long),
            "specificity_labels": torch.tensor(specificity_label, dtype=torch.long)
        }

class QueryProductDataset(Dataset):
    """Dataset for contrastive learning with query-product pairs."""

    def __init__(
        self,
        query_file: str,
        product_file: str,
        config: DataConfig,
        tokenizer=None
    ):
        self.config = config

        # Load data
        self.queries = pd.read_csv(query_file)
        self.products = pd.read_csv(product_file)

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            self.tokenizer = tokenizer

        # Create positive and negative pairs
        self.pairs = self._create_pairs()

        print(f"Created {len(self.pairs)} query-product pairs")

    def _create_pairs(self) -> List[Dict[str, Any]]:
        """Create positive and negative query-product pairs."""
        pairs = []

        # Group products by domain
        domain_products = {}
        for _, product in self.products.iterrows():
            domain = product.get('domain', 'general')
            if domain not in domain_products:
                domain_products[domain] = []
            domain_products[domain].append(product)

        for _, query in self.queries.iterrows():
            query_domain = query.get('domain', 'general')
            query_text = query['text']

            # Positive pairs: same domain products
            if query_domain in domain_products:
                for product in domain_products[query_domain][:5]:  # Limit to 5 products per query
                    pairs.append({
                        'query': query_text,
                        'product': product['text'],
                        'query_domain': query_domain,
                        'product_domain': product.get('domain', 'general'),
                        'is_positive': True
                    })

            # Negative pairs: different domain products
            other_domains = [d for d in domain_products.keys() if d != query_domain]
            if other_domains:
                neg_domain = random.choice(other_domains)
                neg_product = random.choice(domain_products[neg_domain])
                pairs.append({
                    'query': query_text,
                    'product': neg_product['text'],
                    'query_domain': query_domain,
                    'product_domain': neg_product.get('domain', 'general'),
                    'is_positive': False
                })

        return pairs

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and encode text."""
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Tokenize query and product
        query_encoding = self._tokenize_text(pair['query'])
        product_encoding = self._tokenize_text(pair['product'])

        return {
            'query_input_ids': query_encoding['input_ids'],
            'query_attention_mask': query_encoding['attention_mask'],
            'product_input_ids': product_encoding['input_ids'],
            'product_attention_mask': product_encoding['attention_mask'],
            'is_positive': torch.tensor(1 if pair['is_positive'] else 0, dtype=torch.long)
        }

def create_dataloaders(
    train_file: str,
    val_file: str,
    config: DataConfig,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""

    # Create datasets
    train_dataset = SKAWRDataset(train_file, config, mode="train")
    val_dataset = SKAWRDataset(val_file, config, mode="val")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def create_tokenizer(train_file: str, vocab_size: int = 30000) -> AutoTokenizer:
    """Create custom tokenizer trained on the dataset."""
    # For now, use pre-trained multilingual tokenizer
    # In production, you might want to train a custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    return tokenizer