#!/usr/bin/env python3
"""
Kaggle Dataset Downloader for SKAWR Search ML Training

Downloads and processes multiple domain datasets from Kaggle for training
the custom transformer model.
"""

import os
import sys
import json
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class KaggleDataCollector:
    """Downloads and processes datasets from Kaggle for multi-domain training."""

    def __init__(self, data_dir: str = "data/kaggle"):
        self.data_dir = Path(PROJECT_ROOT) / data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()

        # Dataset configurations
        self.datasets = self._get_dataset_configs()

        logger.info(f"Initialized Kaggle collector, data dir: {self.data_dir}")

    def _get_dataset_configs(self) -> Dict:
        """Configure datasets to download for each domain."""
        return {
            "automotive": [
                {
                    "dataset": "austinreese/craigslist-carstrucks-data",
                    "description": "Craigslist car listings with descriptions",
                    "files": ["vehicles.csv"],
                    "text_columns": ["description", "model", "manufacturer"],
                    "category_column": "type",
                    "size_mb": 50
                },
                {
                    "dataset": "nehalbirla/vehicle-dataset-from-cardekho",
                    "description": "Car specifications and prices",
                    "files": ["car data.csv"],
                    "text_columns": ["name", "fuel", "transmission"],
                    "category_column": "fuel",
                    "size_mb": 5
                },
                {
                    "dataset": "doaaalsenani/usa-cers-dataset",
                    "description": "USA cars dataset with specifications",
                    "files": ["USA_cars_datasets.csv"],
                    "text_columns": ["brand", "model", "title"],
                    "category_column": "brand",
                    "size_mb": 20
                }
            ],
            "electronics": [
                {
                    "dataset": "srolka/ecommerce-customers",
                    "description": "E-commerce product data",
                    "files": ["Ecommerce Customers"],
                    "text_columns": ["description", "category"],
                    "category_column": "category",
                    "size_mb": 2
                },
                {
                    "dataset": "rmisra/amazon-product-dataset",
                    "description": "Amazon product reviews and metadata",
                    "files": ["amazon_products.csv"],
                    "text_columns": ["title", "description", "features"],
                    "category_column": "main_category",
                    "size_mb": 100
                }
            ],
            "fashion": [
                {
                    "dataset": "paramaggarwal/fashion-product-images-dataset",
                    "description": "Fashion product images and metadata",
                    "files": ["styles.csv"],
                    "text_columns": ["productDisplayName", "articleType", "baseColour"],
                    "category_column": "masterCategory",
                    "size_mb": 15
                }
            ],
            "general": [
                {
                    "dataset": "snap/amazon-fine-food-reviews",
                    "description": "Amazon product reviews (general)",
                    "files": ["Reviews.csv"],
                    "text_columns": ["Summary", "Text"],
                    "category_column": "ProductId",
                    "size_mb": 300
                }
            ]
        }

    def download_dataset(self, dataset_name: str, domain: str) -> bool:
        """Download a single dataset."""
        dataset_dir = self.data_dir / domain / dataset_name.split("/")[-1]
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading {dataset_name} to {dataset_dir}")

            self.api.dataset_download_files(
                dataset_name,
                path=str(dataset_dir),
                unzip=True
            )

            logger.success(f"Downloaded {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False

    def process_dataset(self, config: Dict, domain: str) -> Optional[pd.DataFrame]:
        """Process a downloaded dataset into standardized format."""
        dataset_name = config["dataset"].split("/")[-1]
        dataset_dir = self.data_dir / domain / dataset_name

        try:
            # Find CSV files in dataset directory
            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {dataset_dir}")
                return None

            # Load the main file
            main_file = csv_files[0]  # Take first CSV file
            logger.info(f"Processing {main_file}")

            df = pd.read_csv(main_file)
            logger.info(f"Loaded {len(df)} rows from {main_file}")

            # Standardize columns
            processed_df = self._standardize_dataframe(df, config, domain)

            # Save processed data
            output_file = self.data_dir / domain / f"{dataset_name}_processed.csv"
            processed_df.to_csv(output_file, index=False)
            logger.success(f"Processed dataset saved to {output_file}")

            return processed_df

        except Exception as e:
            logger.error(f"Failed to process {config['dataset']}: {e}")
            return None

    def _standardize_dataframe(self, df: pd.DataFrame, config: Dict, domain: str) -> pd.DataFrame:
        """Standardize dataframe to common format for training."""

        # Create standardized columns
        standardized_data = []

        for _, row in df.iterrows():
            # Combine text columns
            text_parts = []
            for col in config["text_columns"]:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))

            if not text_parts:
                continue  # Skip rows with no text

            combined_text = " ".join(text_parts)

            # Get category
            category = "unknown"
            if config["category_column"] in df.columns:
                category = str(row[config["category_column"]]) if pd.notna(row[config["category_column"]]) else "unknown"

            standardized_data.append({
                "text": combined_text,
                "domain": domain,
                "category": category,
                "source": config["dataset"],
                "description": config["description"]
            })

        return pd.DataFrame(standardized_data)

    def download_all_domains(self, domains: Optional[List[str]] = None) -> Dict[str, int]:
        """Download and process all datasets for specified domains."""

        if domains is None:
            domains = list(self.datasets.keys())

        results = {}

        for domain in domains:
            logger.info(f"\n=== Processing {domain.upper()} domain ===")
            domain_count = 0

            for config in self.datasets[domain]:
                dataset_name = config["dataset"]

                # Download dataset
                if self.download_dataset(dataset_name, domain):

                    # Process dataset
                    processed_df = self.process_dataset(config, domain)

                    if processed_df is not None:
                        domain_count += len(processed_df)
                        logger.info(f"Added {len(processed_df)} samples from {dataset_name}")

            results[domain] = domain_count
            logger.success(f"Domain {domain}: {domain_count} total samples")

        return results

    def combine_domains(self) -> pd.DataFrame:
        """Combine all processed datasets into single training file."""

        all_data = []

        for domain in self.datasets.keys():
            domain_dir = self.data_dir / domain
            processed_files = list(domain_dir.glob("*_processed.csv"))

            for file_path in processed_files:
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} samples from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        if not all_data:
            logger.error("No processed data found!")
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and clean
        combined_df = combined_df.drop_duplicates(subset=['text'])
        combined_df = combined_df.dropna(subset=['text'])
        combined_df = combined_df[combined_df['text'].str.len() > 10]  # Minimum text length

        # Save combined dataset
        output_file = self.data_dir / "combined_training_data.csv"
        combined_df.to_csv(output_file, index=False)

        logger.success(f"Combined dataset: {len(combined_df)} samples saved to {output_file}")

        # Print statistics
        self._print_dataset_stats(combined_df)

        return combined_df

    def _print_dataset_stats(self, df: pd.DataFrame):
        """Print dataset statistics."""
        logger.info("\n=== DATASET STATISTICS ===")
        logger.info(f"Total samples: {len(df):,}")

        # Domain distribution
        logger.info("\nDomain distribution:")
        domain_counts = df['domain'].value_counts()
        for domain, count in domain_counts.items():
            logger.info(f"  {domain}: {count:,} ({count/len(df)*100:.1f}%)")

        # Text length statistics
        text_lengths = df['text'].str.len()
        logger.info(f"\nText length stats:")
        logger.info(f"  Mean: {text_lengths.mean():.1f} characters")
        logger.info(f"  Median: {text_lengths.median():.1f} characters")
        logger.info(f"  Min: {text_lengths.min()} characters")
        logger.info(f"  Max: {text_lengths.max()} characters")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Kaggle datasets for SKAWR ML training")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["automotive", "electronics", "fashion", "general"],
        help="Domains to download (default: all)"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only combine existing processed data, don't download"
    )

    args = parser.parse_args()

    collector = KaggleDataCollector()

    if not args.combine_only:
        # Download and process datasets
        results = collector.download_all_domains(args.domains)
        logger.info(f"\nDownload results: {results}")

    # Combine all processed data
    combined_df = collector.combine_domains()
    logger.success(f"Data collection complete! {len(combined_df)} total samples ready for training.")


if __name__ == "__main__":
    main()