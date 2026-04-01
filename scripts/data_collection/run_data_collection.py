#!/usr/bin/env python3
"""
Data Collection Orchestrator for SKAWR Search ML

Orchestrates the collection of training data from multiple sources:
- Kaggle datasets
- Reddit discussions
- Wikipedia articles
- Web scraping

Combines all sources into a unified training dataset.
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import collectors
try:
    from scripts.data_collection.kaggle_downloader import KaggleDataCollector
    from scripts.data_collection.reddit_scraper import RedditDataScraper
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

class DataCollectionOrchestrator:
    """Orchestrates data collection from multiple sources."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(PROJECT_ROOT) / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize collectors
        self.kaggle_collector = KaggleDataCollector()
        self.reddit_scraper = RedditDataScraper()

        logger.info(f"Initialized data collection orchestrator")

    def collect_all_data(self, sources: List[str], domains: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect data from all specified sources and domains."""

        collected_data = {}

        # Collect from Kaggle
        if "kaggle" in sources:
            logger.info("\n🔄 Starting Kaggle data collection...")
            try:
                kaggle_results = self.kaggle_collector.download_all_domains(domains)
                kaggle_df = self.kaggle_collector.combine_domains()
                collected_data["kaggle"] = kaggle_df
                logger.success(f"Kaggle collection: {len(kaggle_df)} samples")
            except Exception as e:
                logger.error(f"Kaggle collection failed: {e}")
                collected_data["kaggle"] = pd.DataFrame()

        # Collect from Reddit
        if "reddit" in sources:
            logger.info("\n🔄 Starting Reddit data collection...")
            try:
                reddit_results = self.reddit_scraper.scrape_all_domains(domains)
                reddit_df = self.reddit_scraper.combine_reddit_data()
                collected_data["reddit"] = reddit_df
                logger.success(f"Reddit collection: {len(reddit_df)} samples")
            except Exception as e:
                logger.error(f"Reddit collection failed: {e}")
                collected_data["reddit"] = pd.DataFrame()

        return collected_data

    def combine_all_sources(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from all sources into unified dataset."""

        all_dataframes = []

        for source, df in source_data.items():
            if not df.empty:
                # Add source identifier
                df_copy = df.copy()
                df_copy["data_source"] = source
                all_dataframes.append(df_copy)
                logger.info(f"Adding {len(df_copy)} samples from {source}")

        if not all_dataframes:
            logger.error("No data collected from any source!")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Standardize and clean
        combined_df = self._standardize_dataset(combined_df)

        return combined_df

    def _standardize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the combined dataset."""

        logger.info("🔧 Standardizing combined dataset...")

        # Ensure required columns exist
        required_columns = ["text", "domain", "category", "source", "description"]
        for col in required_columns:
            if col not in df.columns:
                df[col] = "unknown"

        # Clean text data
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() > 10]  # Minimum length
        df = df[df["text"].str.len() < 2000]  # Maximum length

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=["text"])
        duplicates_removed = initial_count - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate samples")

        # Clean text content
        df["text"] = df["text"].str.strip()
        df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
        df["text"] = df["text"].str.replace(r'[^\w\s\u0600-\u06FF.,!?-]', '', regex=True)  # Keep Arabic chars

        # Standardize categories
        df["domain"] = df["domain"].str.lower().str.strip()
        df["category"] = df["category"].str.lower().str.strip()

        # Add metadata
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()

        logger.success(f"Standardization complete: {len(df)} clean samples")
        return df

    def create_training_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits."""

        logger.info("📊 Creating training splits...")

        # Shuffle data
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # Calculate split indices
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Create splits
        splits = {
            "train": df.iloc[:train_end],
            "validation": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:]
        }

        # Log split statistics
        for split_name, split_df in splits.items():
            logger.info(f"{split_name}: {len(split_df)} samples")

            # Domain distribution in split
            domain_dist = split_df["domain"].value_counts()
            logger.info(f"  Domain distribution: {dict(domain_dist)}")

        return splits

    def save_datasets(self, combined_df: pd.DataFrame, splits: Dict[str, pd.DataFrame]):
        """Save all datasets to files."""

        logger.info("💾 Saving datasets...")

        # Save combined dataset
        combined_file = self.output_dir / "combined_training_data.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Combined dataset saved: {combined_file}")

        # Save splits
        for split_name, split_df in splits.items():
            split_file = self.output_dir / f"{split_name}.csv"
            split_df.to_csv(split_file, index=False)
            logger.info(f"{split_name} split saved: {split_file}")

        # Save metadata
        metadata = {
            "total_samples": len(combined_df),
            "domains": list(combined_df["domain"].unique()),
            "sources": list(combined_df["data_source"].unique()) if "data_source" in combined_df.columns else [],
            "domain_distribution": dict(combined_df["domain"].value_counts()),
            "text_stats": {
                "mean_length": float(combined_df["text_length"].mean()),
                "median_length": float(combined_df["text_length"].median()),
                "min_length": int(combined_df["text_length"].min()),
                "max_length": int(combined_df["text_length"].max())
            },
            "splits": {
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "test": len(splits["test"])
            }
        }

        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Dataset metadata saved: {metadata_file}")

    def generate_summary_report(self, combined_df: pd.DataFrame) -> str:
        """Generate a summary report of the collected data."""

        report = []
        report.append("# SKAWR Search ML - Data Collection Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"- **Total samples**: {len(combined_df):,}")
        report.append(f"- **Total domains**: {combined_df['domain'].nunique()}")
        report.append(f"- **Data sources**: {', '.join(combined_df['data_source'].unique()) if 'data_source' in combined_df.columns else 'N/A'}")
        report.append("")

        # Domain distribution
        report.append("## Domain Distribution")
        domain_counts = combined_df['domain'].value_counts()
        for domain, count in domain_counts.items():
            percentage = (count / len(combined_df)) * 100
            report.append(f"- **{domain}**: {count:,} ({percentage:.1f}%)")
        report.append("")

        # Text statistics
        report.append("## Text Statistics")
        text_stats = combined_df['text_length'].describe()
        report.append(f"- **Mean length**: {text_stats['mean']:.1f} characters")
        report.append(f"- **Median length**: {text_stats['50%']:.1f} characters")
        report.append(f"- **Min length**: {int(text_stats['min'])} characters")
        report.append(f"- **Max length**: {int(text_stats['max'])} characters")
        report.append("")

        # Sample data
        report.append("## Sample Data")
        for domain in combined_df['domain'].unique()[:3]:  # Show samples from 3 domains
            domain_samples = combined_df[combined_df['domain'] == domain].head(2)
            report.append(f"### {domain.title()} Samples")
            for _, sample in domain_samples.iterrows():
                report.append(f"- {sample['text'][:100]}...")
            report.append("")

        return "\n".join(report)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Orchestrate SKAWR ML data collection")

    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["kaggle", "reddit", "wikipedia"],
        default=["kaggle", "reddit"],
        help="Data sources to collect from"
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["automotive", "electronics", "fashion", "general"],
        default=["automotive", "electronics"],
        help="Domains to collect data for"
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed data"
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator(args.output_dir)

    logger.info("🚀 Starting SKAWR ML data collection...")
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Domains: {args.domains}")

    # Collect data from all sources
    source_data = orchestrator.collect_all_data(args.sources, args.domains)

    # Combine all sources
    combined_df = orchestrator.combine_all_sources(source_data)

    if combined_df.empty:
        logger.error("❌ No data collected! Exiting.")
        sys.exit(1)

    # Create training splits
    splits = orchestrator.create_training_splits(combined_df)

    # Save all datasets
    orchestrator.save_datasets(combined_df, splits)

    # Generate and save report
    report = orchestrator.generate_summary_report(combined_df)
    report_file = Path(args.output_dir) / "collection_report.md"
    with open(report_file, 'w') as f:
        f.write(report)

    logger.success(f"✅ Data collection complete!")
    logger.success(f"📊 Total samples: {len(combined_df):,}")
    logger.success(f"📁 Data saved to: {args.output_dir}")
    logger.success(f"📋 Report saved to: {report_file}")


if __name__ == "__main__":
    main()