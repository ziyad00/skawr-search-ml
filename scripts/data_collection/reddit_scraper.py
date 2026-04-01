#!/usr/bin/env python3
"""
Reddit Data Scraper for SKAWR Search ML Training

Scrapes relevant subreddits for marketplace and product discussions
to enhance training data with real user queries and conversations.
"""

import os
import sys
import json
import time
import pandas as pd
import praw
from pathlib import Path
from typing import List, Dict, Optional, Set
from loguru import logger
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class RedditDataScraper:
    """Scrapes Reddit for product discussions and marketplace posts."""

    def __init__(self, data_dir: str = "data/reddit"):
        self.data_dir = Path(PROJECT_ROOT) / data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Reddit API (requires credentials in environment)
        self.reddit = self._init_reddit_api()

        # Configure subreddits for each domain
        self.subreddit_configs = self._get_subreddit_configs()

        logger.info(f"Initialized Reddit scraper, data dir: {self.data_dir}")

    def _init_reddit_api(self) -> praw.Reddit:
        """Initialize Reddit API with credentials."""
        # For now, use read-only mode
        try:
            reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID", "dummy"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET", "dummy"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "SKAWR-ML-Scraper/1.0"),
                username=os.getenv("REDDIT_USERNAME"),
                password=os.getenv("REDDIT_PASSWORD")
            )
            # Test connection
            reddit.auth.scopes()
            return reddit
        except Exception as e:
            logger.warning(f"Reddit API not configured: {e}")
            logger.info("Will use demo mode with static data")
            return None

    def _get_subreddit_configs(self) -> Dict:
        """Configure subreddits to scrape for each domain."""
        return {
            "automotive": {
                "subreddits": [
                    "cars", "UsedCars", "whatcarshouldIbuy", "SportsCars",
                    "BMW", "Mercedes", "Toyota", "Ford", "Honda",
                    "CarSales", "AutoTrader", "CarMarket"
                ],
                "keywords": [
                    "sports car", "sedan", "SUV", "truck", "coupe", "convertible",
                    "for sale", "selling", "buying", "price", "specs", "review"
                ],
                "max_posts_per_sub": 100
            },
            "electronics": {
                "subreddits": [
                    "electronics", "gadgets", "BuyItForLife", "techdeals",
                    "laptops", "phones", "apple", "android", "samsung",
                    "buildapc", "consoles", "gaming"
                ],
                "keywords": [
                    "laptop", "phone", "iphone", "android", "gaming", "computer",
                    "for sale", "review", "recommend", "specs", "price"
                ],
                "max_posts_per_sub": 100
            },
            "fashion": {
                "subreddits": [
                    "fashion", "malefashion", "femalefashion", "streetwear",
                    "sneakers", "watches", "jewelry", "bags",
                    "fashionsales", "frugalmalefashion"
                ],
                "keywords": [
                    "dress", "shirt", "shoes", "sneakers", "watch", "bag",
                    "for sale", "outfit", "style", "brand", "price"
                ],
                "max_posts_per_sub": 100
            },
            "general": {
                "subreddits": [
                    "marketplace", "forsale", "deals", "frugal", "BuyItForLife",
                    "whatisthisthing", "HelpMeFind", "tipofmytongue"
                ],
                "keywords": [
                    "looking for", "need", "buy", "sell", "price", "where to find",
                    "recommend", "review", "quality", "best"
                ],
                "max_posts_per_sub": 100
            }
        }

    def scrape_subreddit(self, subreddit_name: str, domain: str, max_posts: int = 100) -> List[Dict]:
        """Scrape posts from a single subreddit."""
        if self.reddit is None:
            return self._generate_demo_data(subreddit_name, domain, max_posts)

        posts_data = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Scraping r/{subreddit_name} (max {max_posts} posts)")

            # Get hot posts
            posts = subreddit.hot(limit=max_posts)

            for post in posts:
                try:
                    # Skip stickied/pinned posts
                    if post.stickied:
                        continue

                    # Extract post data
                    post_data = {
                        "id": post.id,
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "subreddit": subreddit_name,
                        "domain": domain,
                        "url": f"https://reddit.com{post.permalink}",
                        "author": str(post.author) if post.author else "[deleted]"
                    }

                    # Get top comments for additional context
                    post.comments.replace_more(limit=0)  # Remove MoreComments
                    top_comments = []

                    for comment in post.comments[:5]:  # Top 5 comments
                        if hasattr(comment, 'body') and len(comment.body) > 20:
                            top_comments.append(comment.body[:500])  # Limit comment length

                    post_data["top_comments"] = top_comments

                    posts_data.append(post_data)

                    if len(posts_data) % 10 == 0:
                        logger.info(f"Scraped {len(posts_data)} posts from r/{subreddit_name}")

                except Exception as e:
                    logger.warning(f"Error processing post {post.id}: {e}")
                    continue

                # Rate limiting
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to scrape r/{subreddit_name}: {e}")

        logger.success(f"Scraped {len(posts_data)} posts from r/{subreddit_name}")
        return posts_data

    def _generate_demo_data(self, subreddit_name: str, domain: str, max_posts: int) -> List[Dict]:
        """Generate demo data when Reddit API is not available."""
        logger.info(f"Generating demo data for r/{subreddit_name}")

        demo_posts = []
        keywords = self.subreddit_configs[domain]["keywords"]

        for i in range(min(max_posts, 20)):  # Limit demo data
            demo_posts.append({
                "id": f"demo_{subreddit_name}_{i}",
                "title": f"Demo post about {keywords[i % len(keywords)]} in {domain}",
                "selftext": f"This is a demo post discussing {keywords[i % len(keywords)]} for training purposes.",
                "score": 10 + (i % 50),
                "num_comments": i % 20,
                "created_utc": time.time() - (i * 3600),
                "subreddit": subreddit_name,
                "domain": domain,
                "url": f"https://reddit.com/demo/{subreddit_name}/{i}",
                "author": "demo_user",
                "top_comments": [f"Demo comment about {keywords[(i+1) % len(keywords)]}"]
            })

        return demo_posts

    def process_posts_for_training(self, posts_data: List[Dict], domain: str) -> pd.DataFrame:
        """Process scraped posts into training format."""
        processed_data = []

        for post in posts_data:
            # Combine title and text
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            comments = " ".join(post.get("top_comments", []))

            # Create different text combinations for training variety
            text_variants = [
                title,  # Title only
                f"{title} {selftext}".strip(),  # Title + post text
                f"{title} {comments}".strip()  # Title + comments
            ]

            for text in text_variants:
                if len(text.strip()) > 10:  # Minimum length
                    processed_data.append({
                        "text": text.strip(),
                        "domain": domain,
                        "category": post.get("subreddit", "unknown"),
                        "source": f"reddit_r_{post.get('subreddit')}",
                        "description": f"Reddit post from r/{post.get('subreddit')}",
                        "score": post.get("score", 0),
                        "engagement": post.get("num_comments", 0),
                        "post_id": post.get("id")
                    })

        return pd.DataFrame(processed_data)

    def scrape_domain(self, domain: str) -> pd.DataFrame:
        """Scrape all subreddits for a specific domain."""
        config = self.subreddit_configs[domain]
        all_posts = []

        logger.info(f"\n=== Scraping {domain.upper()} domain ===")

        for subreddit_name in config["subreddits"]:
            posts = self.scrape_subreddit(
                subreddit_name,
                domain,
                config["max_posts_per_sub"]
            )
            all_posts.extend(posts)

            # Rate limiting between subreddits
            time.sleep(1)

        # Process all posts for training
        processed_df = self.process_posts_for_training(all_posts, domain)

        # Save domain data
        output_file = self.data_dir / f"{domain}_reddit_data.csv"
        processed_df.to_csv(output_file, index=False)

        logger.success(f"Domain {domain}: {len(processed_df)} training samples saved to {output_file}")
        return processed_df

    def scrape_all_domains(self, domains: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Scrape all configured domains."""
        if domains is None:
            domains = list(self.subreddit_configs.keys())

        results = {}

        for domain in domains:
            try:
                df = self.scrape_domain(domain)
                results[domain] = df
            except Exception as e:
                logger.error(f"Failed to scrape domain {domain}: {e}")
                results[domain] = pd.DataFrame()

        return results

    def combine_reddit_data(self) -> pd.DataFrame:
        """Combine all scraped Reddit data."""
        all_data = []

        # Load all domain files
        for domain in self.subreddit_configs.keys():
            file_path = self.data_dir / f"{domain}_reddit_data.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} samples from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        if not all_data:
            logger.warning("No Reddit data found!")
            return pd.DataFrame()

        # Combine and clean
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        combined_df = combined_df.dropna(subset=['text'])

        # Save combined data
        output_file = self.data_dir / "combined_reddit_data.csv"
        combined_df.to_csv(output_file, index=False)

        logger.success(f"Combined Reddit data: {len(combined_df)} samples saved to {output_file}")
        return combined_df


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Reddit for SKAWR ML training data")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["automotive", "electronics", "fashion", "general"],
        help="Domains to scrape (default: all)"
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Use demo data instead of real Reddit API"
    )

    args = parser.parse_args()

    scraper = RedditDataScraper()

    if args.demo_mode:
        logger.info("Running in demo mode")

    # Scrape domains
    results = scraper.scrape_all_domains(args.domains)

    # Combine all data
    combined_df = scraper.combine_reddit_data()

    logger.success(f"Reddit scraping complete! {len(combined_df)} total samples collected.")


if __name__ == "__main__":
    main()