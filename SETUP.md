# SKAWR Search ML - Setup Instructions

## Quick Start

1. **Test Setup**
   ```bash
   python3 scripts/test_setup.py
   ```

2. **Install Dependencies**
   ```bash
   # Core dependencies (required)
   pip3 install pandas numpy pyyaml tqdm loguru

   # ML dependencies (for training)
   pip3 install torch transformers scikit-learn wandb

   # Data collection dependencies (optional)
   pip3 install kaggle praw beautifulsoup4 requests
   ```

3. **Test Training Pipeline**
   ```bash
   # Create dummy data and test training
   python3 scripts/train_model.py --create-dummy-data --dummy-samples 100
   ```

## Detailed Setup

### 1. Environment Setup

Create virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

**Option A: Install all dependencies**
```bash
pip3 install -r requirements.txt
```

**Option B: Install selectively**
```bash
# Minimal setup (for testing)
pip3 install pandas numpy pyyaml

# Add ML training
pip3 install torch transformers scikit-learn

# Add experiment tracking
pip3 install wandb

# Add data collection
pip3 install kaggle praw beautifulsoup4
```

### 3. Configuration

Copy environment template:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# Kaggle API (for data collection)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_key
```

### 4. Data Collection

**Option A: Use dummy data (quick start)**
```bash
python3 scripts/train_model.py --create-dummy-data
```

**Option B: Collect real data**
```bash
# Demo mode (no API keys required)
python3 scripts/data_collection/run_data_collection.py --sources reddit --demo-mode

# Real data collection (requires API keys)
python3 scripts/data_collection/run_data_collection.py --sources kaggle reddit
```

### 5. Training

**Quick test:**
```bash
python3 scripts/train_model.py --create-dummy-data --dummy-samples 100
```

**Full training:**
```bash
python3 scripts/train_model.py
```

## Hardware Requirements

### Minimum (CPU training)
- 8GB RAM
- 10GB disk space
- Python 3.8+

### Recommended (GPU training)
- 16GB RAM
- NVIDIA GPU with 8GB VRAM
- 50GB disk space
- CUDA 11.0+

### Cloud Options
- Google Colab (free tier with GPU)
- AWS EC2 (p3.2xlarge or similar)
- Google Cloud Platform (compute with GPU)

## Project Structure

```
skawr-search-ml/
├── config/                 # Configuration files
│   └── model_config.yaml
├── data/                   # Data storage (gitignored)
│   ├── processed/          # Training data
│   ├── kaggle/            # Kaggle downloads
│   └── reddit/            # Reddit scraping
├── models/                # Model code
│   ├── architecture/       # Model definitions
│   ├── training/          # Training logic
│   └── evaluation/        # Evaluation metrics
├── scripts/               # Executable scripts
│   ├── data_collection/   # Data gathering
│   ├── train_model.py     # Main training script
│   └── test_setup.py      # Setup verification
├── logs/                  # Training logs
└── requirements.txt       # Dependencies
```

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Make sure you're in the project root
cd /path/to/skawr-search-ml

# Check Python path
python3 scripts/test_setup.py
```

**2. CUDA not available**
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only PyTorch if needed
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Out of memory during training**
```bash
# Reduce batch size in config/model_config.yaml
training:
  batch_size: 16  # Reduce from 32
```

**4. Data collection fails**
```bash
# Use demo mode first
python3 scripts/data_collection/run_data_collection.py --demo-mode
```

### Performance Tips

1. **Use GPU if available** - Training is 10-20x faster
2. **Use mixed precision** - Enables larger batch sizes
3. **Use multiple workers** - Faster data loading
4. **Monitor with wandb** - Track training progress

## Next Steps

1. **Verify setup:** `python3 scripts/test_setup.py`
2. **Test training:** `python3 scripts/train_model.py --create-dummy-data`
3. **Collect real data:** Use Kaggle and Reddit scrapers
4. **Train full model:** Run on GPU with real data
5. **Integrate with SKAWR:** Deploy trained model

## Getting Help

- Check logs in `logs/` directory
- Run `scripts/test_setup.py` to diagnose issues
- Review configuration in `config/model_config.yaml`
- Ensure all dependencies are installed from `requirements.txt`