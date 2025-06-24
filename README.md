# TiRex Fine-tuning for Bitcoin Price Forecasting

A comprehensive fine-tuning implementation of the TiRex (Time Series Regression with xLSTM) model for Bitcoin price forecasting with multi-quantile predictions.

## Overview

This project implements fine-tuning of the pre-trained TiRex model for Bitcoin price forecasting across multiple horizons (6h, 12h, 24h) using a multi-quantile approach. The implementation includes:

- **Data Processing**: Bitcoin 5-minute OHLCV data from 2022-2025
- **Model Architecture**: TiRex with xLSTM encoder and pinball quantile head
- **Training**: Multi-quantile pinball loss with cosine annealing + warmup
- **Evaluation**: Walk-forward backtesting with MAE, MASE, and CRPS metrics

## Features

### Model Architecture
- 35M parameter TiRex model with xLSTM encoder
- Patch size: 32 time-steps
- Context length: 2016 time-steps (7 days of 5-min bars)
- Multi-quantile predictions: Q={0.1, 0.2, ..., 0.9}
- Prediction horizons: 72, 144, 288 steps (6h, 12h, 24h)

### Training Features
- Multi-quantile pinball loss function
- AdamW optimizer with cosine annealing + 5% linear warmup
- Early stopping with validation CRPS monitoring
- Gradient clipping and weight decay regularization
- Mixed precision training support

### Evaluation Metrics
- **MAE**: Mean Absolute Error on median predictions
- **MASE**: Mean Absolute Scaled Error vs. seasonal naive
- **CRPS**: Continuous Ranked Probability Score for probabilistic forecasts

## Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA GPU (optional, CPU supported with performance limitations)

### Setup

1. **Clone and install TiRex**:
```bash
git clone https://github.com/NX-AI/tirex.git
cd tirex
pip install -e .
```

2. **Install additional dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml tqdm
```

3. **Set environment variables**:
```bash
# For macOS/CPU compatibility
export TIREX_NO_CUDA=1

# Optional: Binance API credentials for data fetching
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## Data Preparation

The project expects Bitcoin 5-minute OHLCV data in CSV format:

```
datetime,timestamp,open,high,low,close,volume,number_of_trades,quote_asset_volume,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
2022-01-01 00:00:00,1640995200000,47717.36,47719.37,47658.23,47673.04,23.45678,...
```

### Data Splits
- **Training**: 2022-01-01 → 2024-12-31
- **Validation**: 2025-01-01 → 2025-06-23  
- **Test**: 2025-06-24 → Present

## Usage

### Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Key parameters
data:
  context_length: 2016  # 7 days of 5-min bars
  prediction_lengths: [72, 144, 288]  # 6h, 12h, 24h

training:
  epochs: 10
  batch_size: 16
  learning_rate: 1.0e-4
  patience: 2

model:
  freeze_encoder: false  # Fine-tune entire model
  quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### Training

Run the training script:

```bash
python train.py
```

Training progress:
- Saves checkpoints to `checkpoints/`
- Monitors validation CRPS for early stopping
- Logs metrics every epoch

### Backtesting

After training, run walk-forward backtesting:

```bash
python backtest.py
```

This generates:
- Daily forecasts from 2025-06-24 onwards
- Performance metrics for each horizon
- Detailed results in `backtest_results/`

### Example Output

```
BACKTEST RESULTS SUMMARY
============================================================

72h Forecast (6.0 hours):
  Forecasts: 42
  MAE:  1247.32 ± 156.78
  MASE: 0.89 ± 0.12
  CRPS: -0.234 ± 0.045

144h Forecast (12.0 hours):
  Forecasts: 42  
  MAE:  1456.89 ± 201.34
  MASE: 1.03 ± 0.15
  CRPS: -0.278 ± 0.052

288h Forecast (24.0 hours):
  Forecasts: 42
  MAE:  1678.45 ± 267.91
  MASE: 1.19 ± 0.18
  CRPS: -0.321 ± 0.061
```

## Code Organization

```
├── data.py              # Data loading and preprocessing
├── model.py             # TiRex wrapper with multi-quantile loss
├── train.py             # Training script with validation
├── backtest.py          # Walk-forward backtesting
├── config.yaml          # Configuration file
├── checkpoints/         # Model checkpoints
├── backtest_results/    # Backtest outputs
└── README.md           # This file
```

### Key Components

**data.py**:
- `BitcoinDataLoader`: Loads and preprocesses Bitcoin data
- `TimeSeriesDataset`: PyTorch dataset with sliding windows
- `ValidationDataset`: Validation-specific dataset

**model.py**:
- `TiRexFineTuner`: Main model wrapper
- `PinballLoss`: Multi-quantile pinball loss
- `CRPS`: Continuous Ranked Probability Score

**train.py**:
- `TiRexTrainer`: Training manager with early stopping
- `CosineAnnealingWithWarmup`: Learning rate scheduler
- Validation loop with metric logging

**backtest.py**:
- `WalkForwardBacktest`: Walk-forward validation
- `ForecastMetrics`: MAE, MASE, CRPS calculations
- Results summarization and export

## Technical Details

### Context Window Design
- **Length**: 2016 time-steps (7 days × 288 intervals/day)
- **Divisible by 32**: Required for TiRex patch embedding
- **Frequency**: 5-minute Bitcoin price data

### Multi-quantile Approach
- **Quantiles**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- **Loss**: Pinball loss averaged across quantiles
- **Evaluation**: CRPS using quantile approximation

### Optimizer Configuration
- **AdamW**: lr=1e-4, weight_decay=1e-2
- **Scheduler**: Cosine annealing with 5% linear warmup
- **Final LR**: 1e-5
- **Gradient clipping**: 1.0

## Performance Considerations

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB RAM, CUDA GPU with CC >= 8.0
- **macOS**: CPU-only with `TIREX_NO_CUDA=1`

### Training Time
- **CPU**: ~8-12 hours for 10 epochs
- **GPU**: ~2-4 hours for 10 epochs
- **Memory**: ~4-6GB during training

## Troubleshooting

### Common Issues

1. **CUDA kernel errors on macOS**:
   ```bash
   export TIREX_NO_CUDA=1
   ```

2. **Out of memory**:
   - Reduce batch_size in config.yaml
   - Reduce context_length (must be divisible by 32)

3. **Slow training**:
   - Use GPU if available
   - Reduce validation frequency
   - Use mixed precision training

4. **Poor performance**:
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and normalization

## Future Enhancements

- [ ] Multi-series forecasting (other cryptocurrencies)
- [ ] Feature engineering (technical indicators)
- [ ] Hyperparameter optimization
- [ ] Real-time forecasting API
- [ ] Model ensembling
- [ ] Alternative loss functions

## References

- [TiRex Paper](https://arxiv.org/abs/2505.23719): Zero-Shot Forecasting across Long and Short Horizons
- [xLSTM](https://github.com/NX-AI/xlstm): Extended Long Short-Term Memory
- [TiRex Model](https://huggingface.co/NX-AI/TiRex): Pre-trained time series forecasting model

## License

This project is for educational and research purposes. Please refer to the TiRex license for model usage terms.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{auerTiRexZeroShotForecasting2025,
  title = {{{TiRex}}: {{Zero-Shot Forecasting Across Long}} and {{Short Horizons}} with {{Enhanced In-Context Learning}}},
  author = {Auer, Andreas and Podest, Patrick and Klotz, Daniel and B{\"o}ck, Sebastian and Klambauer, G{\"u}nter and Hochreiter, Sepp},
  journal = {ArXiv},
  volume = {2505.23719},   
  year = {2025}
}
```