import os
import platform

# Only disable CUDA kernels on macOS/non-Windows systems
if platform.system() != 'Windows':
    os.environ['TIREX_NO_CUDA'] = '1'

import torch
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta

from data import BitcoinDataLoader, TimeSeriesDataset
from model import create_model, TiRexFineTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForecastMetrics:
    """Calculate various forecasting metrics."""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 288) -> float:
        """Mean Absolute Scaled Error."""
        # Calculate naive forecast MAE on training data
        if len(y_train) < seasonal_period:
            # Fall back to simple differencing if not enough seasonal data
            naive_mae = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_forecast = y_train[:-seasonal_period]
            naive_mae = np.mean(np.abs(y_train[seasonal_period:] - naive_forecast))
        
        if naive_mae == 0:
            return np.inf
        
        mae = ForecastMetrics.mae(y_true, y_pred)
        return mae / naive_mae
    
    @staticmethod
    def crps_quantile(y_true: np.ndarray, quantile_forecasts: np.ndarray, quantiles: List[float]) -> float:
        """CRPS using quantile forecasts."""
        quantiles = np.array(quantiles)
        
        # Expand dimensions for broadcasting
        y_true_expanded = y_true[..., np.newaxis]  # (..., 1)
        quantiles_expanded = quantiles[np.newaxis, ...]  # (1, num_quantiles)
        
        # Indicator function: 1 if y_true <= quantile_forecast, 0 otherwise
        indicator = (y_true_expanded <= quantile_forecasts).astype(float)
        
        # CRPS calculation
        crps_values = 2 * (y_true_expanded - quantile_forecasts) * (indicator - quantiles_expanded)
        
        # Average over quantiles and time steps
        return np.mean(crps_values)
    
    @staticmethod
    def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Quantile loss for a specific quantile."""
        error = y_true - y_pred
        return np.mean(np.where(error >= 0, quantile * error, (quantile - 1) * error))


class WalkForwardBacktest:
    """Walk-forward backtesting for time series forecasting."""
    
    def __init__(
        self,
        model: TiRexFineTuner,
        data_loader: BitcoinDataLoader,
        config: Dict,
        start_date: str = "2025-06-24",
        context_length: int = 2016
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.start_date = pd.Timestamp(start_date)
        self.context_length = context_length
        self.prediction_lengths = config['data']['prediction_lengths']
        self.quantiles = config['model']['quantiles']
        
        # Load test data (from start_date onwards)
        self.test_data = self._prepare_test_data()
        
        logger.info(f"Backtest initialized from {start_date}")
        logger.info(f"Test data shape: {self.test_data.shape}")
    
    def _prepare_test_data(self) -> pd.Series:
        """Prepare test data starting from backtest start date."""
        # Load all 2025 data
        val_df = pd.read_csv(
            self.config['data']['val_path'],
            parse_dates=["datetime"]
        ).set_index("datetime")
        
        val_df = val_df.asfreq("5min").ffill()
        
        # Filter to test period (from start_date onwards)
        test_df = val_df[val_df.index >= self.start_date]
        
        if len(test_df) == 0:
            raise ValueError(f"No test data available from {self.start_date}")
        
        return test_df['close']
    
    def generate_forecast_windows(self) -> List[Tuple[pd.Timestamp, np.ndarray, Dict[str, np.ndarray]]]:
        """Generate all forecast windows for walk-forward backtesting."""
        windows = []
        
        # We need context_length + max_prediction_length data points for each window
        max_pred_len = max(self.prediction_lengths)
        min_required_length = self.context_length + max_pred_len
        
        # Start from the point where we have enough context
        start_idx = self.context_length
        
        while start_idx + max_pred_len <= len(self.test_data):
            # Get context window
            context_start_idx = start_idx - self.context_length
            context = self.test_data.iloc[context_start_idx:start_idx].values
            
            # Get targets for all prediction lengths
            targets = {}
            forecast_timestamp = self.test_data.index[start_idx]
            
            for pred_len in self.prediction_lengths:
                if start_idx + pred_len <= len(self.test_data):
                    targets[f"target_{pred_len}"] = self.test_data.iloc[start_idx:start_idx + pred_len].values
            
            if targets:  # Only add if we have at least one valid target
                windows.append((forecast_timestamp, context, targets))
            
            # Move to next day (288 5-minute intervals = 1 day)
            start_idx += 288
        
        return windows
    
    def run_backtest(self) -> Dict[str, List[Dict]]:
        """Run the complete walk-forward backtest."""
        logger.info("Starting walk-forward backtest...")
        
        # Generate forecast windows
        windows = self.generate_forecast_windows()
        logger.info(f"Generated {len(windows)} forecast windows")
        
        if len(windows) == 0:
            raise ValueError("No valid forecast windows generated")
        
        # Initialize results storage
        results = {
            'forecasts': [],
            'metrics': {f'horizon_{pl}': [] for pl in self.prediction_lengths}
        }
        
        # Prepare normalization stats from training data
        train_df, _ = self.data_loader.load_data()
        train_series = train_df['close']
        train_mean = train_series.mean()
        train_std = train_series.std()
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (timestamp, context, targets) in enumerate(tqdm(windows, desc="Backtesting")):
                # Normalize context
                context_normalized = (context - train_mean) / train_std
                
                # Convert to tensor
                context_tensor = torch.tensor(context_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # Generate predictions
                predictions = self.model.predict_batch(context_tensor)
                
                # Store forecast
                forecast_entry = {
                    'timestamp': timestamp.isoformat(),
                    'context_start': self.test_data.index[i * 288].isoformat() if i * 288 < len(self.test_data) else None
                }
                
                # Process each prediction horizon
                for pred_len in self.prediction_lengths:
                    pred_key = f"pred_{pred_len}"
                    target_key = f"target_{pred_len}"
                    
                    if pred_key in predictions and target_key in targets:
                        # Get predictions and denormalize
                        pred_tensor = predictions[pred_key].cpu().numpy()[0]  # (pred_len, num_quantiles)
                        pred_denorm = pred_tensor * train_std + train_mean
                        
                        # Get true values
                        y_true = targets[target_key]
                        
                        # Calculate metrics
                        median_pred = pred_denorm[:, len(self.quantiles) // 2]  # 0.5 quantile
                        
                        mae = ForecastMetrics.mae(y_true, median_pred)
                        crps = ForecastMetrics.crps_quantile(y_true, pred_denorm, self.quantiles)
                        
                        # MASE requires training data
                        mase = ForecastMetrics.mase(y_true, median_pred, train_series.values)
                        
                        # Store metrics
                        metric_entry = {
                            'timestamp': timestamp.isoformat(),
                            'horizon': pred_len,
                            'mae': float(mae),
                            'mase': float(mase),
                            'crps': float(crps)
                        }
                        results['metrics'][f'horizon_{pred_len}'].append(metric_entry)
                        
                        # Store forecast details
                        forecast_entry[f'pred_{pred_len}'] = {
                            'quantiles': pred_denorm.tolist(),
                            'median': median_pred.tolist(),
                            'actual': y_true.tolist()
                        }
                
                results['forecasts'].append(forecast_entry)
        
        return results
    
    def summarize_results(self, results: Dict) -> Dict[str, Dict]:
        """Summarize backtest results."""
        summary = {}
        
        for pred_len in self.prediction_lengths:
            horizon_metrics = results['metrics'][f'horizon_{pred_len}']
            
            if horizon_metrics:
                mae_values = [m['mae'] for m in horizon_metrics]
                mase_values = [m['mase'] for m in horizon_metrics if not np.isinf(m['mase'])]
                crps_values = [m['crps'] for m in horizon_metrics]
                
                summary[f'horizon_{pred_len}h'] = {
                    'count': len(horizon_metrics),
                    'mae': {
                        'mean': float(np.mean(mae_values)),
                        'std': float(np.std(mae_values)),
                        'median': float(np.median(mae_values))
                    },
                    'mase': {
                        'mean': float(np.mean(mase_values)) if mase_values else np.inf,
                        'std': float(np.std(mase_values)) if mase_values else 0.0,
                        'median': float(np.median(mase_values)) if mase_values else np.inf
                    },
                    'crps': {
                        'mean': float(np.mean(crps_values)),
                        'std': float(np.std(crps_values)),
                        'median': float(np.median(crps_values))
                    }
                }
        
        return summary


def load_best_model(config: Dict) -> TiRexFineTuner:
    """Load the best trained model."""
    best_checkpoint_path = Path(config['training']['save_dir']) / 'best.pt'
    
    if not best_checkpoint_path.exists():
        logger.warning("Best checkpoint not found, using latest checkpoint")
        best_checkpoint_path = Path(config['training']['save_dir']) / 'latest.pt'
    
    if not best_checkpoint_path.exists():
        raise FileNotFoundError("No model checkpoint found. Please run training first.")
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        prediction_lengths=config['data']['prediction_lengths'],
        quantiles=config['model']['quantiles'],
        freeze_encoder=config['model']['freeze_encoder'],
        device=config['training']['device']
    )
    
    # Load checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from {best_checkpoint_path}")
    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    logger.info(f"Best validation CRPS: {checkpoint['best_val_crps']:.4f}")
    
    return model


def main():
    """Main backtest function."""
    # Load configuration
    config_path = "config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file {config_path} not found!")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info("Loading trained model...")
    model = load_best_model(config)
    
    # Create data loader
    data_loader = BitcoinDataLoader(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path']
    )
    
    # Create backtester
    backtester = WalkForwardBacktest(
        model=model,
        data_loader=data_loader,
        config=config,
        start_date="2025-06-24",  # Start backtest from this date
        context_length=config['data']['context_length']
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Summarize results
    summary = backtester.summarize_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    for horizon, metrics in summary.items():
        pred_len = horizon.split('_')[1]
        print(f"\n{pred_len} Forecast ({int(pred_len[:-1]) * 5/60:.1f} hours):")
        print(f"  Forecasts: {metrics['count']}")
        print(f"  MAE:  {metrics['mae']['mean']:.2f} ± {metrics['mae']['std']:.2f}")
        print(f"  MASE: {metrics['mase']['mean']:.3f} ± {metrics['mase']['std']:.3f}")
        print(f"  CRPS: {metrics['crps']['mean']:.4f} ± {metrics['crps']['std']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / f"backtest_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    with open(results_dir / f"backtest_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    
    print(f"\nDetailed results saved to: backtest_results_{timestamp}.json")
    print(f"Summary saved to: backtest_summary_{timestamp}.json")


if __name__ == "__main__":
    main()