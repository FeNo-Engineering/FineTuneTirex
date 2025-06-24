import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BitcoinDataLoader:
    """Loads and preprocesses Bitcoin price data for TiRex fine-tuning."""
    
    def __init__(self, train_path: str, val_path: str):
        self.train_path = train_path
        self.val_path = val_path
        self.train_df = None
        self.val_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data."""
        logger.info("Loading Bitcoin price data...")
        
        # Load training data (2022-2024)
        self.train_df = pd.read_csv(
            self.train_path,
            parse_dates=["datetime"]
        ).set_index("datetime")
        
        # Load validation data (2025)
        self.val_df = pd.read_csv(
            self.val_path,
            parse_dates=["datetime"]
        ).set_index("datetime")
        
        # Ensure 5-minute frequency and fill gaps
        self.train_df = self.train_df.asfreq("5min").ffill()
        self.val_df = self.val_df.asfreq("5min").ffill()
        
        # Filter to specified date ranges
        self.train_df = self.train_df[
            (self.train_df.index >= "2022-01-01") & 
            (self.train_df.index <= "2024-12-31")
        ]
        self.val_df = self.val_df[
            (self.val_df.index >= "2025-01-01") & 
            (self.val_df.index <= "2025-06-23")
        ]
        
        logger.info(f"Training data: {len(self.train_df)} samples ({self.train_df.index[0]} to {self.train_df.index[-1]})")
        logger.info(f"Validation data: {len(self.val_df)} samples ({self.val_df.index[0]} to {self.val_df.index[-1]})")
        
        return self.train_df, self.val_df
    
    def get_price_series(self, df: pd.DataFrame, feature: str = "close") -> pd.Series:
        """Extract price series for modeling."""
        return df[feature]


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series windows with TiRex specifications."""
    
    def __init__(
        self,
        data: pd.Series,
        context_length: int = 2016,  # 7 days * 288 (5min intervals per day), divisible by 32
        prediction_lengths: List[int] = [72, 144, 288],  # 6h, 12h, 24h
        stride: int = 1,
        normalize: bool = True
    ):
        self.data = data
        self.context_length = context_length
        self.prediction_lengths = prediction_lengths
        self.max_prediction_length = max(prediction_lengths)
        self.stride = stride
        self.normalize = normalize
        
        # Convert to numpy for faster indexing
        self.values = data.values.astype(np.float32)
        self.timestamps = data.index
        
        # Calculate normalization statistics on training data
        if self.normalize:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            self.values = (self.values - self.mean) / self.std
        
        # Calculate valid indices for windowing
        self.valid_indices = self._get_valid_indices()
        
        logger.info(f"Dataset created with {len(self.valid_indices)} windows")
        logger.info(f"Context length: {context_length}, Max prediction length: {self.max_prediction_length}")
        
    def _get_valid_indices(self) -> List[int]:
        """Get valid starting indices for complete windows."""
        min_length = self.context_length + self.max_prediction_length
        valid_indices = []
        
        for i in range(0, len(self.values) - min_length + 1, self.stride):
            valid_indices.append(i)
            
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with context and multiple prediction horizons."""
        start_idx = self.valid_indices[idx]
        
        # Extract context window
        context = self.values[start_idx:start_idx + self.context_length]
        
        # Extract targets for all prediction lengths
        targets = {}
        for pred_len in self.prediction_lengths:
            target_start = start_idx + self.context_length
            target_end = target_start + pred_len
            targets[f"target_{pred_len}"] = self.values[target_start:target_end]
        
        # Convert to tensors
        sample = {
            "context": torch.tensor(context, dtype=torch.float32).unsqueeze(0),  # Add batch dimension
            "timestamp_idx": torch.tensor(start_idx + self.context_length - 1, dtype=torch.long)  # Timestamp index
        }
        
        for key, target in targets.items():
            sample[key] = torch.tensor(target, dtype=torch.float32)
            
        return sample
    
    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values back to original scale."""
        if self.normalize:
            return values * self.std + self.mean
        return values


class ValidationDataset(Dataset):
    """Dataset for sliding window validation over time series."""
    
    def __init__(
        self,
        data: pd.Series,
        context_length: int = 2016,
        prediction_lengths: List[int] = [72, 144, 288],
        step_size: int = 288,  # Step by 1 day for validation
        normalize_stats: Optional[Tuple[float, float]] = None
    ):
        self.data = data
        self.context_length = context_length
        self.prediction_lengths = prediction_lengths
        self.max_prediction_length = max(prediction_lengths)
        self.step_size = step_size
        
        # Convert to numpy
        self.values = data.values.astype(np.float32)
        self.timestamps = data.index
        
        # Use provided normalization stats or calculate from data
        if normalize_stats:
            self.mean, self.std = normalize_stats
            self.values = (self.values - self.mean) / self.std
        else:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            self.values = (self.values - self.mean) / self.std
        
        # Calculate validation windows
        self.valid_indices = self._get_validation_windows()
        
        logger.info(f"Validation dataset created with {len(self.valid_indices)} windows")
    
    def _get_validation_windows(self) -> List[int]:
        """Get validation window starting indices."""
        min_length = self.context_length + self.max_prediction_length
        valid_indices = []
        
        start_idx = self.context_length
        while start_idx + self.max_prediction_length <= len(self.values):
            valid_indices.append(start_idx - self.context_length)
            start_idx += self.step_size
            
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get validation sample."""
        start_idx = self.valid_indices[idx]
        
        # Extract context window
        context = self.values[start_idx:start_idx + self.context_length]
        
        # Extract targets for all prediction lengths
        targets = {}
        for pred_len in self.prediction_lengths:
            target_start = start_idx + self.context_length
            target_end = target_start + pred_len
            if target_end <= len(self.values):
                targets[f"target_{pred_len}"] = self.values[target_start:target_end]
        
        # Convert to tensors
        sample = {
            "context": torch.tensor(context, dtype=torch.float32).unsqueeze(0),
            "timestamp_idx": torch.tensor(start_idx + self.context_length - 1, dtype=torch.long)
        }
        
        for key, target in targets.items():
            sample[key] = torch.tensor(target, dtype=torch.float32)
            
        return sample
    
    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values back to original scale."""
        return values * self.std + self.mean


def create_data_loaders(
    train_path: str,
    val_path: str,
    context_length: int = 2016,
    prediction_lengths: List[int] = [72, 144, 288],
    batch_size: int = 16,
    val_batch_size: int = 1
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Create training and validation data loaders."""
    
    # Load data
    data_loader = BitcoinDataLoader(train_path, val_path)
    train_df, val_df = data_loader.load_data()
    
    # Extract price series
    train_series = data_loader.get_price_series(train_df, "close")
    val_series = data_loader.get_price_series(val_df, "close")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_series,
        context_length=context_length,
        prediction_lengths=prediction_lengths,
        stride=1,
        normalize=True
    )
    
    # Use training dataset normalization stats for validation
    val_dataset = ValidationDataset(
        val_series,
        context_length=context_length,
        prediction_lengths=prediction_lengths,
        step_size=288,  # Daily validation steps
        normalize_stats=(train_dataset.mean, train_dataset.std)
    )
    
    # Create data loaders with platform-specific optimizations
    import platform
    
    # Optimize DataLoader settings based on platform
    if platform.system() == 'Windows':
        num_workers = 4  # Use multiple workers on Windows
        pin_memory = True
        persistent_workers = True
    else:
        num_workers = 0  # Single process for macOS compatibility
        pin_memory = False  # Disable pin_memory on macOS MPS
        persistent_workers = False
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    # Return metadata
    metadata = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "context_length": context_length,
        "prediction_lengths": prediction_lengths,
        "normalization_stats": (train_dataset.mean, train_dataset.std),
        "train_period": (train_series.index[0], train_series.index[-1]),
        "val_period": (val_series.index[0], val_series.index[-1])
    }
    
    return train_loader, val_loader, metadata


if __name__ == "__main__":
    # Test the data loading
    train_path = "/Users/felixtriendl/PycharmProjects/FineTuneTirex/bitcoin_2022_2024_5min.csv"
    val_path = "/Users/felixtriendl/PycharmProjects/FineTuneTirex/bitcoin_2025_5min.csv"
    
    train_loader, val_loader, metadata = create_data_loaders(
        train_path, val_path, batch_size=4
    )
    
    print(f"Metadata: {metadata}")
    
    # Test a training batch
    for batch in train_loader:
        print(f"Training batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        break
    
    # Test a validation batch
    for batch in val_loader:
        print(f"Validation batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        break