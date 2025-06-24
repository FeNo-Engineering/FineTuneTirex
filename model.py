import os
import platform

# Only disable CUDA kernels on macOS/non-Windows systems
if platform.system() != 'Windows':
    os.environ['TIREX_NO_CUDA'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from tirex import load_model, ForecastModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PinballLoss(nn.Module):
    """Multi-quantile pinball loss for probabilistic forecasting."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.num_quantiles = len(quantiles)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss for multiple quantiles.
        
        Args:
            predictions: Shape (batch_size, prediction_length, num_quantiles)
            targets: Shape (batch_size, prediction_length)
        
        Returns:
            loss: Scalar tensor
        """
        # Move quantiles to same device as predictions
        quantiles = self.quantiles.to(predictions.device)
        
        # Expand targets to match prediction dimensions
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)  # (batch_size, pred_len, num_quantiles)
        
        # Calculate errors
        errors = targets_expanded - predictions  # (batch_size, pred_len, num_quantiles)
        
        # Pinball loss for each quantile
        quantiles_expanded = quantiles.view(1, 1, -1).expand_as(errors)
        losses = torch.where(
            errors >= 0,
            quantiles_expanded * errors,
            (quantiles_expanded - 1) * errors
        )
        
        # Average over all dimensions
        return losses.mean()


class CRPS(nn.Module):
    """Continuous Ranked Probability Score for forecast evaluation."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CRPS using quantile forecasts approximation.
        
        Args:
            predictions: Shape (batch_size, prediction_length, num_quantiles)
            targets: Shape (batch_size, prediction_length)
        
        Returns:
            crps: Scalar tensor
        """
        # Move quantiles to same device
        quantiles = self.quantiles.to(predictions.device)
        
        # Expand targets
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
        
        # Calculate CRPS using quantile approximation
        # CRPS = 2 * sum_q (y - q) * (1{y <= q} - q)
        indicator = (targets_expanded <= predictions).float()
        quantiles_expanded = quantiles.view(1, 1, -1).expand_as(predictions)
        
        crps_values = 2 * (targets_expanded - predictions) * (indicator - quantiles_expanded)
        
        # Average over quantiles and time steps, then over batch
        return crps_values.mean(dim=-1).mean(dim=-1).mean()


class TiRexFineTuner(nn.Module):
    """Fine-tuning wrapper for TiRex model with multi-quantile forecasting."""
    
    def __init__(
        self,
        model_name: str = "NX-AI/TiRex",
        prediction_lengths: List[int] = [72, 144, 288],
        quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # Load pre-trained TiRex model with CPU mapping for compatibility
        logger.info(f"Loading TiRex model: {model_name}")
        try:
            self.tirex_model: ForecastModel = load_model(model_name, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load with auto device, trying CPU: {e}")
            self.tirex_model: ForecastModel = load_model(model_name, device="cpu")
        
        self.base_model = self.tirex_model  # TiRex model IS the nn.Module
        
        self.prediction_lengths = prediction_lengths
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Since TiRex doesn't preserve gradients well for fine-tuning,
        # we'll use it as a feature extractor and add trainable adaptation layers
        self.use_adaptation_layers = True
        
        if self.use_adaptation_layers:
            # Add small adaptation layers for each prediction horizon
            self.adaptation_layers = nn.ModuleDict()
            for pred_len in prediction_lengths:
                # Simple linear layer to adapt TiRex outputs
                self.adaptation_layers[f"adapt_{pred_len}"] = nn.Sequential(
                    nn.Linear(self.num_quantiles, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, self.num_quantiles)
                )
        
        # Optionally freeze encoder weights
        if freeze_encoder and not self.use_adaptation_layers:
            self._freeze_encoder()
        
        # Loss functions
        self.pinball_loss = PinballLoss(quantiles)
        self.crps_metric = CRPS(quantiles)
        
        logger.info(f"TiRex fine-tuner initialized with {len(prediction_lengths)} prediction horizons")
        logger.info(f"Quantiles: {quantiles}")
        if self.use_adaptation_layers:
            logger.info("Using adaptation layers for fine-tuning")
        
    def _freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning only the head."""
        for name, param in self.base_model.named_parameters():
            if 'head' not in name.lower():  # Freeze everything except prediction head
                param.requires_grad = False
        logger.info("Encoder weights frozen - only training prediction head")
    
    def forward(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """
        Forward pass with multi-quantile predictions.
        
        Args:
            context: Shape (batch_size, num_series, context_length)
            prediction_length: Number of steps to forecast
        
        Returns:
            predictions: Shape (batch_size, prediction_length, num_quantiles)
        """
        # Use the underlying TiRex model for forecasting
        # The TiRex model expects context in shape (batch_size, context_length) for single series
        batch_size = context.shape[0]
        
        # Squeeze the series dimension if it's 1 (univariate time series)
        if context.shape[1] == 1:
            context_squeezed = context.squeeze(1)  # (batch_size, context_length)
        else:
            context_squeezed = context
        
        # Handle device placement for TiRex model
        if platform.system() == 'Windows':
            # On Windows with CUDA, TiRex can run on GPU
            tirex_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            context_for_tirex = context_squeezed.to(tirex_device)
        else:
            # On macOS, keep TiRex on CPU
            context_for_tirex = context_squeezed.cpu()
        
        # Generate forecast using TiRex (always in inference mode since gradients don't flow)
        with torch.no_grad():
            forecast_result = self.tirex_model.forecast(
                context=context_for_tirex,
                prediction_length=prediction_length
            )
        
        # Extract quantile predictions
        # TiRex returns a tuple (quantiles, mean)
        if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
            quantiles_tensor, mean_tensor = forecast_result
            predictions = quantiles_tensor  # (batch_size, pred_len, num_quantiles)
        else:
            # Fallback for unexpected format
            if hasattr(forecast_result, 'shape'):
                predictions = forecast_result
            else:
                raise ValueError(f"Unexpected forecast result format: {type(forecast_result)}")
        
        # Move predictions back to original device
        original_device = context.device
        predictions = predictions.to(original_device)
        
        # Apply adaptation layer if available and in training mode
        if self.use_adaptation_layers and self.training:
            adapt_key = f"adapt_{prediction_length}"
            if adapt_key in self.adaptation_layers:
                # Get the device of the adaptation layer
                adapt_device = next(self.adaptation_layers[adapt_key].parameters()).device
                
                # Move predictions to adaptation layer device
                predictions_on_device = predictions.to(adapt_device)
                
                # Apply adaptation layer pointwise to each timestep
                # Input: (batch_size, pred_len, num_quantiles)
                batch_size, pred_len, num_quant = predictions_on_device.shape
                
                # Reshape to apply linear layer to each timestep
                predictions_flat = predictions_on_device.view(-1, num_quant)  # (batch_size * pred_len, num_quantiles)
                
                # Apply adaptation
                adapted_flat = self.adaptation_layers[adapt_key](predictions_flat)
                
                # Reshape back and move to original device
                predictions = adapted_flat.view(batch_size, pred_len, num_quant).to(original_device)
        
        return predictions
    
    def _normal_quantile(self, p: float) -> float:
        """Approximate normal distribution quantile using Beasley-Springer-Moro algorithm."""
        if p <= 0 or p >= 1:
            raise ValueError("p must be between 0 and 1")
        
        # Coefficients for Beasley-Springer-Moro approximation
        a = [0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        
        b = [0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        
        c = [0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        
        d = [0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        
        # Split into regions
        p_low = 0.02425
        p_high = 1 - p_low
        
        if p < p_low:
            # Rational approximation for lower region
            q = np.sqrt(-2 * np.log(p))
            return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / \
                   ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        elif p <= p_high:
            # Rational approximation for central region
            q = p - 0.5
            r = q * q
            return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / \
                   (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        else:
            # Rational approximation for upper region
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / \
                   ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for multiple prediction horizons.
        
        Args:
            predictions: Dict with keys like 'pred_72', 'pred_144', 'pred_288'
            targets: Dict with keys like 'target_72', 'target_144', 'target_288'
        
        Returns:
            losses: Dict containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for pred_len in self.prediction_lengths:
            pred_key = f"pred_{pred_len}"
            target_key = f"target_{pred_len}"
            
            if pred_key in predictions and target_key in targets:
                # Ensure predictions and targets are on the same device
                pred = predictions[pred_key]
                target = targets[target_key]
                
                # Move prediction to same device as target
                if pred.device != target.device:
                    pred = pred.to(target.device)
                
                loss = self.pinball_loss(pred, target)
                losses[f"loss_{pred_len}"] = loss
                total_loss += loss
        
        losses["total_loss"] = total_loss / len(self.prediction_lengths)
        return losses
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics (MAE, CRPS).
        
        Args:
            predictions: Dict with quantile predictions
            targets: Dict with ground truth targets
        
        Returns:
            metrics: Dict containing MAE and CRPS for each horizon
        """
        metrics = {}
        
        for pred_len in self.prediction_lengths:
            pred_key = f"pred_{pred_len}"
            target_key = f"target_{pred_len}"
            
            if pred_key in predictions and target_key in targets:
                pred = predictions[pred_key]  # (batch_size, pred_len, num_quantiles)
                target = targets[target_key]  # (batch_size, pred_len)
                
                # Move prediction to same device as target
                if pred.device != target.device:
                    pred = pred.to(target.device)
                
                # MAE using median (0.5 quantile)
                median_idx = len(self.quantiles) // 2
                median_pred = pred[:, :, median_idx]  # (batch_size, pred_len)
                mae = torch.abs(median_pred - target).mean()
                metrics[f"mae_{pred_len}"] = mae
                
                # CRPS
                crps = self.crps_metric(pred, target)
                metrics[f"crps_{pred_len}"] = crps
        
        return metrics
    
    def predict_batch(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for all horizons given context.
        
        Args:
            context: Shape (batch_size, num_series, context_length)
        
        Returns:
            predictions: Dict with predictions for each horizon
        """
        predictions = {}
        
        for pred_len in self.prediction_lengths:
            pred = self.forward(context, pred_len)
            predictions[f"pred_{pred_len}"] = pred
        
        return predictions


def create_model(
    model_name: str = "NX-AI/TiRex",
    prediction_lengths: List[int] = [72, 144, 288],
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    freeze_encoder: bool = False,
    device: str = "auto"
) -> TiRexFineTuner:
    """
    Create and initialize TiRex fine-tuning model.
    
    Args:
        model_name: HuggingFace model name
        prediction_lengths: List of forecast horizons
        quantiles: List of quantiles for probabilistic forecasting
        freeze_encoder: Whether to freeze encoder weights
        device: Device to use ('auto', 'cpu', 'mps', 'cuda')
    
    Returns:
        model: Initialized TiRexFineTuner
    """
    # Create model
    model = TiRexFineTuner(
        model_name=model_name,
        prediction_lengths=prediction_lengths,
        quantiles=quantiles,
        freeze_encoder=freeze_encoder
    )
    
    # Keep TiRex on CPU but move other components to target device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # Move adaptation layers and loss functions to target device
    if hasattr(model, 'adaptation_layers'):
        model.adaptation_layers = model.adaptation_layers.to(device)
    
    model.pinball_loss.quantiles = model.pinball_loss.quantiles.to(device)
    model.crps_metric.quantiles = model.crps_metric.quantiles.to(device)
    
    # On Windows with CUDA, we can also move TiRex to GPU for better performance
    if platform.system() == 'Windows' and device == 'cuda':
        try:
            model.tirex_model = model.tirex_model.to(device)
            logger.info(f"All components moved to device: {device} (including TiRex model)")
        except Exception as e:
            logger.warning(f"Could not move TiRex to GPU: {e}, keeping on CPU")
            logger.info(f"Adaptation layers and loss functions moved to device: {device}, TiRex model kept on CPU")
    else:
        logger.info(f"Adaptation layers and loss functions moved to device: {device}, TiRex model kept on CPU")
    
    return model


if __name__ == "__main__":
    # Test the model creation and forward pass
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Create model
    model = create_model(device=device)
    
    # Test forward pass
    batch_size = 2
    context_length = 2016
    context = torch.randn(batch_size, 1, context_length).to(device)
    
    print(f"Input context shape: {context.shape}")
    
    # Test predictions
    with torch.no_grad():
        predictions = model.predict_batch(context)
        
    print("Prediction shapes:")
    for key, pred in predictions.items():
        print(f"  {key}: {pred.shape}")
    
    # Test loss computation
    targets = {
        "target_72": torch.randn(batch_size, 72).to(device),
        "target_144": torch.randn(batch_size, 144).to(device),
        "target_288": torch.randn(batch_size, 288).to(device)
    }
    
    losses = model.compute_loss(predictions, targets)
    print(f"Losses: {losses}")
    
    metrics = model.compute_metrics(predictions, targets)
    print(f"Metrics: {metrics}")