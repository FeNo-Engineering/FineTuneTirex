import os
import platform

# Only disable CUDA kernels on macOS/non-Windows systems
if platform.system() != 'Windows':
    os.environ['TIREX_NO_CUDA'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import Dict, List, Tuple, Optional

from data import create_data_loaders
from model import create_model, TiRexFineTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 2, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = float('inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, metric: float, model: nn.Module) -> bool:
        """
        Check if training should stop and update best weights.
        
        Args:
            metric: Current validation metric (lower is better)
            model: Model to save weights for
            
        Returns:
            bool: True if training should stop
        """
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                logger.info(f"Restoring best weights from epoch {self.stopped_epoch}")
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class CosineAnnealingWithWarmup:
    """Cosine annealing learning rate scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-5,
        max_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class TiRexTrainer:
    """Training manager for TiRex fine-tuning."""
    
    def __init__(
        self,
        model: TiRexFineTuner,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * config['training']['epochs']
        warmup_steps = int(total_steps * config['training']['warmup_ratio'])
        
        self.scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=config['training']['min_lr'],
            max_lr=config['training']['learning_rate']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_crps = float('inf')
        self.training_history = []
        
        logger.info(f"Trainer initialized with {total_steps} total steps, {warmup_steps} warmup steps")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_losses = {f"loss_{pl}": 0.0 for pl in self.model.prediction_lengths}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move batch to device (except context which is handled in model)
            context = batch['context']  # Keep on original device, model handles CPU conversion
            targets = {}
            
            # Get device from model's quantiles tensor (which gets moved in create_model)
            device = self.model.pinball_loss.quantiles.device
            
            for pred_len in self.model.prediction_lengths:
                target_key = f"target_{pred_len}"
                if target_key in batch:
                    targets[target_key] = batch[target_key].to(device)
            
            # Forward pass
            predictions = self.model.predict_batch(context)
            
            # Compute loss
            losses = self.model.compute_loss(predictions, targets)
            total_loss_batch = losses['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            for key, loss in losses.items():
                if key in total_losses:
                    total_losses[key] += loss.item()
            
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}"
            })
        
        # Average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            **{key: loss / num_batches for key, loss in total_losses.items()}
        }
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_losses = {f"loss_{pl}": 0.0 for pl in self.model.prediction_lengths}
        total_metrics = {}
        for pl in self.model.prediction_lengths:
            total_metrics[f"mae_{pl}"] = 0.0
            total_metrics[f"crps_{pl}"] = 0.0
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                context = batch['context']
                targets = {}
                
                # Get device from model's quantiles tensor
                device = self.model.pinball_loss.quantiles.device
                
                for pred_len in self.model.prediction_lengths:
                    target_key = f"target_{pred_len}"
                    if target_key in batch:
                        targets[target_key] = batch[target_key].to(device)
                
                # Forward pass
                predictions = self.model.predict_batch(context)
                
                # Compute losses and metrics
                losses = self.model.compute_loss(predictions, targets)
                metrics = self.model.compute_metrics(predictions, targets)
                
                # Accumulate
                for key, loss in losses.items():
                    if key.startswith('loss_'):
                        total_losses[key] += loss.item()
                
                for key, metric in metrics.items():
                    total_metrics[key] += metric.item()
                
                num_batches += 1
        
        # Average
        avg_losses = {key: loss / num_batches for key, loss in total_losses.items()}
        avg_metrics = {key: metric / num_batches for key, metric in total_metrics.items()}
        
        # Compute overall CRPS for early stopping
        avg_crps = np.mean([avg_metrics[f"crps_{pl}"] for pl in self.model.prediction_lengths])
        
        return {**avg_losses, **avg_metrics, 'avg_crps': avg_crps}
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_crps': self.best_val_crps,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            logger.info(f"Best model saved with CRPS: {metrics['avg_crps']:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"Val CRPS: {val_metrics['avg_crps']:.4f}")
            
            for pred_len in self.model.prediction_lengths:
                logger.info(f"Val MAE {pred_len}h: {val_metrics[f'mae_{pred_len}']:.4f}")
            
            # Save metrics
            epoch_history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.scheduler.get_lr()
            }
            self.training_history.append(epoch_history)
            
            # Check for best model
            is_best = val_metrics['avg_crps'] < self.best_val_crps
            if is_best:
                self.best_val_crps = val_metrics['avg_crps']
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping check
            if self.early_stopping(val_metrics['avg_crps'], self.model):
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training completed. Best validation CRPS: {self.best_val_crps:.4f}")


def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    # Load configuration
    config_path = "config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file {config_path} not found!")
        return
    
    config = load_config(config_path)
    
    # Set random seeds
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader, metadata = create_data_loaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        context_length=config['data']['context_length'],
        prediction_lengths=config['data']['prediction_lengths'],
        batch_size=config['training']['batch_size'],
        val_batch_size=config['training']['val_batch_size']
    )
    
    logger.info(f"Training samples: {metadata['train_size']}")
    logger.info(f"Validation samples: {metadata['val_size']}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_name=config['model']['name'],
        prediction_lengths=config['data']['prediction_lengths'],
        quantiles=config['model']['quantiles'],
        freeze_encoder=config['model']['freeze_encoder'],
        device=config['training']['device']
    )
    
    # Create trainer
    trainer = TiRexTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config['training']['save_dir']
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()