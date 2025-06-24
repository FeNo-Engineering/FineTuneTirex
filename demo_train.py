import os
os.environ['TIREX_NO_CUDA'] = '1'

import torch
import yaml
import logging
from pathlib import Path

from train import TiRexTrainer
from data import create_data_loaders
from model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_training():
    """Run a quick demo of the training process."""
    
    # Load config but modify for quick demo
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for quick demo
    config['training']['epochs'] = 1
    config['training']['batch_size'] = 2
    config['training']['save_dir'] = 'demo_checkpoints'
    
    # Set random seeds
    torch.manual_seed(42)
    
    logger.info("Creating demo training setup...")
    
    # Create data loaders
    train_loader, val_loader, metadata = create_data_loaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        context_length=config['data']['context_length'],
        prediction_lengths=config['data']['prediction_lengths'],
        batch_size=config['training']['batch_size'],
        val_batch_size=config['training']['val_batch_size']
    )
    
    logger.info(f"Data loaded: {metadata['train_size']} train, {metadata['val_size']} val samples")
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        prediction_lengths=config['data']['prediction_lengths'],
        quantiles=config['model']['quantiles'],
        freeze_encoder=config['model']['freeze_encoder'],
        device="cpu"  # Use CPU for demo
    )
    
    # Create trainer
    trainer = TiRexTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config['training']['save_dir']
    )
    
    logger.info("Starting demo training (first 10 batches only)...")
    
    # Run a few training steps
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Only run 10 batches for demo
            break
            
        trainer.optimizer.zero_grad()
        
        # Get batch data
        context = batch['context']
        targets = {}
        device = trainer.model.pinball_loss.quantiles.device
        
        for pred_len in model.prediction_lengths:
            target_key = f"target_{pred_len}"
            if target_key in batch:
                targets[target_key] = batch[target_key].to(device)
        
        # Forward pass
        predictions = model.predict_batch(context)
        
        # Compute loss
        losses = model.compute_loss(predictions, targets)
        total_loss_batch = losses['total_loss']
        
        # Backward pass
        total_loss_batch.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        
        total_loss += total_loss_batch.item()
        
        if i % 5 == 0:
            logger.info(f"Batch {i+1}: loss = {total_loss_batch.item():.4f}, lr = {trainer.scheduler.get_lr():.2e}")
    
    avg_loss = total_loss / 10
    logger.info(f"Demo training completed. Average loss: {avg_loss:.4f}")
    
    # Run validation
    logger.info("Running validation...")
    val_metrics = trainer.validate()
    
    logger.info("Validation results:")
    for key, value in val_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save demo checkpoint
    Path(config['training']['save_dir']).mkdir(exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'demo_metrics': val_metrics
    }
    torch.save(checkpoint, f"{config['training']['save_dir']}/demo.pt")
    
    logger.info(f"Demo checkpoint saved to {config['training']['save_dir']}/demo.pt")
    logger.info("✓ Demo training completed successfully!")

if __name__ == "__main__":
    demo_training()