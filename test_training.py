import os
os.environ['TIREX_NO_CUDA'] = '1'

import torch
import yaml
import logging
from tqdm import tqdm

from data import create_data_loaders
from model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_steps():
    """Test a few training steps to ensure everything works."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Loading data...")
    train_loader, val_loader, metadata = create_data_loaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        context_length=config['data']['context_length'],
        prediction_lengths=config['data']['prediction_lengths'],
        batch_size=2,  # Small batch size for testing
        val_batch_size=1
    )
    
    logger.info("Creating model...")
    model = create_model(
        model_name=config['model']['name'],
        prediction_lengths=config['data']['prediction_lengths'],
        quantiles=config['model']['quantiles'],
        freeze_encoder=config['model']['freeze_encoder'],
        device="cpu"  # Use CPU for simplicity
    )
    
    # Test a few training steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    logger.info("Testing training steps...")
    
    num_test_steps = 3
    for i, batch in enumerate(train_loader):
        if i >= num_test_steps:
            break
            
        logger.info(f"Step {i+1}/{num_test_steps}")
        
        optimizer.zero_grad()
        
        # Get batch data
        context = batch['context']
        targets = {}
        for pred_len in model.prediction_lengths:
            target_key = f"target_{pred_len}"
            if target_key in batch:
                targets[target_key] = batch[target_key]
        
        # Forward pass
        predictions = model.predict_batch(context)
        print(f"  Predictions keys: {list(predictions.keys())}")
        print(f"  Context shape: {context.shape}")
        
        for key, pred in predictions.items():
            print(f"  {key} shape: {pred.shape}, device: {pred.device}")
        
        # Compute loss
        losses = model.compute_loss(predictions, targets)
        total_loss = losses['total_loss']
        
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print("  ✓ Step completed successfully")
    
    # Test validation step
    logger.info("Testing validation step...")
    model.eval()
    
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        context = val_batch['context']
        targets = {}
        for pred_len in model.prediction_lengths:
            target_key = f"target_{pred_len}"
            if target_key in val_batch:
                targets[target_key] = val_batch[target_key]
        
        predictions = model.predict_batch(context)
        losses = model.compute_loss(predictions, targets)
        metrics = model.compute_metrics(predictions, targets)
        
        print(f"Val loss: {losses['total_loss'].item():.4f}")
        print(f"Val metrics: {[(k, v.item()) for k, v in metrics.items()]}")
        print("✓ Validation step completed successfully")
    
    logger.info("✓ All tests passed! Training should work correctly.")

if __name__ == "__main__":
    test_training_steps()