"""
Training Script for KID (Knowledge Injection based Deepfake Detection)
======================================================================
Complete training pipeline following the paper's implementation details.

From Paper Section 4.1 Implementation Details:
- RetinaFace for face detection and alignment
- Images cropped and aligned, saved as 224x224
- ViT/B-16 pretrained on ImageNet as backbone
- Training for maximum 300 epochs
- AdamW optimizer with weight decay 0.01
- Batch size of 24
- Early stopping (20 epochs without improvement)
- Initial learning rate: 1e-4
- Cosine annealing with lower bound 1e-6
- Data augmentation: flip, hue/sat, brightness, JPEG, blur, SBI
- Hyperparameters: γ_0=0.2, γ_1=0.8, β=1.2, μ=0.1

Inference:
- Randomly extract 32 frames from each video
- Take average of frame-level results for video-level prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import argparse
import json
import time
from tqdm import tqdm

# Import our modules
from kid import KnowledgeInjectionDeepfakeDetector, create_kid_model
from data_augmentation import (
    create_augmentation_pipeline,
    generate_random_fake_image,
    DeepfakeDataset,
)


# =============================================================================
# SECTION 1: Training Configuration
# =============================================================================

class TrainingConfig:
    """
    Training configuration following paper's implementation details.
    
    From Paper Section 4.1:
    All hyperparameters are set according to the paper's specifications.
    """
    
    # Model architecture
    img_size: int = 224                # Paper: "images are saved with size 224x224"
    patch_size: int = 16               # ViT-B/16
    embed_dim: int = 768               # ViT-B/16
    num_layers: int = 12               # ViT-B/16
    num_heads: int = 12                # ViT-B/16
    
    # Training parameters
    max_epochs: int = 300              # Paper: "maximum of 300 epochs"
    batch_size: int = 24               # Paper: "batch size of 24"
    num_workers: int = 4
    
    # Optimizer
    learning_rate: float = 1e-4        # Paper: "initial learning rate is 1e-4"
    min_learning_rate: float = 1e-6   # Paper: "lower bound of 1e-6"
    weight_decay: float = 0.01         # Paper: "weight decay of 0.01"
    
    # Early stopping
    patience: int = 20                 # Paper: "20 consecutive epochs"
    
    # KID hyperparameters
    beta: float = 1.2                  # Paper Eq. 9: suppression loss bound
    mu: float = 0.1                    # Paper Eq. 10: contrast loss margin
    gamma_0: float = 0.2               # Paper Eq. 5: localization lower threshold
    gamma_1: float = 0.8               # Paper Eq. 5: localization upper threshold
    num_shallow_layers: int = 8        # L_0 for suppression loss
    
    # Inference
    num_frames_per_video: int = 32     # Paper: "randomly extract 32 frames"
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# =============================================================================
# SECTION 2: Training Utilities
# =============================================================================

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve.
    
    From Paper:
    "Early stopping is implemented, terminating training when the loss
    doesn't decrease for 20 consecutive epochs."
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class MetricTracker:
    """Track and log training metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def reset(self):
        avg = self.get_average()
        self.history.append(avg)
        self.metrics = {}
        return avg


def compute_auc(
    predictions: torch.Tensor,  # [N] probabilities
    labels: torch.Tensor,       # [N] ground truth
) -> float:
    """
    Compute Area Under ROC Curve.
    
    From Paper:
    "we utilize the area under the receiver operating characteristic curve
    (AUC) as the primary evaluation metric"
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy())
    except ImportError:
        # Fallback: simple accuracy
        preds = (predictions > 0.5).long()
        return (preds == labels).float().mean().item()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_auc: float,
    save_path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auc': best_auc,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    load_path: Path,
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_auc']


# =============================================================================
# SECTION 3: Training and Evaluation Functions
# =============================================================================

def train_one_epoch(
    model: KnowledgeInjectionDeepfakeDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    From Paper:
    The training process updates only:
    - cls_token
    - normalization layers  
    - W^{tilde{Q}} in each I-MSA block
    - localization branch
    - classification head
    """
    model.train()
    metric_tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass with loss computation
        optimizer.zero_grad()
        output = model(images, labels=labels, face_masks=masks, return_loss=True)
        
        # Backward pass
        loss = output['losses']['total']
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        metrics = {k: v.item() for k, v in output['losses'].items()}
        metrics['accuracy'] = (output['pred'] == labels).float().mean().item()
        metric_tracker.update(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'acc': f"{metrics['accuracy']:.4f}"
        })
    
    return metric_tracker.reset()


@torch.no_grad()
def evaluate(
    model: KnowledgeInjectionDeepfakeDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    From Paper:
    "we utilize the area under the receiver operating characteristic curve
    (AUC) as the primary evaluation metric"
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        output = model(images, labels=labels, face_masks=masks, return_loss=True)
        
        # Collect predictions
        probs = output['probs'][:, 1]  # Probability of fake
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(output['pred'].cpu())
        
        total_loss += output['losses']['total'].item()
        num_batches += 1
    
    # Concatenate all predictions
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    
    # Compute metrics
    auc = compute_auc(all_probs, all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / num_batches
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'loss': avg_loss,
    }


def train(
    model: KnowledgeInjectionDeepfakeDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    resume_path: Optional[Path] = None,
):
    """
    Full training loop following paper's specifications.
    
    From Paper Implementation Details:
    - AdamW optimizer with weight decay 0.01
    - Cosine annealing learning rate schedule
    - Early stopping after 20 epochs without improvement
    """
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Freeze pretrained weights
    # Paper: "the original attention branch is frozen"
    model.freeze_pretrained()
    
    # Move model to device
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create optimizer
    # Paper: "AdamW optimizer with weight decay of 0.01"
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Create learning rate scheduler
    # Paper: "cosine annealing for learning rate decay, with lower bound of 1e-6"
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.min_learning_rate,
    )
    
    # Initialize tracking
    start_epoch = 0
    best_auc = 0.0
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Resume from checkpoint if provided
    if resume_path is not None and resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        start_epoch, best_auc = load_checkpoint(model, optimizer, scheduler, resume_path)
        start_epoch += 1
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, config.max_epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['total']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_auc,
                checkpoint_dir / "best_model.pth"
            )
            print(f"  → New best AUC: {best_auc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_auc,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            )
        
        # Early stopping check
        # Paper: "terminating training when loss doesn't decrease for 20 epochs"
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    return best_auc


# =============================================================================
# SECTION 4: Inference Functions
# =============================================================================

@torch.no_grad()
def predict_single_image(
    model: KnowledgeInjectionDeepfakeDetector,
    image: torch.Tensor,  # [C, H, W] or [1, C, H, W]
    device: torch.device,
) -> Dict[str, float]:
    """
    Predict on a single image.
    
    Returns probability of image being fake.
    """
    model.eval()
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    output = model(image)
    
    return {
        'fake_probability': output['probs'][0, 1].item(),
        'real_probability': output['probs'][0, 0].item(),
        'prediction': 'fake' if output['pred'][0].item() == 1 else 'real',
    }


@torch.no_grad()
def predict_video(
    model: KnowledgeInjectionDeepfakeDetector,
    frames: torch.Tensor,  # [T, C, H, W] video frames
    device: torch.device,
    num_frames: int = 32,  # Paper: "randomly extract 32 frames"
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Predict on a video by averaging frame-level predictions.
    
    From Paper Inference Stage:
    "In the inference stage, we randomly extract 32 frames from each video
    for detection and take the average of the frame-level results as the
    video-level result."
    """
    model.eval()
    
    T = frames.shape[0]
    
    # Randomly sample frames if video is longer than num_frames
    if T > num_frames:
        indices = torch.randperm(T)[:num_frames]
        frames = frames[indices]
    
    # Process in batches
    all_probs = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size].to(device)
        output = model(batch)
        probs = output['probs'][:, 1]  # Probability of fake
        all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs)
    
    # Average frame-level predictions for video-level result
    # Paper: "take the average of the frame-level results as video-level result"
    avg_prob = all_probs.mean().item()
    
    return {
        'fake_probability': avg_prob,
        'real_probability': 1 - avg_prob,
        'prediction': 'fake' if avg_prob > 0.5 else 'real',
        'frame_probs': all_probs.tolist(),
    }


# =============================================================================
# SECTION 5: Main Entry Point
# =============================================================================

def create_dummy_data(
    num_samples: int = 100,
    img_size: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy data for testing."""
    images = torch.rand(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 2, (num_samples,))
    return images, labels


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train KID deepfake detector")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--demo', action='store_true', help='Run demo with dummy data')
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating KID model...")
    model = create_kid_model(backbone='vit_base')
    
    if args.demo:
        # Demo mode with dummy data
        print("\n=== Running demo with dummy data ===\n")
        
        # Create dummy datasets
        train_images, train_labels = create_dummy_data(100)
        val_images, val_labels = create_dummy_data(20)
        
        # Create datasets
        # Training: use SBI to generate fakes on-the-fly
        train_dataset = DeepfakeDataset(
            real_images=train_images,
            use_sbi=True,
            sbi_probability=0.5,
        )
        # Validation: use SBI with 50% probability to ensure both classes exist
        # This avoids the "only one class present" warning in AUC computation
        val_dataset = DeepfakeDataset(
            real_images=val_images,
            use_sbi=True,           # Enable SBI to get fake samples
            sbi_probability=0.5,    # 50% real, 50% fake
            augment=False,          # No augmentation for validation
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Quick training test (just 2 epochs)
        config.max_epochs = 2
        config.patience = 5
        
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        
        # Test inference
        print("\n=== Testing inference ===\n")
        
        # Single image prediction
        test_image = torch.rand(3, 224, 224)
        result = predict_single_image(model, test_image, device)
        print(f"Single image prediction: {result}")
        
        # Video prediction
        test_video = torch.rand(50, 3, 224, 224)
        result = predict_video(model, test_video, device)
        print(f"Video prediction: {result['prediction']} (prob: {result['fake_probability']:.4f})")
        
        print("\n=== Demo complete! ===")
    
    else:
        print("\nTo run training with real data:")
        print("1. Prepare your dataset (FF++ or similar)")
        print("2. Create DataLoader with your data")
        print("3. Call train() function")
        print("\nFor a quick demo, run: python train.py --demo")


if __name__ == '__main__':
    main()
