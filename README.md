# Knowledge Injection based Deepfake Detection (KID)

PyTorch implementation of **"Deepfake Detection via Knowledge Injection"** (arXiv:2503.02503v1).

## Overview

This implementation provides the complete KID framework for deepfake detection, following a functional programming style with detailed comments explaining the reasoning from the paper.

### Key Components

1. **Knowledge Injection Module** (Section 3.1)
   - Injection Multi-Head Self-Attention (I-MSA) blocks
   - Authenticity correlation matrix computation
   - Knowledge-augmented attention mechanism

2. **Coarse-Grained Forgery Localization Branch** (Section 3.2)
   - Multi-task learning for improved generalization
   - Patch-wise localization predictions
   - Only used during training

3. **Layer-Wise Suppression and Contrast Losses** (Section 3.3)
   - Suppression loss for shallow layers (preserve real knowledge)
   - Contrast loss for deep layers (separate real/fake distributions)

## File Structure

```
injection_detection/
├── kid.py                 # Core KID model implementation
├── data_augmentation.py   # Data augmentation & SBI fake synthesis
├── train.py               # Training script and utilities
└── README.md              # This file
```

## Installation

```bash
pip install torch torchvision numpy tqdm
# Optional for AUC computation:
pip install scikit-learn
```

## Quick Start

### Demo Mode

Run a quick demo with synthetic data:

```bash
python train.py --demo
```

### Using the Model

```python
import torch
from kid import create_kid_model, KnowledgeInjectionDeepfakeDetector

# Create model
model = create_kid_model(backbone='vit_base')

# Freeze pretrained weights (only trains knowledge injection parameters)
model.freeze_pretrained()

# Forward pass
images = torch.randn(2, 3, 224, 224)  # [B, C, H, W]
labels = torch.tensor([0, 1])          # 0=real, 1=fake
face_masks = torch.rand(2, 224, 224)   # Face region masks

output = model(images, labels=labels, face_masks=face_masks, return_loss=True)

print(output['logits'])      # Classification logits [B, 2]
print(output['pred'])        # Predictions [B]
print(output['losses'])      # Dictionary of loss components
```

### Training

```python
from kid import create_kid_model
from train import train, TrainingConfig
from data_augmentation import DeepfakeDataset
from torch.utils.data import DataLoader

# Create model
model = create_kid_model(backbone='vit_base')

# Prepare your data
train_dataset = DeepfakeDataset(
    real_images=your_real_images,    # [N, 3, 224, 224]
    fake_images=your_fake_images,    # Optional
    use_sbi=True,                    # Generate SBI fakes on-the-fly
)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = ...

# Train
config = TrainingConfig()
train(model, train_loader, val_loader, config, device=torch.device('cuda'))
```

### Inference

```python
from train import predict_single_image, predict_video

# Single image
result = predict_single_image(model, image, device)
print(f"Prediction: {result['prediction']}")
print(f"Fake probability: {result['fake_probability']:.4f}")

# Video (averages 32 random frames)
result = predict_video(model, video_frames, device)
print(f"Video prediction: {result['prediction']}")
```

## Model Architecture

```
Input Image (224x224)
        ↓
    Patch Embedding (16x16 patches → 196 tokens)
        ↓
    + CLS Token + Positional Embedding
        ↓
    ┌─────────────────────────────────────┐
    │   Transformer Block × 12            │
    │   ┌───────────────────────────────┐ │
    │   │  I-MSA (Knowledge Injection)  │ │
    │   │  • Standard Q, K, V (frozen)  │ │
    │   │  • Knowledge Query Q̃ (trained)│ │
    │   │  • Authenticity Correlation   │ │
    │   │  • Augmented Attention        │ │
    │   └───────────────────────────────┘ │
    │   ┌───────────────────────────────┐ │
    │   │         MLP (frozen)          │ │
    │   └───────────────────────────────┘ │
    └─────────────────────────────────────┘
        ↓
    Layer Norm
        ↓
    ┌─────────────────┬──────────────────────────┐
    │                 │                          │
    ↓                 ↓                          │
CLS Token       Patch Tokens                     │
    ↓                 ↓                          │
Classification  Localization Branch              │
Head            (training only)                  │
    ↓                 ↓                          │
Real/Fake      Per-patch scores                  │
                                                 │
Loss = L_CE + L_DICE + L_S + L_D ←───────────────┘
```

## Hyperparameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image size | 224×224 | Input resolution |
| Patch size | 16×16 | ViT patch size |
| Batch size | 24 | Training batch size |
| Learning rate | 1e-4 | Initial LR |
| Min LR | 1e-6 | Cosine annealing lower bound |
| Weight decay | 0.01 | AdamW regularization |
| Max epochs | 300 | Training duration |
| Early stopping | 20 | Patience epochs |
| β | 1.2 | Suppression loss bound (Eq. 9) |
| μ | 0.1 | Contrast loss margin (Eq. 10) |
| γ₀ | 0.2 | Localization lower threshold (Eq. 5) |
| γ₁ | 0.8 | Localization upper threshold (Eq. 5) |
| L₀ | 8 | Number of shallow layers for suppression |

## Loss Functions

### Overall Loss (Equation 11)
```
L = L_CE + L_DICE + L_S + L_D
```

- **L_CE**: Cross-entropy for classification
- **L_DICE**: Dice loss for localization
- **L_S**: Suppression loss (shallow layers, Eq. 8-9)
- **L_D**: Contrast loss (deep layers, Eq. 10)

## Data Augmentation

Following the paper, we apply:
- Horizontal flipping
- Random hue/saturation changes
- Random brightness/contrast changes
- JPEG compression
- Gaussian blur
- **SBI (Self-Blended Images)** fake synthesis

## Supported Backbones

- `vit_small`: ViT-S/16 (384 dim, 12 layers, 6 heads)
- `vit_base`: ViT-B/16 (768 dim, 12 layers, 12 heads) **[Default]**
- `vit_large`: ViT-L/16 (1024 dim, 24 layers, 16 heads)

## Citation

```bibtex
@article{li2025deepfake,
  title={Deepfake Detection via Knowledge Injection},
  author={Li, Tonghui and Guo, Yuanfang and Peng, Heqi and Liu, Zeming and Wang, Yunhong},
  journal={arXiv preprint arXiv:2503.02503},
  year={2025}
}
```

## License

This implementation is for research purposes only.
