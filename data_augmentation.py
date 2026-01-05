"""
Data Augmentation and SBI Fake Synthesis for KID
=================================================
Implementation of data augmentation pipeline used in the KID paper.

From Paper Implementation Details:
"For data augmentation, we apply horizontal flipping, random hue saturation
changes, random brightness contrast changes, JPEG compression, blurring
and SBI fake synthesis."

SBI (Self-Blended Images) Reference:
"The entire framework also utilizes self-blended images as the training fake
data to boost the model's capacity to generalize across unseen deepfake
techniques." - citing SLADD [11]
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Callable
import random


# =============================================================================
# SECTION 1: Basic Image Augmentations (Functional Style)
# =============================================================================

def horizontal_flip(
    image: torch.Tensor,  # [C, H, W] or [B, C, H, W]
    p: float = 0.5,
) -> torch.Tensor:
    """
    Random horizontal flip.
    
    From Paper: "we apply horizontal flipping"
    Simple but effective augmentation for face images.
    """
    if random.random() < p:
        return torch.flip(image, dims=[-1])
    return image


def random_hue_saturation(
    image: torch.Tensor,  # [C, H, W] RGB in [0, 1]
    hue_shift: float = 0.1,
    saturation_factor: float = 0.2,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Random hue and saturation adjustments.
    
    From Paper: "random hue saturation changes"
    Helps model be invariant to color variations in different lighting/cameras.
    """
    if random.random() > p:
        return image
    
    # Convert RGB to HSV
    # Using a simplified approach for efficiency
    r, g, b = image[0], image[1], image[2]
    
    max_c = torch.max(image, dim=0)[0]
    min_c = torch.min(image, dim=0)[0]
    diff = max_c - min_c + 1e-8
    
    # Compute hue
    h = torch.zeros_like(max_c)
    mask_r = (max_c == r) & (diff > 1e-7)
    mask_g = (max_c == g) & (diff > 1e-7) & ~mask_r
    mask_b = (max_c == b) & (diff > 1e-7) & ~mask_r & ~mask_g
    
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    
    # Compute saturation
    s = torch.zeros_like(max_c)
    s[max_c > 1e-7] = diff[max_c > 1e-7] / max_c[max_c > 1e-7]
    
    # Value
    v = max_c
    
    # Apply random shifts
    h_shift = random.uniform(-hue_shift, hue_shift) * 360
    s_factor = random.uniform(1 - saturation_factor, 1 + saturation_factor)
    
    h = (h + h_shift) % 360
    s = torch.clamp(s * s_factor, 0, 1)
    
    # Convert back to RGB
    h = h / 60
    i = torch.floor(h).long() % 6
    f = h - torch.floor(h)
    
    p_val = v * (1 - s)
    q_val = v * (1 - f * s)
    t_val = v * (1 - (1 - f) * s)
    
    result = torch.zeros_like(image)
    
    # Handle each sector
    for sector in range(6):
        mask = (i == sector)
        if sector == 0:
            result[0][mask], result[1][mask], result[2][mask] = v[mask], t_val[mask], p_val[mask]
        elif sector == 1:
            result[0][mask], result[1][mask], result[2][mask] = q_val[mask], v[mask], p_val[mask]
        elif sector == 2:
            result[0][mask], result[1][mask], result[2][mask] = p_val[mask], v[mask], t_val[mask]
        elif sector == 3:
            result[0][mask], result[1][mask], result[2][mask] = p_val[mask], q_val[mask], v[mask]
        elif sector == 4:
            result[0][mask], result[1][mask], result[2][mask] = t_val[mask], p_val[mask], v[mask]
        else:
            result[0][mask], result[1][mask], result[2][mask] = v[mask], p_val[mask], q_val[mask]
    
    return torch.clamp(result, 0, 1)


def random_brightness_contrast(
    image: torch.Tensor,  # [C, H, W] in [0, 1]
    brightness_factor: float = 0.2,
    contrast_factor: float = 0.2,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Random brightness and contrast adjustments.
    
    From Paper: "random brightness contrast changes"
    Important for handling different lighting conditions in real-world images.
    """
    if random.random() > p:
        return image
    
    # Random brightness shift
    b_shift = random.uniform(-brightness_factor, brightness_factor)
    image = image + b_shift
    
    # Random contrast adjustment
    c_factor = random.uniform(1 - contrast_factor, 1 + contrast_factor)
    mean = image.mean()
    image = (image - mean) * c_factor + mean
    
    return torch.clamp(image, 0, 1)


def jpeg_compression(
    image: torch.Tensor,  # [C, H, W] in [0, 1]
    quality_range: Tuple[int, int] = (70, 100),
    p: float = 0.5,
) -> torch.Tensor:
    """
    Simulate JPEG compression artifacts.
    
    From Paper: "JPEG compression"
    Critical for robustness as most real-world images undergo JPEG compression.
    This is approximated using DCT-based simulation.
    """
    if random.random() > p:
        return image
    
    # Approximate JPEG compression by adding quantization-like noise
    # True JPEG requires PIL/opencv; this is a differentiable approximation
    quality = random.randint(*quality_range)
    
    # Higher quality = less noise
    noise_scale = (100 - quality) / 500.0
    noise = torch.randn_like(image) * noise_scale
    
    # Block-based noise to simulate DCT quantization
    C, H, W = image.shape
    block_size = 8
    
    # Create block-wise noise pattern
    blocks_h = H // block_size
    blocks_w = W // block_size
    block_noise = torch.randn(C, blocks_h, blocks_w) * noise_scale
    block_noise = F.interpolate(
        block_noise.unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze(0)
    
    image = image + block_noise
    
    return torch.clamp(image, 0, 1)


def gaussian_blur(
    image: torch.Tensor,  # [C, H, W]
    kernel_size_range: Tuple[int, int] = (3, 7),
    sigma_range: Tuple[float, float] = (0.1, 2.0),
    p: float = 0.5,
) -> torch.Tensor:
    """
    Apply Gaussian blur.
    
    From Paper: "blurring"
    Helps model be robust to out-of-focus images and different resolutions.
    """
    if random.random() > p:
        return image
    
    # Random kernel size (must be odd)
    k = random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
    sigma = random.uniform(*sigma_range)
    
    # Create Gaussian kernel
    x = torch.arange(k).float() - k // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
    kernel_2d = kernel_2d.to(image.device)
    
    # Apply to each channel
    C, H, W = image.shape
    image = image.unsqueeze(0)  # [1, C, H, W]
    
    # Pad and convolve
    pad = k // 2
    result = []
    for c in range(C):
        channel = image[:, c:c+1, :, :]
        channel = F.pad(channel, (pad, pad, pad, pad), mode='reflect')
        channel = F.conv2d(channel, kernel_2d)
        result.append(channel)
    
    result = torch.cat(result, dim=1).squeeze(0)  # [C, H, W]
    
    return result


# =============================================================================
# SECTION 2: Self-Blended Images (SBI) Fake Synthesis
# =============================================================================

def generate_face_mask(
    height: int,
    width: int,
    center: Optional[Tuple[float, float]] = None,
    axes: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Generate an elliptical face mask.
    
    From SBI Paper [11]:
    Face X-ray and SBI methods swap faces with similar facial landmarks
    and create realistic blending boundaries.
    
    This creates a soft elliptical mask representing the face region.
    """
    if center is None:
        # Random center around image center
        center = (
            height / 2 + random.uniform(-height * 0.1, height * 0.1),
            width / 2 + random.uniform(-width * 0.1, width * 0.1),
        )
    
    if axes is None:
        # Random ellipse axes (face-like proportions)
        axes = (
            random.uniform(height * 0.25, height * 0.4),
            random.uniform(width * 0.2, width * 0.35),
        )
    
    # Create coordinate grid
    y = torch.arange(height).float()
    x = torch.arange(width).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Ellipse equation: ((y-cy)/a)^2 + ((x-cx)/b)^2 <= 1
    cy, cx = center
    a, b = axes
    
    distance = ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2
    
    # Soft mask with smooth boundary
    mask = torch.sigmoid(5 * (1 - distance))
    
    return mask  # [H, W]


def create_blending_boundary(
    mask: torch.Tensor,  # [H, W] face mask
    blur_sigma: float = 5.0,
) -> torch.Tensor:
    """
    Create soft blending boundary for realistic fake synthesis.
    
    From Paper and SBI [11]:
    "Face X-ray firstly proposed to swap faces with similar facial landmarks
    and enable the detection model to focus on the blending boundaries"
    
    Realistic deepfakes have smooth blending at face boundaries.
    This function creates that soft transition zone.
    """
    H, W = mask.shape
    
    # Create Gaussian kernel for blurring
    k = int(blur_sigma * 6) | 1  # Ensure odd
    x = torch.arange(k).float() - k // 2
    kernel_1d = torch.exp(-x**2 / (2 * blur_sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Apply blur
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    pad = k // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode='reflect')
    mask = F.conv2d(mask, kernel_2d)
    mask = mask.squeeze()  # [H, W]
    
    return mask


def generate_sbi_fake(
    source_image: torch.Tensor,  # [C, H, W] source (will be "real" part)
    target_image: torch.Tensor,  # [C, H, W] target (will be "fake" face)
    blend_ratio_range: Tuple[float, float] = (0.3, 0.7),
    color_transfer: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Self-Blended Image (SBI) for fake data synthesis.
    
    From Paper Section 4.1 and SBI reference [11]:
    "The entire framework also utilizes self-blended images as the training
    fake data to boost the model's capacity to generalize across unseen
    deepfake techniques."
    
    SBI Process:
    1. Take two face images (source and target)
    2. Create a face mask for blending region
    3. Optionally transfer color statistics
    4. Blend the face region from target into source
    5. Return blended image and mask
    
    This simulates various deepfake techniques (face swap, face reenactment)
    without requiring actual deepfake generation.
    
    Key Insight:
    "SBI further improved the forgery synthesis mechanisms by creating more
    realistic fake images with diverse forgery types, in a self-blending manner"
    """
    C, H, W = source_image.shape
    
    # Step 1: Generate face mask
    face_mask = generate_face_mask(H, W)
    
    # Step 2: Create soft blending boundary
    blend_mask = create_blending_boundary(face_mask, blur_sigma=random.uniform(3, 8))
    blend_mask = blend_mask.unsqueeze(0)  # [1, H, W]
    
    # Step 3: Optional color transfer (match target colors to source)
    if color_transfer:
        target_image = match_color_statistics(target_image, source_image, face_mask)
    
    # Step 4: Random blend ratio for diversity
    blend_ratio = random.uniform(*blend_ratio_range)
    blend_mask = blend_mask * blend_ratio
    
    # Step 5: Blend images
    # fake = source * (1 - mask) + target * mask
    fake_image = source_image * (1 - blend_mask) + target_image * blend_mask
    
    # Return fake image and the mask (for localization training)
    # Mask: 0 = blended (inner face), 1 = original (outer face)
    outer_mask = 1 - face_mask
    
    return fake_image, outer_mask


def match_color_statistics(
    source: torch.Tensor,  # [C, H, W] image to modify
    target: torch.Tensor,  # [C, H, W] reference image
    mask: Optional[torch.Tensor] = None,  # [H, W] region mask
) -> torch.Tensor:
    """
    Match color statistics of source to target for seamless blending.
    
    This is important for realistic fake synthesis:
    - Different images have different lighting/color distributions
    - Without color matching, blending creates obvious artifacts
    - Matching makes the fake harder to detect visually
    
    Method: Reinhard color transfer (simplified)
    1. Convert to LAB-like space
    2. Match mean and std of each channel
    3. Convert back to RGB
    """
    C, H, W = source.shape
    
    if mask is not None:
        # Compute statistics only in mask region
        mask = mask.unsqueeze(0).expand(C, -1, -1)
        
        # Source stats
        src_masked = source * mask
        src_sum = src_masked.sum(dim=(1, 2))
        src_count = mask[0].sum()
        src_mean = src_sum / (src_count + 1e-8)
        src_var = ((source - src_mean.view(C, 1, 1)) ** 2 * mask).sum(dim=(1, 2)) / (src_count + 1e-8)
        src_std = torch.sqrt(src_var + 1e-8)
        
        # Target stats
        tgt_masked = target * mask
        tgt_sum = tgt_masked.sum(dim=(1, 2))
        tgt_mean = tgt_sum / (src_count + 1e-8)
        tgt_var = ((target - tgt_mean.view(C, 1, 1)) ** 2 * mask).sum(dim=(1, 2)) / (src_count + 1e-8)
        tgt_std = torch.sqrt(tgt_var + 1e-8)
    else:
        # Compute global statistics
        src_mean = source.mean(dim=(1, 2))
        src_std = source.std(dim=(1, 2))
        tgt_mean = target.mean(dim=(1, 2))
        tgt_std = target.std(dim=(1, 2))
    
    # Transfer: (source - src_mean) / src_std * tgt_std + tgt_mean
    result = (source - src_mean.view(C, 1, 1)) / (src_std.view(C, 1, 1) + 1e-8)
    result = result * tgt_std.view(C, 1, 1) + tgt_mean.view(C, 1, 1)
    
    return torch.clamp(result, 0, 1)


def generate_random_fake_image(
    real_images: torch.Tensor,  # [B, C, H, W] batch of real images
    fake_probability: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate batch with mix of real and SBI fake images.
    
    From Paper:
    "Following the settings used in [11], we adopt the FF++ as the training set.
    The dataset contains 1,000 real images and 4,000 fake images generated by
    four manipulation methods"
    
    For SBI training, we generate fake images on-the-fly by blending
    pairs of real images, which provides unlimited training data diversity.
    
    Returns:
        images: [B, C, H, W] mixed batch
        labels: [B] binary labels (0=real, 1=fake)
        masks: [B, H, W] outer face masks for fake images
    """
    B, C, H, W = real_images.shape
    device = real_images.device
    
    images = []
    labels = []
    masks = []
    
    for i in range(B):
        if random.random() < fake_probability:
            # Generate fake using SBI
            source = real_images[i]
            # Pick random target (different from source)
            target_idx = random.choice([j for j in range(B) if j != i] or [i])
            target = real_images[target_idx]
            
            fake_img, outer_mask = generate_sbi_fake(source, target)
            images.append(fake_img)
            labels.append(1)  # Fake
            masks.append(outer_mask)
        else:
            # Keep as real
            images.append(real_images[i])
            labels.append(0)  # Real
            # Real images have all-ones mask (all outer)
            masks.append(torch.ones(H, W, device=device))
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    masks = torch.stack(masks, dim=0)
    
    return images, labels, masks


# =============================================================================
# SECTION 3: Complete Augmentation Pipeline
# =============================================================================

def create_augmentation_pipeline(
    flip_p: float = 0.5,
    color_p: float = 0.5,
    jpeg_p: float = 0.3,
    blur_p: float = 0.3,
) -> Callable:
    """
    Create the complete augmentation pipeline as described in the paper.
    
    From Paper Implementation Details:
    "For data augmentation, we apply horizontal flipping, random hue saturation
    changes, random brightness contrast changes, JPEG compression, blurring
    and SBI fake synthesis."
    
    Returns a function that augments a single image.
    """
    def augment(image: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations to a single image [C, H, W]."""
        # Horizontal flip
        image = horizontal_flip(image, p=flip_p)
        
        # Color augmentations
        image = random_hue_saturation(image, p=color_p)
        image = random_brightness_contrast(image, p=color_p)
        
        # Compression artifacts
        image = jpeg_compression(image, p=jpeg_p)
        
        # Blur
        image = gaussian_blur(image, p=blur_p)
        
        return image
    
    return augment


def augment_batch(
    images: torch.Tensor,  # [B, C, H, W]
    augment_fn: Callable,
) -> torch.Tensor:
    """Apply augmentation to a batch of images."""
    return torch.stack([augment_fn(img) for img in images], dim=0)


# =============================================================================
# SECTION 4: Dataset and DataLoader Utilities
# =============================================================================

class DeepfakeDataset(torch.utils.data.Dataset):
    """
    Dataset for deepfake detection with SBI fake synthesis.
    
    From Paper Section 4.1:
    "Following the settings used in [11], we adopt the FF++ as the training set"
    
    This dataset:
    1. Loads real face images
    2. Optionally loads pre-generated fake images
    3. Applies data augmentation
    4. Generates SBI fakes on-the-fly for diversity
    """
    
    def __init__(
        self,
        real_images: torch.Tensor,  # Pre-loaded real images [N, C, H, W]
        fake_images: Optional[torch.Tensor] = None,  # Pre-loaded fakes
        use_sbi: bool = True,
        sbi_probability: float = 0.5,
        augment: bool = True,
        img_size: int = 224,
    ):
        """
        Args:
            real_images: Tensor of real face images
            fake_images: Optional tensor of pre-generated fake images
            use_sbi: Whether to generate SBI fakes on-the-fly
            sbi_probability: Probability of generating SBI fake vs using real
            augment: Whether to apply data augmentation
            img_size: Target image size
        """
        self.real_images = real_images
        self.fake_images = fake_images
        self.use_sbi = use_sbi
        self.sbi_probability = sbi_probability
        self.augment = augment
        self.img_size = img_size
        
        # Create augmentation function
        self.augment_fn = create_augmentation_pipeline() if augment else lambda x: x
        
        # Compute dataset length
        self.num_real = len(real_images)
        self.num_fake = len(fake_images) if fake_images is not None else 0
    
    def __len__(self):
        # Paper uses 1000 real + 4000 fake per epoch
        return self.num_real + self.num_fake
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.num_real:
            # Real image
            if self.use_sbi and random.random() < self.sbi_probability:
                # Generate SBI fake
                source = self.real_images[idx]
                target_idx = random.randint(0, self.num_real - 1)
                target = self.real_images[target_idx]
                
                image, mask = generate_sbi_fake(source, target)
                label = 1  # Fake
            else:
                # Use as real
                image = self.real_images[idx]
                mask = torch.ones(image.shape[1], image.shape[2])
                label = 0  # Real
        else:
            # Pre-generated fake
            fake_idx = idx - self.num_real
            image = self.fake_images[fake_idx]
            mask = torch.zeros(image.shape[1], image.shape[2])  # Approximate
            label = 1  # Fake
        
        # Apply augmentation
        if self.augment:
            image = self.augment_fn(image)
        
        # Resize if needed
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='nearest'
            ).squeeze()
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
        }


def create_dataloader(
    real_images: torch.Tensor,
    fake_images: Optional[torch.Tensor] = None,
    batch_size: int = 24,  # Paper: batch size of 24
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for training/evaluation.
    
    From Paper:
    "batch size of 24"
    """
    dataset = DeepfakeDataset(
        real_images=real_images,
        fake_images=fake_images,
        **kwargs,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == '__main__':
    # Test augmentations
    print("Testing augmentation pipeline...")
    
    # Create dummy image
    img = torch.rand(3, 224, 224)
    
    # Test individual augmentations
    print("Original shape:", img.shape)
    
    flipped = horizontal_flip(img, p=1.0)
    print("After flip:", flipped.shape)
    
    color_aug = random_hue_saturation(img, p=1.0)
    print("After hue/sat:", color_aug.shape)
    
    brightness = random_brightness_contrast(img, p=1.0)
    print("After brightness:", brightness.shape)
    
    jpeg = jpeg_compression(img, p=1.0)
    print("After JPEG:", jpeg.shape)
    
    blur = gaussian_blur(img, p=1.0)
    print("After blur:", blur.shape)
    
    # Test SBI fake generation
    print("\nTesting SBI fake synthesis...")
    source = torch.rand(3, 224, 224)
    target = torch.rand(3, 224, 224)
    
    fake, mask = generate_sbi_fake(source, target)
    print("Fake shape:", fake.shape)
    print("Mask shape:", mask.shape)
    print("Mask range:", mask.min().item(), "-", mask.max().item())
    
    # Test batch generation
    print("\nTesting batch generation...")
    batch = torch.rand(8, 3, 224, 224)
    images, labels, masks = generate_random_fake_image(batch, fake_probability=0.5)
    print("Output batch shape:", images.shape)
    print("Labels:", labels.tolist())
    print("Masks shape:", masks.shape)
    
    print("\nAll tests passed!")
