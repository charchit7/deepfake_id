"""
Knowledge Injection based Deepfake Detection (KID)
===================================================
Implementation of "Deepfake Detection via Knowledge Injection" (arXiv:2503.02503v1)

This implementation follows a functional programming style with detailed comments
explaining the reasoning from the paper for each component.

Paper Summary:
- Proposes a multi-task learning based knowledge injection framework
- Compatible with ViT-based backbone models (ViT, DinoV2, LeViT)
- Three main components:
  1. Knowledge Injection Module (Section 3.1)
  2. Coarse-grained Forgery Localization Branch (Section 3.2)
  3. Layer-wise Suppression and Contrast Losses (Section 3.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
from functools import partial


# =============================================================================
# SECTION 1: Core Functional Operations for Knowledge Injection
# =============================================================================

def compute_qkv(
    features: torch.Tensor,  # Input features from transformer block [B, N, D]
    w_q: nn.Linear,          # Query projection weight
    w_k: nn.Linear,          # Key projection weight  
    w_v: nn.Linear,          # Value projection weight
    num_heads: int,          # Number of attention heads
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, Value projections for multi-head attention.
    
    From Paper Section 3.1:
    "Let P_l represent the input features of the I-MSA block at the l-th layer.
    P_l is firstly splitted into multi-head features, denoted as H_l^i."
    
    The standard QKV computation is preserved from the original ViT to maintain
    the pre-trained knowledge about real image distributions.
    """
    B, N, D = features.shape  # B: batch, N: num_patches + 1 (cls token), D: embed_dim
    head_dim = D // num_heads  # Dimension per attention head
    
    # Project input features to Q, K, V spaces
    # Shape: [B, N, D] -> [B, N, D]
    q = w_q(features)
    k = w_k(features)
    v = w_v(features)
    
    # Reshape for multi-head attention: [B, N, D] -> [B, num_heads, N, head_dim]
    # This split allows parallel computation across heads
    q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    
    return q, k, v


def compute_knowledge_query(
    features: torch.Tensor,  # Multi-head features H_l^i [B, num_heads, N, head_dim]
    w_q_tilde: nn.Linear,    # Knowledge query projection W_l^{\tilde{Q}}
    num_heads: int,
) -> torch.Tensor:
    """
    Compute the additional knowledge query vector for authenticity correlation.
    
    From Paper Section 3.1, Equation (1):
    "\bar{Q} = H_l^i W_l^{\tilde{Q}}"
    
    This is the KEY INNOVATION of the paper. Instead of using standard Q for
    attention, we introduce a separate query specifically designed to learn
    the authenticity correlation between image patches.
    
    The knowledge query learns to identify patterns that distinguish real
    patches from fake patches, independent of the main classification task.
    """
    B, N, D = features.shape
    head_dim = D // num_heads
    
    # Project features through the knowledge-specific query matrix
    # This projection learns what "authenticity" means in feature space
    q_tilde = w_q_tilde(features)  # [B, N, D]
    
    # Reshape to multi-head format for parallel processing
    q_tilde = q_tilde.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    
    return q_tilde  # [B, num_heads, N, head_dim]


def compute_authenticity_correlation(
    q_tilde: torch.Tensor,  # Knowledge query [B, num_heads, N, head_dim]
    k: torch.Tensor,        # Key from standard attention [B, num_heads, N, head_dim]
    scale: float,           # Scaling factor sqrt(d_k)
) -> torch.Tensor:
    """
    Compute the authenticity correlation matrix.
    
    From Paper Section 3.1, Equation (2):
    "\overline{Corr}_l^i = \frac{\bar{Q}_l^i K_l^i}{\sqrt{d_k}}"
    
    This correlation matrix represents the learned knowledge about distributions
    of real and fake data. High correlation between patches suggests they share
    similar authenticity characteristics.
    
    Key Insight from Paper:
    "The authenticity correlation matrix \overline{Corr}_l, which represents
    the learned knowledge about the distributions of real and fake data"
    
    For REAL images: patches should show consistent correlations (uniform patterns)
    For FAKE images: manipulated regions show different correlation patterns
    """
    # Matrix multiplication: Q_tilde @ K^T
    # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
    corr = torch.matmul(q_tilde, k.transpose(-2, -1))
    
    # Scale by sqrt(d_k) to prevent gradients from becoming too small
    # This is standard practice from "Attention is All You Need"
    corr = corr / scale
    
    return corr  # [B, num_heads, N, N] - correlation between all patch pairs


def knowledge_injected_attention(
    q: torch.Tensor,           # Standard query [B, num_heads, N, head_dim]
    k: torch.Tensor,           # Standard key [B, num_heads, N, head_dim]
    v: torch.Tensor,           # Standard value [B, num_heads, N, head_dim]
    corr: torch.Tensor,        # Authenticity correlation [B, num_heads, N, N]
    scale: float,              # Scaling factor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform attention with injected knowledge from authenticity correlation.
    
    From Paper Section 3.1, Equation (3):
    "head_l^i = softmax(\frac{Q_l^i K_l^i}{\sqrt{d_k}} + \overline{Corr}_l^i) V_l^i"
    
    This is the core of knowledge injection. The standard attention scores
    are AUGMENTED with the authenticity correlation matrix.
    
    Key Insight:
    - Standard attention: learns what regions to attend to for classification
    - Authenticity correlation: biases attention based on patch authenticity
    - Combined: model learns to classify while being aware of forgery patterns
    
    From Paper Remark (Equation 6):
    "head_l^i = softmax(\frac{H_l^i(W_l^Q + W_l^{\tilde{Q}})K_l^i}{\sqrt{d_k}})V_l^i"
    This shows that W_l^Q and W_l^{\tilde{Q}} are symmetric in their effect.
    """
    # Compute standard attention scores: Q @ K^T / sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, heads, N, N]
    
    # INJECT KNOWLEDGE: Add authenticity correlation to attention scores
    # This biases the attention based on learned forgery patterns
    # The correlation acts as a learned prior about patch relationships
    augmented_scores = attn_scores + corr  # [B, heads, N, N]
    
    # Softmax normalization to get attention weights
    attn_weights = F.softmax(augmented_scores, dim=-1)  # [B, heads, N, N]
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)  # [B, heads, N, head_dim]
    
    return output, corr  # Return corr for loss computation


def merge_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Merge multi-head outputs back to original dimension.
    
    Reverse operation of head splitting in compute_qkv.
    [B, num_heads, N, head_dim] -> [B, N, D]
    """
    B, _, N, head_dim = x.shape
    # Permute and reshape: [B, heads, N, d] -> [B, N, heads, d] -> [B, N, D]
    x = x.permute(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)
    return x


# =============================================================================
# SECTION 2: Injection Multi-head Self-Attention (I-MSA) Block
# =============================================================================

class InjectionMultiHeadSelfAttention(nn.Module):
    """
    Injection Multi-Head Self-Attention (I-MSA) Block.
    
    From Paper Section 3.1:
    "To integrate our knowledge injection framework into the backbone model,
    we propose an Injection Multi-Head Self-Attention (I-MSA) block to replace
    the regular multi-head self-attention block to perform knowledge injection."
    
    Key Design Choices:
    1. Original attention weights (W^Q, W^K, W^V) are FROZEN during training
       - Preserves pre-trained knowledge about real images from ImageNet
    2. Only W^{\tilde{Q}} is trained
       - Reduces parameters significantly (faster convergence)
       - Forces model to learn only authenticity-related modifications
    
    From Paper:
    "In the training process, the original attention branch is frozen and only
    the weight matrix W_l^{\tilde{Q}} is updated."
    """
    
    def __init__(
        self,
        embed_dim: int,      # Embedding dimension D
        num_heads: int,      # Number of attention heads
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k) for scaling
        
        # Standard QKV projections - these will be FROZEN (loaded from pretrained)
        # Paper: "the original attention branch is frozen"
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Knowledge injection query projection - THIS IS TRAINED
        # Paper Eq. (1): "\bar{Q} = H_l^i W_l^{\tilde{Q}}"
        # This is the only new parameter added to each attention block
        self.w_q_tilde = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Initialize w_q_tilde to near-zero so initial behavior matches original ViT
        # This ensures stable training start
        nn.init.zeros_(self.w_q_tilde.weight)
        if qkv_bias:
            nn.init.zeros_(self.w_q_tilde.bias)
        
        # Output projection (frozen, from pretrained)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def freeze_pretrained_weights(self):
        """
        Freeze the pretrained attention weights.
        
        From Paper:
        "The only updated parameters of the backbone part are class token
        and normalization layers."
        
        This preserves the pre-trained knowledge about real image distributions
        learned from ImageNet.
        """
        for param in self.w_q.parameters():
            param.requires_grad = False
        for param in self.w_k.parameters():
            param.requires_grad = False
        for param in self.w_v.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        x: torch.Tensor,  # Input features [B, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with knowledge injection.
        
        Returns:
            - output: Attention output [B, N, D]
            - corr: Authenticity correlation matrix [B, heads, N, N]
                   (needed for suppression and contrast losses)
        """
        B, N, D = x.shape
        
        # Step 1: Compute standard Q, K, V (using frozen weights)
        q, k, v = compute_qkv(x, self.w_q, self.w_k, self.w_v, self.num_heads)
        
        # Step 2: Compute knowledge query (using trainable w_q_tilde)
        # This is the only trainable part of attention - learns authenticity patterns
        q_tilde = compute_knowledge_query(x, self.w_q_tilde, self.num_heads)
        
        # Step 3: Compute authenticity correlation matrix
        # Paper Eq. (2): measures similarity in authenticity space
        corr = compute_authenticity_correlation(q_tilde, k, self.scale)
        
        # Step 4: Perform attention with injected knowledge
        # Paper Eq. (3): standard attention + authenticity bias
        attn_output, corr = knowledge_injected_attention(q, k, v, corr, self.scale)
        
        # Step 5: Merge heads and project
        attn_output = merge_heads(attn_output, self.num_heads)  # [B, N, D]
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        
        return attn_output, corr


# =============================================================================
# SECTION 3: Transformer Block with Knowledge Injection
# =============================================================================

class KIDTransformerBlock(nn.Module):
    """
    Transformer block with Injection Multi-Head Self-Attention.
    
    This replaces the standard ViT transformer block.
    Architecture: LayerNorm -> I-MSA -> Residual -> LayerNorm -> MLP -> Residual
    
    From Paper Figure 2:
    The I-MSA block is placed where the standard attention would be,
    with the knowledge injection happening inside the attention computation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        
        # Pre-norm architecture (standard in ViT)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # I-MSA block with knowledge injection
        self.attn = InjectionMultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # MLP block
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both features and correlation matrix.
        
        The correlation matrix is needed for layer-wise losses (Section 3.3).
        """
        # I-MSA with residual connection
        attn_out, corr = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, corr


# =============================================================================
# SECTION 4: Coarse-Grained Forgery Localization Branch
# =============================================================================

def compute_localization_features(
    features: torch.Tensor,      # Input features L_l [B, N, D]
    corr: torch.Tensor,          # Authenticity correlation [B, heads, N, N]
    w_k_loc: nn.Linear,          # Localization key projection W^K
    positional_encoding: torch.Tensor,  # PE for position awareness
    layer_norm: nn.LayerNorm,
    num_heads: int,
) -> torch.Tensor:
    """
    Update localization features using authenticity correlation.
    
    From Paper Section 3.2, Equation (4):
    "L_{l+1} = softmax(\overline{Corr}_l) · LN(L_l + PE) · W_l^K"
    
    Breaking down the equation:
    1. L_l + PE: Add positional encoding to localization features
    2. LN(...): Apply layer normalization
    3. softmax(Corr_l): Get attention weights from correlation matrix
    4. softmax(Corr) @ LN(L + PE): Weighted aggregation of features
    5. ... @ W^K: Project to key space
    
    Key Insight from Paper:
    "Different from the original classification branch, our localization branch
    provides a supplementary pathway to constrain the detection model in a
    multi-task learning manner."
    
    The localization branch helps learn:
    - Inner patches: consistent forgery patterns in fake images
    - Outer patches: should look real even in fake images  
    - Boundary patches: where blending artifacts typically appear
    """
    B, N, D = features.shape
    
    # Step 1: Add positional encoding to features (L_l + PE)
    # PE helps the model understand spatial relationships
    features_with_pe = features + positional_encoding  # [B, N, D]
    
    # Step 2: Apply layer normalization LN(L_l + PE)
    features_normed = layer_norm(features_with_pe)  # [B, N, D]
    
    # Step 3: Average correlation across heads for localization
    # [B, heads, N, N] -> [B, N, N]
    corr_avg = corr.mean(dim=1)
    
    # Step 4: Apply softmax to get attention weights from correlation
    # softmax(\overline{Corr}_l)
    corr_weights = F.softmax(corr_avg, dim=-1)  # [B, N, N]
    
    # Step 5: Weighted aggregation: softmax(Corr) @ LN(L + PE)
    # This makes localization features attend to patches with similar authenticity
    aggregated_features = torch.matmul(corr_weights, features_normed)  # [B, N, D]
    
    # Step 6: Project through localization key matrix W^K
    # The paper shows this as right multiplication: ... · W^K
    updated_features = w_k_loc(aggregated_features)  # [B, N, D]
    
    return updated_features


def compute_localization_labels(
    face_masks: torch.Tensor,    # Binary face masks [B, H, W]
    num_patches_h: int,          # Number of patches in height
    num_patches_w: int,          # Number of patches in width
    gamma_0: float = 0.2,        # Lower threshold (paper: 0.2)
    gamma_1: float = 0.8,        # Upper threshold (paper: 0.8)
) -> torch.Tensor:
    """
    Compute coarse-grained localization ground truth labels.
    
    From Paper Section 3.2, Equation (5):
    "y_i = {0,       if Γ_i < γ_0
            1,       if Γ_i > γ_1
            Γ_i,     otherwise}"
    
    Where Γ_i is the percentage of pixels belonging to the outer face
    within each patch.
    
    Label interpretation:
    - 0: Inner face patch (mostly inside the face)
    - 1: Outer face patch (mostly outside the face)
    - Γ_i: Boundary region (mixture of inner and outer)
    
    This three-way labeling helps the model learn:
    - Inner patches: should have consistent forgery patterns in fake images
    - Outer patches: should look real even in fake images
    - Boundary patches: where blending artifacts typically appear
    """
    B, H, W = face_masks.shape
    
    # Reshape mask into patches
    # [B, H, W] -> [B, num_patches_h, patch_h, num_patches_w, patch_w]
    patch_h = H // num_patches_h
    patch_w = W // num_patches_w
    
    masks_patched = face_masks.reshape(B, num_patches_h, patch_h, num_patches_w, patch_w)
    masks_patched = masks_patched.permute(0, 1, 3, 2, 4)  # [B, ph, pw, h, w]
    
    # Compute Γ_i: percentage of outer face pixels per patch
    # Assuming mask: 1 = outer face, 0 = inner face
    gamma_i = masks_patched.float().mean(dim=(-2, -1))  # [B, num_patches_h, num_patches_w]
    gamma_i = gamma_i.reshape(B, -1)  # [B, num_patches]
    
    # Apply thresholding from Equation (5)
    labels = gamma_i.clone()
    labels[gamma_i < gamma_0] = 0.0  # Inner face
    labels[gamma_i > gamma_1] = 1.0  # Outer face
    # Boundary regions keep their Γ_i value
    
    return labels  # [B, num_patches]


class ForgeryLocalizationBranch(nn.Module):
    """
    Coarse-Grained Forgery Localization Branch.
    
    From Paper Section 3.2:
    "To guide the knowledge injection module in learning the knowledge of fake
    data and enhance the location-awareness of injected forgery knowledge, we
    construct a separate coarse-grained forgery localization branch."
    
    This branch is ONLY USED DURING TRAINING (not inference).
    It serves as an auxiliary task to improve the main classification.
    
    Architecture:
    - Takes final layer features
    - Applies MLP to predict patch-wise localization scores
    - Trained with dice loss against ground truth labels
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_patches: int,  # N - 1 (excluding cls token)
    ):
        super().__init__()
        
        # Positional encoding for location awareness
        # Paper: "Positional Encoding(PE) is applied to enhance the perception
        # of positional information"
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Localization key projection W^K from paper
        self.w_k_loc = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm for feature processing
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # MLP for patch classification
        # Paper: "a MLP layer is constructed to conduct coarse-grained
        # classifications on the output image patch features of the last layer"
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),  # Binary classification per patch
            nn.Sigmoid(),
        )
        
        self.num_patches = num_patches
    
    def forward(
        self,
        features: torch.Tensor,  # [B, N, D] from last transformer block
        corr: torch.Tensor,      # [B, heads, N, N] authenticity correlation
    ) -> torch.Tensor:
        """
        Predict patch-wise localization scores.
        
        Returns:
            localization_scores: [B, num_patches] probability of outer face
        """
        B, N, D = features.shape
        
        # Update features using authenticity correlation (Eq. 4)
        loc_features = compute_localization_features(
            features=features,
            corr=corr,
            w_k_loc=self.w_k_loc,
            positional_encoding=self.positional_encoding,
            layer_norm=self.layer_norm,
            num_heads=corr.shape[1],
        )
        
        # Remove cls token for patch-wise prediction
        loc_features = loc_features[:, 1:, :]  # [B, num_patches, D]
        
        # Predict localization scores
        scores = self.mlp(loc_features).squeeze(-1)  # [B, num_patches]
        
        return scores


# =============================================================================
# SECTION 5: Loss Functions
# =============================================================================

def compute_suppression_loss(
    correlations: List[torch.Tensor],  # List of corr matrices from layers
    layer_indices: List[int],          # Which layers to apply (shallow layers)
    beta: float = 1.2,                 # Upper bound parameter (paper: β = 1.2)
) -> torch.Tensor:
    """
    Compute layer-wise suppression loss for shallow layers.
    
    From Paper Section 3.3, Equations (8) and (9):
    "A_l = (1/MN) Σ_i Σ_j |Corr_{l,i,j}|"        (8)
    "L_S = Σ_{l=0}^{L_0} (Σ_{b=0}^{B} max(0, A_l - β)) / B"     (9)
    
    Purpose:
    "This suppression loss constrains the detection model to maintain the
    core understandings of real data distributions at the shallow layers"
    
    Key Insight:
    - Shallow layers should preserve general image understanding
    - We suppress large activations in correlation matrix per sample
    - This prevents the model from injecting too much "fake knowledge" early
    - Maintains the pre-trained representation of real images
    
    The β parameter (1.2) allows some modification but prevents excessive changes.
    """
    loss = torch.tensor(0.0, device=correlations[0].device)
    
    for idx in layer_indices:
        if idx >= len(correlations):
            continue
            
        corr = correlations[idx]  # [B, heads, N, N]
        B = corr.shape[0]
        
        # Compute average activation per sample: A_l = (1/MN) Σ|Corr|
        # M = num_heads * N, N = sequence length
        # Paper Eq. 8: average over heads and spatial dimensions per sample
        A_l_per_sample = corr.abs().mean(dim=(1, 2, 3))  # [B]
        
        # Apply hinge loss with upper bound β per sample
        # Paper Eq. 9: sum over batch, then normalize
        # Only penalize if activation exceeds β
        layer_loss = F.relu(A_l_per_sample - beta).sum() / B
        
        loss = loss + layer_loss
    
    # Sum over layers (no normalization by num_layers in paper)
    return loss


def compute_contrast_loss(
    correlations: List[torch.Tensor],  # Corr matrices from deep layers
    labels: torch.Tensor,              # Binary labels: 0=real, 1=fake [B]
    layer_indices: List[int],          # Which layers (deep layers, last 2)
    mu: float = 0.1,                   # Margin parameter (paper: μ = 0.1)
) -> torch.Tensor:
    """
    Compute contrast loss for deep layers.
    
    From Paper Section 3.3, Equation (10):
    "L_D = Σ_{l=L-2}^{L} (Σ_{b=0}^{B} max(0, A_l^fake - A_l^real + μ)) / B"
    
    Purpose:
    "According to general observations, real images generally exhibit strong
    internal consistencies and correlations across patches, whereas fake images
    often display inconsistencies between manipulated and benign regions"
    
    Key Insight from Paper:
    - The contrast loss is computed based on the authenticity correlation
      matrices (knowledge) from the I-MSA blocks in the final two transformer blocks
    - It effectively improves the knowledge of the distributions of real and
      fake images, establishing a more accurate classification boundary
    
    The μ parameter (0.1) sets minimum acceptable difference between activations.
    """
    loss = torch.tensor(0.0, device=correlations[0].device)
    
    # Separate real and fake samples
    real_mask = (labels == 0)  # Real images
    fake_mask = (labels == 1)  # Fake images
    
    if not real_mask.any() or not fake_mask.any():
        # Need both real and fake samples for contrast
        return loss
    
    B = labels.shape[0]
    
    for idx in layer_indices:
        if idx >= len(correlations):
            continue
            
        corr = correlations[idx]  # [B, heads, N, N]
        
        # Compute average activation per sample: A_l per sample
        # Paper Eq. 8: A_l = (1/MN) Σ|Corr|
        A_per_sample = corr.abs().mean(dim=(1, 2, 3))  # [B]
        
        # Get mean activation for real and fake samples in this batch
        # Paper Eq. 10 computes this per batch
        A_real_mean = A_per_sample[real_mask].mean()
        A_fake_mean = A_per_sample[fake_mask].mean()
        
        # Contrastive hinge loss from Eq. 10:
        # max(0, A_l^fake - A_l^real + μ)
        # This encourages A_real > A_fake + μ (real has higher activation)
        # The loss is active when fake activation is too close to real
        layer_loss = F.relu(A_fake_mean - A_real_mean + mu)
        
        loss = loss + layer_loss
    
    # Sum over layers (paper sums over layers L-2 to L)
    return loss


def compute_dice_loss(
    predictions: torch.Tensor,  # [B, num_patches] predicted scores
    targets: torch.Tensor,      # [B, num_patches] ground truth labels
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute dice loss for forgery localization.
    
    From Paper Section 3.2:
    "At last, a dice loss is calculated to update the coarse-grained forgery
    localization branch."
    
    Dice loss is good for:
    - Handling class imbalance (many inner vs boundary patches)
    - Smooth gradients for probability predictions
    - Commonly used in segmentation tasks
    
    Formula: Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice
    """
    # Flatten predictions and targets
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Dice loss
    return 1.0 - dice


def compute_kid_loss(
    logits: torch.Tensor,              # Classification logits [B, 2]
    labels: torch.Tensor,              # Binary labels [B]
    correlations: List[torch.Tensor],  # All correlation matrices
    loc_predictions: Optional[torch.Tensor] = None,  # [B, num_patches]
    loc_targets: Optional[torch.Tensor] = None,      # [B, num_patches]
    num_layers: int = 12,
    num_shallow_layers: int = 8,       # L_0 from paper: layers 0 to L_0-1
    beta: float = 1.2,                 # Suppression bound
    mu: float = 0.1,                   # Contrast margin
) -> Dict[str, torch.Tensor]:
    """
    Compute overall KID loss.
    
    From Paper Section 3.4, Equation (11):
    "L = L_CE + L_DICE + L_S + L_D"
    
    Components:
    - L_CE: Cross-entropy for classification (main task)
    - L_DICE: Dice loss for localization (auxiliary task, training only)
    - L_S: Suppression loss for shallow layers (preserve real knowledge)
    - L_D: Contrast loss for deep layers (separate real/fake)
    
    From Paper:
    "Note that the entire coarse-grained forgery localization branch is only
    utilized in the training phase."
    """
    # Classification loss (cross-entropy)
    L_CE = F.cross_entropy(logits, labels)
    
    # Suppression loss (shallow layers: 0 to L_0-1)
    shallow_indices = list(range(num_shallow_layers))
    L_S = compute_suppression_loss(correlations, shallow_indices, beta)
    
    # Contrast loss (deep layers: last 2 transformer blocks)
    deep_indices = list(range(num_layers - 2, num_layers))
    L_D = compute_contrast_loss(correlations, labels, deep_indices, mu)
    
    # Localization loss (if provided, training only)
    L_DICE = torch.tensor(0.0, device=logits.device)
    if loc_predictions is not None and loc_targets is not None:
        L_DICE = compute_dice_loss(loc_predictions, loc_targets)
    
    # Total loss
    total_loss = L_CE + L_DICE + L_S + L_D
    
    return {
        'total': total_loss,
        'ce': L_CE,
        'dice': L_DICE,
        'suppression': L_S,
        'contrast': L_D,
    }


# =============================================================================
# SECTION 6: Complete KID Model
# =============================================================================

class KnowledgeInjectionDeepfakeDetector(nn.Module):
    """
    Knowledge Injection based Deepfake Detection (KID) Model.
    
    From Paper Abstract:
    "We propose a simple and novel approach, named Knowledge Injection based
    deepfake Detection (KID), by constructing a multi-task learning based
    knowledge injection framework, which can be easily plugged into existing
    ViT-based backbone models, including foundation models."
    
    Architecture Overview (Figure 2):
    1. Patch Embedding: Image -> Patches -> Linear Projection + CLS token
    2. Transformer Blocks with I-MSA: Standard ViT blocks with knowledge injection
    3. Classification Head: CLS token -> MLP -> Binary classification
    4. Localization Branch: Patch features -> MLP -> Per-patch scores (training only)
    
    Key Design Principles:
    - Minimal additional parameters (only W^Q_tilde per layer)
    - Compatible with any ViT-based backbone
    - Preserves pre-trained knowledge through weight freezing
    - Multi-task learning improves generalization
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,        # ViT-B/16 default
        num_layers: int = 12,        # ViT-B/16 default
        num_heads: int = 12,         # ViT-B/16 default
        mlp_ratio: float = 4.0,
        num_classes: int = 2,        # Real vs Fake
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        # Hyperparameters from paper
        beta: float = 1.2,           # Suppression loss bound (Eq. 9)
        mu: float = 0.1,             # Contrast loss margin (Eq. 10)
        gamma_0: float = 0.2,        # Localization lower threshold (Eq. 5)
        gamma_1: float = 0.8,        # Localization upper threshold (Eq. 5)
        num_shallow_layers: int = 8, # L_0 for suppression loss
    ):
        super().__init__()
        
        # Store hyperparameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2  # 14*14 = 196 for 224x224
        self.beta = beta
        self.mu = mu
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.num_shallow_layers = num_shallow_layers
        
        # === Patch Embedding ===
        # Convert image to sequence of patch embeddings
        # Paper: "images are saved as input with the size of 224 × 224"
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # === CLS Token ===
        # Learnable classification token prepended to patch sequence
        # Paper: "The only updated parameters of the backbone part are class
        # token and normalization layers"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # === Positional Embedding ===
        # Add positional information to patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # === Transformer Blocks with I-MSA ===
        # Paper: "We adopt ViT/B-16 pre-trained on ImageNet as the model backbone"
        self.blocks = nn.ModuleList([
            KIDTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(num_layers)
        ])
        
        # === Final Layer Norm ===
        self.norm = nn.LayerNorm(embed_dim)
        
        # === Classification Head ===
        # Binary classification: Real (0) vs Fake (1)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # === Localization Branch (Training Only) ===
        # Paper Section 3.2: Coarse-grained forgery localization
        self.localization_branch = ForgeryLocalizationBranch(
            embed_dim=embed_dim,
            num_patches=self.num_patches,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ViT conventions."""
        # Initialize patch embedding
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def freeze_pretrained(self):
        """
        Freeze pretrained weights, keeping only KID-specific parameters trainable.
        
        From Paper Implementation Details:
        "The only updated parameters of the backbone part are class token
        and normalization layers"
        
        Trainable parameters after freezing:
        - cls_token
        - All LayerNorm parameters
        - w_q_tilde in each I-MSA block (knowledge injection)
        - Localization branch (auxiliary task)
        - Classification head
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze positional embedding
        self.pos_embed.requires_grad = False
        
        # Freeze pretrained attention weights in each block
        for block in self.blocks:
            block.attn.freeze_pretrained_weights()
            # Keep MLP frozen too (standard ViT practice)
            for param in block.mlp.parameters():
                param.requires_grad = False
        
        # Keep trainable: cls_token, norms, w_q_tilde, localization, head
        # These are already trainable by default
    
    def forward_features(
        self,
        x: torch.Tensor,  # [B, C, H, W]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features and collect correlation matrices.
        
        Returns:
            features: [B, N+1, D] (includes CLS token)
            correlations: List of correlation matrices from each layer
        """
        B = x.shape[0]
        
        # Patch embedding: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, N, D]
        x = self.patch_embed(x)  # [B, D, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, D]
        
        # Prepend CLS token: [B, N, D] -> [B, N+1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through transformer blocks, collecting correlations
        correlations = []
        for block in self.blocks:
            x, corr = block(x)
            correlations.append(corr)
        
        # Final layer norm
        x = self.norm(x)
        
        return x, correlations
    
    def forward(
        self,
        x: torch.Tensor,                          # [B, C, H, W]
        labels: Optional[torch.Tensor] = None,    # [B] for training
        face_masks: Optional[torch.Tensor] = None, # [B, H, W] for localization
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            x: Input images [B, C, H, W]
            labels: Ground truth labels (0=real, 1=fake) for training
            face_masks: Face region masks for localization training
            return_loss: Whether to compute and return losses
        
        Returns:
            Dictionary containing:
            - logits: Classification logits [B, 2]
            - probs: Classification probabilities [B, 2]
            - pred: Predicted class [B]
            - losses: (if return_loss) Dictionary of loss components
        """
        # Extract features and correlations
        features, correlations = self.forward_features(x)
        
        # Classification from CLS token
        cls_token = features[:, 0]  # [B, D]
        logits = self.head(cls_token)  # [B, 2]
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
            'pred': pred,
        }
        
        # Compute losses if in training mode
        if return_loss and labels is not None:
            # Localization predictions (training only)
            loc_predictions = None
            loc_targets = None
            
            if face_masks is not None:
                # Get last layer correlation for localization
                last_corr = correlations[-1]
                loc_predictions = self.localization_branch(features, last_corr)
                
                # Compute localization targets
                num_patches_side = self.img_size // self.patch_size
                loc_targets = compute_localization_labels(
                    face_masks,
                    num_patches_side,
                    num_patches_side,
                    self.gamma_0,
                    self.gamma_1,
                )
            
            # Compute all losses
            losses = compute_kid_loss(
                logits=logits,
                labels=labels,
                correlations=correlations,
                loc_predictions=loc_predictions,
                loc_targets=loc_targets,
                num_layers=self.num_layers,
                num_shallow_layers=self.num_shallow_layers,
                beta=self.beta,
                mu=self.mu,
            )
            
            output['losses'] = losses
        
        return output


# =============================================================================
# SECTION 7: Utility Functions
# =============================================================================

def load_pretrained_vit_weights(
    model: KnowledgeInjectionDeepfakeDetector,
    pretrained_path: str,
) -> None:
    """
    Load pretrained ViT weights into KID model.
    
    From Paper Implementation Details:
    "We adopt ViT/B-16 pre-trained on ImageNet as the model backbone"
    
    This function maps standard ViT weights to our KID architecture,
    which has the same structure but with additional w_q_tilde parameters.
    """
    # Load pretrained state dict
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different formats (timm, torchvision, etc.)
    if 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model']
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    
    # Get model state dict
    model_dict = model.state_dict()
    
    # Map pretrained weights to our model
    # Standard ViT uses qkv combined, we use separate q, k, v
    mapped_dict = {}
    
    for key, value in pretrained_dict.items():
        # Skip weights that don't match our architecture
        if key not in model_dict:
            # Try to find matching key
            new_key = key
            # Add mappings here based on your pretrained format
            if new_key in model_dict and model_dict[new_key].shape == value.shape:
                mapped_dict[new_key] = value
        else:
            if model_dict[key].shape == value.shape:
                mapped_dict[key] = value
    
    # Load matched weights
    model.load_state_dict(mapped_dict, strict=False)
    print(f"Loaded {len(mapped_dict)}/{len(model_dict)} weights from pretrained model")


def create_kid_model(
    backbone: str = 'vit_base',
    pretrained: bool = True,
    **kwargs,
) -> KnowledgeInjectionDeepfakeDetector:
    """
    Factory function to create KID model with different backbones.
    
    From Paper Table 5:
    "We also apply the knowledge injection framework to the Vit-based backbone
    models DinoV2 and the more lightweight LeVit"
    
    Supported backbones:
    - vit_base: ViT-B/16 (768 dim, 12 layers, 12 heads)
    - vit_large: ViT-L/16 (1024 dim, 24 layers, 16 heads)
    - vit_small: ViT-S/16 (384 dim, 12 layers, 6 heads)
    """
    configs = {
        'vit_small': {'embed_dim': 384, 'num_layers': 12, 'num_heads': 6},
        'vit_base': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
        'vit_large': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16},
    }
    
    if backbone not in configs:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(configs.keys())}")
    
    # Merge config with kwargs
    config = {**configs[backbone], **kwargs}
    
    # Create model
    model = KnowledgeInjectionDeepfakeDetector(**config)
    
    return model


# =============================================================================
# SECTION 8: Example Usage and Training Loop
# =============================================================================

def train_step(
    model: KnowledgeInjectionDeepfakeDetector,
    images: torch.Tensor,
    labels: torch.Tensor,
    face_masks: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    Single training step.
    
    From Paper Implementation Details:
    "In the training stage, the model is trained for a maximum of 300 epochs
    using the AdamW optimizer, with a weight decay of 0.01 and a batch size
    of 24. Early stopping is implemented, terminating training when the loss
    doesn't decrease for 20 consecutive epochs. The initial learning rate is
    set to 1 × 10^−4, and we utilize cosine annealing for learning rate decay,
    with a lower bound of 1 × 10^−6."
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass with loss computation
    output = model(images, labels=labels, face_masks=face_masks, return_loss=True)
    
    # Backward pass
    loss = output['losses']['total']
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Return loss values for logging
    return {k: v.item() for k, v in output['losses'].items()}


@torch.no_grad()
def evaluate(
    model: KnowledgeInjectionDeepfakeDetector,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    From Paper:
    "We utilize the area under the receiver operating characteristic curve
    (AUC) as the primary evaluation metric"
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        output = model(images)
        probs = output['probs'][:, 1]  # Probability of fake
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
    except ImportError:
        # Fallback: compute accuracy
        preds = (all_probs > 0.5).long()
        auc = (preds == all_labels).float().mean().item()
        print("sklearn not available, returning accuracy instead of AUC")
    
    return {'auc': auc}


if __name__ == '__main__':
    # Example usage
    print("Creating KID model...")
    model = create_kid_model(backbone='vit_base')
    
    # Freeze pretrained weights (as per paper)
    model.freeze_pretrained()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])  # One real, one fake
    face_masks = torch.randint(0, 2, (2, 224, 224)).float()
    
    output = model(x, labels=labels, face_masks=face_masks, return_loss=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Predictions: {output['pred']}")
    print(f"Losses: {output['losses']}")
