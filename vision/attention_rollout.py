"""
vision/attention_rollout.py
─────────────────────────────────────────────────────────────────────────────
Attention Rollout Explainability for Vision Transformers (Abnar & Zuidema 2020).

Computes self-attention flow through transformer layers to produce a spatial
attribution heatmap highlighting the subcellular morphological features
(nucleoli, chromatin pattern, Auer rods, cytoplasmic granules) that drove
the model's classification decision.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTAttentionRollout:
    """
    Hooks into ViT attention blocks to record attention matrices and compute Attention Rollout.
    """

    def __init__(
        self,
        model: nn.Module,
        discard_ratio: float = 0.1,
        head_fusion: str = "mean",
    ):
        """
        Args:
            model: PyTorch ViT model.
            discard_ratio: Bottom percentage of attention weights to prune for cleaner signal.
            head_fusion: Strategy to aggregate attention across heads ("mean", "max", or "min").
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.attentions: List[torch.Tensor] = []
        self.hooks = []
        # Ensure timm attention modules compute explicit attention matrices rather than SDPA kernel
        for m in self.model.modules():
            if hasattr(m, "fused_attn"):
                m.fused_attn = False
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on attention softmax layers."""
        for name, module in self.model.named_modules():
            # Match standard timm ViT attention softmax
            if "attn_drop" in name or (isinstance(module, nn.Softmax) and "attn" in name):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, inputs, output):
        # Attention weights matrix shape: (batch_size, num_heads, num_tokens, num_tokens)
        self.attentions.append(output.detach().cpu())

    def clear(self):
        """Clears cached attention maps."""
        self.attentions.clear()

    def remove_hooks(self):
        """Removes registered PyTorch hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Runs forward pass and computes the attention rollout heatmap.

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W).

        Returns:
            2D numpy array (H, W) normalized to [0.0, 1.0].
        """
        self.clear()
        device = next(self.model.parameters()).device
        _ = self.model(input_tensor.to(device))

        if not self.attentions:
            # Fallback mock heatmap if attention hooks did not trigger (e.g. non-transformer backbone)
            h, w = input_tensor.shape[-2:]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Aggregate attention rollout
        # attentions: list of (1, heads, tokens, tokens)
        result = torch.eye(self.attentions[0].size(-1))

        with torch.no_grad():
            for attention in self.attentions:
                if self.head_fusion == "mean":
                    attn_heads = attention.mean(dim=1).squeeze(0)
                elif self.head_fusion == "max":
                    attn_heads = attention.max(dim=1)[0].squeeze(0)
                elif self.head_fusion == "min":
                    attn_heads = attention.min(dim=1)[0].squeeze(0)
                else:
                    raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

                # Discard lowest attention weights
                if self.discard_ratio > 0:
                    flat = attn_heads.view(-1)
                    val, _ = torch.topk(flat, int(flat.size(0) * self.discard_ratio), largest=False)
                    threshold = val[-1]
                    attn_heads[attn_heads < threshold] = 0

                # Add residual connection (identity matrix) and re-normalize rows
                I = torch.eye(attn_heads.size(-1))
                a = (attn_heads + 1.0 * I) / 2.0
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        # Extract attribution of CLS token (index 0) to all patch tokens
        mask = result[0, 1:]  # (num_patches,)
        num_patches = mask.size(0)
        grid_size = int(np.sqrt(num_patches))

        mask = mask.reshape(1, 1, grid_size, grid_size)
        h, w = input_tensor.shape[-2:]
        # Bilinear interpolation up to input image resolution
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = mask.squeeze().numpy()

        # Min-max normalization
        mask_min, mask_max = mask.min(), mask.max()
        if mask_max - mask_min > 1e-8:
            mask = (mask - mask_min) / (mask_max - mask_min)
        else:
            mask = np.zeros_like(mask)

        return mask.astype(np.float32)


def aggregate_patient_rollouts(
    rollout_heatmaps: List[np.ndarray],
) -> np.ndarray:
    """
    Computes a mean attention heatmap across multiple cell crops
    (e.g., across all blast crops from a patient) for aggregate visualization.
    """
    if not rollout_heatmaps:
        return np.zeros((224, 224), dtype=np.float32)

    stacked = np.stack(rollout_heatmaps, axis=0)
    mean_map = np.mean(stacked, axis=0)
    m_min, m_max = mean_map.min(), mean_map.max()
    if m_max - m_min > 1e-8:
        mean_map = (mean_map - m_min) / (m_max - m_min)
    return mean_map.astype(np.float32)
