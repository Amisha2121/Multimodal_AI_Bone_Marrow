
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    timm = None
    _TIMM_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    _PEFT_AVAILABLE = False

from vision.taxonomy import MORPH_CLASSES


# Default LoRA target modules for timm vit_small_patch16_224.
# The attention QKV projection in timm ViTs is fused as "qkv"; we target
# "query" and "value" via name-based matching from PEFT.
# In practice, timm vit_small uses module names like:
#   blocks.N.attn.qkv  (fused QKV linear)
# PEFT matches substrings, so we target "qkv" to hit the fused layer.
_DEFAULT_LORA_TARGET_MODULES = ["qkv"]


class BoneMarrowViT(nn.Module):
    """
    ViT-Small/16 with LoRA and an 8-class morphology classifier head.

    Only the LoRA adapters and the final head are trainable by default.
    All other backbone parameters are frozen.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        num_classes: int = len(MORPH_CLASSES),
        pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            backbone_name: timm model identifier.
            num_classes: Number of output classes (default: 8 MORPH_CLASSES).
            pretrained: Load ImageNet pre-trained weights.
            use_lora: Apply LoRA adapters to attention projections.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling alpha.
            lora_dropout: Dropout probability within LoRA adapters.
            target_modules: Module name substrings to apply LoRA to.
                Defaults to ['qkv'] (timm's fused QKV projection).
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_lora = use_lora

        if not _TIMM_AVAILABLE:
            # Minimal fallback for environments without timm — used in unit tests only
            logger.warning("timm not available — using minimal stub backbone.")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.head = nn.Linear(64, num_classes)
            return

        # Load backbone with num_classes=0 so timm returns features, not logits.
        # We attach our own head to ensure exact num_classes control.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,   # removes the default classifier head
        )

        # Freeze all backbone parameters before applying LoRA
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to attention projections
        if use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required for LoRA fine-tuning. "
                    "Install with: pip install peft>=0.7.0"
                )
            if target_modules is None:
                target_modules = _DEFAULT_LORA_TARGET_MODULES

            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                # task_type is not set: PEFT auto-handles feature extractor models
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            logger.info(
                f"LoRA applied: r={lora_r}, alpha={lora_alpha}, "
                f"targets={target_modules}"
            )

        # Feature dimension from the backbone (ViT-Small/16 → 384)
        feat_dim = self._probe_feature_dim()

        # Fresh classification head (always trainable)
        self.head = nn.Linear(feat_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _probe_feature_dim(self) -> int:
        """Runs a single dummy forward pass to determine backbone output dim."""
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            try:
                out = self.backbone(dummy)
                return out.shape[-1]
            except Exception:
                # Default ViT-Small/16 feature dim
                return 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Image batch of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        feats = self.backbone(x)
        return self.head(feats)

    # ── Parameter statistics ──────────────────────────────────────────────────

    def get_trainable_param_stats(self) -> Dict[str, Union[int, float]]:
        """Returns counts of total vs. trainable parameters."""
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        return {
            "total_params":        total_params,
            "trainable_params":    trainable_params,
            "trainable_percentage": round(pct, 2),
        }

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Union[str, Path], extra_metadata: Optional[Dict] = None) -> None:
        """Saves a lightweight checkpoint with state dict and config metadata."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict":    self.state_dict(),
            "backbone_name": self.backbone_name,
            "num_classes":   self.num_classes,
            "use_lora":      self.use_lora,
            "class_names":   MORPH_CLASSES,
            "metadata":      extra_metadata or {},
        }
        torch.save(checkpoint, out_path)
        logger.info(f"Checkpoint saved → {out_path}")

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: str = "cpu") -> "BoneMarrowViT":
        """Loads model from a checkpoint saved by save_checkpoint()."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            backbone_name=checkpoint.get("backbone_name", "vit_small_patch16_224"),
            num_classes=checkpoint.get("num_classes", len(MORPH_CLASSES)),
            pretrained=False,   # weights loaded from checkpoint
            use_lora=checkpoint.get("use_lora", False),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
