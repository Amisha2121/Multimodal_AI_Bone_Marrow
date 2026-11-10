"""
vision/calibration.py
─────────────────────────────────────────────────────────────────────────────
Temperature Scaling and Expected Calibration Error (ECE) for ViT Classifier.

References:
    Guo et al. (2017): On Calibration of Modern Neural Networks
    Naeini et al. (2015): Obtaining Well Calibrated Probabilities Using
                          Bayesian Binning Into Quantiles
    Minderer et al. (2021): Revisiting the Calibration of Modern Neural Networks

Clinical Rationale:
    Calibrated confidence scores are critical for medical AI:
    - A 90% confidence should be correct 90% of the time
    - Prevents overconfident wrong predictions
    - Helps downstream decision-making (Critic agent thresholds)
    - Required for FDA submission and clinical deployment
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TemperatureScalingLayer(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    
    Given logits z, temperature-scaled probabilities are:
        p_calibrated = softmax(z / T)
    
    where T > 0 is learned on a validation set to minimize NLL.
    
    Properties:
        - Preserves ranking (argmax unchanged)
        - Single scalar parameter
        - Fast to optimize (convex in log T)
    """
    
    def __init__(self, temperature: float = 1.5):
        """
        Args:
            temperature: Initial temperature value (typically 1.0-2.0).
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.
        
        Args:
            logits: Raw model logits (B, C).
        
        Returns:
            Temperature-scaled logits (B, C).
        """
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """Returns current temperature value."""
        return self.temperature.item()


def optimize_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01,
) -> TemperatureScalingLayer:
    """
    Optimizes temperature on validation set to minimize NLL.
    
    Args:
        model: Trained classifier (outputs logits).
        val_loader: Validation data loader.
        device: torch device.
        max_iter: Max optimization iterations.
        lr: Learning rate for LBFGS.
    
    Returns:
        Optimized TemperatureScalingLayer.
    """
    # Collect all logits and labels from validation set
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0).to(device)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0).to(device)  # (N,)
    
    # Initialize temperature scaling layer
    temp_layer = TemperatureScalingLayer(temperature=1.5).to(device)
    
    # Optimize temperature via LBFGS (fast convergence for single parameter)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [temp_layer.temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn='strong_wolfe'
    )
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_layer(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    initial_loss = criterion(all_logits, all_labels).item()
    optimizer.step(closure)
    final_loss = criterion(temp_layer(all_logits), all_labels).item()
    
    T_opt = temp_layer.get_temperature()
    logger.info(
        f"Temperature optimization complete: T = {T_opt:.4f} "
        f"(NLL: {initial_loss:.4f} → {final_loss:.4f})"
    )
    
    return temp_layer


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy:
        ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
    
    where B_m are confidence bins.
    
    Args:
        confidences: Max softmax probabilities (N,).
        predictions: Predicted class indices (N,).
        labels: Ground truth class indices (N,).
        n_bins: Number of confidence bins (typically 10-20).
    
    Returns:
        Dict with:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error (worst bin)
            - bin_accuracies: Accuracy per bin
            - bin_confidences: Average confidence per bin
            - bin_counts: Sample count per bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (predictions == labels)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        
        if count > 0:
            acc = np.mean(accuracies[in_bin])
            conf = np.mean(confidences[in_bin])
            
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
            
            # Weighted contribution to ECE
            ece += (count / len(confidences)) * abs(acc - conf)
            
            # Track maximum calibration error
            mce = max(mce, abs(acc - conf))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "n_bins": n_bins,
    }


def plot_reliability_diagram(
    ece_results: Dict,
    save_path: Union[str, Path],
    title: str = "Reliability Diagram"
) -> None:
    """
    Plots reliability diagram showing confidence vs accuracy.
    
    A well-calibrated model should lie on the diagonal (confidence = accuracy).
    
    Args:
        ece_results: Output from compute_ece().
        save_path: Path to save figure.
        title: Plot title.
    """
    bin_confs = ece_results["bin_confidences"]
    bin_accs = ece_results["bin_accuracies"]
    bin_counts = ece_results["bin_counts"]
    n_bins = ece_results["n_bins"]
    ece = ece_results["ece"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot bins as bars
    bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + (1 / (2 * n_bins))
    widths = 1.0 / n_bins
    
    # Color by gap (darker = worse calibration)
    gaps = np.abs(bin_accs - bin_confs)
    colors = plt.cm.RdYlGn_r(gaps / max(0.3, np.max(gaps)))  # Red = bad, Green = good
    
    for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accs, bin_counts)):
        if count > 0:
            ax.bar(
                center, acc, width=widths * 0.9,
                color=colors[i], edgecolor='black', linewidth=1,
                alpha=0.7
            )
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Gap bars (show miscalibration)
    for i, (center, conf, acc, count) in enumerate(
        zip(bin_centers, bin_confs, bin_accs, bin_counts)
    ):
        if count > 0:
            ax.plot([center, center], [conf, acc], 'r-', linewidth=2, alpha=0.6)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Reliability diagram saved → {save_path}")


@torch.no_grad()
def evaluate_calibration(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    temp_layer: Optional[TemperatureScalingLayer] = None,
    n_bins: int = 15,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates calibration on a dataset.
    
    Args:
        model: Trained classifier.
        data_loader: Data loader (typically validation or test).
        device: torch device.
        temp_layer: Optional temperature scaling layer.
        n_bins: Number of confidence bins for ECE.
    
    Returns:
        Tuple of:
            - ece_results: Dict with ECE metrics
            - confidences: Max probabilities (N,)
            - predictions: Predicted classes (N,)
            - labels: Ground truth classes (N,)
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    for images, labels, _, _ in data_loader:
        images = images.to(device)
        logits = model(images)
        
        # Apply temperature scaling if provided
        if temp_layer is not None:
            logits = temp_layer(logits)
        
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        all_probs.append(max_probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    
    confidences = np.concatenate(all_probs)
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    ece_results = compute_ece(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        n_bins=n_bins
    )
    
    return ece_results, confidences, predictions, labels


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Union[str, Path],
    n_bins: int = 15,
) -> TemperatureScalingLayer:
    """
    Full calibration pipeline: optimize temperature + compute ECE + plot.
    
    Args:
        model: Trained classifier.
        val_loader: Validation data loader.
        device: torch device.
        save_dir: Directory to save results.
        n_bins: Number of bins for ECE.
    
    Returns:
        Optimized TemperatureScalingLayer.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("TEMPERATURE SCALING CALIBRATION")
    logger.info("=" * 70)
    
    # 1. Evaluate BEFORE calibration
    logger.info("Evaluating pre-calibration metrics...")
    ece_before, confs_before, preds_before, labels = evaluate_calibration(
        model=model,
        data_loader=val_loader,
        device=device,
        temp_layer=None,
        n_bins=n_bins,
    )
    
    acc_before = np.mean(preds_before == labels)
    logger.info(f"Before calibration:")
    logger.info(f"  Accuracy: {acc_before:.4f}")
    logger.info(f"  ECE:      {ece_before['ece']:.4f}")
    logger.info(f"  MCE:      {ece_before['mce']:.4f}")
    
    # 2. Optimize temperature
    logger.info("\nOptimizing temperature parameter...")
    temp_layer = optimize_temperature(
        model=model,
        val_loader=val_loader,
        device=device,
    )
    
    # 3. Evaluate AFTER calibration
    logger.info("\nEvaluating post-calibration metrics...")
    ece_after, confs_after, preds_after, _ = evaluate_calibration(
        model=model,
        data_loader=val_loader,
        device=device,
        temp_layer=temp_layer,
        n_bins=n_bins,
    )
    
    acc_after = np.mean(preds_after == labels)
    logger.info(f"After calibration:")
    logger.info(f"  Accuracy:    {acc_after:.4f}")
    logger.info(f"  ECE:         {ece_after['ece']:.4f}")
    logger.info(f"  MCE:         {ece_after['mce']:.4f}")
    logger.info(f"  Temperature: {temp_layer.get_temperature():.4f}")
    
    # 4. Plot reliability diagrams
    plot_reliability_diagram(
        ece_results=ece_before,
        save_path=save_dir / "reliability_before_calibration.png",
        title="Before Temperature Scaling"
    )
    
    plot_reliability_diagram(
        ece_results=ece_after,
        save_path=save_dir / "reliability_after_calibration.png",
        title="After Temperature Scaling"
    )
    
    # 5. Save calibration results
    calibration_report = {
        "temperature": temp_layer.get_temperature(),
        "before": {
            "accuracy": float(acc_before),
            "ece": float(ece_before["ece"]),
            "mce": float(ece_before["mce"]),
            "avg_confidence": float(np.mean(confs_before)),
        },
        "after": {
            "accuracy": float(acc_after),
            "ece": float(ece_after["ece"]),
            "mce": float(ece_after["mce"]),
            "avg_confidence": float(np.mean(confs_after)),
        },
        "improvement": {
            "ece_reduction": float(ece_before["ece"] - ece_after["ece"]),
            "ece_reduction_pct": float(
                100 * (ece_before["ece"] - ece_after["ece"]) / max(0.0001, ece_before["ece"])
            ),
        },
        "n_bins": n_bins,
        "n_samples": len(labels),
    }
    
    with open(save_dir / "calibration_report.json", "w") as f:
        json.dump(calibration_report, f, indent=2)
    
    logger.info(f"\nCalibration report saved → {save_dir / 'calibration_report.json'}")
    logger.info(f"ECE reduction: {calibration_report['improvement']['ece_reduction']:.4f} "
                f"({calibration_report['improvement']['ece_reduction_pct']:.1f}%)")
    logger.info("=" * 70 + "\n")
    
    # 6. Save temperature layer
    torch.save(
        temp_layer.state_dict(),
        save_dir / "temperature_layer.pt"
    )
    logger.info(f"Temperature layer saved → {save_dir / 'temperature_layer.pt'}")
    
    return temp_layer


class CalibratedBoneMarrowViT(nn.Module):
    """
    Wrapper that combines BoneMarrowViT + TemperatureScalingLayer.
    
    Use this for inference to get calibrated probabilities.
    """
    
    def __init__(self, base_model: nn.Module, temp_layer: TemperatureScalingLayer):
        """
        Args:
            base_model: Trained BoneMarrowViT.
            temp_layer: Optimized temperature scaling layer.
        """
        super().__init__()
        self.base_model = base_model
        self.temp_layer = temp_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Image batch (B, 3, H, W).
        
        Returns:
            Temperature-scaled logits (B, num_classes).
        """
        logits = self.base_model(x)
        return self.temp_layer(logits)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns calibrated probabilities.
        
        Args:
            x: Image batch (B, 3, H, W).
        
        Returns:
            Calibrated probabilities (B, num_classes).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def get_temperature(self) -> float:
        """Returns current temperature value."""
        return self.temp_layer.get_temperature()
    
    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        temp_path: Union[str, Path],
        device: str = "cpu"
    ) -> "CalibratedBoneMarrowViT":
        """
        Loads calibrated model from separate checkpoints.
        
        Args:
            model_path: Path to BoneMarrowViT checkpoint.
            temp_path: Path to temperature layer checkpoint.
            device: torch device.
        
        Returns:
            CalibratedBoneMarrowViT instance.
        """
        from vision.model import BoneMarrowViT
        
        # Load base model
        base_model = BoneMarrowViT.load_checkpoint(model_path, device=device)
        
        # Load temperature layer
        temp_layer = TemperatureScalingLayer()
        temp_layer.load_state_dict(torch.load(temp_path, map_location=device))
        temp_layer.to(device)
        
        return cls(base_model, temp_layer)


def print_calibration_summary(report_path: Union[str, Path]) -> None:
    """Prints a formatted calibration summary from JSON report."""
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Temperature: {report['temperature']:.4f}")
    print(f"Samples:     {report['n_samples']}")
    print(f"Bins:        {report['n_bins']}")
    print()
    print("Metrics Comparison:")
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print("  " + "-" * 60)
    print(f"  {'Accuracy':<20} {report['before']['accuracy']:>10.4f} "
          f"{report['after']['accuracy']:>10.4f} {report['after']['accuracy'] - report['before']['accuracy']:>10.4f}")
    print(f"  {'ECE':<20} {report['before']['ece']:>10.4f} "
          f"{report['after']['ece']:>10.4f} {report['after']['ece'] - report['before']['ece']:>10.4f}")
    print(f"  {'MCE':<20} {report['before']['mce']:>10.4f} "
          f"{report['after']['mce']:>10.4f} {report['after']['mce'] - report['before']['mce']:>10.4f}")
    print(f"  {'Avg Confidence':<20} {report['before']['avg_confidence']:>10.4f} "
          f"{report['after']['avg_confidence']:>10.4f} {report['after']['avg_confidence'] - report['before']['avg_confidence']:>10.4f}")
    print()
    print(f"ECE Reduction: {report['improvement']['ece_reduction_pct']:.1f}%")
    print("=" * 70 + "\n")
