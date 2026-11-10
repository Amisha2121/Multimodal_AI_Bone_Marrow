"""
tests/test_calibration.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for temperature scaling calibration module.
"""

import sys
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from vision.calibration import (
    TemperatureScalingLayer,
    compute_ece,
    optimize_temperature,
    evaluate_calibration,
    calibrate_model,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 8)
    
    def forward(self, x):
        return self.fc(x)


def test_temperature_scaling_layer():
    """Test TemperatureScalingLayer initialization and forward pass."""
    print("\nTest: TemperatureScalingLayer")
    
    temp_layer = TemperatureScalingLayer(temperature=1.5)
    assert temp_layer.get_temperature() == 1.5, "Initial temperature mismatch"
    
    logits = torch.randn(4, 8)
    scaled = temp_layer(logits)
    
    # Check scaling
    expected = logits / 1.5
    assert torch.allclose(scaled, expected, atol=1e-5), "Scaling incorrect"
    
    print("  ✓ Temperature layer works correctly")


def test_compute_ece_perfect_calibration():
    """Test ECE computation with perfectly calibrated predictions."""
    print("\nTest: ECE with perfect calibration")
    
    # Perfect calibration: confidence = accuracy
    np.random.seed(42)
    n = 1000
    
    # Generate predictions with confidence matching accuracy
    confidences = np.random.uniform(0.5, 1.0, n)
    labels = np.random.randint(0, 8, n)
    
    # Make predictions correct according to confidence
    predictions = labels.copy()
    for i in range(n):
        if np.random.rand() > confidences[i]:
            predictions[i] = (labels[i] + 1) % 8  # Wrong prediction
    
    ece_results = compute_ece(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        n_bins=10
    )
    
    # ECE should be reasonably low for this construction
    assert ece_results["ece"] >= 0.0, "ECE should be non-negative"
    assert ece_results["mce"] >= ece_results["ece"], "MCE should be >= ECE"
    assert ece_results["n_bins"] == 10, "Number of bins mismatch"
    
    print(f"  ✓ ECE = {ece_results['ece']:.4f}, MCE = {ece_results['mce']:.4f}")


def test_compute_ece_overconfident():
    """Test ECE computation with overconfident predictions."""
    print("\nTest: ECE with overconfident model")
    
    n = 1000
    
    # Overconfident: high confidence but only 60% accurate
    confidences = np.random.uniform(0.8, 1.0, n)
    labels = np.random.randint(0, 8, n)
    predictions = labels.copy()
    
    # Make 40% wrong
    wrong_indices = np.random.choice(n, size=int(0.4 * n), replace=False)
    for idx in wrong_indices:
        predictions[idx] = (labels[idx] + 1) % 8
    
    ece_results = compute_ece(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        n_bins=10
    )
    
    # Should have high ECE (confidence >> accuracy)
    assert ece_results["ece"] > 0.1, f"Expected high ECE for overconfident model, got {ece_results['ece']}"
    
    print(f"  ✓ High ECE detected: {ece_results['ece']:.4f}")


def test_optimize_temperature():
    """Test temperature optimization."""
    print("\nTest: Temperature optimization")
    
    torch.manual_seed(42)
    
    # Create dummy model that outputs overconfident logits
    model = DummyModel()
    
    # Create dummy data
    n = 500
    X = torch.randn(n, 10)
    y = torch.randint(0, 8, (n,))
    
    dataset = TensorDataset(X, y, torch.zeros(n), torch.zeros(n))  # Match expected format
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Optimize temperature
    temp_layer = optimize_temperature(
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        max_iter=10,
    )
    
    T = temp_layer.get_temperature()
    assert T > 0, "Temperature must be positive"
    assert 0.5 < T < 5.0, f"Temperature {T} seems unreasonable"
    
    print(f"  ✓ Optimized temperature: T = {T:.4f}")


def test_evaluate_calibration():
    """Test calibration evaluation on dataset."""
    print("\nTest: Calibration evaluation")
    
    torch.manual_seed(42)
    
    model = DummyModel()
    temp_layer = TemperatureScalingLayer(temperature=1.3)
    
    # Create dummy data
    n = 200
    X = torch.randn(n, 10)
    y = torch.randint(0, 8, (n,))
    
    dataset = TensorDataset(X, y, torch.zeros(n), torch.zeros(n))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Evaluate without temperature
    ece_before, confs_before, preds_before, labels = evaluate_calibration(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        temp_layer=None,
        n_bins=10,
    )
    
    # Evaluate with temperature
    ece_after, confs_after, preds_after, _ = evaluate_calibration(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        temp_layer=temp_layer,
        n_bins=10,
    )
    
    # Predictions should not change (argmax invariant)
    assert np.array_equal(preds_before, preds_after), "Temperature scaling changed predictions!"
    
    # Confidences should change
    assert not np.allclose(confs_before, confs_after), "Temperature did not affect confidences"
    
    # Average confidence should decrease (T > 1)
    assert np.mean(confs_after) < np.mean(confs_before), "T > 1 should reduce confidence"
    
    print(f"  ✓ ECE before: {ece_before['ece']:.4f}, after: {ece_after['ece']:.4f}")
    print(f"  ✓ Avg conf before: {np.mean(confs_before):.4f}, after: {np.mean(confs_after):.4f}")


def test_full_calibration_pipeline():
    """Test complete calibration pipeline."""
    print("\nTest: Full calibration pipeline")
    
    torch.manual_seed(42)
    
    model = DummyModel()
    
    # Create dummy validation set
    n = 300
    X = torch.randn(n, 10)
    y = torch.randint(0, 8, (n,))
    
    dataset = TensorDataset(X, y, torch.zeros(n), torch.zeros(n))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Run calibration
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_layer = calibrate_model(
            model=model,
            val_loader=loader,
            device=torch.device("cpu"),
            save_dir=tmpdir,
            n_bins=10,
        )
        
        # Check outputs exist
        tmppath = Path(tmpdir)
        assert (tmppath / "calibration_report.json").exists(), "Report not saved"
        assert (tmppath / "temperature_layer.pt").exists(), "Temperature layer not saved"
        assert (tmppath / "reliability_before_calibration.png").exists(), "Before diagram not saved"
        assert (tmppath / "reliability_after_calibration.png").exists(), "After diagram not saved"
        
        print(f"  ✓ All outputs generated successfully")
        print(f"  ✓ Final temperature: T = {temp_layer.get_temperature():.4f}")


def run_all_tests():
    """Run all calibration tests."""
    print("=" * 70)
    print("CALIBRATION MODULE TESTS")
    print("=" * 70)
    
    try:
        test_temperature_scaling_layer()
        test_compute_ece_perfect_calibration()
        test_compute_ece_overconfident()
        test_optimize_temperature()
        test_evaluate_calibration()
        test_full_calibration_pipeline()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
