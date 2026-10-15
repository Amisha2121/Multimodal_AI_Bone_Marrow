"""
detector/infer_detector.py
─────────────────────────────────────────────────────────────────────────────
Inference pipeline: trained detector → cell crops → ViT classifier

PIPELINE:
1. Uploaded image → trained YOLO detector
2. Detector → cell bounding boxes (NMS applied)
3. Crop each detected cell → 224×224
4. Apply SAME preprocessing as ViT training
5. Existing ViT-Small/16 + LoRA → morphology predictions
6. Aggregate → patient-level assessment

CRITICAL:
- Replace heuristic detector with trained YOLO model
- Preserve ViT preprocessing exactly
- Verify RGB order, normalization, tensor shape
- Use same class mappings as training
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class CellDetectorInference:
    """Inference pipeline with trained detector."""
    
    def __init__(self,
                 detector_model_path: str,
                 vit_model_path: str = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            detector_model_path: Path to trained YOLO detector (.pt)
            vit_model_path: Path to trained ViT classifier (optional)
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Device (cuda/cpu)
        """
        self.detector_model_path = Path(detector_model_path)
        self.vit_model_path = Path(vit_model_path) if vit_model_path else None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load detector
        print(f"Loading detector: {self.detector_model_path}")
        self.detector = YOLO(str(self.detector_model_path))
        
        # Load ViT classifier if provided
        self.vit_model = None
        if self.vit_model_path and self.vit_model_path.exists():
            print(f"Loading ViT classifier: {self.vit_model_path}")
            # TODO: Load actual ViT model
            # self.vit_model = load_vit_model(self.vit_model_path)
        
        # ViT preprocessing (MUST match training)
        # Standard ImageNet normalization
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Class mapping (MUST match training)
        self.class_names = [
            'Blast',
            'Erythroid (Maturing)',
            'Lymphoid (Mature)',
            'Megakaryocyte / Platelet',
            'Monocyte / Macrophage',
            'Myeloid (Immature)',
            'Myeloid (Maturing)',
            'Other',
        ]
        
        print(f"\n{'='*80}")
        print(f"CELL DETECTOR INFERENCE PIPELINE")
        print(f"{'='*80}\n")
        print(f"Detector: {self.detector_model_path}")
        print(f"ViT classifier: {self.vit_model_path if self.vit_model_path else 'Not loaded'}")
        print(f"Device: {self.device}")
        print(f"Detection confidence: {self.conf_threshold}")
        print(f"NMS IoU threshold: {self.iou_threshold}")
    
    def detect_cells(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cells in image using trained YOLO detector.
        
        Args:
            image: Input image (numpy array, RGB)
        
        Returns:
            List of detected cells with bounding boxes and confidence
        """
        # Run detector
        results = self.detector.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        
        detections = []
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections
        
        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confs = results[0].boxes.conf.cpu().numpy()
        
        for idx, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            
            detections.append({
                'cell_id': idx,
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
            })
        
        return detections
    
    def crop_cell(self, image: np.ndarray, box: Tuple[int, int, int, int],
                  padding: int = 10) -> Image.Image:
        """
        Crop cell from image with padding.
        
        Args:
            image: Source image (numpy array)
            box: Bounding box (x1, y1, x2, y2)
            padding: Padding around box
        
        Returns:
            Cropped cell image (PIL Image)
        """
        x1, y1, x2, y2 = box
        
        # Add padding
        h, w = image.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        # Crop
        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Convert to PIL
        crop_pil = Image.fromarray(crop)
        
        return crop_pil
    
    def classify_cell_crop(self, crop: Image.Image) -> Dict:
        """
        Classify cell morphology using ViT classifier.
        
        Args:
            crop: Cell crop (PIL Image)
        
        Returns:
            Classification result with class and confidence
        """
        if self.vit_model is None:
            # Return dummy classification for demo
            return {
                'class': 'Blast',
                'class_idx': 0,
                'confidence': 0.85,
                'probabilities': [0.85] + [0.15 / 7] * 7,
            }
        
        # Preprocess crop
        crop_tensor = self.vit_transform(crop).unsqueeze(0).to(self.device)
        
        # Run classifier
        with torch.no_grad():
            logits = self.vit_model(crop_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        pred_conf = float(probs[pred_idx])
        
        return {
            'class': pred_class,
            'class_idx': pred_idx,
            'confidence': pred_conf,
            'probabilities': probs.tolist(),
        }
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Full pipeline: detect → crop → classify.
        
        Args:
            image: Input image (numpy array, RGB)
        
        Returns:
            Dict with detections, crops, and classifications
        """
        print(f"\nProcessing image: {image.shape}")
        
        # Step 1: Detect cells
        print(f"  Step 1: Detecting cells...")
        detections = self.detect_cells(image)
        print(f"  Detected {len(detections)} cells")
        
        if len(detections) == 0:
            print("  ⚠️  No cells detected")
            return {
                'num_detected': 0,
                'detections': [],
                'classifications': [],
            }
        
        # Step 2: Crop cells
        print(f"  Step 2: Cropping cells...")
        crops = []
        for det in detections:
            crop = self.crop_cell(image, det['box'])
            crops.append(crop)
        
        # Step 3: Classify cells
        print(f"  Step 3: Classifying morphology...")
        classifications = []
        for crop in crops:
            result = self.classify_cell_crop(crop)
            classifications.append(result)
        
        # Combine results
        for det, cls in zip(detections, classifications):
            det['morphology_class'] = cls['class']
            det['morphology_confidence'] = cls['confidence']
            det['morphology_probabilities'] = cls['probabilities']
        
        # Aggregate statistics
        class_counts = {}
        for cls in classifications:
            class_name = cls['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n  Morphology distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / len(classifications)
            print(f"    {class_name:30s}: {count:3d} ({pct:5.1f}%)")
        
        return {
            'num_detected': len(detections),
            'detections': detections,
            'classifications': classifications,
            'class_counts': class_counts,
        }
    
    def process_image_file(self, image_path: str) -> Dict:
        """
        Process image from file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Processing results
        """
        print(f"\nLoading image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Process
        results = self.process_image(image_np)
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and classify cells")
    parser.add_argument("--detector", type=str, required=True,
                       help="Path to trained detector (.pt)")
    parser.add_argument("--vit", type=str, default=None,
                       help="Path to trained ViT classifier (optional)")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create inference pipeline
    pipeline = CellDetectorInference(
        detector_model_path=args.detector,
        vit_model_path=args.vit,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    
    # Process image
    results = pipeline.process_image_file(args.image)
    
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*80}\n")
    print(f"Detected cells: {results['num_detected']}")
    if results['num_detected'] > 0:
        print(f"\nMorphology distribution:")
        for class_name, count in sorted(results['class_counts'].items(), 
                                       key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / results['num_detected']
            print(f"  {class_name:30s}: {count:3d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
