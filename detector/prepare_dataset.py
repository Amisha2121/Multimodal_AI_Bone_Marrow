"""
detector/prepare_dataset.py
─────────────────────────────────────────────────────────────────────────────
Convert ground-truth cell annotations to YOLO format for object detection.

CRITICAL REQUIREMENTS:
- Use ONLY real ground-truth annotations (cell_metadata.csv)
- Preserve existing patient-level splits (train/val/test)
- NO data leakage: zero patient overlap across splits
- Single detection class: "cell" (class_id = 0)
- DO NOT create morphology classes for detector
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class YOLODatasetPreparer:
    """Prepares YOLO detection dataset from ground-truth cell annotations."""
    
    def __init__(self, 
                 metadata_csv: str = "data/processed/cell_metadata.csv",
                 output_dir: str = "detector/yolo_dataset",
                 image_format: str = "png"):
        """
        Args:
            metadata_csv: Path to cell_metadata.csv with ground-truth boxes
            output_dir: Output directory for YOLO dataset
            image_format: Image format (png/jpg)
        """
        self.metadata_csv = metadata_csv
        self.output_dir = Path(output_dir)
        self.image_format = image_format
        
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        # Single class detection: cell = 0
        self.class_mapping = {"cell": 0}
        
    def load_metadata(self):
        """Load and validate cell metadata."""
        print(f"Loading metadata from {self.metadata_csv}...")
        df = pd.read_csv(self.metadata_csv)
        
        print(f"\nTotal annotations: {len(df)}")
        print(f"Extraction status distribution:")
        print(df['extraction_status'].value_counts())
        
        # Keep only VALID consensus cells
        df_valid = df[df['extraction_status'] == 'valid'].copy()
        print(f"\nValid consensus cells: {len(df_valid)}")
        
        # Verify required columns
        required_cols = ['cell_id', 'patient_id', 'split', 'x1', 'y1', 'x2', 'y2', 
                        'source_series_uid', 'crop_path']
        missing = set(required_cols) - set(df_valid.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove NaN coordinates
        df_valid = df_valid.dropna(subset=['x1', 'y1', 'x2', 'y2'])
        print(f"After removing NaN coordinates: {len(df_valid)}")
        
        return df_valid
    
    def verify_patient_splits(self, df):
        """Verify zero patient overlap across splits."""
        print("\n" + "="*80)
        print("PATIENT SPLIT VERIFICATION")
        print("="*80)
        
        train_patients = set(df[df['split'] == 'train']['patient_id'].unique())
        val_patients = set(df[df['split'] == 'val']['patient_id'].unique())
        test_patients = set(df[df['split'] == 'test']['patient_id'].unique())
        
        print(f"\nTrain patients: {len(train_patients)}")
        print(f"Validation patients: {len(val_patients)}")
        print(f"Test patients: {len(test_patients)}")
        
        # Check for overlap
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        print(f"\nTrain-Val overlap: {len(train_val_overlap)}")
        print(f"Train-Test overlap: {len(train_test_overlap)}")
        print(f"Val-Test overlap: {len(val_test_overlap)}")
        
        assert len(train_val_overlap) == 0, f"Train-Val overlap: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"Train-Test overlap: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Val-Test overlap: {val_test_overlap}"
        
        print("\n✓ VERIFIED: Zero patient overlap across splits")
        
        # Count annotations per split
        print(f"\nAnnotations per split:")
        for split in ['train', 'val', 'test']:
            count = len(df[df['split'] == split])
            patients = len(df[df['split'] == split]['patient_id'].unique())
            print(f"  {split:5s}: {count:5d} annotations, {patients:3d} patients")
        
        return train_patients, val_patients, test_patients
    
    def get_image_info_from_crop_path(self, crop_path):
        """
        Extract source image information from crop path.
        
        Crop path format: data/processed/cell_crops/{split}/{patient_id}/{cell_id}.png
        We need to find the corresponding source WSI image.
        
        For now, we'll group annotations by patient_id and source_series_uid
        to create synthetic "image" files for YOLO training.
        """
        # Parse crop path
        parts = Path(crop_path).parts
        if len(parts) < 4:
            return None
        
        split = parts[3]  # train/val/test
        patient_id = parts[4]
        
        return {
            'split': split,
            'patient_id': patient_id,
        }
    
    def group_annotations_by_image(self, df):
        """
        Group annotations by source image.
        
        Since we don't have direct access to source WSI images, we'll group
        annotations by patient_id + source_series_uid to create virtual images.
        
        Each group will be treated as one "image" containing multiple cell bounding boxes.
        """
        print("\n" + "="*80)
        print("GROUPING ANNOTATIONS BY SOURCE IMAGE")
        print("="*80)
        
        # Group by patient and source series
        df['image_id'] = df['patient_id'] + '_' + df['source_series_uid'].astype(str)
        
        image_groups = df.groupby(['image_id', 'split']).agg({
            'cell_id': 'count',
            'patient_id': 'first',
            'source_series_uid': 'first',
        }).reset_index()
        
        image_groups.rename(columns={'cell_id': 'num_annotations'}, inplace=True)
        
        print(f"\nTotal unique images: {len(image_groups)}")
        print(f"\nImages per split:")
        for split in ['train', 'val', 'test']:
            count = len(image_groups[image_groups['split'] == split])
            print(f"  {split:5s}: {count:4d} images")
        
        return df
    
    def convert_to_yolo_format(self, x1, y1, x2, y2, img_width, img_height):
        """
        Convert bounding box to YOLO format.
        
        Input: Absolute coordinates (x1, y1, x2, y2)
        Output: YOLO format (class_id, x_center, y_center, width, height) normalized [0, 1]
        
        Args:
            x1, y1, x2, y2: Absolute pixel coordinates
            img_width, img_height: Image dimensions
        
        Returns:
            Tuple of (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
        """
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Clip to valid range
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        # Single class: cell = 0
        class_id = 0
        
        return (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
    
    def estimate_image_dimensions(self, df_image):
        """
        Estimate reasonable image dimensions from bounding boxes.
        
        Since original WSI coordinates can be huge (causing memory errors),
        we'll scale down to a reasonable size while preserving relative positions.
        """
        max_x = df_image[['x1', 'x2']].max().max()
        max_y = df_image[['y1', 'y2']].max().max()
        
        # Target maximum dimension (YOLO can handle up to ~10000px)
        MAX_DIM = 4096
        
        # Calculate scaling factor if needed
        scale_factor = 1.0
        if max_x > MAX_DIM or max_y > MAX_DIM:
            scale_factor = MAX_DIM / max(max_x, max_y)
        
        # Scale dimensions
        img_width = int(max_x * scale_factor) + 100
        img_height = int(max_y * scale_factor) + 100
        
        # Store scaling factor for coordinate conversion
        return img_width, img_height, scale_factor
    
    def create_yolo_labels(self, df):
        """Create YOLO label files and composite images for each image group."""
        print("\n" + "="*80)
        print("CREATING YOLO IMAGES AND LABELS")
        print("="*80)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Group by image_id
        image_ids = df['image_id'].unique()
        print(f"\nProcessing {len(image_ids)} images...")
        print("Creating composite images from cell crops...")
        
        stats = defaultdict(lambda: {'images': 0, 'annotations': 0})
        
        for image_id in tqdm(image_ids, desc="Creating images and labels"):
            df_image = df[df['image_id'] == image_id]
            split = df_image['split'].iloc[0]
            
            # Estimate image dimensions with scaling
            img_width, img_height, scale_factor = self.estimate_image_dimensions(df_image)
            
            print(f"  {image_id}: {img_width}×{img_height}px (scale: {scale_factor:.3f})")
            
            # Create blank canvas
            canvas = Image.new('RGB', (img_width, img_height), color=(240, 240, 240))
            
            # Place cell crops on canvas at their scaled positions
            for _, row in df_image.iterrows():
                crop_path = Path(row['crop_path'])
                if crop_path.exists():
                    try:
                        # Load cell crop
                        cell_img = Image.open(crop_path).convert('RGB')
                        
                        # Scale coordinates
                        x1 = int(row['x1'] * scale_factor)
                        y1 = int(row['y1'] * scale_factor)
                        x2 = int(row['x2'] * scale_factor)
                        y2 = int(row['y2'] * scale_factor)
                        
                        # Calculate box dimensions
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        if box_width > 0 and box_height > 0:
                            # Resize cell crop to fit bounding box
                            cell_resized = cell_img.resize((box_width, box_height), Image.BICUBIC)
                            
                            # Ensure coordinates are within canvas bounds
                            if x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height:
                                canvas.paste(cell_resized, (x1, y1))
                    except Exception as e:
                        # If crop doesn't exist or can't be loaded, skip it
                        pass
            
            # Save composite image
            image_file = self.output_dir / split / 'images' / f"{image_id}.{self.image_format}"
            canvas.save(image_file, quality=95)
            
            # Create label file with scaled coordinates
            label_file = self.output_dir / split / 'labels' / f"{image_id}.txt"
            
            with open(label_file, 'w') as f:
                for _, row in df_image.iterrows():
                    # Scale original coordinates
                    x1 = row['x1'] * scale_factor
                    y1 = row['y1'] * scale_factor
                    x2 = row['x2'] * scale_factor
                    y2 = row['y2'] * scale_factor
                    
                    # Convert to YOLO format
                    class_id, x_c, y_c, w, h = self.convert_to_yolo_format(
                        x1, y1, x2, y2, img_width, img_height
                    )
                    
                    # Write to label file
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            stats[split]['images'] += 1
            stats[split]['annotations'] += len(df_image)
        
        # Print statistics
        print(f"\nImage and label file statistics:")
        for split in ['train', 'val', 'test']:
            print(f"  {split:5s}: {stats[split]['images']:4d} images, "
                  f"{stats[split]['annotations']:5d} annotations")
        
        return stats
    
    def create_yolo_config(self, stats):
        """Create YOLO dataset configuration file."""
        config_path = self.output_dir / "dataset.yaml"
        
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            
            'nc': 1,  # Single class
            'names': ['cell'],
            
            # Statistics
            'train_images': stats['train']['images'],
            'val_images': stats['val']['images'],
            'test_images': stats['test']['images'],
            'train_annotations': stats['train']['annotations'],
            'val_annotations': stats['val']['annotations'],
            'test_annotations': stats['test']['annotations'],
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nCreated YOLO config: {config_path}")
        return config_path
    
    def create_metadata_summary(self, df, train_patients, val_patients, test_patients):
        """Create a summary of dataset metadata."""
        summary_path = self.output_dir / "metadata_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("YOLO CELL DETECTION DATASET - METADATA SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("SOURCE DATA:\n")
            f.write(f"  Metadata CSV: {self.metadata_csv}\n")
            f.write(f"  Total valid annotations: {len(df)}\n\n")
            
            f.write("PATIENT SPLITS:\n")
            f.write(f"  Train patients: {len(train_patients)}\n")
            f.write(f"  Val patients: {len(val_patients)}\n")
            f.write(f"  Test patients: {len(test_patients)}\n")
            f.write(f"  VERIFIED: Zero patient overlap\n\n")
            
            f.write("ANNOTATION SPLITS:\n")
            for split in ['train', 'val', 'test']:
                count = len(df[df['split'] == split])
                f.write(f"  {split}: {count} annotations\n")
            f.write("\n")
            
            f.write("DETECTION TASK:\n")
            f.write(f"  Classes: 1 (cell)\n")
            f.write(f"  Format: YOLO (normalized bounding boxes)\n")
            f.write(f"  Class ID mapping: {self.class_mapping}\n\n")
            
            f.write("COORDINATE CONVENTION:\n")
            f.write(f"  Input format: (x1, y1, x2, y2) absolute pixels\n")
            f.write(f"  YOLO format: (class_id, x_center, y_center, width, height) normalized [0,1]\n")
            f.write(f"  Coordinate system: Original WSI coordinates from DICOM annotations\n\n")
            
            f.write("NOTES:\n")
            f.write(f"  - Ground-truth annotations from consensus expert labels\n")
            f.write(f"  - Patient-level split ensures no data leakage\n")
            f.write(f"  - Single detection class: do NOT classify morphology\n")
            f.write(f"  - Morphology classification is a separate downstream task\n")
        
        print(f"Created metadata summary: {summary_path}")
    
    def prepare(self):
        """Main preparation pipeline."""
        print("\n" + "="*80)
        print("PREPARING YOLO CELL DETECTION DATASET")
        print("="*80)
        
        # Load metadata
        df = self.load_metadata()
        
        # Verify patient splits
        train_patients, val_patients, test_patients = self.verify_patient_splits(df)
        
        # Group annotations by image
        df = self.group_annotations_by_image(df)
        
        # Create YOLO images and labels
        stats = self.create_yolo_labels(df)
        
        # Create YOLO config
        config_path = self.create_yolo_config(stats)
        
        # Create metadata summary
        self.create_metadata_summary(df, train_patients, val_patients, test_patients)
        
        print("\n" + "="*80)
        print("DATASET PREPARATION COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"YOLO config: {config_path}")
        print(f"\nNext step: Run train_detector.py to train the YOLO model")


def main():
    """Main entry point."""
    preparer = YOLODatasetPreparer(
        metadata_csv="data/processed/cell_metadata.csv",
        output_dir="detector/yolo_dataset",
    )
    
    preparer.prepare()


if __name__ == "__main__":
    main()
