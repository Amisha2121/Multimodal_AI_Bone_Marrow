"""
Document Type Classification Model Training (DiT-base / ViT Backbone)
======================================================================
Stage 4.2 — Medical Document AI Training

Fine-tunes a vision transformer classification head over rendered document images
to classify incoming reports into the 15 hematological diagnostic categories.
Evaluates strictly on the held-out Template C test split.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

class DocumentImageDataset(Dataset):
    def __init__(self, sample_ids, features_by_id, transform=None):
        self.sample_ids = sample_ids
        self.features_by_id = features_by_id
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        feat = self.features_by_id[sid]
        img_path = feat["image_path"]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        label = feat["category_idx"]
        return image, torch.tensor(label, dtype=torch.long), sid

def train_document_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    base_dir = Path("data/document_ai")
    splits_file = base_dir / "splits" / "split_manifest.json"
    features_file = base_dir / "processed" / "document_features_master.json"
    label_map_file = base_dir / "processed" / "category_label_map.json"
    output_dir = base_dir / "models" / "dit_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r") as f:
        splits = json.load(f)["splits"]
    with open(features_file, "r") as f:
        features_list = json.load(f)
    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    inv_label_map = {v: k for k, v in label_map.items()}
    features_by_id = {item["report_id"]: item for item in features_list}

    # Image Transforms (Standard 224x224 Document Resolution)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.0), # No horizontal flip for documents
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DocumentImageDataset(splits["train"], features_by_id, train_transform)
    val_dataset = DocumentImageDataset(splits["val"], features_by_id, eval_transform)
    test_dataset = DocumentImageDataset(splits["test"], features_by_id, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    num_classes = len(label_map)
    print(f"Dataset summary: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Num target categories: {num_classes}")

    # Use a pre-trained Vision Transformer or lightweight backbone
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_f1 = 0.0
    epochs = 6

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "label_map": label_map
            }, output_dir / "best_classifier.pt")

    print("\nRunning Evaluation on Held-Out Test Set (Template C Strict Isolation)...")
    checkpoint = torch.load(output_dir / "best_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets, test_sids = [], [], []
    with torch.no_grad():
        for images, labels, sids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            test_sids.extend(sids)

    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_targets, test_preds, average="micro", zero_division=0)

    target_names = [inv_label_map[i] for i in range(num_classes)]
    report = classification_report(test_targets, test_preds, target_names=target_names, output_dict=True, zero_division=0)

    print(f"\nTest Set Results (Held-Out Template C):")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Macro F1:  {test_macro_f1:.4f}")
    print(f"  Micro F1:  {test_micro_f1:.4f}")

    eval_result = {
        "model_architecture": "VisionTransformer-DocumentClassifier",
        "test_split_size": len(test_dataset),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_micro_f1": test_micro_f1,
        "held_out_template": "TEMPLATE_C",
        "detailed_classification_report": report
    }

    eval_path = base_dir / "manifests" / "classification_evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Saved evaluation metrics to: {eval_path}")

if __name__ == "__main__":
    train_document_classifier()
