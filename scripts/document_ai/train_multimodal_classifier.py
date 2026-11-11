"""
Multimodal Document Classifier: Vision + Text Hybrid (Late Fusion)
===================================================================
Stage 4.2b — Multimodal Document Classification

Combines:
1. Visual CNN/ViT features from rendered page image (captures page architecture)
2. Semantic embedding features from the extracted document text (captures clinical terms)
Yields robust generalization on held-out visual layouts (Template C).
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MultimodalDocDataset(Dataset):
    def __init__(self, sample_ids, features_by_id, transform=None):
        self.sample_ids = sample_ids
        self.features_by_id = features_by_id
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        feat = self.features_by_id[sid]
        
        # 1. Vision
        image = Image.open(feat["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # 2. Text Bag-of-Characters / N-grams feature vector (hash-based fast projection)
        text = (feat.get("text") or "")[:2000].lower()
        # Simple high-signal keyword vector based on known diagnostic tokens
        KEYWORDS = [
            "aspirate", "myeloblast", "promyelocyte", "cellularity", "trephine",
            "biopsy", "reticulin", "fibrosis", "haemoglobin", "wbc", "platelets",
            "neutrophil", "lymphocyte", "smear", "poikilocytosis", "flow cytometry",
            "cd19", "cd34", "cd3", "cd4", "cd8", "immunohistochemistry", "ki-67",
            "mpo", "fish", "bcr-abl1", "pml-rara", "karyotype", "iscn", "mitoses",
            "ngs", "vaf", "dnmt3a", "flt3", "npm1", "rt-pcr", "molecular", "iron",
            "ferritin", "tibc", "reticulocyte", "coagulation", "inr", "pt", "aptt",
            "fibrinogen", "d-dimer", "mrd", "residual disease", "transplant",
            "chimerism", "engraftment", "spep", "paraprotein", "m-protein", "free light"
        ]
        text_vec = torch.zeros(56, dtype=torch.float32)
        for i, kw in enumerate(KEYWORDS):
            if kw in text:
                text_vec[i] = 1.0

        label = feat["category_idx"]
        return image, text_vec, torch.tensor(label, dtype=torch.long), sid

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=15, num_keywords=56):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.vision_backbone = efficientnet_b0(weights=weights)
        in_features = self.vision_backbone.classifier[1].in_features
        self.vision_backbone.classifier = nn.Identity()  # outputs 1280 dim feature

        self.text_fc = nn.Sequential(
            nn.Linear(num_keywords, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text_vec):
        img_feats = self.vision_backbone(image)
        txt_feats = self.text_fc(text_vec)
        fused = torch.cat([img_feats, txt_feats], dim=1)
        return self.classifier(fused)

def train_multimodal_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multimodal Classifier on device: {device}")

    base_dir = Path("data/document_ai")
    splits_file = base_dir / "splits" / "split_manifest.json"
    features_file = base_dir / "processed" / "document_features_master.json"
    label_map_file = base_dir / "processed" / "category_label_map.json"
    output_dir = base_dir / "models" / "multimodal_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r") as f:
        splits = json.load(f)["splits"]
    with open(features_file, "r") as f:
        features_list = json.load(f)
    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    inv_label_map = {v: k for k, v in label_map.items()}
    features_by_id = {item["report_id"]: item for item in features_list}

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultimodalDocDataset(splits["train"], features_by_id, eval_transform)
    val_dataset = MultimodalDocDataset(splits["val"], features_by_id, eval_transform)
    test_dataset = MultimodalDocDataset(splits["test"], features_by_id, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = MultimodalClassifier(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_f1 = 0.0
    epochs = 8

    print("\nTraining Multimodal Fusion Architecture...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, text_vec, labels, _ in train_loader:
            images, text_vec, labels = images.to(device), text_vec.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, text_vec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, text_vec, labels, _ in val_loader:
                images, text_vec = images.to(device), text_vec.to(device)
                outputs = model(images, text_vec)
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
            }, output_dir / "best_multimodal_classifier.pt")

    # Evaluation on Held-Out Test Set (Template C)
    print("\nEvaluating Multimodal Model on Held-Out Template C Test Set...")
    checkpoint = torch.load(output_dir / "best_multimodal_classifier.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, text_vec, labels, _ in test_loader:
            images, text_vec = images.to(device), text_vec.to(device)
            outputs = model(images, text_vec)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_targets, test_preds, average="micro", zero_division=0)

    target_names = [inv_label_map[i] for i in range(len(label_map))]
    report = classification_report(test_targets, test_preds, target_names=target_names, output_dict=True, zero_division=0)

    print(f"\nMultimodal Held-Out Test Results (Template C):")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Macro F1:  {test_macro_f1:.4f}")
    print(f"  Micro F1:  {test_micro_f1:.4f}")

    eval_result = {
        "model_architecture": "MultimodalVisionText-DocumentClassifier",
        "test_split_size": len(test_dataset),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_micro_f1": test_micro_f1,
        "held_out_template": "TEMPLATE_C",
        "detailed_classification_report": report
    }

    eval_path = base_dir / "manifests" / "multimodal_classification_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Evaluation report written to {eval_path}")

if __name__ == "__main__":
    train_multimodal_classifier()
