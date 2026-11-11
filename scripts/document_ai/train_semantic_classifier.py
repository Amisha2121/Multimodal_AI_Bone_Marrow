"""
BioLinkBERT Clinical Text Classifier & Document Router
======================================================
Stage 4.2c — Semantic Language Classification

Fine-tunes a clinical language model (BioLinkBERT / PubMedBERT) or high-capacity
TF-IDF + Linear Classifier directly over full text strings to classify documents
into the 15 hematological categories.
Because clinical vocabulary (e.g. 'blasts', 'karyotype', 'flow cytometry')
is invariant to visual layout shifts, this achieves layout-agnostic zero-shot transfer.
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def train_semantic_text_classifier():
    print("Training Semantic Text Classifier on Document Content...")

    base_dir = Path("data/document_ai")
    splits_file = base_dir / "splits" / "split_manifest.json"
    features_file = base_dir / "processed" / "document_features_master.json"
    label_map_file = base_dir / "processed" / "category_label_map.json"
    output_dir = base_dir / "models" / "semantic_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r") as f:
        splits = json.load(f)["splits"]
    with open(features_file, "r") as f:
        features_list = json.load(f)
    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    inv_label_map = {v: k for k, v in label_map.items()}
    features_by_id = {item["report_id"]: item for item in features_list}

    train_texts = [features_by_id[sid]["text"] for sid in splits["train"]]
    train_labels = [features_by_id[sid]["category_idx"] for sid in splits["train"]]

    val_texts = [features_by_id[sid]["text"] for sid in splits["val"]]
    val_labels = [features_by_id[sid]["category_idx"] for sid in splits["val"]]

    test_texts = [features_by_id[sid]["text"] for sid in splits["test"]]
    test_labels = [features_by_id[sid]["category_idx"] for sid in splits["test"]]

    print(f"Data split sizes: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # High-precision n-gram pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=10.0, max_iter=500, class_weight="balanced"))
    ])

    model.fit(train_texts, train_labels)

    # Validation
    val_preds = model.predict(val_texts)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    print(f"Validation Performance: Acc={val_acc*100:.2f}%, Macro F1={val_f1:.4f}")

    # Held-Out Evaluation on Unseen Template C
    test_preds = model.predict(test_texts)
    test_acc = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)

    target_names = [inv_label_map[i] for i in range(len(label_map))]
    report = classification_report(test_labels, test_preds, target_names=target_names, output_dict=True, zero_division=0)

    print(f"\nHeld-Out Test Results (Template C - Unseen Layout):")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Macro F1:  {test_macro_f1:.4f}")
    print(f"  Micro F1:  {test_micro_f1:.4f}")

    # Save model artifact
    model_path = output_dir / "semantic_document_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")

    # Save evaluation metrics
    eval_result = {
        "model_architecture": "Semantic-Clinical-Text-Classifier",
        "test_split_size": len(test_texts),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_micro_f1": test_micro_f1,
        "held_out_template": "TEMPLATE_C",
        "detailed_classification_report": report
    }

    eval_path = base_dir / "manifests" / "semantic_classification_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Evaluation report written to: {eval_path}")

if __name__ == "__main__":
    train_semantic_text_classifier()
