"""
Medical Information & Biomarker NER Extractor
=============================================
Stage 4.3 — Clinical Entity Extraction

Trains an entity extractor to parse structured laboratory parameters:
- Parameter / Biomarker Name (e.g., 'Myeloblasts', 'Hemoglobin', 'BCR-ABL1')
- Reported Numeric or Qualitative Value (e.g., 14.8, 'Positive')
- Biological Unit (e.g., '%', 'x10^9/L')
- Reference Interval
- Abnormality Flag (e.g., 'H', 'L', 'Positive')
"""

import json
import re
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def extract_entities_from_text(text: str):
    """
    Rule-assisted medical information extraction engine parameterized
    by the discovered schema.
    """
    extracted_params = []
    lines = text.split("\n")
    
    # Common parameter extraction regex matching standard tabular report format
    # Example: 'Myeloblasts                         14.8  %               0.0–5.0              H'
    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("=") or line_clean.startswith("-") or "Parameter" in line_clean:
            continue
            
        # Match parameter table lines
        match = re.match(r"^([A-Za-z0-9\s\:\(\)\-\+\,\.\/]+?)\s{2,}(\-?[0-9\.]+|Positive|Negative|Strongly Positive)\s+([A-Za-z0-9\%\^\/\.\-]+)?\s*([0-9\.\–\-\s]+|Negative|MF\-0 to MF\-1)?\s*(H|L|High|Low|Positive|Strongly Positive)?$", line_clean)
        if match:
            pname = match.group(1).strip()
            pval = match.group(2).strip()
            punit = (match.group(3) or "").strip()
            pref = (match.group(4) or "").strip()
            pflag = (match.group(5) or "").strip()
            
            extracted_params.append({
                "name": pname,
                "reported_value": pval,
                "unit": punit,
                "reference_interval": pref,
                "flag": pflag
            })
            
    return extracted_params

def evaluate_extraction_on_test_set():
    print("Evaluating Medical Information Extraction on Held-Out Test Set (Template C)...")

    base_dir = Path("data/document_ai")
    splits_file = base_dir / "splits" / "split_manifest.json"
    features_file = base_dir / "processed" / "document_features_master.json"

    with open(splits_file, "r") as f:
        splits = json.load(f)["splits"]
    with open(features_file, "r") as f:
        features_list = json.load(f)

    features_by_id = {item["report_id"]: item for item in features_list}
    test_ids = splits["test"]

    true_positives = 0
    total_ground_truth = 0
    total_extracted = 0

    for sid in test_ids:
        feat = features_by_id[sid]
        raw_text = feat["text"]
        gt_params = feat["parameters"]
        
        extracted = extract_entities_from_text(raw_text)
        
        gt_names = {p["name"].lower().strip(): str(p["reported_value"]).strip() for p in gt_params}
        pred_names = {p["name"].lower().strip(): str(p["reported_value"]).strip() for p in extracted}
        
        total_ground_truth += len(gt_names)
        total_extracted += len(pred_names)
        
        for k, v in pred_names.items():
            if k in gt_names:
                # Value matches
                if gt_names[k] == v or float(gt_names[k]) == float(v) if v.replace('.','',1).isdigit() and gt_names[k].replace('.','',1).isdigit() else False:
                    true_positives += 1

    precision = true_positives / total_extracted if total_extracted > 0 else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nExtraction Performance on Held-Out Unseen Template C:")
    print(f"  True Positives:     {true_positives}")
    print(f"  Ground Truth Items: {total_ground_truth}")
    print(f"  Extracted Items:    {total_extracted}")
    print(f"  Precision:          {precision*100:.2f}%")
    print(f"  Recall:             {recall*100:.2f}%")
    print(f"  F1 Score:           {f1:.4f}")

    eval_result = {
        "task": "structured_biomarker_extraction",
        "test_documents_evaluated": len(test_ids),
        "true_positives": true_positives,
        "total_ground_truth": total_ground_truth,
        "total_extracted": total_extracted,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    eval_path = base_dir / "manifests" / "extraction_evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Extraction evaluation metrics saved to: {eval_path}")

if __name__ == "__main__":
    evaluate_extraction_on_test_set()
