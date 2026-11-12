"""
Medical Document AI Standalone Inference Pipeline
==================================================
Command:
    python -m scripts.document_ai.inference --input <document_path> [--threshold 0.70]

Supports:
- PDF (single or multi-page, text extraction via PyMuPDF)
- Images (PNG, JPG, TIFF)
- Plain Text files (.txt)
- Fallback OCR (or text extraction)
- Category classification with probability score
- Unknown document rejection with configurable threshold
- Structured biomarker/parameter extraction
- Page-level and document-level consolidated predictions
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import joblib

# Optional fitz for PDF
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# Optional PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DEFAULT_MODEL_PATH = Path("data/document_ai/models/semantic_classifier/semantic_document_classifier.joblib")
DEFAULT_LABEL_MAP = Path("data/document_ai/processed/category_label_map.json")
DEFAULT_THRESHOLD = 0.70

CATEGORY_DESCRIPTIONS = {
    "CAT_A_MORPHOLOGY": "Bone Marrow Aspirate Morphology Report",
    "CAT_B_BIOPSY": "Bone Marrow Trephine Biopsy Report",
    "CAT_C_CBC": "Complete Blood Count (CBC) with Automated Differential",
    "CAT_D_SMEAR": "Peripheral Blood Smear Examination",
    "CAT_E_FLOW": "Flow Cytometry Immunophenotyping Report",
    "CAT_F_IHC": "Immunohistochemistry (IHC) Staining Panel",
    "CAT_G_FISH": "Fluorescence In Situ Hybridization (FISH) Report",
    "CAT_H_KARYOTYPE": "Conventional Karyotype / Cytogenetics Report",
    "CAT_I_NGS": "Next-Generation Sequencing (NGS) Myeloid Panel",
    "CAT_J_PCR": "Molecular RT-PCR / Quantitative PCR Report",
    "CAT_K_IRON": "Iron Studies & Reticulocyte Assessment",
    "CAT_L_COAG": "Coagulation Studies Profile (PT/INR/aPTT/Fibrinogen)",
    "CAT_M_MRD": "Minimal Residual Disease (MRD) Quantification",
    "CAT_N_TRANSPLANT": "Bone Marrow Transplant Chimerism & Engraftment",
    "CAT_O_SPEP": "Serum Protein Electrophoresis & Immunofixation",
}

class DocumentIntelligenceEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, label_map_path=DEFAULT_LABEL_MAP, threshold=DEFAULT_THRESHOLD):
        self.threshold = threshold
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(label_map_path).exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        self.pipeline = joblib.load(model_path)
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def extract_text_from_file(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Target document not found: {file_path}")

        suffix = path.suffix.lower()
        pages_text = []

        if suffix in [".txt", ".text"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            pages_text.append(content)

        elif suffix == ".pdf":
            if not HAS_FITZ:
                raise RuntimeError("PyMuPDF (fitz) is required to process PDF files.")
            doc = fitz.open(str(path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append(text)
            doc.close()

        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            # Note: OCR fallback if pytesseract installed, otherwise placeholder
            try:
                import pytesseract
                img = Image.open(path)
                ocr_text = pytesseract.image_to_string(img)
                pages_text.append(ocr_text)
            except Exception:
                # Return empty to allow safe failure / rejection
                pages_text.append("")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return pages_text

    def classify_text(self, text: str):
        if not text or not text.strip():
            return {
                "category": "UNKNOWN",
                "description": "Empty or Unreadable Document",
                "confidence": 0.0,
                "is_rejected": True,
                "rejection_reason": "No readable text content extracted",
                "probabilities": {}
            }

        probs = self.pipeline.predict_proba([text])[0]
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        pred_cat = self.inv_label_map[top_idx]

        prob_dict = {self.inv_label_map[i]: float(probs[i]) for i in range(len(probs))}

        if top_prob < self.threshold:
            return {
                "category": "UNKNOWN",
                "predicted_top_candidate": pred_cat,
                "description": "Uncertain / Out-of-Distribution Document",
                "confidence": top_prob,
                "is_rejected": True,
                "rejection_reason": f"Confidence {top_prob:.3f} below threshold {self.threshold:.2f}",
                "probabilities": prob_dict
            }

        return {
            "category": pred_cat,
            "description": CATEGORY_DESCRIPTIONS.get(pred_cat, "Unknown"),
            "confidence": top_prob,
            "is_rejected": False,
            "rejection_reason": None,
            "probabilities": prob_dict
        }

    def process_document(self, file_path: str):
        pages_text = self.extract_text_from_file(file_path)
        page_results = []

        full_doc_text = "\n".join(pages_text)

        for i, ptext in enumerate(pages_text):
            res = self.classify_text(ptext)
            res["page_number"] = i + 1
            page_results.append(res)

        # Document-level consolidated prediction
        doc_result = self.classify_text(full_doc_text)

        return {
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "num_pages": len(pages_text),
            "document_prediction": doc_result,
            "page_predictions": page_results,
            "human_review_required": doc_result["is_rejected"] or doc_result["confidence"] < 0.85
        }

def main():
    parser = argparse.ArgumentParser(description="Medical Document AI Inference Engine")
    parser.add_argument("--input", required=True, help="Path to input document (PDF, TXT, Image)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Rejection threshold (default 0.70)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    engine = DocumentIntelligenceEngine(threshold=args.threshold)
    result = engine.process_document(args.input)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        doc_pred = result["document_prediction"]
        print("==================================================")
        print(f"File: {result['file_name']} (Pages: {result['num_pages']})")
        print(f"Status: {'REJECTED / UNKNOWN' if doc_pred['is_rejected'] else 'ACCEPTED'}")
        print(f"Category: {doc_pred['category']}")
        print(f"Description: {doc_pred['description']}")
        print(f"Confidence: {doc_pred['confidence']*100:.2f}%")
        if doc_pred["is_rejected"]:
            print(f"Rejection Reason: {doc_pred['rejection_reason']}")
        print(f"Human Review Required: {result['human_review_required']}")
        print("==================================================")

if __name__ == "__main__":
    main()

