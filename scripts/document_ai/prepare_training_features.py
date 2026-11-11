"""
Feature Preparation & Preprocessing Pipeline
=============================================
Stage 4.1 — Medical Document AI Training

Renders document pages from PDFs into images for vision models (DiT/LayoutLMv3),
extracts token sequences, word-level bounding boxes, and BIO entity tags.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

def process_corpus_features(base_dir: Path):
    corpus_dir = base_dir / "synthetic_corpus"
    annot_dir = base_dir / "annotations"
    processed_dir = base_dir / "processed"
    images_dir = processed_dir / "images"

    images_dir.mkdir(parents=True, exist_ok=True)

    annot_files = sorted(list(annot_dir.glob("*_annotation.json")))
    print(f"Preparing features for {len(annot_files)} documents...")

    category_mapping = {}
    features = []

    for fpath in annot_files:
        with open(fpath, "r", encoding="utf-8") as f:
            ann = json.load(f)

        rid = ann["report_id"]
        cat = ann["category"]
        if cat not in category_mapping:
            category_mapping[cat] = len(category_mapping)

        cat_idx = category_mapping[cat]

        # 1. Render First Page to PNG
        pdf_path = corpus_dir / f"{rid}.pdf"
        img_out_path = images_dir / f"{rid}_page1.png"

        width, height = 595, 842  # standard default A4
        words_with_boxes = []

        if pdf_path.exists():
            doc = fitz.open(str(pdf_path))
            if len(doc) > 0:
                page = doc[0]
                rect = page.rect
                width, height = int(rect.width), int(rect.height)
                
                # Render at 150 DPI for visual transformer input
                pix = page.get_pixmap(dpi=150)
                pix.save(str(img_out_path))

                # Extract words and normalized bounding boxes (0-1000 range)
                page_words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
                for w in page_words:
                    x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                    # Normalize to 0-1000
                    norm_box = [
                        int(min(max(0, x0 / width * 1000), 1000)),
                        int(min(max(0, y0 / height * 1000), 1000)),
                        int(min(max(0, x1 / width * 1000), 1000)),
                        int(min(max(0, y1 / height * 1000), 1000)),
                    ]
                    words_with_boxes.append({"word": word, "bbox": norm_box})
            doc.close()

        # 2. Extract Plain Text & BIO Tags for NER
        txt_path = corpus_dir / f"{rid}.txt"
        text_content = ""
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as tf:
                text_content = tf.read()

        features.append({
            "report_id": rid,
            "category": cat,
            "category_idx": cat_idx,
            "template_id": ann.get("template_id"),
            "image_path": str(img_out_path),
            "image_width": width,
            "image_height": height,
            "text": text_content,
            "words": words_with_boxes,
            "parameters": ann.get("parameters", []),
            "conclusion": ann.get("conclusion", "")
        })

    # Save category label map
    label_map_file = processed_dir / "category_label_map.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(category_mapping, f, indent=2)

    # Save master features manifest
    features_file = processed_dir / "document_features_master.json"
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print(f"Features extraction complete. Processed {len(features)} documents.")
    print(f"Categories mapped: {len(category_mapping)}")
    print(f"Artifacts saved to: {processed_dir}")

if __name__ == "__main__":
    process_corpus_features(Path("data/document_ai"))
