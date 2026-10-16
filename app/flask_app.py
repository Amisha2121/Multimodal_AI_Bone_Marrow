# -*- coding: utf-8 -*-
"""
Flask Backend API for Bone Marrow Analysis
"""

import os
import sys
import math
import pickle
from pathlib import Path
from datetime import datetime
import hashlib
import json
import traceback
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.new_patient_engine import analyze_new_case
from retrieval.retrieve import HybridRetriever
from agents.llm_provider import get_llm_provider

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = REPO_ROOT / 'data' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------
STORAGE_DIR = REPO_ROOT / 'data' / 'app_storage'
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSES_PKL = STORAGE_DIR / 'analyses.pkl'
CHATS_JSON   = STORAGE_DIR / 'chats.json'


def _safe_float(val):
    """Return float; replace NaN/Inf with 0.0."""
    try:
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _cell_to_dict(cell):
    """Serialise a CellInferenceResult to a plain JSON-safe dict."""
    path = cell.image_path if cell.image_path else None
    if path and not Path(path).exists():
        path = None
    return {
        'id':         cell.cell_id,
        'class':      cell.predicted_class,
        'confidence': _safe_float(cell.confidence),
        'image_path': path,
        'bbox':       list(cell.bbox) if cell.bbox else None,
    }


def load_analyses():
    if ANALYSES_PKL.exists():
        try:
            with open(ANALYSES_PKL, 'rb') as f:
                data = pickle.load(f)
            migrated = {}
            for aid, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                result = entry.get('analysis')
                if result is None:
                    continue
                # Backfill keys added in newer schema
                entry.setdefault('id', aid)
                entry.setdefault('cells_count',
                    getattr(result, 'total_cells_detected', 0))
                entry.setdefault('blast_pct',
                    _safe_float(getattr(result, 'blast_fraction', 0.0)))
                entry.setdefault('morphology',
                    getattr(result, 'morphology_counts', {}))
                entry.setdefault('morphology_percentages',
                    getattr(result, 'morphology_percentages', {}))
                entry.setdefault('detection_images', {})
                entry.setdefault('cell_images', [])
                entry.setdefault('images', [])
                entry.setdefault('first_image', None)
                migrated[aid] = entry
            print(f"Loaded {len(migrated)} persisted analyses")
            return migrated
        except Exception as e:
            print(f"Could not load analyses.pkl: {e}")
    return {}


def save_analyses(analyses_dict):
    try:
        with open(ANALYSES_PKL, 'wb') as f:
            pickle.dump(analyses_dict, f)
    except Exception as e:
        print(f"Could not save analyses.pkl: {e}")


def load_chats():
    if CHATS_JSON.exists():
        try:
            with open(CHATS_JSON, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # Migrate old format: {q, a, time} -> {question, answer, timestamp}
            migrated = {}
            for sid, msgs in raw.items():
                new_msgs = []
                for m in msgs:
                    if 'q' in m and 'question' not in m:
                        new_msgs.append({
                            'question':  m.get('q', ''),
                            'answer':    m.get('a', ''),
                            'timestamp': m.get('time', ''),
                        })
                    else:
                        new_msgs.append(m)
                migrated[sid] = new_msgs
            return migrated
        except Exception as e:
            print(f"Could not load chats.json: {e}")
    return {}


def save_chats(chats_dict):
    try:
        with open(CHATS_JSON, 'w', encoding='utf-8') as f:
            json.dump(chats_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Could not save chats.json: {e}")


analyses     = load_analyses()
chat_history = load_chats()

# ---------------------------------------------------------------------------
# RAG / LLM initialisation (with smoke-test validation)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Initialising RAG system...")
RAG_AVAILABLE = False
retriever    = None
llm_provider = None

try:
    retriever = HybridRetriever()
    print("Retriever OK")
except Exception as e:
    print(f"Retriever FAILED: {e}")
    traceback.print_exc()

try:
    llm_provider = get_llm_provider()
    # Smoke-test: actually call the API so we catch bad keys immediately
    test_out = llm_provider.generate(
        [{"role": "user", "content": "Say OK"}],
        max_tokens=5, temperature=0.0,
    )
    print(f"LLM OK  ({llm_provider.provider}/{llm_provider.model_name}  -> {test_out.strip()!r})")
    if retriever is not None:
        RAG_AVAILABLE = True
        print("RAG system READY")
    else:
        print("LLM OK but retriever failed -- RAG disabled")
except Exception as e:
    print(f"LLM FAILED: {e}")
    traceback.print_exc()
    llm_provider = None

print("=" * 70)


# ===========================================================================
# API ENDPOINTS
# ===========================================================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health():
    return jsonify({
        'status':       'ok',
        'rag_available': RAG_AVAILABLE,
        'llm_provider': llm_provider.provider   if llm_provider else None,
        'llm_model':    llm_provider.model_name if llm_provider else None,
    })


@app.route('/api/upload', methods=['POST'])
def upload_analysis():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    saved_files = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)
            saved_files.append(filepath)

    try:
        print(f"Analysing {len(saved_files)} file(s)...")
        result = analyze_new_case(
            uploaded_files=saved_files,
            age_years=None, sex=None, sample_site=None,
        )
        print(f"Done: {result.case_id}  ({result.total_cells_detected} cells)")

        analysis_id = hashlib.md5(
            f"{result.case_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Detection images (3 views)
        detection_images = {'raw': None, 'filtered': None, 'contact': None}
        first_thumbnail  = None
        if result.image_infos:
            info = result.image_infos[0]
            detection_images = {
                'raw':      info.debug_raw_path      or None,
                'filtered': info.debug_filtered_path or None,
                'contact':  info.debug_contact_sheet_path or None,
            }
            first_thumbnail = info.debug_filtered_path or None

        # Cell list (all cells, path=None when crop is missing)
        cell_list = [_cell_to_dict(c) for c in result.cell_predictions]

        # Per-image metadata
        images_meta = [
            {
                'filename':       info.filename,
                'cells':          info.cells_detected,
                'filtered_path':  info.debug_filtered_path      or None,
                'raw_path':       info.debug_raw_path            or None,
                'contact_sheet':  info.debug_contact_sheet_path  or None,
                'annotated_path': info.annotated_image_path      or None,
                'width':          info.width_px,
                'height':         info.height_px,
            }
            for info in result.image_infos
        ]

        analyses[analysis_id] = {
            'id':                     analysis_id,
            'case_id':                result.case_id,
            'timestamp':              datetime.now().isoformat(),
            'analysis':               result,
            'cells_count':            result.total_cells_detected,
            'blast_pct':              _safe_float(result.blast_fraction),
            'morphology':             result.morphology_counts,
            'morphology_percentages': result.morphology_percentages,
            'detection_images':       detection_images,
            'cell_images':            cell_list,
            'images':                 images_meta,
            'first_image':            first_thumbnail,
        }
        chat_history.setdefault(analysis_id, [])

        save_analyses(analyses)
        save_chats(chat_history)

        print(f"Stored as {analysis_id}")
        return jsonify({
            'success':       True,
            'analysis_id':   analysis_id,
            'case_id':       result.case_id,
            'cells_detected': result.total_cells_detected,
            'blast_fraction': _safe_float(result.blast_fraction),
        })

    except Exception as e:
        print(f"Upload FAILED: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyses')
def get_analyses():
    return jsonify({
        'analyses': [
            {
                'id':        aid,
                'case_id':   d['case_id'],
                'timestamp': d['timestamp'],
                'cells':     d['cells_count'],
                'blasts':    _safe_float(d['blast_pct']),
                'thumbnail': d.get('first_image'),
            }
            for aid, d in analyses.items()
        ]
    })


@app.route('/api/analysis/<analysis_id>')
def get_analysis(analysis_id):
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    d      = analyses[analysis_id]
    result = d['analysis']

    return jsonify({
        'id':        analysis_id,
        'case_id':   d['case_id'],
        'timestamp': d['timestamp'],
        'metrics': {
            'images_processed': result.total_images_processed,
            'cells_detected':   result.total_cells_detected,
            'cells_classified': result.cells_classified,
            'blast_fraction':   _safe_float(result.blast_fraction),
        },
        'morphology': [
            {
                'type':       ct,
                'count':      cnt,
                'percentage': _safe_float(d['morphology_percentages'].get(ct, 0)),
            }
            for ct, cnt in result.morphology_counts.items()
        ],
        'detection_images': d.get('detection_images', {}),
        'images':           d['images'],
        'cell_images':      d.get('cell_images', []),
    })


@app.route('/api/cells/<analysis_id>')
def get_cells(analysis_id):
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    d = analyses[analysis_id]
    cells = d.get('cell_images')
    if not cells:
        cells = [_cell_to_dict(c) for c in d['analysis'].cell_predictions]
    return jsonify({'cells': cells})


@app.route('/api/analysis-report/<analysis_id>')
def get_analysis_report(analysis_id):
    if not RAG_AVAILABLE:
        return jsonify({
            'error': 'RAG system unavailable',
            'hint':  'Check server console -- LLM or retriever failed to initialise.',
        }), 503

    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    # CACHE: Return instantly if report was already generated
    force_regen = request.args.get('regenerate', '') == '1'
    if not force_regen and 'cached_report' in analyses[analysis_id]:
        print(f"Returning CACHED report for {analysis_id}")
        return jsonify(analyses[analysis_id]['cached_report'])

    d      = analyses[analysis_id]
    result = d['analysis']

    llm_failure = None

    def rag_gen(query, prompt, max_tokens=500):
        nonlocal llm_failure
        retrieved = retriever.retrieve(query=query, top_k=3)
        evidence  = "\n\n".join(
            f"[{r['chunk_id']}] {r['text'][:350]}" for r in retrieved
        )
        full = (
            prompt.strip()
            + "\n\nCLINICAL EVIDENCE FROM GUIDELINES:\n"
            + (evidence or "Use standard hematopathology principles.")
        )
        if llm_failure:
            return _report_generation_fallback(evidence), retrieved

        try:
            text = llm_provider.generate(
                [{"role": "user", "content": full}],
                temperature=0.1, max_tokens=max_tokens,
            ).strip()
            return text, retrieved
        except Exception as exc:
            llm_failure = exc
            print(f"Report LLM generation failed: {exc}; continuing with retrieved-evidence fallback")
            return _report_generation_fallback(evidence), retrieved

    def _report_generation_fallback(evidence):
        fallback = (
            "Automated narrative generation is unavailable. This section is not a clinical interpretation; "
            "review the retrieved guideline evidence with a qualified hematopathologist."
        )
        if evidence:
            fallback += "\n\nRetrieved guideline evidence:\n" + evidence
        return fallback

    try:
        sections     = []
        all_src_ids  = []

        dist_brief = ", ".join(
            f"{k} ({v})" for k, v in
            sorted(result.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        # --- Case Summary ---
        text, ev = rag_gen(
            query  = f"bone marrow cytomorphology {result.blast_fraction:.1%} blasts case summary",
            prompt = (
                "You are a clinical haematopathologist. Write a formal case summary paragraph (4-5 sentences).\n\n"
                f"Case ID: {result.case_id}\n"
                f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
                f"Total nucleated cells: {result.total_cells_detected}\n"
                f"Blast proportion: {result.blast_fraction:.1%}\n"
                f"Cell distribution: {dist_brief}\n\n"
                "Include: sample type (bone marrow aspirate), automated ViT classifier, total cells, blast proportion, "
                "predominant cell types. Use formal pathology language."
            ),
        )
        all_src_ids += [r['chunk_id'] for r in ev]
        sections.append({'title': 'Case Summary', 'content': text, 'cell_images': []})
        print("Report: Case Summary done")

        # --- Analytical Methodology (static — no LLM call needed) ---
        sections.append({'title': 'Analytical Methodology', 'content': (
            "This analysis employs a two-stage automated pipeline for bone marrow cytomorphology assessment. "
            "In the first stage, adaptive thresholding combined with watershed segmentation identifies and isolates "
            "individual nucleated cells from the aspirate smear image. In the second stage, a Vision Transformer "
            "(ViT-Small/16) with LoRA fine-tuning classifies each detected cell into morphological categories. "
            "The model was trained and validated on annotated paediatric bone marrow aspirate specimens and provides "
            "consistent, reproducible results that complement traditional manual microscopy. All results should be "
            "interpreted in conjunction with clinical findings, complete blood count, and expert pathologist review."
        ), 'cell_images': []})
        print("Report: Methodology done (static — 0 LLM calls)")

        # --- Overall Morphological Assessment ---
        dist_full = ", ".join(
            f"{ct} ({cnt} cells, {_safe_float(d['morphology_percentages'].get(ct, 0))*100:.1f}%)"
            for ct, cnt in sorted(result.morphology_counts.items(), key=lambda x: x[1], reverse=True)
        )
        text, ev = rag_gen(
            query  = f"bone marrow {result.blast_fraction:.1%} blasts morphological assessment differential",
            prompt = (
                "Write a detailed overall morphological assessment paragraph (4-5 sentences).\n\n"
                f"Total cells: {result.total_cells_detected}\n"
                f"Blast %: {result.blast_fraction:.1%}\n"
                f"Full distribution: {dist_full}\n\n"
                "Cover: overall cellularity, blast significance, maturation spectrum, differential diagnosis context. "
                "Use WHO 2022 thresholds where relevant."
            ),
        )
        all_src_ids += [r['chunk_id'] for r in ev]
        sections.append({'title': 'Overall Morphological Assessment', 'content': text, 'cell_images': []})
        print("Report: Morphological Assessment done")

        # --- Per cell-type sections ---
        top_types = sorted(result.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for cell_type, count in top_types:
            if count == 0:
                continue
            pct = _safe_float(d['morphology_percentages'].get(cell_type, 0))
            text, ev = rag_gen(
                query  = f"{cell_type} bone marrow cytomorphology clinical significance diagnostic criteria",
                prompt = (
                    f"Write a clinical paragraph (3-4 sentences) about {cell_type} cells found in this sample.\n\n"
                    f"Count: {count} cells  ({pct*100:.1f}% of total)\n"
                    f"Case: {result.case_id}\n\n"
                    "Cover: morphological characteristics, significance of this proportion, differential diagnostic "
                    "implications. Use haematopathology terminology."
                ),
                max_tokens=300,
            )
            all_src_ids += [r['chunk_id'] for r in ev]
            imgs = [
                c['image_path'] for c in d.get('cell_images', [])
                if c['class'] == cell_type and c.get('image_path')
            ][:4]
            sections.append({
                'title':       f'{cell_type} ({count} cells, {pct*100:.1f}%)',
                'content':     text,
                'cell_images': imgs,
            })
            print(f"Report: {cell_type} done ({len(imgs)} images)")

        # --- Clinical Interpretation ---
        major = ", ".join(
            f"{k} ({v})" for k, v in list(result.morphology_counts.items())[:3]
        )
        text, ev = rag_gen(
            query  = f"bone marrow {result.blast_fraction:.1%} blasts clinical interpretation recommendations",
            prompt = (
                "Write a clinical interpretation and recommendations paragraph (4-5 sentences).\n\n"
                f"Blast %: {result.blast_fraction:.1%}\n"
                f"Main cell types: {major}\n"
                f"Total cells: {result.total_cells_detected}\n\n"
                "Cover: blast % in clinical context (WHO threshold >=20%), differential diagnoses, recommended "
                "additional workup (flow cytometry, cytogenetics, molecular panel), any urgent actions, caveat that "
                "automated assessment supplements manual review."
            ),
        )
        all_src_ids += [r['chunk_id'] for r in ev]
        sections.append({'title': 'Clinical Interpretation and Recommendations', 'content': text, 'cell_images': []})
        print("Report: Clinical Interpretation done")

        report_data = {
            'sections':     sections,
            'case_id':      result.case_id,
            'generated_at': datetime.now().isoformat(),
            'sources':      list(dict.fromkeys(all_src_ids)),
            'warning':      (
                'The LLM provider could not complete report generation. Affected sections contain retrieved '
                'guideline evidence only and are not model-generated interpretations. Retry after the provider '
                'rate limit clears for a fully generated report.'
            ) if llm_failure else None,
        }

        # Cache for instant future access
        analyses[analysis_id]['cached_report'] = report_data
        save_analyses(analyses)
        print(f"Report CACHED for {analysis_id}")

        return jsonify(report_data)

    except Exception as e:
        print(f"Report FAILED: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    if not RAG_AVAILABLE:
        return jsonify({'error': 'RAG system unavailable'}), 503

    body        = request.json or {}
    analysis_id = body.get('analysis_id')
    question    = (body.get('question') or '').strip()

    if not analysis_id or not question:
        return jsonify({'error': 'Missing analysis_id or question'}), 400
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    d      = analyses[analysis_id]
    result = d['analysis']

    dist_brief = ", ".join(
        f"{ct} ({cnt} cells, {_safe_float(d['morphology_percentages'].get(ct, 0))*100:.1f}%)"
        for ct, cnt in sorted(result.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    context = (
        f"Case: {result.case_id}\n"
        f"Total cells: {result.total_cells_detected}\n"
        f"Blast %: {result.blast_fraction:.1%}\n"
        f"Distribution: {dist_brief}"
    )

    try:
        retrieved = retriever.retrieve(query=question, top_k=3)
        evidence  = "\n\n".join(
            f"[{r['chunk_id']}] {r['text'][:400]}" for r in retrieved
        )
        prompt = (
            f"{context}\n\n"
            f"CLINICAL GUIDELINES:\n{evidence or 'Use standard hematopathology principles.'}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer in 2-4 clear sentences using professional clinical language."
        )
        answer = llm_provider.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=500,
        ).strip()

        entry = {'question': question, 'answer': answer, 'timestamp': datetime.now().isoformat()}
        chat_history.setdefault(analysis_id, []).append(entry)
        save_chats(chat_history)

        return jsonify({'answer': answer, 'sources': [r['chunk_id'] for r in retrieved]})

    except Exception as e:
        print(f"Chat FAILED: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/<analysis_id>')
def get_chat_history(analysis_id):
    return jsonify({'history': chat_history.get(analysis_id, [])})


@app.route('/api/analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    if analysis_id in analyses:
        del analyses[analysis_id]
        chat_history.pop(analysis_id, None)
        save_analyses(analyses)
        save_chats(chat_history)
        return jsonify({'success': True})
    return jsonify({'error': 'Analysis not found'}), 404


@app.route('/api/image/<path:filepath>')
def serve_image(filepath):
    from urllib.parse import unquote
    filepath = unquote(filepath)
    candidates = [
        Path(filepath),
        REPO_ROOT / filepath,
        REPO_ROOT / 'data' / 'outputs' / Path(filepath).name,
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return send_from_directory(str(p.parent), p.name)
        except Exception:
            continue
    return jsonify({'error': 'Image not found', 'path': filepath}), 404


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
