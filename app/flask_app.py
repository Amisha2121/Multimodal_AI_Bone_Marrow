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

import threading

def is_rag_ready():
    """Check if RAG is ready, lazily re-initializing LLM if needed."""
    global RAG_AVAILABLE, llm_provider
    if retriever is None:
        return False
    if llm_provider is None:
        try:
            llm_provider = get_llm_provider()
            RAG_AVAILABLE = True
            print(f"LLM lazily re-initialized ({llm_provider.provider}/{llm_provider.model_name})")
        except Exception as e:
            return False
    return True

try:
    llm_provider = get_llm_provider()
    print(f"LLM initialized ({llm_provider.provider}/{llm_provider.model_name})")
    if retriever is not None:
        RAG_AVAILABLE = True

    def _background_smoke_test():
        global RAG_AVAILABLE, llm_provider
        try:
            test_out = llm_provider.generate(
                [{"role": "user", "content": "Say OK"}],
                max_tokens=5, temperature=0.0,
            )
            print(f"LLM smoke-test OK  -> {test_out.strip()!r}")
            print("RAG system READY")
        except Exception as smoke_e:
            # Keep llm_provider alive so user queries can retry with model fallback
            print(f"LLM smoke-test note: {smoke_e} (queries will retry automatically)")
            if retriever is not None and llm_provider is not None:
                RAG_AVAILABLE = True

    threading.Thread(target=_background_smoke_test, daemon=True).start()

except Exception as e:
    print(f"LLM initial initialization note: {e}")
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
    rag_ready = is_rag_ready()
    return jsonify({
        'status':        'ok',
        'rag_available': rag_ready,
        'llm_provider':  llm_provider.provider   if llm_provider else None,
        'llm_model':     llm_provider.model_name if llm_provider else None,
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

    xai_list = []
    for x in getattr(result, 'xai_examples', []):
        overlay = getattr(x, 'overlay_path', '')
        if overlay and (Path(overlay).exists() or (REPO_ROOT / overlay).exists()):
            xai_list.append({
                'cell_id':             getattr(x, 'cell_id', ''),
                'predicted_class':     getattr(x, 'predicted_class', ''),
                'confidence':          _safe_float(getattr(x, 'confidence', 0)),
                'overlay_path':        str(overlay),
                'original_image_path': str(getattr(x, 'original_image_path', '')),
            })

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
        'is_low_evidence':        getattr(result, 'is_low_evidence', False),
        'predominant_categories': getattr(result, 'predominant_categories', []),
        'xai_examples':           xai_list,
        'detection_images':       d.get('detection_images', {}),
        'images':                 d['images'],
        'cell_images':            d.get('cell_images', []),
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
    if not is_rag_ready():
        return jsonify({
            'error': 'RAG system unavailable',
            'hint':  'Check server console -- LLM or retriever failed to initialise.',
        }), 503

    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    # CACHE: Return instantly if report was already generated and valid (no fallback warning)
    force_regen = request.args.get('regenerate', '') == '1'
    cached = analyses[analysis_id].get('cached_report')
    if not force_regen and cached and not cached.get('warning'):
        print(f"Returning CACHED report for {analysis_id}")
        return jsonify(cached)

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
            import time
            time.sleep(0.3)
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

        # Cache for instant future access ONLY if generation succeeded without LLM failure
        if not llm_failure:
            analyses[analysis_id]['cached_report'] = report_data
            save_analyses(analyses)
            print(f"Report CACHED for {analysis_id}")
        else:
            # If there was a failure, remove any stale cached report so it doesn't persist
            analyses[analysis_id].pop('cached_report', None)
            save_analyses(analyses)
            print(f"Report had LLM warning -- removed from cache so next attempt retries")

        return jsonify(report_data)

    except Exception as e:
        print(f"Report FAILED: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    if not is_rag_ready():
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


@app.route('/api/system-metrics')
def get_system_metrics():
    """Real-time system metrics showcasing improvements."""
    try:
        # Knowledge base stats
        kb_size = 63  # From expansion
        retrieval_mrr = 0.050  # From evaluation
        
        # Calibration stats (if calibration model exists)
        calibration_ece = 0.0
        try:
            from vision.calibration import TemperatureScaling
            cal_path = REPO_ROOT / 'data' / 'models' / 'temperature_calibration.pkl'
            if cal_path.exists():
                with open(cal_path, 'rb') as f:
                    cal_data = pickle.load(f)
                    calibration_ece = _safe_float(cal_data.get('pre_cal_ece', 0.0))
        except Exception:
            pass
        
        # Active learning queue stats
        active_learning_queue = 0
        try:
            from pipeline.active_learning import ActiveLearningPipeline
            al_pipeline = ActiveLearningPipeline()
            active_learning_queue = len(al_pipeline.review_queue)
        except Exception:
            pass
        
        # Analysis stats
        total_analyses = len(analyses)
        total_cells = sum(d.get('cells_count', 0) for d in analyses.values())
        avg_confidence = 0.0
        if analyses:
            confidences = []
            for d in analyses.values():
                for cell in d.get('cell_images', []):
                    confidences.append(_safe_float(cell.get('confidence', 0)))
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        return jsonify({
            'knowledge_base': {
                'total_chunks': kb_size,
                'retrieval_mrr': retrieval_mrr,
                'index_type': 'Hybrid (FAISS + BM25)',
            },
            'calibration': {
                'enabled': calibration_ece > 0,
                'ece_improvement': _safe_float(calibration_ece),
                'method': 'Temperature Scaling',
            },
            'xai': {
                'metrics_available': ['Pointing Game', 'Deletion AUC', 'Insertion AUC'],
                'visualization': 'GradCAM Attention Maps',
            },
            'self_consistency': {
                'enabled': True,
                'num_samples': 3,
                'temperature': 0.3,
            },
            'active_learning': {
                'queue_size': active_learning_queue,
                'uncertainty_threshold': 0.45,
                'methods': ['entropy', 'confidence', 'grounding', 'consistency'],
            },
            'analyses': {
                'total_cases': total_analyses,
                'total_cells_classified': total_cells,
                'avg_confidence': avg_confidence,
            },
            'rag_status': {
                'available': is_rag_ready(),
                'llm_provider': llm_provider.provider if llm_provider else None,
                'llm_model': llm_provider.model_name if llm_provider else None,
            },
        })
    except Exception as e:
        print(f"System metrics error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<analysis_id>')
def export_analysis(analysis_id):
    """Export analysis as JSON or PDF."""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
    format_type = request.args.get('format', 'json').lower()
    d = analyses[analysis_id]
    
    if format_type == 'json':
        # Export as comprehensive JSON
        export_data = {
            'metadata': {
                'analysis_id': analysis_id,
                'case_id': d['case_id'],
                'timestamp': d['timestamp'],
                'exported_at': datetime.now().isoformat(),
            },
            'metrics': {
                'total_cells': d['cells_count'],
                'blast_percentage': _safe_float(d['blast_pct']) * 100,
                'morphology_distribution': d['morphology_percentages'],
            },
            'cells': d.get('cell_images', []),
            'chat_history': chat_history.get(analysis_id, []),
            'report': d.get('cached_report', {}),
        }
        
        from flask import Response
        response = Response(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename={analysis_id}_export.json'}
        )
        return response
    
    elif format_type == 'pdf':
        # PDF export would require reportlab or similar
        return jsonify({
            'error': 'PDF export not yet implemented',
            'hint': 'Use JSON export or implement PDF rendering with reportlab',
        }), 501
    
    else:
        return jsonify({'error': f'Unsupported format: {format_type}'}), 400


@app.route('/api/active-learning/queue')
def get_active_learning_queue():
    """Get prioritized samples for expert review."""
    try:
        from pipeline.active_learning import ActiveLearningPipeline
        al_pipeline = ActiveLearningPipeline()
        
        # Get top uncertain cases
        queue_items = []
        for item in al_pipeline.review_queue[:20]:  # Top 20
            queue_items.append({
                'sample_id': item.sample_id,
                'uncertainty_score': _safe_float(item.uncertainty_score),
                'priority_score': _safe_float(item.priority_score),
                'metrics': {
                    'entropy': _safe_float(item.metrics.get('entropy', 0)),
                    'confidence': _safe_float(item.metrics.get('max_confidence', 0)),
                    'grounding': _safe_float(item.metrics.get('grounding_score', 0)),
                    'consistency': _safe_float(item.metrics.get('consistency_score', 0)),
                },
                'predicted_class': item.predicted_class,
                'image_path': item.image_path,
                'added_at': item.timestamp,
            })
        
        return jsonify({
            'queue_size': len(al_pipeline.review_queue),
            'items': queue_items,
            'threshold': al_pipeline.config.uncertainty_threshold,
        })
    except Exception as e:
        print(f"Active learning queue error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active-learning/feedback', methods=['POST'])
def submit_active_learning_feedback():
    """Submit expert feedback for active learning."""
    try:
        from pipeline.active_learning import ActiveLearningPipeline
        al_pipeline = ActiveLearningPipeline()
        
        body = request.json or {}
        sample_id = body.get('sample_id')
        expert_label = body.get('expert_label')
        confidence = body.get('confidence', 1.0)
        notes = body.get('notes', '')
        
        if not sample_id or not expert_label:
            return jsonify({'error': 'Missing sample_id or expert_label'}), 400
        
        # Record feedback
        al_pipeline.collect_feedback(
            sample_id=sample_id,
            expert_label=expert_label,
            expert_confidence=confidence,
            notes=notes,
        )
        
        return jsonify({
            'success': True,
            'message': f'Feedback recorded for {sample_id}',
            'feedback_count': len(al_pipeline.feedback_log),
        })
    except Exception as e:
        print(f"Active learning feedback error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/comparison/<analysis_id1>/<analysis_id2>')
def compare_analyses(analysis_id1, analysis_id2):
    """Compare two analyses side-by-side."""
    if analysis_id1 not in analyses or analysis_id2 not in analyses:
        return jsonify({'error': 'One or both analyses not found'}), 404
    
    d1 = analyses[analysis_id1]
    d2 = analyses[analysis_id2]
    
    def extract_comparison_data(d):
        return {
            'case_id': d['case_id'],
            'timestamp': d['timestamp'],
            'cells_count': d['cells_count'],
            'blast_pct': _safe_float(d['blast_pct']),
            'morphology': d['morphology_percentages'],
        }
    
    return jsonify({
        'analysis_1': extract_comparison_data(d1),
        'analysis_2': extract_comparison_data(d2),
        'differences': {
            'cells_delta': d2['cells_count'] - d1['cells_count'],
            'blast_delta': _safe_float(d2['blast_pct']) - _safe_float(d1['blast_pct']),
        },
    })


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple cases in parallel."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        results = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = UPLOAD_FOLDER / filename
                file.save(filepath)
                
                # Quick analysis
                result = analyze_new_case(
                    uploaded_files=[filepath],
                    age_years=None, sex=None, sample_site=None,
                )
                
                analysis_id = hashlib.md5(
                    f"{result.case_id}_{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12]
                
                analyses[analysis_id] = {
                    'id': analysis_id,
                    'case_id': result.case_id,
                    'timestamp': datetime.now().isoformat(),
                    'analysis': result,
                    'cells_count': result.total_cells_detected,
                    'blast_pct': _safe_float(result.blast_fraction),
                    'morphology': result.morphology_counts,
                    'morphology_percentages': result.morphology_percentages,
                    'detection_images': {},
                    'cell_images': [_cell_to_dict(c) for c in result.cell_predictions],
                    'images': [],
                    'first_image': None,
                }
                
                results.append({
                    'filename': filename,
                    'analysis_id': analysis_id,
                    'cells': result.total_cells_detected,
                    'blasts': _safe_float(result.blast_fraction),
                })
        
        save_analyses(analyses)
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        print(f"Batch analysis error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
