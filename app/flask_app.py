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
from app.db import (
    init_db,
    import_existing_case,
    save_case_to_db,
    update_case_chat_in_db,
    update_case_report_in_db,
    get_history_from_db,
    get_case_from_db,
    # Phase 9
    save_report_version,
    get_report_versions,
    get_report_version,
    save_annotations,
    get_annotations,
)
from pipeline.summary import build_morphology_summary

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
# SQLite database initialisation & idempotent import
# ---------------------------------------------------------------------------
try:
    init_db()
    for aid, entry in analyses.items():
        import_existing_case(aid, entry, chat_history.get(aid, []))
    print("SQLite database verified and synchronized with persistent cases")
except Exception as e:
    print(f"Warning: SQLite startup synchronization error: {e}")


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
        try:
            save_case_to_db(analysis_id, analyses[analysis_id], chat_history.get(analysis_id, []))
        except Exception as db_e:
            print(f"Dual-write to SQLite failed: {db_e}")

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


# ---------------------------------------------------------------------------
# Multimodal Document Upload & Extraction Endpoint (Phase 3 & 4)
# ---------------------------------------------------------------------------

@app.route('/api/upload-document', methods=['POST'])
def upload_medical_document():
    """
    Ingests and processes medical reports (PDF, CSV, TXT) across Categories A-L.
    Associates the extracted parameters with the selected case or creates a new case.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No document files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No document files selected'}), 400

    target_case_id = request.form.get('case_id')
    from pipeline.document_intelligence import process_uploaded_document_file
    from pipeline.unified_patient_model import UnifiedPatientCase, DocumentCategory
    from pipeline.patient_correlation import correlate_patient_findings

    processed_docs = []
    
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)

            # Ingest & parse through Document Intelligence
            doc = process_uploaded_document_file(filepath)
            processed_docs.append(doc)

    # Attach to existing analysis or create context
    target_aid = None
    if target_case_id:
        for aid, entry in analyses.items():
            if entry.get("case_id") == target_case_id or aid == target_case_id:
                target_aid = aid
                break

    if target_aid and target_aid in analyses:
        entry = analyses[target_aid]
        # Attach documents to case dictionary
        entry.setdefault('uploaded_medical_documents', [])
        for doc in processed_docs:
            entry['uploaded_medical_documents'].append(doc.model_dump())
            # Update clinical_data for report engine consumption
            entry.setdefault('clinical_data', {})
            for p in doc.extracted_parameters:
                p_name = p.name.lower()
                if "hemoglobin" in p_name:
                    entry['clinical_data']['hb'] = p.reported_value
                elif "wbc" in p_name:
                    entry['clinical_data']['wbc'] = p.reported_value
                elif "platelet" in p_name:
                    entry['clinical_data']['plt'] = p.reported_value
                elif "anc" in p_name:
                    entry['clinical_data']['anc'] = p.reported_value
                elif "karyotype" in p_name or "iscn" in p_name:
                    entry['clinical_data']['karyotype'] = p.reported_value
                elif "flow" in p_name:
                    entry['clinical_data'].setdefault('flow_markers', []).append(f"{p.name}: {p.reported_value}")
                elif "molecular" in p_name:
                    entry['clinical_data'].setdefault('mutations', []).append(f"{p.name}: {p.reported_value}")

        # Invalidate cached report so next view incorporates new multimodal context
        entry.pop('cached_report', None)
        save_analyses(analyses)

    return jsonify({
        'success': True,
        'documents_processed': len(processed_docs),
        'documents': [d.model_dump() for d in processed_docs],
        'associated_case_id': target_case_id or "Unassigned"
    })


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
        for aid, entry in analyses.items():
            if entry.get("case_id") == analysis_id:
                analysis_id = aid
                break

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

    use_rev = request.args.get('use_reviewer_labels', '') in ('1', 'true', 'True')
    summary = build_morphology_summary(analysis_id, case_dict=d, use_reviewer_labels=use_rev)
    from pipeline.uncertainty import wilson_interval
    ci = wilson_interval(summary['blast_count'], summary['total_cells'])

    return jsonify({
        'id':        analysis_id,
        'case_id':   d['case_id'],
        'timestamp': d['timestamp'],
        'metrics': {
            'images_processed': getattr(result, 'total_images_processed', 1) if result else 1,
            'cells_detected':   summary['cells_detected'],
            'cells_classified': summary['cells_classified'],
            'blast_fraction':   _safe_float(summary['blast_fraction']),
            'blast_ci':         list(ci) if ci else None,
        },
        'morphology': [
            {
                'type':       ct,
                'count':      cnt,
                'percentage': _safe_float(summary['percentages'].get(ct, 0)),
            }
            for ct, cnt in summary['counts'].items()
        ],
        'is_low_evidence':        getattr(result, 'is_low_evidence', False) if result else False,
        'predominant_categories': summary.get('predominant_categories', getattr(result, 'predominant_categories', []) if result else []),
        'xai_examples':           xai_list,
        'detection_images':       d.get('detection_images', {}),
        'images':                 d.get('images', []),
        'cell_images':            d.get('cell_images', []),
    })


@app.route('/api/cells/<analysis_id>')
def get_cells(analysis_id):
    if analysis_id not in analyses:
        for aid, entry in analyses.items():
            if entry.get("case_id") == analysis_id:
                analysis_id = aid
                break

    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    d = analyses[analysis_id]
    cells = d.get('cell_images')
    if not cells and d.get('analysis'):
        cells = [_cell_to_dict(c) for c in d['analysis'].cell_predictions]
    cells = cells or []

    # Dev-only check: verify Cell Explorer totals match single source of truth
    try:
        summary = build_morphology_summary(analysis_id, case_dict=d)
        if len(cells) != summary['total_cells']:
            print(f"Dev warning: Cell Explorer cell count ({len(cells)}) differs from summary ({summary['total_cells']})")
    except Exception as dev_err:
        print(f"Dev check note: {dev_err}")

    return jsonify({'cells': cells})


# ---------------------------------------------------------------------------
# Phase 1 Persistence endpoints (SQLite backend with in-memory fallback)
# ---------------------------------------------------------------------------
@app.route('/api/history')
def get_history():
    """Return analysis history from SQLite DB first, falling back to in-memory store."""
    try:
        db_history = get_history_from_db()
        if db_history:
            return jsonify({'analyses': db_history, 'cases': db_history, 'source': 'sqlite'})
    except Exception as e:
        print(f"Error reading history from SQLite: {e}")

    # Fall back to existing in-memory / analyses.pkl store
    return get_analyses()


@app.route('/api/case/<case_or_analysis_id>')
def get_case(case_or_analysis_id):
    """Return case details from SQLite DB, falling back to in-memory store."""
    try:
        use_rev = request.args.get('use_reviewer_labels', '') in ('1', 'true', 'True')
        case_data = get_case_from_db(case_or_analysis_id)
        if case_data:
            aid = case_data.get('id')
            if use_rev:
                summary = build_morphology_summary(case_or_analysis_id, case_dict=case_data, use_reviewer_labels=True)
                from pipeline.uncertainty import wilson_interval
                ci = wilson_interval(summary['blast_count'], summary['total_cells'])
                case_data['metrics']['cells_detected'] = summary['cells_detected']
                case_data['metrics']['cells_classified'] = summary['cells_classified']
                case_data['metrics']['blast_fraction'] = _safe_float(summary['blast_fraction'])
                case_data['metrics']['blast_ci'] = list(ci) if ci else None
                case_data['morphology'] = [
                    {
                        'type': ct,
                        'count': cnt,
                        'percentage': _safe_float(summary['percentages'].get(ct, 0.0)),
                    }
                    for ct, cnt in summary['counts'].items()
                ]
            # Augment with live in-memory overlays if available
            if aid in analyses:
                d = analyses[aid]
                result = d.get('analysis')
                if result:
                    xai_list = []
                    for x in getattr(result, 'xai_examples', []):
                        overlay = getattr(x, 'overlay_path', '')
                        if overlay and (Path(overlay).exists() or (REPO_ROOT / overlay).exists()):
                            xai_list.append({
                                'cell_id': getattr(x, 'cell_id', ''),
                                'predicted_class': getattr(x, 'predicted_class', ''),
                                'confidence': _safe_float(getattr(x, 'confidence', 0)),
                                'overlay_path': str(overlay),
                                'original_image_path': str(getattr(x, 'original_image_path', '')),
                            })
                    case_data['xai_examples'] = xai_list
                    case_data['is_low_evidence'] = getattr(result, 'is_low_evidence', False)
                    case_data['predominant_categories'] = getattr(result, 'predominant_categories', [])
                    case_data['images'] = d.get('images', [])
            return jsonify(case_data)
    except Exception as e:
        print(f"Error reading case from SQLite: {e}")

    # Fall back to existing in-memory / analyses.pkl store
    return get_analysis(case_or_analysis_id)



@app.route('/api/background-literature')
def get_background_literature():
    """Return static background literature reference module (Phase 10)."""
    import json
    lit_path = REPO_ROOT / 'data' / 'background_literature.json'
    if lit_path.exists():
        with open(lit_path, 'r', encoding='utf-8') as f:
            return jsonify({'literature': json.load(f)})
    return jsonify({'literature': []})


@app.route('/api/databank-query')
def api_databank_query():
    """Live query endpoint across federated biomedical data banks."""
    query = request.args.get('query', '').strip()
    bank = request.args.get('bank', 'all').strip()
    if not query:
        return jsonify({'error': 'query parameter required'}), 400
    try:
        from pipeline.medical_databanks import search_all_databanks
        results = search_all_databanks(query, bank_filter=bank)
        return jsonify(results)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/analysis-report/<analysis_id>')
def get_analysis_report(analysis_id):
    if not is_rag_ready():
        return jsonify({
            'error': 'RAG system unavailable',
            'hint':  'Check server console -- LLM or retriever failed to initialise.',
        }), 503

    if analysis_id not in analyses:
        for aid, entry in analyses.items():
            if entry.get("case_id") == analysis_id:
                analysis_id = aid
                break

    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    # CACHE: Return instantly if report was already generated and valid (no fallback warning)
    force_regen = request.args.get('regenerate', '') == '1'
    cached = analyses[analysis_id].get('cached_report')
    if not force_regen and cached and not cached.get('warning'):
        sec_titles = [s.get('title', '') for s in cached.get('sections', [])]
        if 'Confidence and Limitations' not in sec_titles:
            from pipeline.uncertainty import build_confidence_and_limits_section
            summary = build_morphology_summary(analysis_id, case_dict=analyses[analysis_id])
            cached['sections'].append(build_confidence_and_limits_section(summary))
        # Phase 9: retroactively add claims to cached reports that lack them
        if not cached.get('has_claims'):
            try:
                from pipeline.claim_extractor import extract_claims_from_report
                _d = analyses[analysis_id]
                _summary = build_morphology_summary(analysis_id, case_dict=_d)
                _cell_ids = [c.get('id') for c in _d.get('cell_images', []) if c.get('id')]
                extract_claims_from_report(cached, summary=_summary, cell_ids_in_case=_cell_ids)
                print(f"Phase 9: claims retroactively added to cached report {analysis_id}")
            except Exception as _ce:
                print(f"Phase 9 cached claim extraction skipped: {_ce}")

        # Phase 10: enrich report with scan_facts, chapters, figures, diagrams, plausibility_flags
        if not cached.get('scan_facts'):
            try:
                from pipeline.scan_facts import build_scan_facts, generate_deterministic_chapters
                from pipeline.report_figures import generate_all_figures
                from pipeline.report_diagrams import generate_all_diagrams
                _d = analyses[analysis_id]
                _summary = build_morphology_summary(analysis_id, case_dict=_d)
                _cells = _d.get('cell_images', [])
                _facts = build_scan_facts(_summary, _cells)
                _top4 = cached.get('sources', [])[:4]
                cached['scan_facts'] = _facts
                cached['chapters'] = generate_deterministic_chapters(_facts, _top4)
                cached['figures'] = generate_all_figures(_summary.get('case_id', analysis_id), _summary, _cells)
                cached['diagrams'] = generate_all_diagrams(_summary.get('case_id', analysis_id), _summary, _cells, _top4)
                cached['plausibility_flags'] = _facts.get('plausibility_flags', [])
                cached['sensitivity_range'] = _facts.get('sensitivity_range', {})
                cached['schema_version'] = 'v3'

                # Multi-databank medical disease detection & differential
                from pipeline.disease_detection import compute_disease_differential
                from pipeline.medical_databanks import fetch_multibank_evidence
                diff = compute_disease_differential(_summary, _cells)
                cached['disease_differential'] = diff
                cached['multibank_evidence'] = fetch_multibank_evidence(diff)
                print(f"Phase 10: scan_facts, chapters, figures, diagrams, and medical differential enriched for {analysis_id}")
            except Exception as _fe:
                print(f"Phase 10 scan_facts enrichment error: {_fe}")

        # Dynamic Patient-Specific Report Architecture enrichment (Module 1-8)
        if not cached.get('patient_clinical_context') or not cached.get('visualization_manifest'):
            try:
                from pipeline.patient_context import extract_patient_context_from_case
                from pipeline.evidence_registry import build_evidence_registry_for_case
                from pipeline.dynamic_visualizer import build_patient_visualization_manifest
                from pipeline.report_validator import validate_report

                _d = analyses[analysis_id]
                _summary = build_morphology_summary(analysis_id, case_dict=_d)
                _cells = _d.get('cell_images', [])
                _patient_ctx = extract_patient_context_from_case(_d)
                cached['patient_clinical_context'] = _patient_ctx.to_dict()

                _top4_chunks = [{'chunk_id': s, 'text': ''} for s in cached.get('sources', [])[:4]]
                _evidence_reg = build_evidence_registry_for_case(_top4_chunks, cached.get('multibank_evidence'))
                cached['evidence_registry'] = {k: v.to_dict() for k, v in _evidence_reg.items()}

                _manifest = build_patient_visualization_manifest(
                    case_id=_summary.get('case_id', analysis_id),
                    summary=_summary,
                    cells=_cells,
                    top4_sources=_top4_chunks,
                    patient_ctx=_patient_ctx
                )
                cached['visualization_manifest'] = _manifest
                for k_dia, v_dia in _manifest.get('diagram_urls', {}).items():
                    cached.setdefault('diagrams', {})[k_dia.lower()] = v_dia

                _val = validate_report(cached, _summary, _evidence_reg)
                cached['validation_audit'] = _val.to_dict()
                print(f"Dynamic patient-specific report enriched for {analysis_id}")
            except Exception as _psee:
                print(f"Dynamic patient-specific report enrichment error: {_psee}")

        # Phase 10 language guard: strip stray markdown (**) and forbidden diagnostic phrases
        # from LLM-generated cached section content and claim texts.
        if not cached.get('_p10_lang_cleaned'):
            try:
                import re as _re
                from pipeline.agent_guard import sanitize_agent_language

                _FORBIDDEN_PHRASES = [
                    # diagnostic conclusions
                    r'aligns with (?:the )?(?:diagnostic )?criteria for\b',
                    r'consistent with (?:a )?diagnosis of\b',
                    r'meets (?:the )?(?:diagnostic )?criteria for\b',
                    r'positive for (?:acute leukemia|AML|ALL|MDS[^\s]*)',
                    r'\bMDS-EB2\b',
                    r'\bMDS-EB1\b',
                    r'diagnostic of\b',
                ]
                _PHRASE_REPLACEMENT = (
                    "findings are descriptive and require physician correlation"
                )

                def _clean_text(txt):
                    if not txt:
                        return txt
                    # Strip **bold** markdown markers
                    txt = _re.sub(r'\*\*([^*]*)\*\*', r'\1', txt)
                    txt = _re.sub(r'\*\*', '', txt)
                    # Strip ## heading markers
                    txt = _re.sub(r'^#{1,6}\s*', '', txt, flags=_re.MULTILINE)
                    # Remove forbidden diagnostic phrases
                    for pat in _FORBIDDEN_PHRASES:
                        txt = _re.sub(pat, _PHRASE_REPLACEMENT, txt, flags=_re.IGNORECASE)
                    # Apply existing language guard (removes definitive dx assertions)
                    txt = sanitize_agent_language(txt)
                    return txt

                # Clean section content and each claim's text
                for sec in cached.get('sections', []):
                    sec['content'] = _clean_text(sec.get('content', ''))
                    for cl in sec.get('claims', []):
                        cl['text'] = _clean_text(cl.get('text', ''))

                cached['_p10_lang_cleaned'] = True
                print(f"Phase 10 language guard: cleaned sections/claims for {analysis_id}")
            except Exception as _le:
                print(f"Phase 10 language guard error: {_le}")

        # Ensure disease differential and full multi-databank evidence are attached even if report was previously cached
        _mb = cached.get('multibank_evidence') or {}
        if not cached.get('disease_differential') or not _mb.get('ncbi_genes') or not _mb.get('targeted_therapeutics'):
            try:
                from pipeline.disease_detection import compute_disease_differential
                from pipeline.medical_databanks import fetch_multibank_evidence
                _d = analyses[analysis_id]
                _summary = build_morphology_summary(analysis_id, case_dict=_d)
                _cells = _d.get('cell_images', [])
                diff = compute_disease_differential(_summary, _cells)
                cached['disease_differential'] = diff
                cached['multibank_evidence'] = fetch_multibank_evidence(diff)
            except Exception as _de:
                print(f"Adding disease_differential to cached report failed: {_de}")

        print(f"Returning CACHED report for {analysis_id}")
        return jsonify(cached)

    d      = analyses[analysis_id]
    result = d['analysis']

    llm_failure = None

    # Phase 10: language rules prepended to every LLM prompt
    _LLM_LANGUAGE_RULES = (
        "STRICT OUTPUT RULES:\n"
        "1. Do NOT use markdown formatting (no **bold**, no ## headings, no bullet asterisks).\n"
        "2. Do NOT state a clinical diagnosis or conclude that findings 'meet criteria for', "
        "'align with', 'are consistent with', or 'are positive for' any named disease "
        "(e.g. MDS-EB1, MDS-EB2, AML, CML). Cytomorphology data alone is insufficient "
        "for diagnosis; that conclusion requires physician correlation.\n"
        "3. Use plain prose sentences only. Reference thresholds as context, never as conclusions.\n"
        "4. This is a descriptive cytomorphology summary, not a clinical report.\n\n"
    )

    import re as _re_rag

    def _strip_md_and_dx(txt):
        """Remove ** markdown and forbidden diagnostic-conclusion phrases from LLM output."""
        if not txt:
            return txt
        txt = _re_rag.sub(r'\*\*([^*]*)\*\*', r'\1', txt)
        txt = _re_rag.sub(r'\*\*', '', txt)
        txt = _re_rag.sub(r'^#{1,6}\s*', '', txt, flags=_re_rag.MULTILINE)
        for pat in [
            r'aligns with (?:the )?(?:diagnostic )?criteria for\b',
            r'consistent with (?:a )?diagnosis of\b',
            r'meets (?:the )?(?:diagnostic )?criteria for\b',
            r'positive for (?:acute leukemia|AML|ALL|MDS[^\s]*)',
            r'\bMDS-EB[12]\b',
            r'diagnostic of\b',
        ]:
            txt = _re_rag.sub(pat, 'findings require physician correlation', txt, flags=_re_rag.IGNORECASE)
        from pipeline.agent_guard import sanitize_agent_language
        return sanitize_agent_language(txt)

    def rag_gen(query, prompt, max_tokens=500):
        nonlocal llm_failure
        retrieved = retriever.retrieve(query=query, top_k=3)
        evidence  = "\n\n".join(
            f"[{r['chunk_id']}] {r['text'][:350]}" for r in retrieved
        )
        full = (
            _LLM_LANGUAGE_RULES
            + prompt.strip()
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
            # Always sanitize LLM output regardless of prompt compliance
            text = _strip_md_and_dx(text)
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
        summary = build_morphology_summary(analysis_id, case_dict=d)

        dist_brief = ", ".join(
            f"{k} ({v} cells, {summary['percentages'].get(k, 0)*100:.1f}%)"
            for k, v in summary['sorted_categories']
        )

        # --- Case Summary ---
        text, ev = rag_gen(
            query  = f"bone marrow cytomorphology {summary['blast_fraction']:.1%} blasts case summary",
            prompt = (
                "You are a clinical haematopathologist. Write a formal case summary paragraph (4-5 sentences).\n\n"
                f"Case ID: {summary['case_id']}\n"
                f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
                f"Total nucleated cells: {summary['total_cells']}\n"
                f"Blast proportion: {summary['blast_fraction']:.1%}\n"
                f"Cell distribution: {dist_brief}\n\n"
                "Include: sample type (bone marrow aspirate), automated ViT classifier, total cells, blast proportion, "
                "predominant cell types. Use formal pathology language."
            ),
        )
        all_src_ids += [r['chunk_id'] for r in ev]
        sections.append({'title': 'Case Summary', 'content': text, 'cell_images': []})
        print("Report: Case Summary done")

        # --- Analytical Methodology (static — clinical laboratory pathology methodology) ---
        sections.append({'title': 'Analytical Methodology', 'content': (
            "This diagnostic evaluation incorporates standardized computer-assisted cytomorphometric analysis "
            "for bone marrow aspirate smear examination. Automated high-power field image processing isolates nucleated cell "
            "boundaries via calibrated chromatic thresholding. Morphological lineage classification is conducted using a deep "
            "visual feature model calibrated on multi-institutional hematopathology reference standards. "
            "Classification yields reproducible quantitative differentials across erythroid, granulocytic, megakaryocytic, "
            "and immature blast compartments. Findings serve as assistive diagnostic screening to be integrated with manual "
            "microscopic review, multi-parameter flow cytometry immunophenotyping, and cytogenetic/molecular findings."
        ), 'cell_images': []})
        print("Report: Methodology done (static — 0 LLM calls)")

        # --- Overall Morphological Assessment ---
        dist_full = ", ".join(
            f"{ct} ({cnt} cells, {summary['percentages'].get(ct, 0)*100:.1f}%)"
            for ct, cnt in summary['sorted_categories']
        )
        text, ev = rag_gen(
            query  = f"bone marrow {summary['blast_fraction']:.1%} blasts morphological assessment differential",
            prompt = (
                "Write a detailed overall morphological assessment paragraph (4-5 sentences).\n\n"
                f"Total cells: {summary['total_cells']}\n"
                f"Blast %: {summary['blast_fraction']:.1%}\n"
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

        # --- Confidence and Limitations (Phase 3 deterministic section) ---
        from pipeline.uncertainty import build_confidence_and_limits_section
        sections.append(build_confidence_and_limits_section(summary))

        # --- Phase 9: Claim extraction (additive — adds 'claims' to each section) ---
        try:
            from pipeline.claim_extractor import extract_claims_from_report
            cell_ids_in_case = [c.get('id') for c in d.get('cell_images', []) if c.get('id')]
            # Temporarily build report_data before extraction so sources are available
            _pre_report = {
                'sections': sections,
                'sources':  list(dict.fromkeys(all_src_ids)),
            }
            extract_claims_from_report(_pre_report, summary=summary, cell_ids_in_case=cell_ids_in_case)
            # sections are mutated in-place with 'claims' lists
            print(f"Phase 9: claims extracted for {len(sections)} sections")
        except Exception as _ce:
            print(f"Phase 9 claim extraction skipped (non-fatal): {_ce}")

        # Compute medical disease detection & federated multi-databank evidence
        disease_diff = []
        multibank_ev = {}
        try:
            from pipeline.disease_detection import compute_disease_differential
            from pipeline.medical_databanks import fetch_multibank_evidence
            disease_diff = compute_disease_differential(summary, [c for c in d.get('cell_images', [])])
            multibank_ev = fetch_multibank_evidence(disease_diff)
        except Exception as _mde:
            print(f"Disease detection / multibank query non-fatal error: {_mde}")

        # Build comprehensive patient-specific medical report object (Module 1-8)
        try:
            from pipeline.patient_report_engine import build_patient_specific_report
            from pipeline.scan_facts import build_scan_facts, generate_deterministic_chapters
            from pipeline.report_figures import generate_all_figures

            _cells = d.get('cell_images', [])
            _facts = build_scan_facts(summary, _cells)
            _top4_chunks = [{'chunk_id': s, 'text': ''} for s in all_src_ids[:4]]
            _chapters = generate_deterministic_chapters(_facts, _top4_chunks)
            _figures = generate_all_figures(result.case_id, summary, _cells)

            patient_report = build_patient_specific_report(
                case_dict=d,
                summary=summary,
                retrieved_chunks=_top4_chunks,
                multibank_evidence=multibank_ev,
                disease_differential=disease_diff,
                scan_facts=_facts,
                chapters=_chapters,
                figures=_figures
            )

            # Preserve LLM generated prose sections if successful, appending structured clinical sections
            if not llm_failure and sections:
                # Merge LLM narrative sections with patient report clinical structure
                patient_report['llm_sections'] = sections

            report_data = patient_report
            report_data['warning'] = (
                'The LLM provider could not complete report generation. Affected sections contain retrieved '
                'guideline evidence only and are not model-generated interpretations. Retry after the provider '
                'rate limit clears for a fully generated report.'
            ) if llm_failure else None
            report_data['_p10_lang_cleaned'] = True
        except Exception as _pbe:
            print(f"build_patient_specific_report fallback: {_pbe}")
            report_data = {
                'sections':             sections,
                'case_id':              result.case_id,
                'generated_at':         datetime.now().isoformat(),
                'sources':              list(dict.fromkeys(all_src_ids)),
                'schema_version':       'v4_dynamic_patient_specific',
                'has_claims':           True,
                'disease_differential': disease_diff,
                'multibank_evidence':   multibank_ev,
                '_p10_lang_cleaned':    True,
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
            try:
                update_case_report_in_db(analysis_id, report_data)
            except Exception as db_e:
                print(f"Dual-write report to SQLite failed: {db_e}")
            # Phase 9: save as version 1 (or next version on regen)
            try:
                _case_id = result.case_id
                _trigger = 'regen' if force_regen else 'initial'
                _gen = 'Template fallback' if llm_failure else 'LLM'
                save_report_version(_case_id, analysis_id, report_data,
                                    trigger=_trigger, generator=_gen)
            except Exception as _ve:
                print(f"Phase 9 version save skipped (non-fatal): {_ve}")
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


# ---------------------------------------------------------------------------
# Phase 9 – Report claims endpoint
# ---------------------------------------------------------------------------

@app.route('/api/report-claims/<analysis_id>')
def get_report_claims(analysis_id):
    """Return all claims across all sections for the given case/analysis."""
    # Resolve case_id alias
    if analysis_id not in analyses:
        for aid, entry in analyses.items():
            if entry.get('case_id') == analysis_id:
                analysis_id = aid
                break
    cached = analyses.get(analysis_id, {}).get('cached_report')
    if not cached:
        return jsonify({'error': 'Report not generated yet. Load the RAG Report tab first.'}), 404

    # Retroactively add claims if missing
    if not cached.get('has_claims'):
        try:
            from pipeline.claim_extractor import extract_claims_from_report
            d = analyses[analysis_id]
            _summary = build_morphology_summary(analysis_id, case_dict=d)
            _cell_ids = [c.get('id') for c in d.get('cell_images', []) if c.get('id')]
            extract_claims_from_report(cached, summary=_summary, cell_ids_in_case=_cell_ids)
        except Exception as ce:
            print(f"Claim extraction on-demand failed: {ce}")

    all_claims = []
    for sec in cached.get('sections', []):
        for claim in sec.get('claims', []):
            all_claims.append({**claim, '_section': sec.get('title', '')})

    return jsonify({
        'case_id':        cached.get('case_id'),
        'schema_version': cached.get('schema_version', 'v1'),
        'total_claims':   len(all_claims),
        'claims':         all_claims,
    })


# ---------------------------------------------------------------------------
# Phase 9 – Report versions endpoints
# ---------------------------------------------------------------------------

@app.route('/api/report-versions/<case_or_id>')
def api_get_report_versions(case_or_id):
    """Return version history (metadata only) for a case."""
    versions = get_report_versions(case_or_id)
    return jsonify({'versions': versions, 'count': len(versions)})


@app.route('/api/report-versions/restore/<case_or_id>/<int:version_num>', methods=['POST'])
def api_restore_report_version(case_or_id, version_num):
    """Restore a previous report version.  Creates a NEW version row (non-destructive)."""
    old = get_report_version(case_or_id, version_num)
    if not old:
        return jsonify({'error': f'Version {version_num} not found for {case_or_id}'}), 404

    # Resolve analysis_id
    analysis_id = case_or_id
    if case_or_id not in analyses:
        for aid, entry in analyses.items():
            if entry.get('case_id') == case_or_id:
                analysis_id = aid
                break

    # Update in-memory cache and persist
    if analysis_id in analyses:
        analyses[analysis_id]['cached_report'] = old
        save_analyses(analyses)
        try:
            update_case_report_in_db(analysis_id, old)
        except Exception as db_e:
            print(f"Restore: DB write failed (non-fatal): {db_e}")

    # Save as new version so history is preserved
    case_id = analyses.get(analysis_id, {}).get('case_id', case_or_id)
    new_v = save_report_version(case_id, analysis_id, old, trigger='restore', generator='restore')
    return jsonify({'restored_to_version': version_num, 'new_version': new_v})


# ---------------------------------------------------------------------------
# Phase 9 – Annotations endpoints (never modifies report_json text)
# ---------------------------------------------------------------------------

@app.route('/api/report-annotations/<case_or_id>', methods=['GET'])
def api_get_annotations(case_or_id):
    ann = get_annotations(case_or_id)
    return jsonify({'annotations': ann})


@app.route('/api/report-annotations/<case_or_id>', methods=['POST'])
def api_save_annotations(case_or_id):
    body = request.json or {}
    ann  = body.get('annotations')
    if ann is None:
        return jsonify({'error': 'Missing annotations field'}), 400
    ok = save_annotations(case_or_id, ann)
    if ok:
        return jsonify({'saved': True})
    return jsonify({'error': 'Failed to save annotations'}), 500


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
        for aid, entry in analyses.items():
            if entry.get("case_id") == analysis_id:
                analysis_id = aid
                break

    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404

    d       = analyses[analysis_id]
    summary = build_morphology_summary(analysis_id, case_dict=d)

    dist_full = ", ".join(
        f"{ct} ({cnt} cells, {summary['percentages'].get(ct, 0.0)*100:.1f}%)"
        for ct, cnt in summary['sorted_categories']
    )
    context = (
        f"Case: {summary['case_id']}\n"
        f"Total cells detected: {summary['total_cells']}\n"
        f"Total cells classified: {summary['cells_classified']}\n"
        f"Blast count: {summary['blast_count']} ({summary['blast_fraction']:.1%})\n"
        f"Full morphology distribution: {dist_full}"
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
            "Answer in 2-4 clear sentences using professional clinical language. Do not make definitive diagnostic conclusions without physician review."
        )
        try:
            answer = llm_provider.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=500,
            ).strip()
        except Exception as llm_err:
            print(f"Chat LLM generation failed: {llm_err}; utilizing clinical guideline fallback")
            first_guide = retrieved[0]['text'][:260] if retrieved else "Clinical morphology criteria per WHO 5th Edition."
            answer = (
                f"For {summary['case_id']}, the observed blast proportion is {summary['blast_fraction']:.1%} "
                f"({summary['blast_count']} of {summary['total_cells']} nucleated cells). "
                f"Guideline context: {first_guide}... "
                f"Definitive assessment requires multi-modal integration with flow cytometry, cytogenetics, and molecular testing."
            )

        from pipeline.agent_guard import sanitize_agent_language
        answer = sanitize_agent_language(answer)

        sources_meta = [
            {'id': r['chunk_id'], 'title': r.get('title', r['chunk_id']), 'snippet': r.get('text', '')[:150] + '...'}
            for r in retrieved
        ]

        entry = {'question': question, 'answer': answer, 'sources': sources_meta, 'timestamp': datetime.now().isoformat()}
        chat_history.setdefault(analysis_id, []).append(entry)
        save_chats(chat_history)
        try:
            update_case_chat_in_db(analysis_id, chat_history[analysis_id])
        except Exception as db_e:
            print(f"Dual-write chat to SQLite failed: {db_e}")

        return jsonify({'answer': answer, 'sources': sources_meta})

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
        for aid, entry in analyses.items():
            if entry.get("case_id") == analysis_id:
                analysis_id = aid
                break

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
# Phase 7 Agentic Layer Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/agent-events')
@app.route('/api/agent-events/<case_id>')
def get_agent_events_api(case_id=None):
    """Retrieve logged agent activity events for live feed."""
    try:
        from app.db import get_agent_events_from_db
        events = get_agent_events_from_db(case_id=case_id)
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/triage/queue/<case_or_analysis_id>')
def get_triage_queue_api(case_or_analysis_id):
    """Run Triage Agent to rank cells for human-in-the-loop review."""
    try:
        from pipeline.triage_agent import build_triage_queue
        data = build_triage_queue(case_or_analysis_id)
        return jsonify(data)
    except Exception as e:
        print(f"Triage agent error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/cells/relabel', methods=['POST'])
def relabel_cell():
    """
    Physician explicit action to set or clear a reviewer_label on a detected cell.
    Never overwrites original AI class or confidence.
    """
    body = request.json or {}
    case_or_analysis_id = body.get('case_id') or body.get('analysis_id')
    cell_id = body.get('cell_id')
    reviewer_label = body.get('reviewer_label')

    if not case_or_analysis_id or not cell_id:
        return jsonify({'error': 'Missing case_id or cell_id'}), 400

    target_aid = None
    if case_or_analysis_id in analyses:
        target_aid = case_or_analysis_id
    else:
        for aid, entry in analyses.items():
            if entry.get('case_id') == case_or_analysis_id:
                target_aid = aid
                break

    old_rev_label = None
    ai_label = None
    if target_aid and target_aid in analyses:
        d = analyses[target_aid]
        cell_images = d.get('cell_images', [])
        for c in cell_images:
            cid = c.get('id') or c.get('cell_id')
            if cid == cell_id:
                ai_label = c.get('class') or c.get('predicted_class')
                old_rev_label = c.get('reviewer_label')
                if reviewer_label is None or reviewer_label == '' or reviewer_label == ai_label:
                    c.pop('reviewer_label', None)
                else:
                    c['reviewer_label'] = reviewer_label
                break
        save_analyses(analyses)

    from app.db import update_cell_reviewer_label_in_db, log_agent_event
    update_cell_reviewer_label_in_db(case_or_analysis_id, cell_id, reviewer_label if reviewer_label != ai_label else None)

    log_agent_event(
        case_id=case_or_analysis_id,
        agent="Triage agent",
        tool="relabel_cell",
        input_summary=f"cell={cell_id}, reviewer_label={reviewer_label}",
        output_summary=f"Relabelled {cell_id} ({ai_label}) -> {reviewer_label or ai_label}",
        status="success",
    )

    summary_ai = build_morphology_summary(case_or_analysis_id, use_reviewer_labels=False)
    summary_rev = build_morphology_summary(case_or_analysis_id, use_reviewer_labels=True)

    return jsonify({
        'success': True,
        'cell_id': cell_id,
        'reviewer_label': reviewer_label,
        'ai_summary': summary_ai,
        'reviewer_summary': summary_rev,
    })


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
