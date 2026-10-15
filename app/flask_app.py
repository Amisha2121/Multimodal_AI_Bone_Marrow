"""
Flask Backend API for Bone Marrow Analysis
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.new_patient_engine import analyze_new_case
from retrieval.retrieve import HybridRetriever
from agents.llm_provider import get_llm_provider

# Initialize Flask
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = REPO_ROOT / 'data' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Initialize RAG system
try:
    retriever = HybridRetriever()
    llm_provider = get_llm_provider()
    RAG_AVAILABLE = True
except Exception as e:
    print(f"RAG init failed: {e}")
    retriever = None
    llm_provider = None
    RAG_AVAILABLE = False

# In-memory storage
analyses = {}
chat_history = {}

# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'rag_available': RAG_AVAILABLE
    })

@app.route('/api/upload', methods=['POST'])
def upload_analysis():
    """Upload and analyze bone marrow images"""
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Save files temporarily
    saved_files = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)
            saved_files.append(filepath)
    
    try:
        # Run analysis
        result = analyze_new_case(
            uploaded_files=saved_files,
            age_years=None,
            sex=None,
            sample_site=None
        )
        
        # Generate analysis ID
        analysis_id = hashlib.md5(f"{result.case_id}_{datetime.now()}".encode()).hexdigest()[:12]
        
        # Get all image paths
        detection_images = {}
        if result.image_infos:
            info = result.image_infos[0]
            detection_images = {
                'raw': info.debug_raw_path if hasattr(info, 'debug_raw_path') else None,
                'filtered': info.debug_filtered_path,
                'contact_sheet': info.debug_contact_sheet_path if hasattr(info, 'debug_contact_sheet_path') else None
            }
        
        # Store analysis
        analyses[analysis_id] = {
            'id': analysis_id,
            'case_id': result.case_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': result,
            'cells_count': result.total_cells_detected,
            'blast_pct': result.blast_fraction,
            'morphology': result.morphology_counts,
            'detection_images': detection_images,
            'images': [
                {
                    'filename': info.filename,
                    'cells': info.cells_detected,
                    'filtered_path': info.debug_filtered_path,
                    'raw_path': info.debug_raw_path if hasattr(info, 'debug_raw_path') else None,
                    'contact_sheet': info.debug_contact_sheet_path if hasattr(info, 'debug_contact_sheet_path') else None,
                    'annotated_path': info.annotated_image_path,
                    'width': info.width_px,
                    'height': info.height_px
                }
                for info in result.image_infos
            ],
            'first_image': result.image_infos[0].debug_filtered_path if result.image_infos and result.image_infos[0].debug_filtered_path else None
        }
        
        # Initialize chat for this analysis
        chat_history[analysis_id] = []
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'case_id': result.case_id,
            'cells_detected': result.total_cells_detected,
            'blast_fraction': result.blast_fraction
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses')
def get_analyses():
    """Get all analyses"""
    return jsonify({
        'analyses': [
            {
                'id': aid,
                'case_id': data['case_id'],
                'timestamp': data['timestamp'],
                'cells': data['cells_count'],
                'blasts': data['blast_pct'],
                'thumbnail': data.get('first_image')
            }
            for aid, data in analyses.items()
        ]
    })

@app.route('/api/analysis/<analysis_id>')
def get_analysis(analysis_id):
    """Get specific analysis details"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
    data = analyses[analysis_id]
    result = data['analysis']
    
    return jsonify({
        'id': analysis_id,
        'case_id': data['case_id'],
        'timestamp': data['timestamp'],
        'metrics': {
            'images_processed': result.total_images_processed,
            'cells_detected': result.total_cells_detected,
            'cells_classified': result.cells_classified,
            'blast_fraction': result.blast_fraction
        },
        'morphology': [
            {
                'type': cell_type,
                'count': count,
                'percentage': result.morphology_percentages.get(cell_type, 0)
            }
            for cell_type, count in result.morphology_counts.items()
        ],
        'detection_images': data.get('detection_images', {}),
        'images': data['images']
    })

@app.route('/api/cells/<analysis_id>')
def get_cells(analysis_id):
    """Get all cells for an analysis"""
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
    result = analyses[analysis_id]['analysis']
    
    cells = [
        {
            'id': cell.cell_id,
            'class': cell.predicted_class,
            'confidence': cell.confidence,
            'image_path': str(cell.image_path) if cell.image_path else None,
            'bbox': cell.bbox
        }
        for cell in result.cell_predictions
    ]
    
    return jsonify({'cells': cells})

@app.route('/api/analysis-report/<analysis_id>')
def get_analysis_report(analysis_id):
    """Generate RAG-based analysis report"""
    if not RAG_AVAILABLE:
        return jsonify({'error': 'RAG system unavailable'}), 503
    
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        result = analyses[analysis_id]['analysis']
        
        # Build comprehensive analysis using RAG
        sections = []
        
        # Overview section
        overview_query = f"Bone marrow cytomorphology interpretation with {result.blast_fraction:.1%} blasts"
        overview_retrieved = retriever.retrieve(query=overview_query, top_k=2)
        overview_evidence = "\n".join([r['text'][:400] for r in overview_retrieved])
        
        overview_prompt = f"""Generate a clinical overview paragraph for this bone marrow sample:
        
Sample: {result.case_id}
Total cells: {result.total_cells_detected}
Blast fraction: {result.blast_fraction:.1%}
Distribution: {', '.join([f'{k}: {v}' for k, v in list(result.morphology_counts.items())[:5]])}

Guidelines:
{overview_evidence}

Write a clinical paragraph (4-5 sentences) describing the overall findings."""
        
        overview_text = llm_provider.generate(
            [{"role": "user", "content": overview_prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        sections.append({
            'title': 'Overview',
            'content': overview_text,
            'cell_images': []
        })
        
        # Cell-specific sections
        for cell_type, count in sorted(result.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = result.morphology_percentages.get(cell_type, 0)
            
            # Retrieve evidence for this cell type
            cell_query = f"Cytomorphology and clinical significance of {cell_type} in bone marrow"
            cell_retrieved = retriever.retrieve(query=cell_query, top_k=2)
            cell_evidence = "\n".join([r['text'][:350] for r in cell_retrieved])
            
            # Generate cell-specific analysis
            cell_prompt = f"""Describe the {cell_type} cells found in this bone marrow sample:

Observations:
- {count} cells detected ({pct:.1%} of total)
- Sample: {result.case_id}

Clinical guidelines:
{cell_evidence if cell_evidence else "Standard bone marrow cell evaluation"}

Write a clinical paragraph (3-4 sentences) about the significance of these cells."""
            
            cell_text = llm_provider.generate(
                [{"role": "user", "content": cell_prompt}],
                temperature=0.1,
                max_tokens=250
            )
            
            # Get sample cell images of this type
            cell_images = [
                str(cell.image_path) for cell in result.cell_predictions
                if cell.predicted_class == cell_type and cell.image_path
            ][:4]  # Max 4 images per section
            
            sections.append({
                'title': f'{cell_type} ({count} cells, {pct:.1%})',
                'content': cell_text,
                'cell_images': cell_images
            })
        
        # Clinical interpretation
        interp_query = f"Clinical interpretation bone marrow {result.blast_fraction:.1%} blasts differential diagnosis"
        interp_retrieved = retriever.retrieve(query=interp_query, top_k=2)
        interp_evidence = "\n".join([r['text'][:350] for r in interp_retrieved])
        
        interp_prompt = f"""Provide clinical interpretation and recommendations:

Findings: {result.total_cells_detected} cells, {result.blast_fraction:.1%} blasts
Distribution: {', '.join([f'{k}: {v}' for k, v in list(result.morphology_counts.items())[:3]])}

Guidelines:
{interp_evidence}

Write a clinical interpretation paragraph (3-4 sentences) with recommendations for additional testing."""
        
        interp_text = llm_provider.generate(
            [{"role": "user", "content": interp_prompt}],
            temperature=0.1,
            max_tokens=250
        )
        
        sections.append({
            'title': 'Clinical Interpretation',
            'content': interp_text,
            'cell_images': []
        })
        
        return jsonify({
            'sections': sections,
            'case_id': result.case_id,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Ask question about analysis using RAG"""
    if not RAG_AVAILABLE:
        return jsonify({'error': 'RAG system unavailable'}), 503
    
    data = request.json
    analysis_id = data.get('analysis_id')
    question = data.get('question')
    
    if not analysis_id or not question:
        return jsonify({'error': 'Missing analysis_id or question'}), 400
    
    if analysis_id not in analyses:
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        result = analyses[analysis_id]['analysis']
        
        # Build context
        context = f"""Sample: {result.case_id}
Cells: {result.total_cells_detected}
Blasts: {result.blast_fraction:.1%}
Distribution: {', '.join([f'{k}: {v}' for k, v in list(result.morphology_counts.items())[:3]])}"""
        
        # Retrieve evidence
        retrieved = retriever.retrieve(query=question, top_k=2)
        evidence = "\n".join([f"{r['text'][:250]}" for r in retrieved])
        
        # Generate answer
        prompt = f"""{context}

Guidelines: {evidence if evidence else "None"}

Question: {question}

Provide a brief clinical answer (2-3 sentences):"""
        
        answer = llm_provider.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        # Store in chat history
        chat_entry = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        chat_history[analysis_id].append(chat_entry)
        
        return jsonify({
            'answer': answer,
            'sources': [r['chunk_id'] for r in retrieved]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/<analysis_id>')
def get_chat_history(analysis_id):
    """Get chat history for an analysis"""
    if analysis_id not in chat_history:
        return jsonify({'history': []})
    
    return jsonify({'history': chat_history[analysis_id]})

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    """Serve analysis images"""
    # Try multiple possible directories
    possible_paths = [
        REPO_ROOT / filename,
        REPO_ROOT / 'data' / 'outputs' / filename,
        Path(filename)
    ]
    
    for path in possible_paths:
        if path.exists():
            return send_from_directory(path.parent, path.name)
    
    # If not found, return 404
    return jsonify({'error': 'Image not found'}), 404

# ═══════════════════════════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
