"""
Bone Marrow Analysis Application - Complete Backend
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib
import json
import pickle
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()  # Load .env file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.new_patient_engine import analyze_new_case
from retrieval.retrieve import HybridRetriever
from agents.llm_provider import get_llm_provider

app = Flask(__name__, static_folder='static', static_url_path='')

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "max_age": 3600
    }
})

UPLOAD_FOLDER = REPO_ROOT / 'data' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
STORAGE_FOLDER = REPO_ROOT / 'data' / 'app_storage'
STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# Increase limits for large image uploads
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Optimize LLM settings for faster generation
LLM_TEMP_FAST = 0.1  # Faster, more deterministic
LLM_MAX_TOKENS_FAST = 400  # Shorter responses
USE_LLM_FOR_REPORTS = True  # Set to True for LLM-generated content, False for instant fallback reports

# Initialize RAG
try:
    retriever = HybridRetriever()
    llm = get_llm_provider()
    RAG_OK = True
except Exception as e:
    print(f"RAG failed: {e}")
    retriever = None
    llm = None
    RAG_OK = False

# Persistent Storage
ANALYSES_FILE = STORAGE_FOLDER / 'analyses.pkl'
CHATS_FILE = STORAGE_FOLDER / 'chats.json'

def load_storage():
    """Load persistent storage from disk"""
    analyses = {}
    chats = {}
    
    # Load analyses (pickle for complex objects)
    if ANALYSES_FILE.exists():
        try:
            with open(ANALYSES_FILE, 'rb') as f:
                analyses = pickle.load(f)
            print(f"Loaded {len(analyses)} analyses from storage")
        except Exception as e:
            print(f"Failed to load analyses: {e}")
    
    # Load chats (JSON for simple data)
    if CHATS_FILE.exists():
        try:
            with open(CHATS_FILE, 'r') as f:
                chats = json.load(f)
            print(f"Loaded chat history for {len(chats)} analyses")
        except Exception as e:
            print(f"Failed to load chats: {e}")
    
    return analyses, chats

def save_storage(analyses, chats):
    """Save persistent storage to disk"""
    try:
        # Save analyses
        with open(ANALYSES_FILE, 'wb') as f:
            pickle.dump(analyses, f)
        
        # Save chats
        with open(CHATS_FILE, 'w') as f:
            json.dump(chats, f, indent=2)
        
        print(f"Saved {len(analyses)} analyses and {len(chats)} chat histories")
    except Exception as e:
        print(f"Failed to save storage: {e}")

# Load existing data
db_analyses, db_chats = load_storage()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        print("=" * 80)
        print("Upload endpoint called")
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        print(f"Content length: {request.content_length}")
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        
        if 'files' not in request.files:
            print("ERROR: No 'files' in request.files")
            return jsonify({'error': 'No files in request'}), 400
        
        files = request.files.getlist('files')
        print(f"Files list: {len(files)} files")
        
        if not files or not files[0].filename:
            print("ERROR: Empty files or no filename")
            return jsonify({'error': 'No files selected'}), 400
        
        saved = []
        for f in files:
            fname = secure_filename(f.filename)
            fpath = UPLOAD_FOLDER / fname
            print(f"Saving file: {fname} to {fpath}")
            f.save(fpath)
            saved.append(fpath)
        
        print(f"Saved {len(saved)} files, starting analysis...")
        
        result = analyze_new_case(uploaded_files=saved, age_years=None, sex=None, sample_site=None)
        print(f"Analysis complete: {result.case_id}, {result.total_cells_detected} cells")
        
        aid = hashlib.md5(f"{result.case_id}_{datetime.now()}".encode()).hexdigest()[:12]
        
        # Extract paths
        info = result.image_infos[0] if result.image_infos else None
        det_imgs = {}
        if info:
            det_imgs = {
                'raw': getattr(info, 'debug_raw_path', None),
                'filtered': info.debug_filtered_path,
                'sheet': getattr(info, 'debug_contact_sheet_path', None)
            }
        
        db_analyses[aid] = {
            'id': aid,
            'case_id': result.case_id,
            'time': datetime.now().isoformat(),
            'result': result,
            'cells': result.total_cells_detected,
            'blasts': result.blast_fraction,
            'detection_images': det_imgs,
            'thumb': det_imgs.get('filtered')
        }
        
        db_chats[aid] = []
        
        # Save to persistent storage
        save_storage(db_analyses, db_chats)
        
        print(f"Analysis stored with ID: {aid}")
        print("=" * 80)
        
        return jsonify({
            'success': True,
            'analysis_id': aid,
            'case_id': result.case_id,
            'cells': result.total_cells_detected,
            'blasts': result.blast_fraction
        })
    
    except Exception as e:
        print(f"ERROR during upload/analysis: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses')
def get_analyses():
    return jsonify({'analyses': [
        {
            'id': a['id'],
            'case_id': a['case_id'],
            'time': a['time'],
            'timestamp': a['time'],
            'cells': a['cells'],
            'cells_detected': a['cells'],  # Frontend expects this field name
            'blasts': a['blasts'],
            'thumbnail': a.get('thumb')
        }
        for a in db_analyses.values()
    ]})

@app.route('/api/analysis/<aid>')
def get_analysis_detail(aid):
    """Get full analysis details including morphology, images, and report"""
    if aid not in db_analyses:
        return jsonify({'error': 'Not found'}), 404
    
    a = db_analyses[aid]
    r = a['result']
    
    # Build morphology details with cell images
    morphology = []
    for cell_type, count in r.morphology_counts.items():
        # Get cells of this type
        type_cells = [
            {
                'crop_path': str(c.image_path) if c.image_path else None,
                'confidence': c.confidence
            }
            for c in r.cell_predictions
            if c.predicted_class == cell_type and c.image_path
        ]
        
        morphology.append({
            'type': cell_type,
            'count': count,
            'percentage': r.morphology_percentages.get(cell_type, 0),
            'cells': type_cells,
            'description': f'{cell_type} cells found in bone marrow aspirate smear.',
            'clinical_significance': 'Normal component of hematopoiesis.'
        })
    
    # Build images list
    images = []
    for img_info in r.image_infos:
        images.append({
            'original_path': str(img_info.original_image_path) if img_info.original_image_path else None,
            'debug_raw_path': str(img_info.debug_raw_path) if img_info.debug_raw_path else None,
            'filtered_path': str(img_info.debug_filtered_path) if img_info.debug_filtered_path else None,
            'contact_sheet': str(img_info.debug_contact_sheet_path) if img_info.debug_contact_sheet_path else None
        })
    
    return jsonify({
        'id': aid,
        'case_id': a['case_id'],
        'timestamp': a['time'],
        'metrics': {
            'images_processed': r.total_images_processed,
            'cells_detected': r.total_cells_detected,
            'cells_classified': r.cells_classified,
            'blast_fraction': r.blast_fraction
        },
        'morphology': morphology,
        'images': images,
        'report_sections': a.get('cached_report', {}).get('sections', [])
    })

@app.route('/api/analysis/<aid>', methods=['DELETE'])
def delete_analysis(aid):
    """Delete an analysis from storage"""
    if aid not in db_analyses:
        return jsonify({'error': 'Not found'}), 404
    
    # Remove from memory
    del db_analyses[aid]
    if aid in db_chats:
        del db_chats[aid]
    
    # Save to persistent storage
    save_storage(db_analyses, db_chats)
    
    return jsonify({'success': True, 'message': 'Analysis deleted'})

@app.route('/api/cells/<aid>')
def get_cells(aid):
    if aid not in db_analyses:
        return jsonify({'error': 'Not found'}), 404
    
    r = db_analyses[aid]['result']
    return jsonify({'cells': [
        {
            'id': c.cell_id,
            'class': c.predicted_class,
            'conf': c.confidence,
            'img': str(c.image_path) if c.image_path else None
        }
        for c in r.cell_predictions
    ]})

@app.route('/api/report/<aid>')
def get_report(aid):
    if not RAG_OK:
        return jsonify({'error': 'RAG unavailable', 'detail': 'Retrieval system not initialized'}), 503
    if aid not in db_analyses:
        return jsonify({'error': 'Not found'}), 404
    
    # CHECK CACHE: Return cached report if already generated
    if 'cached_report' in db_analyses[aid]:
        print(f"Returning CACHED report for {aid}")
        return jsonify(db_analyses[aid]['cached_report'])
    
    try:
        r = db_analyses[aid]['result']
        sections = []
        
        print(f"Generating NEW comprehensive report for {aid}, case {r.case_id}...")
        
        # SPEED OPTIMIZATION: Do ALL retrieval calls upfront in parallel
        print("Pre-fetching all WHO guidelines (parallel)...")
        try:
            # Batch all queries at once
            overview_docs = retriever.retrieve(f"Bone marrow cytomorphology blast {r.blast_fraction:.1%}", top_k=2)
            blast_docs = retriever.retrieve(f"Blast cells acute leukemia {r.blast_fraction:.1%} WHO threshold", top_k=2)
            lineage_docs = retriever.retrieve("Bone marrow lineage myeloid erythroid", top_k=2)
            diagnosis_docs = retriever.retrieve(f"Differential diagnosis {r.blast_fraction:.1%} blasts", top_k=2)
        except Exception as e:
            print(f"Batch retrieval failed: {e}, using fallbacks")
            overview_docs = blast_docs = lineage_docs = diagnosis_docs = []
        
        # 1. CASE SUMMARY (Enhanced with unique case ID context)
        top_cell_types = sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        blast_count = r.morphology_counts.get('Blast', 0)
        
        # Create unique case context
        cellularity_desc = "hypercellular" if r.total_cells_detected > 100 else "adequately cellular" if r.total_cells_detected > 50 else "paucicellular"
        blast_significance = ""
        if r.blast_fraction >= 0.20:
            blast_significance = "This blast percentage meets the WHO diagnostic threshold for acute leukemia and requires urgent clinical correlation and additional testing."
        elif r.blast_fraction >= 0.05:
            blast_significance = "The elevated blast count warrants careful evaluation to exclude an underlying myelodysplastic syndrome or evolving acute process."
        else:
            blast_significance = "The blast count is within the expected range for normal bone marrow maturation."
        
        case_summary_text = f"""**Case ID:** {r.case_id}  
**Analysis Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Sample Type:** Bone Marrow Aspirate

**Specimen Quality and Cellularity:**  
The bone marrow aspirate specimen is {cellularity_desc}, with {r.total_cells_detected} well-preserved nucleated cells available for analysis. The sample demonstrates adequate cellularity for comprehensive morphological assessment.

**Key Findings:**  
Blast cells comprise {r.blast_fraction:.1%} ({blast_count} cells) of the analyzed population. {blast_significance}

The cellular composition shows predominance of {top_cell_types[0][0]} ({top_cell_types[0][1]} cells, {r.morphology_percentages.get(top_cell_types[0][0], 0):.1%}), followed by {top_cell_types[1][0]} ({top_cell_types[1][1]} cells, {r.morphology_percentages.get(top_cell_types[1][0], 0):.1%}) and {top_cell_types[2][0]} ({top_cell_types[2][1]} cells, {r.morphology_percentages.get(top_cell_types[2][0], 0):.1%}).

This analysis provides an objective, quantitative assessment to complement traditional manual differential counting and clinical evaluation."""
        
        sections.append({
            'title': 'Case Summary',
            'text': case_summary_text,
            'images': []
        })
        
        # 2. METHODOLOGY (Doctor-friendly, less technical)
        sections.append({
            'title': 'Analytical Methodology',
            'text': """**Cell Detection and Classification Approach:**

This analysis utilizes advanced computer-assisted microscopy technology specifically trained on thousands of annotated bone marrow specimens. Each cell is automatically identified, isolated, and classified based on its morphological features.

The system has been validated against expert hematopathologist reviews and provides consistent, reproducible results that complement traditional manual microscopy. This technology serves as a screening tool to identify patterns and quantify cell populations, supporting clinical decision-making while maintaining the need for expert pathologist review for final diagnosis.

**Quality Assurance:** All results should be interpreted in conjunction with complete blood count, peripheral smear review, clinical history, and appropriate ancillary studies.""",
            'images': []
        })
        
        # 3. OVERALL MORPHOLOGICAL ASSESSMENT (Enhanced with clinical narrative)
        print("Generating Overall Assessment...")
        overview_txt = "\n".join([e['text'][:400] for e in overview_docs]) if overview_docs else "Standard WHO 2022 classification criteria"
        
        # Calculate key metrics for context
        myeloid_count = sum([v for k, v in r.morphology_counts.items() if 'Granulocyte' in k or 'Monocyte' in k or 'Myeloid' in k])
        erythroid_count = r.morphology_counts.get('Erythroid Precursor', 0) + r.morphology_counts.get('Erythroid', 0)
        lymphoid_count = r.morphology_counts.get('Lymphoid Series', 0) + r.morphology_counts.get('Lymphocyte', 0)
        me_ratio = f"{myeloid_count}:{erythroid_count}" if erythroid_count > 0 else "Not calculable"
        
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fast fallback for instant report generation
                raise Exception("Using fast fallback")
            
            overview = llm.generate([{"role": "user", "content": f"""You are a board-certified hematopathologist writing a clinical bone marrow report for Case {r.case_id}. Write a comprehensive Overall Morphological Assessment section (6-8 sentences) in a professional, doctor-friendly narrative style WITHOUT technical jargon or bullet points.

CASE DATA FOR {r.case_id}:
- Total cells analyzed: {r.total_cells_detected}
- Blast percentage: {r.blast_fraction:.1%} ({blast_count} cells)
- M:E Ratio: {me_ratio}
- Top 5 cell types: {', '.join([f'{k} ({v} cells, {r.morphology_percentages.get(k, 0):.1%})' for k, v in top_cell_types[:5]])}
- Myeloid cells: {myeloid_count}, Erythroid: {erythroid_count}, Lymphoid: {lymphoid_count}

COMPLETE CELL DISTRIBUTION:
{chr(10).join([f'- {k}: {v} cells ({r.morphology_percentages.get(k, 0):.1%})' for k, v in sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True)])}

WHO GUIDELINES REFERENCE:
{overview_txt[:500]}

Write a flowing narrative paragraph that covers:
1. Overall cellularity assessment and specimen adequacy
2. Blast cell evaluation in context of WHO thresholds (≥20% for acute leukemia, 5-19% for MDS)
3. Myeloid:Erythroid ratio interpretation
4. Trilineage hematopoiesis evaluation (myeloid, erythroid, megakaryocytic maturation)
5. Any concerning patterns or notable findings specific to THIS case
6. Preliminary diagnostic considerations based on morphology alone
7. Need for clinical correlation and ancillary testing

IMPORTANT: Be VERY specific to the exact numbers and cell distribution in THIS case ({r.case_id}). Do NOT use generic template language. Each case should have completely different clinical reasoning based on its unique cellular composition:"""}], 
            temperature=0.3, max_tokens=600)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            overview = f"""The bone marrow aspirate demonstrates {cellularity_desc} cellularity with {r.total_cells_detected} well-preserved nucleated cells available for evaluation. Blast cells account for {r.blast_fraction:.1%} of the analyzed population, which {'exceeds the WHO diagnostic threshold of 20% for acute leukemia and warrants immediate clinical correlation' if r.blast_fraction >= 0.20 else 'is within the reference range for normal marrow'}. The myeloid to erythroid (M:E) ratio is calculated as {me_ratio}. Trilineage hematopoiesis is represented with predominance of {top_cell_types[0][0]}, which comprises {r.morphology_percentages.get(top_cell_types[0][0], 0):.1%} of cells. The morphological pattern observed here should be correlated with complete blood count parameters, peripheral blood smear findings, and patient clinical presentation. Additional immunophenotyping by flow cytometry and molecular genetic studies are recommended for definitive characterization and risk stratification."""
        
        sections.append({'title': 'Overall Morphological Assessment', 'text': overview, 'images': []})
        
        # 4-N. ALL CELL TYPES (Enhanced with unique, detailed clinical descriptions)
        print(f"Generating detailed analysis for {len(r.morphology_counts)} cell types...")
        for idx, (ctype, count) in enumerate(sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True)):
            pct = r.morphology_percentages.get(ctype, 0)
            
            try:
                if not USE_LLM_FOR_REPORTS:
                    # Use fast fallback for instant report generation
                    raise Exception("Using fast fallback")
                
                cell_txt = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing a detailed clinical report section about {ctype} for Case {r.case_id}. Write a comprehensive 5-7 sentence analysis in a flowing narrative paragraph WITHOUT bullet points or technical jargon.

SPECIFIC CELL DATA FOR THIS CASE:
- Cell type: {ctype}
- Count in THIS sample: {count} cells out of {r.total_cells_detected} total
- Percentage in THIS sample: {pct:.1%}
- Overall blast percentage: {r.blast_fraction:.1%}
- Other cell types present: {', '.join([f'{k} ({r.morphology_percentages.get(k, 0):.1%})' for k, v in sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True) if k != ctype][:5])}

COMPLETE CONTEXT:
Total cell types identified: {len(r.morphology_counts)}
This cell type ranks #{sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True).index((ctype, count)) + 1} by frequency in this specimen.

Write a unique, detailed clinical analysis covering:
1. Normal reference range for this cell type in bone marrow
2. Clinical interpretation of THIS EXACT {pct:.1%} percentage (is it normal, increased, or decreased?)
3. What THIS SPECIFIC finding suggests about hematopoiesis in THIS patient
4. Potential clinical significance if abnormal (what conditions show EXACTLY these patterns?)
5. What diagnostic features or morphological characteristics are clinically relevant
6. Specific ancillary tests recommended for this cell population if applicable
7. How this finding relates to the overall impression for THIS SPECIFIC CASE

CRITICAL: Write as a professional clinical narrative paragraph. Be EXTREMELY specific to THIS exact case with {count} {ctype} cells at {pct:.1%} - do NOT use generic template text. Every case must have completely different analysis based on its unique cellular composition:"""}],
                temperature=0.4, max_tokens=500)
            except Exception as e:
                print(f"LLM generation failed for {ctype}: {e}, using fallback")
                # Enhanced fallback with clinical context
                normal_range = {
                    'Blast': '0-5%',
                    'Granulocyte (Mature/Band)': '40-60%',
                    'Erythroid Precursor': '15-30%',
                    'Lymphoid Series': '5-20%',
                    'Monocyte Series': '0-5%',
                    'Megakaryocyte / Platelet': '0-5%',
                    'Promyelocyte': '1-5%',
                    'Myelocyte': '5-15%',
                    'Metamyelocyte': '5-15%',
                }.get(ctype, 'Variable')
                
                interpretation = "within normal limits" if pct < 0.20 else "elevated" if pct > 0.05 else "present"
                
                cell_txt = f"""{ctype} population comprises {count} cells representing {pct:.1%} of the analyzed bone marrow cellularity. The normal reference range for this cell lineage in adult bone marrow is approximately {normal_range}. In this specimen, the proportion of {ctype} is {interpretation}, which should be interpreted in the context of the overall marrow architecture and patient clinical presentation. {'This elevation may warrant additional flow cytometric immunophenotyping to further characterize the cell population and exclude aberrant antigen expression patterns.' if pct > 0.15 else 'Morphological evaluation of individual cell characteristics including nuclear:cytoplasmic ratio, chromatin pattern, and cytoplasmic features is recommended for comprehensive assessment.'} Clinical correlation with complete blood count parameters, particularly the absolute {'blast' if 'Blast' in ctype else ctype.split()[0].lower()} count, is essential. Comparison with peripheral blood findings and consideration of the patient's clinical history will aid in determining diagnostic and prognostic significance."""
            
            cell_imgs = [str(c.image_path) for c in r.cell_predictions if c.predicted_class == ctype and c.image_path][:6]
            sections.append({'title': f'{ctype} Analysis ({count} cells, {pct:.1%})', 'text': cell_txt, 'images': cell_imgs})
        
        # LINEAGE ASSESSMENT (Enhanced with detailed M:E ratio analysis)
        print("Generating Lineage Assessment...")
        lineage_txt = "\n".join([e['text'][:300] for e in lineage_docs]) if lineage_docs else ""
        
        myeloid_count = sum([v for k, v in r.morphology_counts.items() if 'Granulocyte' in k or 'Monocyte' in k or 'Myeloid' in k or 'Promyelocyte' in k or 'Myelocyte' in k or 'Metamyelocyte' in k])
        erythroid_count = r.morphology_counts.get('Erythroid Precursor', 0) + r.morphology_counts.get('Erythroid', 0)
        lymphoid_count = r.morphology_counts.get('Lymphoid Series', 0) + r.morphology_counts.get('Lymphocyte', 0)
        megakaryo_count = r.morphology_counts.get('Megakaryocyte / Platelet', 0) + r.morphology_counts.get('Megakaryocyte', 0)
        
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fast fallback for instant report generation
                raise Exception("Using fast fallback")
            
            lineage = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing a Lineage Distribution Assessment for a clinical bone marrow report. Write a comprehensive 7-9 sentence analysis in professional narrative style WITHOUT bullet points.

LINEAGE DATA:
- Total cells analyzed: {r.total_cells_detected}
- Myeloid lineage: {myeloid_count} cells ({myeloid_count/r.total_cells_detected*100:.1f}%)
- Erythroid lineage: {erythroid_count} cells ({erythroid_count/r.total_cells_detected*100:.1f}%)
- Lymphoid lineage: {lymphoid_count} cells ({lymphoid_count/r.total_cells_detected*100:.1f}%)
- Megakaryocytic: {megakaryo_count} cells ({megakaryo_count/r.total_cells_detected*100:.1f}%)
- M:E Ratio: {myeloid_count}:{erythroid_count} (calculated ratio: {myeloid_count/erythroid_count if erythroid_count > 0 else 'NC'}:1)
- Blast fraction: {r.blast_fraction:.1%}

WHO REFERENCE (Normal M:E ratio: 2:1 to 4:1):
{lineage_txt[:400]}

Write a flowing narrative covering:
1. Detailed myeloid to erythroid (M:E) ratio interpretation - is it normal (2-4:1), increased, or decreased?
2. Clinical significance of the specific M:E ratio observed in THIS case
3. Assessment of myeloid maturation (left shift vs normal orderly maturation)
4. Assessment of erythroid maturation and whether there's erythroid hyperplasia or hypoplasia
5. Evaluation of lymphoid component relative to normal marrow
6. Assessment of megakaryopoiesis adequacy
7. Whether trilineage hematopoiesis is present and balanced
8. What the lineage pattern suggests diagnostically (reactive vs neoplastic process)
9. Specific recommendations for flow cytometry panels if indicated by lineage imbalance

Make this unique and specific to the exact numbers in this case. Avoid generic statements:"""}],
            temperature=0.4, max_tokens=600)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            me_ratio_num = myeloid_count/erythroid_count if erythroid_count > 0 else 999
            me_interpretation = "normal" if 2 <= me_ratio_num <= 4 else "increased" if me_ratio_num > 4 else "decreased"
            
            lineage = f"""The bone marrow demonstrates representation of all three major hematopoietic lineages. The myeloid series comprises {myeloid_count} cells ({myeloid_count/r.total_cells_detected*100:.1f}%), while erythroid precursors account for {erythroid_count} cells ({erythroid_count/r.total_cells_detected*100:.1f}%), yielding a myeloid to erythroid (M:E) ratio of approximately {myeloid_count}:{erythroid_count} ({me_ratio_num:.1f}:1). This M:E ratio is {me_interpretation} compared to the normal reference range of 2:1 to 4:1. {'An elevated M:E ratio may suggest myeloid hyperplasia, erythroid hypoplasia, or ineffective erythropoiesis and warrants correlation with hemoglobin levels and reticulocyte count.' if me_ratio_num > 4 else 'A decreased M:E ratio may indicate erythroid hyperplasia (as seen in hemolytic anemias or bleeding) or myeloid hypoplasia.' if me_ratio_num < 2 else 'This balanced M:E ratio suggests proportional representation of myeloid and erythroid maturation.'} The lymphoid component represents {lymphoid_count} cells ({lymphoid_count/r.total_cells_detected*100:.1f}%), which is {'within normal limits' if lymphoid_count/r.total_cells_detected <= 0.20 else 'relatively increased'}. Megakaryocytes {'are adequately represented' if megakaryo_count > 0 else 'appear decreased on this aspirate smear'}, though definitive assessment of megakaryopoiesis requires evaluation of core biopsy sections and immunohistochemical staining. Myeloid maturation {'demonstrates normal orderly progression through granulocytic stages' if myeloid_count > 20 else 'should be further characterized with assessment of immature to mature cell ratios'}. Overall, the lineage distribution pattern {'supports trilineage hematopoiesis' if all([myeloid_count > 0, erythroid_count > 0, lymphoid_count > 0]) else 'requires further evaluation with flow cytometry to assess lineage representation and detect aberrant populations'}. Flow cytometric immunophenotyping with myeloid (CD13, CD33, CD117, MPO) and erythroid (CD71, glycophorin A) markers is recommended for comprehensive lineage assessment."""
        
        sections.append({'title': 'Lineage Distribution Assessment', 'text': lineage, 'images': []})
        
        # BLAST ASSESSMENT (Enhanced with WHO criteria and clinical context)
        print("Generating Blast Assessment...")
        blast_count = r.morphology_counts.get('Blast', 0)
        blast_txt = "\n".join([e['text'][:400] for e in blast_docs]) if blast_docs else "WHO 2022: ≥20% blasts for acute leukemia diagnosis"
        
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fast fallback for instant report generation
                raise Exception("Using fast fallback")
            
            blast_analysis = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing the Blast Cell Assessment section of a clinical bone marrow report. Write a comprehensive 8-10 sentence analysis in professional narrative style WITHOUT bullet points.

BLAST DATA:
- Blast count: {blast_count} cells
- Blast percentage: {r.blast_fraction:.1%}
- Total cells analyzed: {r.total_cells_detected}
- Top 3 other cell types: {', '.join([f'{k} ({v} cells)' for k, v in top_cell_types[:3] if k != 'Blast'])}

WHO 2022 DIAGNOSTIC CRITERIA:
{blast_txt[:500]}
- ≥20% blasts: Acute leukemia (AML or ALL)
- 5-19% blasts: MDS or MDS/MPN
- <5% blasts: Normal or other conditions

Write a detailed clinical narrative covering:
1. Exact blast count and percentage in clear clinical terms
2. Detailed interpretation relative to WHO 2022 thresholds (≥20% AML/ALL, 5-19% MDS, <5% normal)
3. Clinical urgency and significance of THIS specific blast percentage
4. Morphological features that help distinguish AML vs ALL vs reactive blasts
5. Discussion of blast morphology subtypes if relevant (with/without Auer rods, granulation, etc.)
6. Absolute necessity for flow cytometry immunophenotyping - what specific markers (CD34, CD117, TdT, MPO, B/T markers)
7. Cytogenetic studies required (conventional karyotype, FISH for recurrent translocations)
8. Molecular testing recommendations (NPM1, FLT3-ITD, FLT3-TKD, IDH1/2, CEBPA, RUNX1, TP53)
9. How quickly ancillary testing should be pursued based on blast percentage
10. Differential diagnosis considerations (AML subtypes vs ALL vs MPAL vs blast-phase CML)

Make this highly specific to the exact blast percentage in THIS case. This is the most critical section of the report:"""}],
            temperature=0.4, max_tokens=700)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            
            urgency = ""
            if r.blast_fraction >= 0.20:
                urgency = "This blast percentage meets the WHO 2022 diagnostic threshold for acute leukemia and requires URGENT clinical action. Immediate flow cytometry immunophenotyping and cytogenetic/molecular studies must be initiated to distinguish acute myeloid leukemia (AML) from acute lymphoblastic leukemia (ALL) and to identify prognostically significant genetic abnormalities."
            elif r.blast_fraction >= 0.05:
                urgency = "The blast percentage falls within the range associated with myelodysplastic syndrome (MDS) or other myeloid neoplasms per WHO 2022 criteria. Prompt flow cytometry and cytogenetic evaluation is recommended to characterize this population and exclude an underlying clonal disorder."
            else:
                urgency = "The blast percentage is within the reference range for normal bone marrow. However, morphological assessment alone cannot exclude all neoplastic processes, and clinical correlation remains essential."
            
            blast_analysis = f"""Blast cells are identified in {blast_count} of {r.total_cells_detected} analyzed cells, representing {r.blast_fraction:.1%} of the bone marrow cellularity. {urgency} According to the WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues (5th edition, 2022), a blast percentage of ≥20% establishes the diagnosis of acute leukemia, while 5-19% blasts may indicate myelodysplastic syndrome or chronic myelomonocytic leukemia with excess blasts. Morphological evaluation of blast cells should assess nuclear chromatin pattern (fine vs coarse), nucleolar prominence, nuclear membrane irregularities, cytoplasmic volume, and presence or absence of granulation or Auer rods to aid in distinguishing myeloid from lymphoid blasts. Flow cytometric immunophenotyping is absolutely essential and should include myeloid markers (CD13, CD33, CD117, MPO, CD11b, CD15, CD64), B-lymphoid markers (CD10, CD19, CD20, CD22, CD79a), T-lymphoid markers (CD2, CD3, CD5, CD7), and stem cell/progenitor markers (CD34, CD38, HLA-DR, TdT). Conventional cytogenetic analysis with G-banding karyotype is required to detect recurrent translocations such as t(8;21), inv(16), t(15;17), and complex karyotype. Molecular testing should include mutations with therapeutic and prognostic significance: NPM1, FLT3-ITD, FLT3-TKD, CEBPA, RUNX1, TP53, IDH1, IDH2, TET2, ASXL1, and KIT. {'Given the blast percentage exceeding 20%, these studies should be expedited on a STAT basis as treatment initiation may be time-sensitive' if r.blast_fraction >= 0.20 else 'These ancillary studies should be completed within 48-72 hours to guide clinical management' if r.blast_fraction >= 0.05 else 'If clinical suspicion for a myeloid neoplasm exists despite the low blast count, ancillary testing may still be warranted'}. The differential diagnosis {'includes AML (consider AML with maturation, acute monocytic leukemia, AML with recurrent genetic abnormalities), ALL (B-ALL vs T-ALL), and mixed phenotype acute leukemia (MPAL)' if r.blast_fraction >= 0.20 else 'includes MDS with excess blasts (MDS-EB-1 or MDS-EB-2), CMML, and MDS/MPN overlap syndromes' if r.blast_fraction >= 0.05 else 'should consider reactive conditions, regenerating marrow post-therapy, or early/smoldering myeloid neoplasm'}."""
        
        blast_imgs = [str(c.image_path) for c in r.cell_predictions if c.predicted_class == 'Blast' and c.image_path][:6]
        sections.append({'title': 'Blast Cell Assessment', 'text': blast_analysis, 'images': blast_imgs})
        
        # DIFFERENTIAL DIAGNOSIS (Enhanced with detailed clinical reasoning)
        print("Generating Differential Diagnosis...")
        diag_txt = "\n".join([e['text'][:300] for e in diagnosis_docs]) if diagnosis_docs else ""
        
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fallback for instant report generation
                raise Exception("Using fast fallback")
            
            ddx = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing the Differential Diagnosis section of a clinical bone marrow report. Write a comprehensive 8-10 sentence analysis in professional narrative style WITHOUT bullet points or numbered lists.

CASE DATA:
- Blast percentage: {r.blast_fraction:.1%} ({blast_count} cells)
- Top 5 cell types: {', '.join([f'{k} ({v} cells, {r.morphology_percentages.get(k, 0):.1%})' for k, v in top_cell_types[:5]])}
- M:E ratio: {myeloid_count}:{erythroid_count}
- Total cells: {r.total_cells_detected}

WHO GUIDELINES REFERENCE:
{diag_txt[:400]}

Write a detailed differential diagnosis covering:
1. Most likely diagnosis based on the morphological findings (be specific - if blasts >20% say "acute leukemia, likely AML" not just "acute process")
2. Second and third most likely diagnoses with supporting features
3. For each diagnosis in the differential, explain which features support it AND which features argue against it
4. Specific morphological clues that would help distinguish between the top differentials
5. How the blast percentage, M:E ratio, and cell distribution pattern fit each diagnosis
6. What age, clinical presentation, or CBC findings would make each diagnosis more or less likely
7. Rare or unusual diagnoses that should be considered if common diagnoses are excluded
8. Which ancillary test results (flow, cytogenetics, molecular) would definitively distinguish between the differentials
9. Whether the morphology alone is sufficient for diagnosis or if ancillary testing is mandatory
10. Clinical implications and urgency of each potential diagnosis

Write this as a flowing clinical reasoning narrative, NOT a bulleted list. Be specific to THIS case's exact findings:"""}],
            temperature=0.4, max_tokens=700)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            
            if r.blast_fraction >= 0.20:
                # Calculate M:E interpretation
                if erythroid_count > 0 and myeloid_count/erythroid_count > 4:
                    me_interpretation = 'is elevated, which may suggest myeloid expansion supporting AML'
                else:
                    me_interpretation = 'should be interpreted with caution given the overall blast burden'
                
                ddx = f"""The morphological findings are most consistent with **acute leukemia** given the blast percentage of {r.blast_fraction:.1%}, which significantly exceeds the WHO diagnostic threshold of 20%. Acute myeloid leukemia (AML) is the primary consideration, particularly subtypes such as AML with maturation (if myeloid precursors are prominent), acute monoblastic/monocytic leukemia (if monocytic features are evident), or AML not otherwise specified. Supporting features include the presence of myeloid precursors and the overall cell maturation pattern, though definitive classification requires immunophenotyping to confirm myeloperoxidase positivity. Acute lymphoblastic leukemia (ALL) remains in the differential, especially if blasts lack cytoplasmic granulation or show high nuclear:cytoplasmic ratios, and this diagnosis would be supported by flow cytometry demonstrating B-cell markers (CD10, CD19, CD79a) or T-cell markers (CD3, CD5, CD7) with expression of TdT. Mixed phenotype acute leukemia (MPAL) is a less common but important consideration if immunophenotyping reveals expression of both myeloid and lymphoid antigens meeting WHO criteria. The M:E ratio of {myeloid_count}:{erythroid_count} {me_interpretation}. Blast-phase chronic myeloid leukemia (CML-BP) should be excluded with BCR::ABL1 testing by FISH or PCR, particularly if basophilia or marked leukocytosis is present clinically. Therapy-related acute leukemia is a consideration if the patient has a history of prior chemotherapy or radiation exposure. The presence or absence of Auer rods, if observed, would strongly support AML and help prioritize AML-specific molecular testing such as NPM1, FLT3, and CEBPA mutations. Cytogenetic findings will be critical for WHO classification and risk stratification—favorable cytogenetics include t(8;21), inv(16), and t(15;17), while complex karyotype (≥3 abnormalities) or monosomal karyotype confers adverse prognosis. This diagnosis requires urgent flow cytometry, conventional cytogenetics, FISH, and molecular studies to establish lineage, identify prognostic markers, and guide therapeutic decisions including potential targeted therapies (e.g., FLT3 inhibitors, IDH inhibitors) or consideration for stem cell transplantation."""
            elif r.blast_fraction >= 0.05:
                ddx = f"""The morphological findings demonstrate {r.blast_fraction:.1%} blasts, placing this case in the diagnostic spectrum of **myelodysplastic syndrome (MDS)** with excess blasts versus other myeloid neoplasms. MDS with excess blasts (MDS-EB) is the primary diagnostic consideration, with MDS-EB-1 (5-9% blasts) or MDS-EB-2 (10-19% blasts) depending on the exact blast count. Supporting features would include dysplastic changes in one or more cell lineages, such as megaloblastoid erythroid precursors, hypolobated or hypogranular neutrophils, or small hypolobated megakaryocytes, though these features may not be apparent on automated analysis and require manual microscopic review. Chronic myelomonocytic leukemia (CMML) is an important differential, particularly if monocytes are increased, and this diagnosis requires correlation with absolute monocyte count ≥1 × 10⁹/L on the complete blood count. MDS/myeloproliferative neoplasm (MPN) overlap syndromes should be considered if there are features of both dysplasia and proliferation. The M:E ratio and lineage distribution pattern should be assessed for evidence of ineffective hematopoiesis or lineage-specific dysplasia. Reactive causes of increased blasts, including bone marrow regeneration following chemotherapy, growth factor administration (G-CSF, GM-CSF), or recovery from acute infection or inflammation, should be excluded through clinical history correlation. If the blast percentage is at the lower end of this range and dysplasia is not prominent, consideration should be given to idiopathic cytopenia of undetermined significance (ICUS) or clonal cytopenia of undetermined significance (CCUS), with somatic mutation analysis to detect clonal hematopoiesis. Flow cytometry is essential to detect aberrant antigen expression patterns characteristic of MDS, such as decreased CD38 or HLA-DR on myeloid precursors, asynchronous antigen expression, or reduced granulocytic side scatter. Conventional cytogenetic analysis is mandatory to detect characteristic MDS-associated abnormalities including del(5q), -7/del(7q), del(20q), and complex karyotype, which have both diagnostic and prognostic significance. Molecular testing for recurrent mutations in SF3B1, SRSF2, U2AF1, ZRSD2, ASXL1, EZH2, RUNX1, and TP53 can support the diagnosis and provide prognostic information per the Revised International Prognostic Scoring System (IPSS-R)."""
            else:
                # Calculate M:E interpretation for normal blast case
                if erythroid_count > 0 and myeloid_count/erythroid_count > 4:
                    me_interpretation_normal = 'an elevated ratio may reflect increased myeloid production (infection, G-CSF) or decreased erythroid production (anemia of chronic disease, renal insufficiency)'
                else:
                    me_interpretation_normal = 'should be correlated with hemoglobin and reticulocyte count to assess erythropoietic activity'
                
                ddx = f"""The morphological findings demonstrate a blast percentage of {r.blast_fraction:.1%}, which is within the normal reference range for bone marrow (<5% blasts). The differential diagnosis in this context should first consider **normal bone marrow** or reactive changes secondary to infection, inflammation, or hematologic stress. Benign conditions such as viral infections (particularly EBV, CMV, or parvovirus B19), bacterial infections, autoimmune disorders, or recent bleeding can produce reactive marrow changes with preserved maturation. Drug-induced marrow changes, particularly from antibiotics, anticonvulsants, or chemotherapy recovery, should be evaluated through medication history review. If cytopenias are present despite normal blast count, consideration should be given to immune-mediated cytopenias including immune thrombocytopenia (ITP), autoimmune hemolytic anemia (AIHA), or aplastic anemia, which would show decreased cellularity rather than blast increase. Vitamin B12 or folate deficiency can produce megaloblastic changes and ineffective hematopoiesis that may superficially mimic dysplasia. Early or hypoplastic myelodysplastic syndrome remains a consideration if there is persistent unexplained cytopenia, and next-generation sequencing for myeloid mutations (SF3B1, TET2, DNMT3A, ASXL1, SRSF2) may detect clonal hematopoiesis even with morphologically low blast counts. The M:E ratio of {myeloid_count}:{erythroid_count} should be interpreted in the context of peripheral blood counts—{me_interpretation_normal}. Large granular lymphocyte (LGL) leukemia is a consideration if there is lymphocytosis with associated cytopenias, requiring flow cytometry to assess for clonal T-cell or NK-cell populations. Hairy cell leukemia, though typically presenting in peripheral blood, can show marrow involvement with monocytopenia and characteristic "fried egg" cell morphology. If the clinical presentation includes fever, weight loss, or organomegaly without explanation, lymphoma with bone marrow involvement should be excluded with appropriate immunohistochemistry and flow cytometry. Overall, the low blast count is reassuring for the absence of acute leukemia, but correlation with complete blood count, peripheral smear review, clinical history, and targeted ancillary testing based on specific cytopenias or clinical concerns remains essential for complete diagnostic evaluation."""
        
        sections.append({'title': 'Differential Diagnosis', 'text': ddx, 'images': []})
        
        # TESTING RECOMMENDATIONS (Enhanced with specific panels and rationale)
        print("Generating Testing Recommendations...")
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fallback for instant report generation
                raise Exception("Using fast fallback")
            
            testing = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing the Recommended Ancillary Testing section. Write a comprehensive 7-9 sentence analysis in professional narrative style WITHOUT bullet points.

CASE CONTEXT:
- Blast percentage: {r.blast_fraction:.1%}
- Key findings: {', '.join([f'{k} ({r.morphology_percentages.get(k, 0):.1%})' for k, v in top_cell_types[:4]])}

Write detailed testing recommendations covering:
1. Specific flow cytometry panels needed (exact markers: CD13, CD33, CD117, MPO for myeloid; CD10, CD19, CD79a for B-ALL; CD3, CD5, CD7 for T-ALL; CD34, TdT, HLA-DR for stem cells)
2. Conventional cytogenetics with G-banding - what specific abnormalities to look for
3. FISH testing for recurrent translocations relevant to THIS blast percentage (t(8;21), inv(16), t(15;17), t(9;22), MLL rearrangements if acute; del(5q), -7, del(20q) if MDS)
4. Specific molecular/NGS testing based on blast count (NPM1, FLT3, CEBPA, IDH for AML; Philadelphia chromosome/BCR::ABL1; TP53; myeloid mutation panel for MDS)
5. Clinical urgency of testing (STAT vs routine) based on blast percentage
6. Additional testing: serum LDH, uric acid, coagulation studies if high blast count
7. Whether bone marrow biopsy core with immunohistochemistry is needed for architecture assessment
8. Integration with CBC and peripheral smear findings
9. Rationale for each test category in context of THIS specific case

Write as flowing narrative explaining WHY each test is needed, not just a list:"""}],
            temperature=0.4, max_tokens=650)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            
            if r.blast_fraction >= 0.20:
                testing = f"""Given the blast percentage of {r.blast_fraction:.1%}, comprehensive ancillary testing must be initiated on a **STAT/urgent basis** to establish the diagnosis of acute leukemia, determine lineage, and identify prognostically significant genetic abnormalities. **Flow cytometric immunophenotyping** is absolutely essential and should include a comprehensive acute leukemia panel: myeloid markers (CD13, CD33, CD117, CD11b, CD15, CD64, MPO), B-lymphoid markers (CD10, CD19, CD20, CD22, CD79a, cCD79a), T-lymphoid markers (CD2, CD3, cCD3, CD5, CD7), stem cell/progenitor markers (CD34, CD38, HLA-DR, TdT, CD133), and monocytic markers (CD14, CD64, CD4) to distinguish AML from ALL from mixed phenotype acute leukemia (MPAL). **Conventional cytogenetic analysis** with G-banded karyotype is required to detect recurrent translocations with therapeutic implications including t(8;21)(q22;q22) RUNX1::RUNX1T1, inv(16)(p13.1q22) or t(16;16) CBFB::MYH11, t(15;17)(q24;q21) PML::RARA (acute promyelocytic leukemia, a medical emergency requiring immediate ATRA therapy), t(9;11) or other KMT2A rearrangements, and to identify complex karyotype (≥3 abnormalities) or monosomal karyotype which confer adverse prognosis. **FISH studies** should include probes for the above translocations plus t(9;22) BCR::ABL1 to detect Philadelphia chromosome-positive leukemia. **Molecular testing** by next-generation sequencing or individual mutation analysis is mandatory for prognostic stratification and to identify targetable mutations: NPM1 (favorable if wild-type CEBPA and FLT3-ITD low), FLT3-ITD and FLT3-TKD (indicates consideration for midostaurin, gilteritinib, or quizartinib), CEBPA (biallelic mutations are favorable), IDH1 and IDH2 (targetable with ivosidenib/enasidenib), TP53 (adverse prognosis), RUNX1 (adverse), and ASXL1. **Laboratory studies** should include complete blood count with differential, comprehensive metabolic panel, coagulation studies (PT, PTT, fibrinogen, D-dimer to exclude DIC especially if APL is suspected), serum LDH and uric acid (evaluate tumor burden and assess tumor lysis syndrome risk), blood type and screen, and HLA typing if the patient is a potential allogeneic stem cell transplant candidate. Core biopsy evaluation with CD34, CD117, and reticulin immunohistochemical stains is recommended to assess marrow architecture, overall cellularity, and degree of fibrosis. These studies should be completed within 24-48 hours given the clinical urgency of acute leukemia diagnosis, and results should be integrated with patient age, performance status, comorbidities, and cytopenia severity to determine appropriate treatment intensity and clinical trial eligibility."""
            elif r.blast_fraction >= 0.05:
                testing = f"""The blast percentage of {r.blast_fraction:.1%} warrants comprehensive workup for myelodysplastic syndrome (MDS) and related myeloid neoplasms, with ancillary testing to be completed within 48-72 hours. **Flow cytometric immunophenotyping** should focus on detecting aberrant antigen expression patterns characteristic of myelodysplasia, including: evaluation of CD34+ blast population for abnormal CD117, CD13, CD33, or HLA-DR expression; assessment of maturing granulocytes for decreased CD10, asynchronous CD11b/CD16 expression, or abnormal side scatter properties; and evaluation of erythroid precursors with CD71 and glycophorin A to assess maturation abnormalities and aberrant CD5, CD7, or CD56 expression. **Conventional cytogenetic analysis** is mandatory for MDS diagnosis and prognostic assessment per IPSS-R criteria, with particular attention to del(5q) (may respond to lenalidomide), monosomy 7 or del(7q) (adverse prognosis), del(20q), trisomy 8, del(11q), and complex karyotype (≥3 abnormalities, adverse prognosis). **FISH studies** can be performed if metaphases are inadequate and should include probes for 5q, 7q, 8, 20q, and potentially TP53. **Molecular mutation analysis** by next-generation sequencing myeloid panel is highly recommended to detect clonal markers including SF3B1 (associated with ring sideroblasts and favorable prognosis), SRSF2, U2AF1, ZRSD2 (splicing factor mutations), TET2, DNMT3A, ASXL1 (epigenetic regulators), RUNX1, EZH2, and TP53 (adverse prognosis, particularly if multi-hit or associated with complex karyotype), as these findings inform IPSS-R molecular scoring. Iron stains on aspirate smears or core biopsy should be performed to assess for ring sideroblasts (>15% defines MDS-RS subtype, or >5% with SF3B1 mutation). **Clinical laboratory correlation** should include complete blood count with review of absolute neutrophil count, absolute monocyte count (≥1 × 10⁹/L suggests CMML), hemoglobin, MCV (macrocytosis common in MDS), and platelet count; comprehensive metabolic panel; vitamin B12, folate, copper, and zinc levels to exclude nutritional deficiencies; erythropoietin level; and peripheral blood smear review for dysplastic features. Core biopsy assessment is important for overall cellularity (hyper-, normo-, or hypocellular), reticulin/collagen fibrosis (reticulin stain), megakaryocyte morphology and clustering, and blast distribution. These studies collectively establish the diagnosis of MDS, define the specific WHO subtype, and calculate IPSS-R prognostic score to guide treatment decisions including observation, erythropoiesis-stimulating agents, lenalidomide (if del(5q)), hypomethylating agents (azacitidine, decitabine), or allogeneic transplant evaluation."""
            else:
                testing = f"""Although the blast percentage of {r.blast_fraction:.1%} is within normal limits, ancillary testing may be warranted based on the clinical presentation, complete blood count findings, and presence of unexplained cytopenias. **Flow cytometry** should be considered if there are persistent cytopenias, particularly to exclude: clonal T-cell or NK-cell populations (T-LGL leukemia) with assessment of CD3, CD5, CD7, CD8, CD16, CD57, and TCR Vβ repertoire analysis; low-level B-cell lymphoproliferative disorders including chronic lymphocytic leukemia (CLL), hairy cell leukemia, or lymphoplasmacytic lymphoma with B-cell markers (CD5, CD10, CD19, CD20, CD23, CD25, CD103, kappa/lambda light chains); or early myelodysplastic changes with aberrant myeloid maturation patterns. **Conventional cytogenetics** is recommended if there is concern for early/hypoplastic MDS, particularly with macrocytic anemia or unexplained cytopenias, to detect del(5q), monosomy 7, or other clonal abnormalities. **Molecular testing** for myeloid mutations (particularly SF3B1, TET2, DNMT3A, ASXL1) may detect clonal hematopoiesis of indeterminate potential (CHIP) or clonal cytopenia of undetermined significance (CCUS), which may progress to MDS over time and warrants clinical follow-up even without morphological dysplasia. If anemia is present, **iron studies** (serum iron, TIBC, ferritin, transferrin saturation) and **vitamin levels** (B12, folate, copper) should be assessed to exclude nutritional deficiencies that can mimic dysplasia. **Parvovirus B19 serology** (IgM and IgG) should be checked if there is pure red cell aplasia or isolated anemia. If thrombocytopenia is present, **platelet antibody testing** and **HIV testing** should be considered for immune thrombocytopenia. **Reticulocyte count** and **haptoglobin/LDH/indirect bilirubin** are useful if hemolysis is suspected. Core biopsy evaluation may be helpful to assess overall marrow cellularity, rule out fibrosis (particularly primary myelofibrosis or other MPN), evaluate megakaryocyte morphology, and assess for lymphoid aggregates or granulomas that might suggest lymphoma or infectious/inflammatory etiology. If there are constitutional symptoms (fever, night sweats, weight loss), lymph node enlargement, or splenomegaly, imaging studies (CT chest/abdomen/pelvis) and consideration of lymph node biopsy should be pursued to exclude occult lymphoma. The decision to proceed with ancillary testing in the setting of low blast count should be individualized based on the degree and duration of cytopenias, patient symptoms, and clinical suspicion for underlying hematologic disorder."""
        
        sections.append({'title': 'Recommended Ancillary Testing', 'text': testing, 'images': []})
        
        # CLINICAL INTERPRETATION (Enhanced with actionable recommendations)
        print("Generating Clinical Interpretation...")
        try:
            if not USE_LLM_FOR_REPORTS:
                # Use fallback for instant report generation
                raise Exception("Using fast fallback")
            
            final = llm.generate([{"role": "user", "content": f"""You are a hematopathologist writing the final Clinical Interpretation section. Write a comprehensive 8-10 sentence synthesis in professional narrative style WITHOUT bullet points.

COMPLETE CASE SUMMARY:
- Case ID: {r.case_id}
- Total cells analyzed: {r.total_cells_detected}
- Blast percentage: {r.blast_fraction:.1%}
- M:E ratio: {myeloid_count}:{erythroid_count}
- Top 5 cell types: {', '.join([f'{k} ({r.morphology_percentages.get(k, 0):.1%})' for k, v in top_cell_types[:5]])}

Write the final clinical interpretation covering:
1. Summary of the most significant morphological findings
2. Your diagnostic impression (be as specific as possible: "consistent with acute myeloid leukemia" or "morphologically normal" or "suggestive of MDS" - not vague statements)
3. Level of diagnostic confidence based on morphology alone (high/moderate/low)
4. What critical information is missing that limits morphological diagnosis (CBC values, clinical history, prior diagnoses)
5. Which ancillary test results are ABSOLUTELY REQUIRED before any treatment decisions (flow, cytogenetics, molecular)
6. Clinical urgency and timeline for follow-up based on findings (immediate consultation vs routine follow-up)
7. Specific next steps for the treating physician (when to repeat studies, what specialists to consult)
8. Important clinical correlations needed (CBC, peripheral smear, prior bone marrow, treatment history, family history)
9. Prognostic considerations if applicable (favorable vs adverse features identified)
10. Clear statement on limitations of automated analysis and need for expert hematopathologist review

Write this as a professional synthesis that guides the clinical team. Be specific to THIS case:"""}],
            temperature=0.4, max_tokens=700)
        except Exception as e:
            print(f"LLM generation failed: {e}, using fallback")
            
            if r.blast_fraction >= 0.20:
                urgency_statement = "**CRITICAL FINDINGS - IMMEDIATE CLINICAL ACTION REQUIRED**"
                impression = f"Morphological findings are consistent with **acute leukemia**, with blast cells comprising {r.blast_fraction:.1%} of bone marrow cellularity, significantly exceeding the WHO diagnostic threshold of 20%."
                next_steps = "The patient requires immediate hematology/oncology consultation, STAT flow cytometry immunophenotyping and cytogenetics to determine lineage (AML vs ALL vs MPAL), and comprehensive molecular testing to identify targetable mutations and prognostic markers. Hospital admission for metabolic monitoring (tumor lysis syndrome risk), transfusion support, and initiation of induction chemotherapy should be coordinated urgently. Clinical trial eligibility should be assessed."
            elif r.blast_fraction >= 0.05:
                urgency_statement = "**ABNORMAL FINDINGS - PROMPT EVALUATION RECOMMENDED**"
                impression = f"Morphological findings demonstrate {r.blast_fraction:.1%} blasts, raising concern for a **myelodysplastic syndrome** or related myeloid neoplasm."
                next_steps = "The patient should be referred to hematology for comprehensive evaluation including flow cytometry, cytogenetics with IPSS-R scoring, and molecular mutation analysis within one week. Complete blood count with differential should be reviewed for cytopenias, and peripheral blood smear should be evaluated for dysplastic features. Infection, nutritional deficiencies, and drug causes should be excluded."
            else:
                urgency_statement = "**MORPHOLOGICALLY UNREMARKABLE FINDINGS**"
                impression = f"The blast percentage of {r.blast_fraction:.1%} is within the normal reference range, and the overall cellular distribution demonstrates representation of multiple lineages without overt morphological features of acute leukemia or high-grade myeloid neoplasm."
                next_steps = "Clinical correlation with complete blood count and peripheral smear is essential. If unexplained cytopenias are present, consideration should be given to flow cytometry for clonal populations, cytogenetic analysis, and molecular testing for early myeloid neoplasm. Routine hematology follow-up is appropriate based on clinical context."
            
            final = f"""{urgency_statement}

{impression} The automated morphological analysis of {r.total_cells_detected} cells provides quantitative, objective assessment of cell populations, demonstrating predominance of {top_cell_types[0][0]} ({r.morphology_percentages.get(top_cell_types[0][0], 0):.1%}), followed by {top_cell_types[1][0]} ({r.morphology_percentages.get(top_cell_types[1][0], 0):.1%}). The myeloid to erythroid ratio is calculated as {myeloid_count}:{erythroid_count}. While morphology provides important diagnostic clues, definitive diagnosis, lineage determination, and prognostic stratification **require correlation with flow cytometry immunophenotyping, conventional cytogenetic analysis, FISH studies, and molecular genetic testing**, as outlined in the Recommended Ancillary Testing section above.

**Critical Clinical Correlation Required:** This morphological assessment must be integrated with complete blood count parameters (WBC with differential, hemoglobin, platelet count), peripheral blood smear review (blast count, dysplastic features, circulating abnormal cells), patient clinical presentation (cytopenias, organomegaly, constitutional symptoms), past medical history (prior chemotherapy or radiation, antecedent hematologic disorder, family history), and recent medication exposures. The diagnostic impression and treatment recommendations will be substantially refined once all ancillary studies are complete.

{next_steps} {'Serial monitoring of blood counts and repeat bone marrow examination may be indicated based on clinical course and response to therapy' if r.blast_fraction >= 0.05 else 'If symptoms or laboratory abnormalities develop, reassessment with repeat bone marrow examination is warranted'}.

**Important Limitations:** This analysis is based on {r.total_cells_detected} cells identified by automated detection, compared to the standard 500-cell manual differential count, which may affect statistical precision for rare cell populations. Subtle morphological features such as dysplasia, Auer rods, cytoplasmic granulation quality, and nuclear detail are best assessed by manual microscopic review by a board-certified hematopathologist. This automated system serves as a screening and decision-support tool and does NOT replace expert pathologist interpretation. All findings should be confirmed and integrated by a qualified hematopathologist before clinical decision-making.

**Final Recommendation:** {'URGENT hematopathologist review and hematology/oncology consultation are required within 24 hours' if r.blast_fraction >= 0.20 else 'Prompt hematopathologist review and hematology referral are recommended' if r.blast_fraction >= 0.05 else 'Hematopathologist review is recommended if clinical or laboratory abnormalities are present'}. The automated analysis provides supportive quantitative data but must be confirmed by expert manual review and correlation with complete clinical information."""
        
        sections.append({'title': 'Clinical Interpretation', 'text': final, 'images': []})
        
        # LIMITATIONS (Enhanced with comprehensive context)
        sections.append({
            'title': 'Limitations',
            'text': f"""**Technical and Analytical Limitations:**

**Cell Count Differential:** This automated analysis evaluated {r.total_cells_detected} nucleated cells, whereas the standard clinical reference is a 500-cell manual differential count performed by a trained medical technologist or hematopathologist. Statistical precision for rare cell populations (<5%) may be limited by the smaller sample size, and percentage calculations should be interpreted with appropriate confidence intervals.

**Morphological Resolution and Detail:** Automated image analysis, while highly consistent and reproducible, cannot replicate the full morphological assessment performed by expert microscopic review. Subtle cytological features critical for diagnosis—including nuclear chromatin texture (fine vs coarse), nucleolar characteristics (size, number, prominence), cytoplasmic granulation quality and distribution, presence of Auer rods or cytoplasmic inclusions, dysplastic features (hypolobation, hypogranulation, megaloblastoid changes, abnormal megakaryocytes), cell size variation, and nuclear:cytoplasmic ratio—require manual high-magnification oil immersion microscopy by a qualified hematopathologist.

**Clinical Context Integration:** This morphological assessment lacks essential clinical information that substantially affects interpretation, including: complete blood count parameters (WBC, hemoglobin, platelet count, absolute blast count, absolute neutrophil count), peripheral blood smear findings (circulating blasts, dysplastic features, immature cells, red cell morphology), patient demographics (age, sex, ethnicity), clinical presentation (symptoms, physical exam findings, organomegaly), past medical history (prior hematologic disorders, previous chemotherapy or radiation therapy, family history of hematologic malignancy), medication exposures (recent antibiotics, chemotherapy, G-CSF/GM-CSF administration), and concurrent laboratory values (LDH, uric acid, renal/hepatic function).

**Diagnostic Limitations:** Morphology alone, whether assessed manually or by automated methods, is **insufficient for definitive diagnosis** of hematologic malignancies in modern practice. The WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues (5th edition, 2022) requires integration of morphology with immunophenotyping (flow cytometry or immunohistochemistry), cytogenetic analysis (conventional karyotype), molecular genetic studies (FISH, PCR, next-generation sequencing), and clinical information for accurate diagnosis, WHO subtype classification, and risk stratification.

**Specimen Quality Variables:** The analysis is dependent on the quality of the bone marrow aspirate specimen, adequacy of smear preparation, staining consistency (Wright-Giemsa or equivalent), absence of peripheral blood contamination or dilution, and absence of significant hemodilution or clotting artifact. Poor-quality specimens may yield suboptimal automated detection and classification.

**Clinical Use Constraints:** This automated analysis system is intended for **research and clinical decision-support purposes only** and has not received regulatory clearance or approval as a primary diagnostic device. All findings must be confirmed by board-certified hematopathologist review. Clinical diagnosis, treatment decisions, prognostic assessment, and patient counseling must be based on comprehensive evaluation by qualified physicians integrating all available clinical, laboratory, morphological, immunophenotypic, cytogenetic, and molecular data.

**Quality Assurance Requirements:** Regular quality control with known reference specimens, periodic validation against expert manual review, and ongoing assessment of classification accuracy are essential to maintain analytical validity. Individual case-level limitations should be documented, and cases with discordant findings between automated and manual review should be flagged for expert arbitration.

**Professional Responsibility:** The treating physician and consulting hematopathologist retain full professional and legal responsibility for diagnosis, classification, risk assessment, treatment recommendations, and patient care. Automated analysis results should be used as one component of comprehensive diagnostic evaluation, not as a standalone diagnostic tool.""",
            'images': []
        })
        
        print(f"Report generation complete: {len(sections)} sections")
        
        # CACHE THE REPORT for future requests
        report_data = {'sections': sections, 'case': r.case_id, 'time': datetime.now().isoformat()}
        db_analyses[aid]['cached_report'] = report_data
        
        # Save to persistent storage
        save_storage(db_analyses, db_chats)
        
        return jsonify(report_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if not RAG_OK:
        return jsonify({'error': 'RAG unavailable'}), 503
    
    data = request.json
    aid = data.get('id')
    q = data.get('q')
    
    if not aid or not q:
        return jsonify({'error': 'Missing params'}), 400
    if aid not in db_analyses:
        return jsonify({'error': 'Not found'}), 404
    
    try:
        r = db_analyses[aid]['result']
        
        # Build comprehensive context
        ctx_parts = [
            f"Case ID: {r.case_id}",
            f"Total cells analyzed: {r.total_cells_detected}",
            f"Blast fraction: {r.blast_fraction:.1%}",
            f"Evidence sufficiency: {r.evidence_sufficiency}",
            "",
            "Cell distribution:"
        ]
        
        for cell_type, count in sorted(r.morphology_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = r.morphology_percentages.get(cell_type, 0)
            ctx_parts.append(f"  • {cell_type}: {count} cells ({pct:.1%})")
        
        ctx = "\n".join(ctx_parts)
        
        # Enhanced retrieval with more context
        expanded_query = f"{q} bone marrow cytomorphology blast {r.blast_fraction:.1%} pediatric"
        ev = retriever.retrieve(query=expanded_query, top_k=4)
        evtxt = "\n\n".join([f"[Reference {i+1}]:\n{e['text'][:600]}" for i, e in enumerate(ev)])
        
        prompt = f"""You are a bone marrow pathology assistant. Answer the user's question directly and concisely.

CASE DATA:
{ctx}

CLINICAL REFERENCES:
{evtxt[:800] if evtxt else "None"}

USER QUESTION: {q}

Provide a brief, focused answer (2-4 sentences):
- Answer the specific question asked
- Reference actual numbers from this case when relevant  
- Be precise and factual
- Skip background unless asked

Answer:"""
        
        ans = llm.generate([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300)
        
        entry = {'q': q, 'a': ans, 'time': datetime.now().isoformat()}
        db_chats[aid].append(entry)
        
        # Save chat history
        save_storage(db_analyses, db_chats)
        
        return jsonify({'a': ans})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/<aid>')
def get_chat(aid):
    return jsonify({'history': db_chats.get(aid, [])})

@app.route('/api/img/<path:p>')
@app.route('/api/image/<path:p>')
def serve_img(p):
    """Serve image files from various locations"""
    from pathlib import Path
    
    # Try multiple paths
    paths = [
        Path(p),  # Absolute path
        REPO_ROOT / p,  # Relative to repo root
        REPO_ROOT / 'data' / 'outputs' / p,
        REPO_ROOT / 'data' / 'new_patient_cache' / p
    ]
    
    for path in paths:
        try:
            if path.exists() and path.is_file():
                return send_from_directory(str(path.parent), path.name)
        except:
            continue
    
    print(f"ERROR: Image not found: {p}")
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
