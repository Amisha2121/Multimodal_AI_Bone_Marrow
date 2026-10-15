"""
Bone Marrow Cytomorphology Assistant - Clean Medical Interface
4-Tab Navigation System
"""

import logging
from pathlib import Path
import sys
from typing import Optional
from datetime import datetime
import hashlib

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_model import PatientAnalysis
from pipeline.new_patient_engine import analyze_new_case
from retrieval.retrieve import HybridRetriever
from agents.llm_provider import get_llm_provider

logging.basicConfig(level=logging.ERROR)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bone Marrow Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# CLEAN STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
* {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
.main .block-container {padding: 1rem 1.5rem; max-width: 100%;}

h1 {font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0; color: #1a1a1a;}
h2 {font-size: 1.25rem; font-weight: 600; margin: 0.75rem 0 0.5rem 0; color: #2a2a2a;}
h3 {font-size: 1.1rem; font-weight: 500; margin: 0.5rem 0; color: #3a3a3a;}

.stButton>button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}
.stButton>button:hover {background: #1d4ed8;}

[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 600; color: #1a1a1a;}
[data-testid="stMetricLabel"] {font-size: 0.875rem; color: #666;}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: white;
    border-bottom: 2px solid #e5e7eb;
    padding: 0 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    color: #6b7280;
    border: none;
}

.stTabs [aria-selected="true"] {
    color: #2563eb;
    border-bottom: 3px solid #2563eb;
}

.history-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    cursor: pointer;
}

.history-card:hover {
    border-color: #2563eb;
    background: #f0f7ff;
}

.history-active {
    border-color: #2563eb;
    background: #eff6ff;
    font-weight: 600;
}

.cell-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.cell-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.5rem;
    text-align: center;
}

.cell-name {
    font-weight: 600;
    font-size: 0.875rem;
    margin: 0.5rem 0 0.25rem 0;
    color: #1a1a1a;
}

.cell-desc {
    font-size: 0.75rem;
    color: #6b7280;
    line-height: 1.3;
}

.chat-msg {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.user-q {
    background: #f0f7ff;
    border-left: 3px solid #2563eb;
}

.assistant-a {
    background: #f9fafb;
    border-left: 3px solid #6b7280;
}

img {border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if 'history' not in st.session_state:
    st.session_state.history = []

if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None
    st.session_state.llm = None

if 'chats' not in st.session_state:
    st.session_state.chats = {}

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def init_rag():
    if st.session_state.retriever is None:
        try:
            st.session_state.retriever = HybridRetriever()
            st.session_state.llm = get_llm_provider()
        except Exception as e:
            st.error(f"RAG init failed: {e}")

def get_selected():
    if not st.session_state.selected_id:
        return None
    for item in st.session_state.history:
        if item['id'] == st.session_state.selected_id:
            return item['analysis']
    return None

def add_to_history(analysis: PatientAnalysis):
    hist_id = hashlib.md5(f"{analysis.case_id}_{datetime.now()}".encode()).hexdigest()[:8]
    
    thumb = None
    if analysis.image_infos:
        info = analysis.image_infos[0]
        if info.debug_filtered_path and Path(info.debug_filtered_path).exists():
            thumb = info.debug_filtered_path
    
    st.session_state.history.insert(0, {
        'id': hist_id,
        'case_id': analysis.case_id,
        'time': datetime.now(),
        'analysis': analysis,
        'thumb': thumb,
        'cells': analysis.total_cells_detected,
        'blast': analysis.blast_fraction
    })
    
    st.session_state.selected_id = hist_id
    st.session_state.chats[hist_id] = []

def get_cell_desc(cell_class: str) -> str:
    desc = {
        "Blast": "Immature cell",
        "Lymphoid Series": "Mature lymphocytes",
        "Granulocyte (Mature)": "Mature neutrophils",
        "Granulocyte (Immature)": "Early myeloid",
        "Erythroid Precursor": "RBC precursor",
        "Monocyte": "Large phagocyte",
        "Megakaryocyte/Platelet": "Platelet producer",
        "Other/Artifact": "Unclear"
    }
    return desc.get(cell_class, "BM cell")

def ask_rag(question: str, analysis: PatientAnalysis) -> str:
    init_rag()
    
    if not st.session_state.retriever or not st.session_state.llm:
        return "RAG unavailable"
    
    context = f"""Sample: {analysis.case_id}
Cells: {analysis.total_cells_detected}
Blasts: {analysis.blast_fraction:.1%}"""
    
    try:
        retrieved = st.session_state.retriever.retrieve(query=question, top_k=2)
        evidence = "\n".join([f"{r['text'][:200]}" for r in retrieved])
    except:
        evidence = ""
    
    try:
        prompt = f"""{context}

Guidelines: {evidence if evidence else "None"}

Q: {question}

Answer (2 sentences):"""
        
        return st.session_state.llm.generate([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200)
    except Exception as e:
        return f"Error: {e}"

# ═══════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.title("🔬 Bone Marrow Cytomorphology Analysis")

# Main 4 tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "💬 Ask Questions", "📊 View Details", "🔍 Cell Explorer"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    col_main, col_history = st.columns([3, 1])
    
    with col_main:
        st.subheader("Upload Bone Marrow Image")
        
        files = st.file_uploader(
            "Select microscopy images",
            type=['png', 'jpg', 'jpeg', 'tiff'],
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔍 Analyze Sample", type="primary", use_container_width=True):
                if not files:
                    st.error("Please upload at least one image")
                else:
                    with st.spinner("Analyzing..."):
                        try:
                            result = analyze_new_case(
                                uploaded_files=files,
                                age_years=None,
                                sex=None,
                                sample_site=None
                            )
                            add_to_history(result)
                            st.success(f"✅ Complete! {result.total_cells_detected} cells detected")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
        
        with col2:
            with st.expander("Metadata (optional)"):
                age = st.number_input("Age", 0.0, 120.0, 0.0)
                sex = st.selectbox("Sex", ["N/A", "F", "M"])
        
        st.divider()
        
        # Show current analysis
        current = get_selected()
        if current:
            st.subheader(f"Current: {current.case_id}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Images", current.total_images_processed)
            col2.metric("Cells", current.total_cells_detected)
            col3.metric("Classified", current.cells_classified)
            col4.metric("Blasts", f"{current.blast_fraction:.1%}")
            
            if current.image_infos:
                info = current.image_infos[0]
                if info.debug_filtered_path and Path(info.debug_filtered_path).exists():
                    st.image(info.debug_filtered_path, use_container_width=True)
            
            # Cell distribution
            st.subheader("Cell Distribution")
            if current.morphology_counts:
                chart_df = pd.DataFrame({
                    "Type": list(current.morphology_counts.keys()),
                    "Count": list(current.morphology_counts.values())
                })
                st.bar_chart(chart_df.set_index("Type"), height=300)
    
    with col_history:
        st.subheader("History")
        if not st.session_state.history:
            st.info("No analyses yet")
        else:
            for item in st.session_state.history:
                is_active = item['id'] == st.session_state.selected_id
                
                if st.button(
                    f"**{item['case_id'][:12]}**\n{item['time'].strftime('%H:%M')}\n{item['cells']} cells • {item['blast']:.0%} blasts",
                    key=f"h1_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.selected_id = item['id']
                    st.rerun()
                
                if item['thumb']:
                    st.image(item['thumb'], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: ASK QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    col_main, col_history = st.columns([3, 1])
    
    with col_main:
        current = get_selected()
        
        if not current:
            st.info("📤 Upload and analyze a sample first")
        else:
            st.subheader(f"Ask About: {current.case_id}")
            st.caption(f"{current.total_cells_detected} cells • {current.blast_fraction:.1%} blasts")
            
            st.divider()
            
            # Quick questions
            st.write("**Quick Questions:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("What does blast % indicate?", use_container_width=True):
                    st.session_state.pending_q = "What does the blast percentage indicate?"
            with col2:
                if st.button("Is this acute leukemia?", use_container_width=True):
                    st.session_state.pending_q = "Is this consistent with acute leukemia?"
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("What tests needed?", use_container_width=True):
                    st.session_state.pending_q = "What additional tests are needed?"
            with col4:
                if st.button("Explain distribution", use_container_width=True):
                    st.session_state.pending_q = "Explain the cell distribution"
            
            st.divider()
            
            # Chat interface
            question = st.text_area("Your Question:", value=st.session_state.get('pending_q', ''), height=100)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔍 Ask", type="primary", use_container_width=True):
                    if question.strip():
                        with st.spinner("Thinking..."):
                            answer = ask_rag(question, current)
                            st.session_state.chats[st.session_state.selected_id].append({'q': question, 'a': answer})
                            st.session_state.pending_q = ''
                            st.rerun()
            with col2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.chats[st.session_state.selected_id] = []
                    st.rerun()
            
            # Chat history
            if st.session_state.selected_id in st.session_state.chats:
                chats = st.session_state.chats[st.session_state.selected_id]
                if chats:
                    st.divider()
                    st.subheader("Conversation")
                    for chat in reversed(chats):
                        st.markdown(f'<div class="chat-msg user-q"><strong>Q:</strong> {chat["q"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-msg assistant-a"><strong>A:</strong> {chat["a"]}</div>', unsafe_allow_html=True)
    
    with col_history:
        st.subheader("History")
        if st.session_state.history:
            for item in st.session_state.history:
                is_active = item['id'] == st.session_state.selected_id
                if st.button(
                    f"{item['case_id'][:12]}\n{item['cells']} cells",
                    key=f"h2_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.selected_id = item['id']
                    st.rerun()
                if item['thumb']:
                    st.image(item['thumb'], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: VIEW DETAILS
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    col_main, col_history = st.columns([3, 1])
    
    with col_main:
        current = get_selected()
        
        if not current:
            st.info("📤 Select a sample from history")
        else:
            st.subheader(f"Details: {current.case_id}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Images", current.total_images_processed)
            col2.metric("Cells", current.total_cells_detected)
            col3.metric("Classified", current.cells_classified)
            col4.metric("Blasts", f"{current.blast_fraction:.1%}")
            
            st.divider()
            
            # Cell distribution table
            st.subheader("Cell Morphology")
            if current.morphology_counts:
                df = pd.DataFrame([
                    {"Cell Type": k, "Count": v, "Percentage": f"{current.morphology_percentages.get(k, 0):.1%}"}
                    for k, v in current.morphology_counts.items()
                ]).sort_values("Count", ascending=False)
                st.dataframe(df, hide_index=True, use_container_width=True)
            
            st.divider()
            
            # Images
            st.subheader("Detection Images")
            for info in current.image_infos:
                with st.expander(f"📷 {info.filename} ({info.cells_detected} cells)"):
                    if info.debug_filtered_path and Path(info.debug_filtered_path).exists():
                        st.image(info.debug_filtered_path, use_container_width=True)
    
    with col_history:
        st.subheader("History")
        if st.session_state.history:
            for item in st.session_state.history:
                is_active = item['id'] == st.session_state.selected_id
                if st.button(
                    f"{item['case_id'][:12]}\n{item['cells']} cells",
                    key=f"h3_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.selected_id = item['id']
                    st.rerun()
                if item['thumb']:
                    st.image(item['thumb'], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: CELL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    col_main, col_history = st.columns([3, 1])
    
    with col_main:
        current = get_selected()
        
        if not current:
            st.info("📤 Select a sample from history")
        elif not current.cell_predictions:
            st.warning("No cells detected in this sample")
        else:
            st.subheader(f"Cells: {current.case_id}")
            st.caption(f"{len(current.cell_predictions)} cells detected")
            
            # Filter
            cell_types = ["All"] + list(set([c.predicted_class for c in current.cell_predictions]))
            selected_type = st.selectbox("Filter by type:", cell_types)
            
            st.divider()
            
            # Get cells to show
            if selected_type == "All":
                cells = current.cell_predictions
            else:
                cells = [c for c in current.cell_predictions if c.predicted_class == selected_type]
            
            st.write(f"**Showing {len(cells)} cells**")
            
            # Display cells in grid
            cols_per_row = 5
            for i in range(0, len(cells), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(cells):
                        cell = cells[i + j]
                        with col:
                            if cell.image_path and Path(cell.image_path).exists():
                                st.image(cell.image_path, use_container_width=True)
                            
                            st.markdown(f"""
                            <div class="cell-card">
                                <div class="cell-name">{cell.predicted_class}</div>
                                <div class="cell-desc">{get_cell_desc(cell.predicted_class)}</div>
                                <div style="font-size: 0.7rem; color: #999; margin-top: 0.25rem;">
                                    {cell.confidence:.0%} • {cell.cell_id.split('_')[-1]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    with col_history:
        st.subheader("History")
        if st.session_state.history:
            for item in st.session_state.history:
                is_active = item['id'] == st.session_state.selected_id
                if st.button(
                    f"{item['case_id'][:12]}\n{item['cells']} cells",
                    key=f"h4_{item['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.selected_id = item['id']
                    st.rerun()
                if item['thumb']:
                    st.image(item['thumb'], use_container_width=True)
