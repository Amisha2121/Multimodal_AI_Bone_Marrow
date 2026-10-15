"""
app/interactive_features.py
─────────────────────────────────────────────────────────────────────────────
Interactive Features for Enhanced Bone Marrow Analysis

Features:
1. Cell-by-Cell Explorer: Detailed analysis of each detected cell with RAG-based explanations
2. Interactive Chatbot: RAG-powered Q&A about the bone marrow sample
"""

import logging
from typing import List, Dict, Any, Optional
import streamlit as st
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.retrieve import HybridRetriever
from agents.llm_provider import get_llm_provider
from pipeline.data_model import PatientAnalysis, CellInferenceResult

logger = logging.getLogger("interactive_features")


# ============================================================================
# FEATURE 1: Cell-by-Cell Explorer
# ============================================================================

def get_cell_specific_knowledge(
    cell_result: CellInferenceResult,
    retriever: HybridRetriever,
    patient_age_years: Optional[float] = None
) -> Dict[str, Any]:
    """
    Retrieve knowledge base information specific to a detected cell's morphology.
    
    Args:
        cell_result: Individual cell inference result
        retriever: Hybrid retriever instance
        patient_age_years: Patient age for context-aware retrieval
        
    Returns:
        Dict with cell-specific knowledge from WHO guidelines
    """
    # Build cell-specific query
    cell_class = cell_result.predicted_class
    confidence = cell_result.confidence
    
    # Create detailed query based on cell type
    query_parts = [f"Cytomorphology characteristics of {cell_class}"]
    
    # Add age context if available
    if patient_age_years is not None:
        if patient_age_years < 18:
            query_parts.append("pediatric context")
        else:
            query_parts.append("adult context")
    
    # Add specific features based on cell type
    if "Blast" in cell_class:
        query_parts.append("high nuclear-to-cytoplasmic ratio fine chromatin nucleoli")
    elif "Lymphoid" in cell_class:
        query_parts.append("small mature lymphocytes condensed chromatin")
    elif "Granulocyte" in cell_class:
        if "Mature" in cell_class:
            query_parts.append("segmented neutrophils bands mature granulocytes")
        else:
            query_parts.append("promyelocytes myelocytes metamyelocytes immature")
    elif "Erythroid" in cell_class:
        query_parts.append("normoblasts erythroid precursors red blood cell development")
    elif "Monocyte" in cell_class:
        query_parts.append("large cells convoluted nuclei gray cytoplasm")
    elif "Megakaryocyte" in cell_class:
        query_parts.append("large polylobulated nuclei abundant cytoplasm platelet production")
    
    query = " ".join(query_parts)
    
    # Retrieve relevant passages
    try:
        results = retriever.retrieve(query=query, top_k=2)  # Top 2 most relevant
        return {
            "query": query,
            "passages": results,
            "cell_class": cell_class,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error retrieving cell-specific knowledge: {e}")
        return {
            "query": query,
            "passages": [],
            "cell_class": cell_class,
            "confidence": confidence,
            "error": str(e)
        }


def generate_cell_explanation(
    cell_result: CellInferenceResult,
    knowledge: Dict[str, Any],
    llm_provider
) -> str:
    """
    Generate natural language explanation for a specific cell using LLM + RAG.
    
    Args:
        cell_result: Cell inference result
        knowledge: Retrieved knowledge from WHO guidelines
        llm_provider: LLM provider instance
        
    Returns:
        Natural language explanation of the cell
    """
    cell_class = cell_result.predicted_class
    confidence = cell_result.confidence
    passages = knowledge.get("passages", [])
    
    # Build context from retrieved passages
    context = "\n\n".join([
        f"[{p['chunk_id']}] {p['text'][:500]}..."
        for p in passages[:2]
    ])
    
    # Create prompt for LLM
    system_prompt = """You are a clinical hematopathology assistant. 
Generate a brief, clear explanation of a detected bone marrow cell based on its classification and WHO guidelines.
Focus on:
1. Key morphological features of this cell type
2. Clinical significance
3. What this tells us about the sample

Keep it concise (2-3 sentences), accurate, and cite the guideline chunks."""
    
    user_prompt = f"""Cell Classification: {cell_class}
Confidence: {confidence:.1%}

Retrieved WHO Guidelines:
{context if context else "No specific guidelines retrieved for this cell type."}

Task: Explain what this {cell_class} cell tells us about the bone marrow sample. 
Be specific, cite guidelines using [chunk_id], and keep it brief (2-3 sentences)."""
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        explanation = llm_provider.generate(messages, temperature=0.1, max_tokens=300)
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating cell explanation: {e}")
        # Fallback to template-based explanation
        return f"This {cell_class} cell (confidence: {confidence:.1%}) represents {_get_fallback_description(cell_class)}."


def _get_fallback_description(cell_class: str) -> str:
    """Fallback descriptions if LLM fails."""
    descriptions = {
        "Blast": "immature hematopoietic cells with high nuclear-to-cytoplasmic ratio, which may indicate acute leukemia if present above 20%",
        "Lymphoid Series": "mature lymphocytes, part of normal immune cell population",
        "Granulocyte (Mature)": "mature neutrophils and bands, indicating normal myeloid maturation",
        "Granulocyte (Immature)": "early myeloid precursors (promyelocytes, myelocytes) showing myeloid proliferation",
        "Erythroid Precursor": "red blood cell precursors, indicating erythropoietic activity",
        "Monocyte": "large mononuclear phagocytes with gray cytoplasm",
        "Megakaryocyte/Platelet": "platelet-producing cells or platelet fragments",
        "Other/Artifact": "unclear cell morphology or staining artifact"
    }
    return descriptions.get(cell_class, "a bone marrow cell component")


# ============================================================================
# FEATURE 2: Interactive Chatbot
# ============================================================================

class BoneMarrowChatbot:
    """RAG-powered chatbot for answering questions about a specific bone marrow sample."""
    
    def __init__(self, analysis: PatientAnalysis, retriever: HybridRetriever, llm_provider):
        self.analysis = analysis
        self.retriever = retriever
        self.llm = llm_provider
        
        # Build sample context
        self.sample_context = self._build_sample_context()
        
    def _build_sample_context(self) -> str:
        """Build a summary of the analyzed sample for context."""
        a = self.analysis
        
        context_parts = [
            f"Case ID: {a.case_id}",
            f"Total cells detected: {a.total_cells_detected}",
            f"Blast fraction: {a.blast_fraction:.1%}",
            f"Morphology distribution:",
        ]
        
        # Add cell counts
        if a.morphology_counts:
            for cell_type, count in sorted(a.morphology_counts.items(), key=lambda x: x[1], reverse=True):
                pct = a.morphology_percentages.get(cell_type, 0)
                context_parts.append(f"  - {cell_type}: {count} cells ({pct:.1%})")
        
        # Add patient demographics if provided
        if a.age_years:
            context_parts.append(f"Patient age: {a.age_years} years")
        if a.sex:
            context_parts.append(f"Patient sex: {a.sex}")
            
        # Add evidence quality
        context_parts.append(f"Evidence quality: {a.evidence_sufficiency or 'MODERATE'}")
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the bone marrow sample using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant guidelines
        try:
            retrieved = self.retriever.retrieve(query=question, top_k=3)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            retrieved = []
        
        # Step 2: Build prompt with sample context + retrieved knowledge
        system_prompt = """You are a clinical hematopathology assistant answering questions about a specific bone marrow sample.

CRITICAL RULES:
1. Answer ONLY based on:
   - The actual sample data provided (cell counts, blast fraction, detected cells)
   - Retrieved WHO guideline passages
2. NEVER fabricate information about:
   - CBC values (not provided)
   - Cytogenetics (not provided)
   - Flow cytometry (not provided)
   - Molecular testing (not provided)
3. If the answer requires information not available, say "This information is not available from the sample morphology alone. [Specify what test is needed]."
4. Cite guideline passages using [chunk_id]
5. Keep answers concise and clinical (3-4 sentences)

Remember: You're analyzing ONE specific sample, not giving general knowledge."""
        
        # Build context
        context_text = f"""SAMPLE DATA:
{self.sample_context}

RETRIEVED WHO GUIDELINES:
"""
        
        if retrieved:
            for i, passage in enumerate(retrieved, 1):
                context_text += f"\n[{passage['chunk_id']}] {passage['text'][:400]}...\n"
        else:
            context_text += "\nNo specific guidelines retrieved for this question.\n"
        
        user_prompt = f"""{context_text}

QUESTION: {question}

Answer the question based ONLY on the sample data and retrieved guidelines above. 
If information is not available, clearly state what additional tests would be needed."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            answer = self.llm.generate(messages, temperature=0.1, max_tokens=500)
            
            return {
                "question": question,
                "answer": answer,
                "sources": [p['chunk_id'] for p in retrieved],
                "sample_context": self.sample_context,
                "retrieved_passages": retrieved
            }
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}. Please try rephrasing your question.",
                "sources": [],
                "error": str(e)
            }
    
    def suggest_questions(self) -> List[str]:
        """Generate suggested questions based on the sample."""
        suggestions = [
            "What does the blast fraction tell us about this sample?",
            "Is this sample consistent with acute leukemia?",
            "What additional tests should be performed?",
        ]
        
        # Add cell-type specific questions
        if self.analysis.morphology_counts:
            predominant = max(self.analysis.morphology_counts.items(), key=lambda x: x[1])[0]
            if "Blast" in predominant:
                suggestions.append("What are the diagnostic criteria for acute leukemia?")
            elif "Megakaryocyte" in predominant:
                suggestions.append("What conditions show increased megakaryocytes?")
            elif "Lymphoid" in predominant:
                suggestions.append("Could this be reactive lymphocytosis?")
        
        # Age-specific questions
        if self.analysis.age_years and self.analysis.age_years < 18:
            suggestions.append("What are pediatric-specific considerations for this sample?")
            suggestions.append("Should haematogones be considered in this case?")
        
        return suggestions[:5]  # Max 5 suggestions


# ============================================================================
# Streamlit UI Components
# ============================================================================

def render_cell_explorer(analysis: PatientAnalysis, retriever: HybridRetriever, llm_provider):
    """Render the Cell-by-Cell Explorer interface."""
    st.markdown("### 🔬 Cell-by-Cell Analysis Explorer")
    st.caption("Explore each detected cell with AI-powered explanations from WHO guidelines")
    
    if not analysis.cell_predictions:
        st.warning("No cell predictions available. Run analysis first.")
        return
    
    # Get top cells to display (mix of classes)
    cells_to_show = _select_diverse_cells(analysis.cell_predictions, max_cells=20)
    
    # Create a grid layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select a cell to explore:**")
        
        # Cell selector
        cell_options = [
            f"{c.cell_id} - {c.predicted_class} ({c.confidence:.0%})"
            for c in cells_to_show
        ]
        
        selected_idx = st.selectbox(
            "Cell Selection",
            range(len(cell_options)),
            format_func=lambda i: cell_options[i],
            label_visibility="collapsed"
        )
        
        selected_cell = cells_to_show[selected_idx]
        
        # Display cell image if available
        if selected_cell.image_path and Path(selected_cell.image_path).exists():
            st.image(selected_cell.image_path, caption=f"{selected_cell.cell_id}", use_column_width=True)
        
        # Cell metrics
        st.markdown("**Cell Metrics:**")
        st.metric("Classification", selected_cell.predicted_class)
        st.metric("Confidence", f"{selected_cell.confidence:.1%}")
        
        # Top 3 probabilities
        if selected_cell.probabilities:
            st.markdown("**Top Predictions:**")
            sorted_probs = sorted(selected_cell.probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            for class_name, prob in sorted_probs:
                st.text(f"{class_name}: {prob:.1%}")
    
    with col2:
        st.markdown("**AI-Powered Explanation:**")
        
        with st.spinner("Retrieving WHO guidelines and generating explanation..."):
            # Get cell-specific knowledge
            knowledge = get_cell_specific_knowledge(
                selected_cell,
                retriever,
                analysis.age_years
            )
            
            # Generate explanation
            explanation = generate_cell_explanation(
                selected_cell,
                knowledge,
                llm_provider
            )
            
            # Display explanation
            st.markdown(f"""
            <div class="clinical-card">
                {explanation}
            </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved evidence
            if knowledge.get("passages"):
                with st.expander("📚 Retrieved WHO Guideline Evidence"):
                    for passage in knowledge["passages"]:
                        st.markdown(f"**[{passage['chunk_id']}]** - Score: {passage.get('score', 0):.3f}")
                        st.text(passage['text'][:300] + "...")
                        st.divider()


def render_chatbot(analysis: PatientAnalysis, retriever: HybridRetriever, llm_provider):
    """Render the Interactive Chatbot interface."""
    st.markdown("### 💬 Interactive Q&A About This Sample")
    st.caption("Ask questions about this specific bone marrow sample - answers are based on detected cells and WHO guidelines")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state or st.session_state.get('current_case_id') != analysis.case_id:
        st.session_state.chatbot = BoneMarrowChatbot(analysis, retriever, llm_provider)
        st.session_state.chat_history = []
        st.session_state.current_case_id = analysis.case_id
    
    chatbot = st.session_state.chatbot
    
    # Display sample context
    with st.expander("📊 Sample Overview"):
        st.text(chatbot.sample_context)
    
    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    suggestions = chatbot.suggest_questions()
    
    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        if col.button(f"❓", key=f"suggest_{i}", help=suggestion):
            st.session_state.current_question = suggestion
    
    # Display suggestions as chips
    for suggestion in suggestions:
        st.caption(f"• {suggestion}")
    
    st.divider()
    
    # Chat interface
    user_question = st.text_input(
        "Ask a question about this bone marrow sample:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., What does the blast percentage indicate?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.current_question = ''
        st.rerun()
    
    if ask_button and user_question.strip():
        with st.spinner("Thinking... Retrieving guidelines and formulating answer..."):
            # Get answer
            response = chatbot.answer_question(user_question)
            
            # Add to history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response["answer"],
                "sources": response.get("sources", [])
            })
            
            # Clear question
            st.session_state.current_question = ''
    
    # Display chat history (most recent first)
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 12px; border-radius: 8px; margin: 8px 0;">
                    <strong>Q:</strong> {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: white; padding: 16px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #1A237E;">
                    <strong>A:</strong> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                if chat.get('sources'):
                    st.caption(f"📚 Sources: {', '.join(chat['sources'])}")
                
                st.divider()


def _select_diverse_cells(cells: List[CellInferenceResult], max_cells: int = 20) -> List[CellInferenceResult]:
    """Select a diverse set of cells across different classes for display."""
    # Group by class
    by_class = {}
    for cell in cells:
        class_name = cell.predicted_class
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(cell)
    
    # Select top confident cells from each class
    selected = []
    cells_per_class = max(1, max_cells // len(by_class))
    
    for class_name, class_cells in by_class.items():
        # Sort by confidence
        sorted_cells = sorted(class_cells, key=lambda c: c.confidence, reverse=True)
        selected.extend(sorted_cells[:cells_per_class])
    
    # If we have less than max_cells, add more high-confidence cells
    if len(selected) < max_cells:
        all_cells_sorted = sorted(cells, key=lambda c: c.confidence, reverse=True)
        for cell in all_cells_sorted:
            if cell not in selected:
                selected.append(cell)
                if len(selected) >= max_cells:
                    break
    
    return selected[:max_cells]
