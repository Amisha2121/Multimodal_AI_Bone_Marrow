"""
app/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
Bone Marrow Cytomorphology Assistant
AI-assisted research decision support.

Primary workflow: "Analyze New Sample"
  1. Upload one or more bone-marrow microscopy images.
  2. Automatic cell detection (OpenCV watershed).
  3. Individual cell crops → existing ViT-Small/16 + LoRA classifier.
  4. Sample-level morphology aggregation.
  5. Attention-rollout XAI on detected cells.
  6. Case-specific hybrid RAG (age-aware).
  7. Multi-agent LangGraph assessment (grounding + sufficiency).
  8. Clean, clinical user-facing result.

Secondary workflow: "Research / Dataset Evaluation"
  Inspect the 113-patient consensus foundation cohort.
  Completely isolated from fresh-case analysis.

IMPORTANT UI rules:
  - Backend details (agent names, FAISS, BM25, scores) are HIDDEN by default.
  - They are accessible under a collapsible "Technical Details" expander.
  - Missing metadata is shown as "Not provided" — never fabricated.
  - Previous case results cannot leak into a new case (keyed by case_id).
"""

import json
import logging
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from PIL import Image
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_model import PatientAnalysis
from pipeline.new_patient_engine import analyze_new_case
from context.assembler import assemble_patient_context, get_available_patient_ids
from vision.taxonomy import MORPH_CLASSES

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bone Marrow Cytomorphology Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title { font-size: 28px; font-weight: 700; color: #1E3A8A; margin-bottom: 2px; }
.sub-title  { font-size: 15px; color: #4B5563; margin-bottom: 16px; }
.section-header { font-size: 18px; font-weight: 600; color: #1F2937; margin: 18px 0 8px 0; }
.badge-ok   { background:#D1FAE5; color:#065F46; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
.badge-warn { background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
.badge-err  { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:12px; font-size:13px; font-weight:600; }
.meta-row   { display:flex; gap:24px; flex-wrap:wrap; margin-bottom:12px; }
.meta-item  { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 16px; min-width:130px; }
.meta-label { font-size:11px; color:#6B7280; text-transform:uppercase; letter-spacing:0.04em; }
.meta-value { font-size:16px; font-weight:600; color:#111827; }
.disclaimer { background:#FEF3C7; border-left:4px solid #F59E0B; padding:12px 16px; border-radius:4px; margin:12px 0; font-size:13px; }
.not-provided { color: #9CA3AF; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Cytomorphology Assistant")
st.sidebar.caption("AI-Assisted Bone Marrow Decision Support")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Mode:",
    options=["Analyze New Sample", "Research / Dataset Evaluation"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:12px; color:#6B7280;'>
<b>Model:</b> ViT-Small/16 + LoRA<br>
<b>Training:</b> 113-patient pediatric cohort<br>
<b>Classes:</b> 8 morphology categories<br>
<b>Purpose:</b> Research decision support
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Bone Marrow Cytomorphology Assistant</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<b>RESEARCH DECISION-SUPPORT PROTOTYPE ONLY.</b>
Outputs are generated for computational pathology research and do <b>NOT</b>
constitute clinical diagnosis or medical advice. All AI-generated findings
require evaluation by a board-certified haematologist/pathologist alongside
flow cytometry, cytogenetics, and appropriate laboratory testing.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# WORKFLOW 1: ANALYZE NEW SAMPLE
# =============================================================================
if app_mode == "Analyze New Sample":

    st.markdown("---")

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Upload Bone-Marrow Image(s)</div>', unsafe_allow_html=True)
    st.caption(
        "Upload one or more bone-marrow microscopy images. "
        "The system will automatically detect individual cells and classify each one. "
        "Do NOT resize or crop cells manually — upload the full microscopy image."
    )

    uploaded_files = st.file_uploader(
        "Drop bone-marrow image files here:",
        type=None,
        accept_multiple_files=True,
        help=(
            "Supported formats: PNG, JPG, TIFF, BMP, WEBP, etc. "
            "Upload raw bone-marrow smear images. Each image will be scanned "
            "for individual cells automatically."
        ),
        label_visibility="collapsed",
    )

    # ── Optional metadata ─────────────────────────────────────────────────────
    with st.expander("Optional Patient Metadata", expanded=False):
        st.caption("All fields are optional. Leave blank if unknown — values will not be fabricated.")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            age_raw = st.number_input(
                "Patient Age (years):", min_value=0.0, max_value=120.0, value=0.0, step=0.5,
                help="Leave at 0 if not known."
            )
            parsed_age: Optional[float] = float(age_raw) if age_raw > 0 else None
        with col_m2:
            sex_raw = st.selectbox("Biological Sex:", options=["Not provided", "Female", "Male"])
            parsed_sex: Optional[str] = sex_raw if sex_raw != "Not provided" else None
        with col_m3:
            site_raw = st.selectbox(
                "Aspirate / Biopsy Site:",
                options=["Not provided", "Posterior Iliac Crest", "Anterior Iliac Crest", "Sternum", "Other"],
            )
            parsed_site: Optional[str] = site_raw if site_raw != "Not provided" else None

    # ── Analyze button ────────────────────────────────────────────────────────
    analyze_clicked = st.button("Analyze Sample", type="primary", use_container_width=True)

    # Session state
    if "case_result" not in st.session_state:
        st.session_state.case_result = None
    if "last_file_count" not in st.session_state:
        st.session_state.last_file_count = 0

    # Clear previous result when new files are added
    if uploaded_files and len(uploaded_files) != st.session_state.last_file_count:
        st.session_state.case_result = None
        st.session_state.last_file_count = len(uploaded_files)

    if analyze_clicked:
        if not uploaded_files:
            st.error("Please upload at least one bone-marrow microscopy image.")
        else:
            progress_bar = st.progress(0, text="Initialising case...")
            with st.status("Processing sample...", expanded=True) as status:
                st.write("Validating uploaded images...")
                progress_bar.progress(10, text="Validating images...")

                st.write("Detecting individual cells in each image...")
                progress_bar.progress(25, text="Detecting cells...")

                st.write("Running ViT-Small/16 + LoRA morphology classifier on each cell...")
                progress_bar.progress(50, text="Classifying cells...")

                st.write("Aggregating morphology distribution across all detected cells...")
                progress_bar.progress(65, text="Aggregating...")

                st.write("Generating attention-rollout explainability for representative cells...")
                progress_bar.progress(75, text="Generating XAI...")

                st.write("Retrieving relevant clinical guideline evidence...")
                progress_bar.progress(85, text="Retrieving evidence...")

                st.write("Running multi-agent clinical assessment...")
                progress_bar.progress(95, text="Running agents...")

                result = analyze_new_case(
                    uploaded_files=uploaded_files,
                    age_years=parsed_age,
                    sex=parsed_sex,
                    sample_site=parsed_site,
                )
                st.session_state.case_result = result
                progress_bar.progress(100, text="Complete.")
                status.update(label="Analysis complete", state="complete", expanded=False)

    analysis: Optional[PatientAnalysis] = st.session_state.case_result

    # ── Results ───────────────────────────────────────────────────────────────
    if analysis is not None:
        st.markdown("---")

        # ── Case banner ───────────────────────────────────────────────────────
        col_id, col_badge = st.columns([4, 1])
        with col_id:
            st.markdown(f"## Case: `{analysis.case_id}`")
        with col_badge:
            if analysis.is_zero_cells:
                st.markdown('<span class="badge-err">No cells detected</span>', unsafe_allow_html=True)
            elif analysis.is_low_evidence:
                st.markdown('<span class="badge-warn">Low evidence</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-ok">Analysis complete</span>', unsafe_allow_html=True)
        st.markdown("### Visual Evidence & Detection Mapping")
        st.caption("Microscopy smear scanned across field of view. Detected cells are outlined with Cell IDs.")

        for idx, info in enumerate(analysis.image_infos):
            c_img, c_qc = st.columns([3, 1])
            with c_img:
                tab_filt, tab_raw, tab_sheet = st.tabs([
                    "Filtered Detections (View 2)",
                    "Raw Candidates (View 1)",
                    "50-Crop Contact Sheet (View 3)",
                ])
                raw_path = getattr(info, "debug_raw_path", "")
                filt_path = getattr(info, "debug_filtered_path", "")
                sheet_path = getattr(info, "debug_contact_sheet_path", "")
                ann_path = getattr(info, "annotated_image_path", "")
                orig_path = getattr(info, "original_image_path", "")

                with tab_filt:
                    if filt_path and Path(filt_path).exists():
                        st.image(
                            filt_path,
                            caption=f"{info.image_id} ({info.filename}) — Filtered genuine individual cells (Cyan: standard, Orange: uncertain)",
                            use_container_width=True,
                        )
                    elif ann_path and Path(ann_path).exists():
                        st.image(
                            ann_path,
                            caption=f"{info.image_id} ({info.filename}) — Bounding boxes with Cell IDs",
                            use_container_width=True,
                        )
                    elif orig_path and Path(orig_path).exists():
                        st.image(
                            orig_path,
                            caption=f"{info.image_id} — {info.status}",
                            use_container_width=True,
                        )
                with tab_raw:
                    if raw_path and Path(raw_path).exists():
                        st.image(
                            raw_path,
                            caption=f"{info.image_id} — Raw candidate detections prior to strict morphological filtering & NMS",
                            use_container_width=True,
                        )
                    else:
                        st.info("Raw candidate debug view not available for this run.")
                with tab_sheet:
                    if sheet_path and Path(sheet_path).exists():
                        st.image(
                            sheet_path,
                            caption=f"{info.image_id} — Contact sheet of individual cell crops with Cell ID, BBox dimensions, and confidence",
                            use_container_width=True,
                        )
                    else:
                        st.info("Detection crop contact sheet not available for this run.")

            with c_qc:
                st.markdown(f"**Image Detection Summary**")
                st.markdown(f"- **Filename:** `{info.filename}`")
                st.markdown(f"- **Dimensions:** {info.width_px} × {info.height_px} px")
                st.markdown(f"- **Cells accepted:** **{info.cells_detected}**")
                st.markdown(f"- **Accepted for ViT:** **{info.cells_detected}**")
                st.markdown(f"- **Detection status:** `{info.status}`")

                # Show detection QC metrics if available
                det_results = getattr(analysis, "_detection_results", [])
                curr_det = next((d for d in det_results if d.image_id == info.image_id), None)
                if curr_det and curr_det.qc:
                    qc = curr_det.qc
                    st.markdown("---")
                    st.markdown("**Diagnostic Metrics**")
                    raw_c = qc.get("raw_candidates_count", 0)
                    acc_c = qc.get("accepted_cells", info.cells_detected)
                    valid_rate = (acc_c / max(1, raw_c)) * 100.0
                    st.markdown(f"- **Raw Candidates:** {raw_c}")
                    st.markdown(f"- **Valid Detection Rate:** **{valid_rate:.1f}%**")
                    st.markdown(f"- **Rejected Count:** {qc.get('rejected_count', 0)}")
                    if "median_bbox_dims" in qc:
                        med_w, med_h = qc["median_bbox_dims"]
                        st.markdown(f"- **Median BBox:** {med_w:.0f} x {med_h:.0f} px")
                    if "median_coverage_pct" in qc:
                        st.markdown(f"- **Median Coverage:** {qc['median_coverage_pct']:.3f}%")
                    if "median_foreground_pct" in qc:
                        st.markdown(f"- **Median Foreground%:** {qc['median_foreground_pct']:.1f}%")
                    if "median_aspect_ratio" in qc:
                        st.markdown(f"- **Median Aspect Ratio:** {qc['median_aspect_ratio']:.2f}")
                    border_clipped = qc.get("border_clipped_cells", 0)
                    overlap_cells  = qc.get("cells_with_overlaps", 0)
                    if border_clipped > 0:
                        st.markdown(f"- **Border-Clipped:** {border_clipped} cells")
                    if overlap_cells > 0:
                        st.markdown(f"- **Cells With Neighbour Overlaps:** {overlap_cells}")

                if info.status == "detection_failed":
                    st.warning(f" {info.error_message}")
                elif info.status == "error":
                    st.error(f" {info.error_message}")

        # ── Key Summary Metrics ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images Analyzed", analysis.total_images_processed)
        m2.metric("Cells Detected", analysis.total_cells_detected)
        m3.metric("Cells Classified", analysis.cells_classified)
        blast_pct = f"{analysis.blast_fraction:.1%}" if analysis.cells_classified > 0 else "N/A"
        m4.metric("Detected-Cell Blast Proportion", blast_pct, help="Proportion among detected cells only. Does NOT replace clinical 500-cell differential.")

        # ── Zero-cell / low-evidence alerts ──────────────────────────────────
        if analysis.is_zero_cells:
            st.error(
                "**No cells detected.** The detector could not identify individual cells in the submitted image(s). "
                "Downstream morphology analysis and AI assessment are not possible.\n\n"
                f"**Details:** {analysis.status_message}"
            )
            st.markdown(analysis.final_report)
            st.stop()

        elif analysis.is_low_evidence:
            st.warning(
                f"**LOW VISUAL EVIDENCE:** Only **{analysis.cells_classified} cells** were reliably detected "
                f"(threshold: 50 cells). Population-level morphology statistics carry **high sampling uncertainty**. "
                "Individual cell classifications are displayed below, but population assessment is marked as insufficient."
            )

        st.markdown("---")

        # ── Cell Morphology Distribution ──────────────────────────────────────
        st.markdown("### Cell Morphology")
        st.caption(
            f"Based on {analysis.cells_classified} individually detected and classified cells across {analysis.total_images_processed} image(s). "
            "Detected-cell proportion — does not replace a clinical 500-cell aspirate differential."
        )

        morph_df = pd.DataFrame([
            {
                "Morphology": k,
                "Cells": analysis.morphology_counts.get(k, 0),
                "Percentage": f"{analysis.morphology_percentages.get(k, 0.0):.1%}",
            }
            for k in MORPH_CLASSES
            if analysis.morphology_counts.get(k, 0) > 0
        ])

        col_tbl, col_chart = st.columns([1, 1])
        with col_tbl:
            st.dataframe(morph_df, use_container_width=True, hide_index=True)

            dominant = analysis.predominant_categories[0] if analysis.predominant_categories else "—"
            if not analysis.is_low_evidence:
                st.info(
                    f"**Predominant observed morphology:** {dominant}  \n"
                    f"**Detected-cell blast proportion:** {analysis.blast_fraction:.1%}  \n"
                    "*(Calculated exclusively from detected cells in this image — not a clinical blast %)*"
                )
            else:
                st.warning(
                    f"**Observed distribution:** {dominant} ({analysis.morphology_counts.get(dominant, 0)} cells).  \n"
                    "Insufficient visual evidence for population-level conclusions."
                )

        with col_chart:
            chart_data = {k: analysis.morphology_counts.get(k, 0) for k in MORPH_CLASSES if analysis.morphology_counts.get(k, 0) > 0}
            if chart_data:
                chart_df = pd.DataFrame.from_dict({"Count": chart_data})
                st.bar_chart(chart_df)

        st.markdown("---")

        # ── Representative Cell Crops & XAI ───────────────────────────────────
        st.markdown("### Representative Cell Crops & Explainability (XAI)")
        st.caption(
            "Attention rollout shows where the model's visual representation attended within each individual detected cell crop. "
            "It is an explainability visualization, not proof of a biological or clinical diagnosis."
        )

        if not analysis.xai_examples:
            st.info("XAI overlays could not be generated for this case.")
        else:
            xai_cols = st.columns(min(len(analysis.xai_examples), 3))
            for i, xai in enumerate(analysis.xai_examples):
                col = xai_cols[i % len(xai_cols)]
                with col:
                    if Path(xai.overlay_path).exists():
                        st.image(
                            xai.overlay_path,
                            caption=(
                                f"**{xai.cell_id}**\n"
                                f"Prediction: {xai.predicted_class} · Cell Confidence: {xai.confidence:.1%}"
                            ),
                            use_container_width=True,
                        )

        st.markdown("---")

        # ── AI-Assisted Assessment ────────────────────────────────────
        st.markdown("### AI-Assisted Assessment")
        st.caption(
            "Generated dynamically from current detected cells + retrieved medical guidelines. "
            "Based strictly on evidence available for this case."
        )

        if analysis.final_report:
            st.markdown(analysis.final_report)
        else:
            st.info("Assessment not yet generated.")

        st.markdown("---")

        # ── Medical Evidence ──────────────────────────────────────────────────
        st.markdown("### Medical Evidence")
        st.caption(f"Retrieved guideline passages relevant to detected morphological profile (Query: `{analysis.retrieved_query}`)")

        if not analysis.retrieved_evidence:
            st.info("No guideline passages retrieved for this case.")
        else:
            for chunk in analysis.retrieved_evidence[:3]:
                with st.expander(f"**{chunk.get('title', 'Passage')}** — {chunk.get('source', '')}", expanded=False):
                    st.write(chunk.get("text", ""))
                    if chunk.get("blast_threshold_rule"):
                        st.caption(f"Relevant rule criteria: {chunk.get('blast_threshold_rule')}")

        st.markdown("---")

        # ── Limitations ───────────────────────────────────────────────────────
        st.markdown("### Limitations")

        lim_col1, lim_col2 = st.columns(2)
        with lim_col1:
            st.markdown("""
**Ancillary modalities not provided:**
- Complete Blood Count (CBC):  Not provided
- Flow Cytometric Immunophenotyping:  Not provided
- Cytogenetics / Karyotype:  Not provided
- Molecular / NGS:  Not provided
""")
        with lim_col2:
            st.warning(
                " **Model Training Context:**  \n"
                "The vision model was developed using a pediatric bone-marrow research cohort "
                "(113 patients, ~20,852 annotated cells). It has not been validated for adult patients "
                "or general clinical deployment. A fresh case without age metadata is treated as age-unspecified."
            )

        # ── Technical Details (collapsed by default) ──────────────────────────
        with st.expander(" Technical Details", expanded=False):
            st.markdown("*This section is intended for research and development review only.*")

            # Evidence quality
            st.markdown("**Evidence Quality**")
            tq1, tq2, tq3 = st.columns(3)
            tq1.metric("Grounding Score", f"{analysis.critic_grounding_score:.2f} / 1.00")
            tq2.metric("Evidence Sufficiency", analysis.evidence_sufficiency)
            tq3.metric("Verification", "APPROVED" if analysis.critic_approved else "REQUIRES REVIEW")

            # Detection QC per image
            st.markdown("**Cell Detection Quality Control & Debug Metrics**")
            for det in getattr(analysis, "_detection_results", []):
                qc = getattr(det, "qc", {})
                if qc:
                    med_w, med_h = qc.get("median_bbox_dims", (0, 0))
                    min_w, min_h = qc.get("min_bbox_dims", (0, 0))
                    max_w, max_h = qc.get("max_bbox_dims", (0, 0))
                    st.markdown(f"- **Image `{qc.get('image_id')}` ({qc.get('image_dims', (0,0))[0]}×{qc.get('image_dims', (0,0))[1]} px)**")
                    st.markdown(
                        f"  - **Raw candidates generated:** {qc.get('raw_candidates_count', 0)}  \n"
                        f"  - **Accepted individual cells:** **{qc.get('accepted_cells', 0)}**  \n"
                        f"  - **Rejected candidates:** {qc.get('rejected_count', 0)} {qc.get('rejected_reasons', {})}  \n"
                        f"  - **Flagged uncertain:** {qc.get('uncertain_cells', 0)}  \n"
                        f"  - **Median box dimensions:** {med_w:.1f} × {med_h:.1f} px  \n"
                        f"  - **Bounding box range:** {min_w}×{min_h} px (min) to {max_w}×{max_h} px (max)  \n"
                        f"  - **Median image coverage:** {qc.get('median_coverage_pct', 0):.3f}% (max: {qc.get('max_coverage_pct', 0):.2f}%)"
                    )

            if not getattr(analysis, "_detection_results", []):
                for info in analysis.image_infos:
                    st.markdown(f"- **{info.image_id}** (`{info.filename}`): {info.cells_detected} cells accepted · Status: `{info.status}` · Dims: {info.width_px}x{info.height_px}px")

            # Per-cell detection quality table
            all_pcq = []
            for det in getattr(analysis, "_detection_results", []):
                pcq = (det.qc or {}).get("per_cell_quality", [])
                all_pcq.extend(pcq)
            if all_pcq:
                st.markdown("**Per-Cell Detection Quality (first 100)**")
                st.caption(
                    "Each row = one automatically detected region. "
                    "Aspect Ratio near 1.0 = compact/round. FG% = foreground pixel fraction. "
                    "Ctr = cell not clipped by image border. Ovlp = neighbour box overlaps."
                )
                pcq_df = pd.DataFrame([
                    {
                        "Cell ID":      q["cell_id"].split("_")[-1],
                        "BBox (x,y,w,h)": str(q["bbox"]),
                        "Area (px)": q["area_px"],
                        "Aspect Ratio": f"{q['aspect_ratio']:.2f}",
                        "FG%": f"{q['foreground_pct']:.0f}%",
                        "Nuc px": q["nuc_pixel_count"],
                        "Ctr": "Yes" if q["is_centered"] else "No",
                        "Ovlp": q["neighbour_overlaps"],
                        "Score": f"{q['confidence']:.0%}",
                        "Uncertain": "Yes" if q["is_uncertain"] else "No",
                    }
                    for q in all_pcq[:100]
                ])
                st.dataframe(pcq_df, use_container_width=True, hide_index=True)
                if len(all_pcq) > 100:
                    st.caption(f"... and {len(all_pcq) - 100} more detected regions.")

            # Model confidence
            if analysis.confidence_summary:
                st.markdown("**Model Confidence Summary**")
                conf = analysis.confidence_summary
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Confidence", f"{conf.get('mean_confidence', 0):.1%}")
                c2.metric("Min Confidence", f"{conf.get('min_confidence', 0):.1%}")
                c3.metric("Max Confidence", f"{conf.get('max_confidence', 0):.1%}")

            # Individual cell predictions table
            if analysis.cell_predictions:
                st.markdown("**Individual Cell Predictions (first 50)**")
                cell_df = pd.DataFrame([
                    {
                        "Cell ID": cp.cell_id,
                        "Source Image": cp.image_id,
                        "BBox (x,y,w,h)": str(cp.bbox),
                        "Predicted Class": cp.predicted_class,
                        "Confidence": f"{cp.confidence:.1%}",
                        "Uncertain": "Yes" if cp.is_uncertain else "No",
                    }
                    for cp in analysis.cell_predictions[:50]
                ])
                st.dataframe(cell_df, use_container_width=True, hide_index=True)
                if len(analysis.cell_predictions) > 50:
                    st.caption(f"... and {len(analysis.cell_predictions) - 50} more cells.")

            # Agent raw draft
            if analysis.draft_differential:
                st.markdown("**Agent Draft (before critic)**")
                st.text_area("Draft", analysis.draft_differential, height=200, disabled=True)

            # RAG technical detail
            if analysis.retrieved_evidence:
                st.markdown("**Retrieved Passages (full)**")
                for chunk in analysis.retrieved_evidence:
                    st.json({
                        "chunk_id": chunk.get("chunk_id"),
                        "title": chunk.get("title"),
                        "source": chunk.get("source"),
                        "hybrid_rrf_score": chunk.get("hybrid_rrf_score"),
                    })


# =============================================================================
# WORKFLOW 2: RESEARCH / DATASET EVALUATION (preserved, isolated)
# =============================================================================
else:
    st.markdown("---")
    st.markdown("###  Research / Dataset Evaluation Mode")
    st.caption(
        "Inspect and evaluate the 113-patient consensus foundation cohort used to train the vision model. "
        "This mode is completely isolated from the 'Analyze New Sample' workflow."
    )
    st.info(
        " This mode accesses pre-computed outputs from the fixed 113-patient research cohort. "
        "Results here reflect annotated ground-truth labels and are not equivalent to fresh-case analysis."
    )

    available_patients = get_available_patient_ids()
    if not available_patients:
        st.error("No patients found in the research cohort. Run the data pipeline first.")
        st.stop()

    default_patient = "02522A2B2726FB0A03BB19F2D8D9524D"
    if default_patient not in available_patients:
        default_patient = available_patients[0]
    default_idx = available_patients.index(default_patient)

    selected_pid = st.selectbox(
        "Select Research Cohort Patient:",
        options=available_patients,
        index=default_idx,
        help="These are the 113 patients from the BoneMarrowWSI-PediatricLeukemia consensus cohort.",
    )

    with st.spinner(f"Loading cohort data for {selected_pid[:8]}..."):
        ctx = assemble_patient_context(patient_id=selected_pid, include_rag=True, include_agent=True)

    d_demog = ctx["demographics"]
    d_morph = ctx["morphology"]

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Patient ID", f"`{selected_pid[:12]}...`")
        st.caption(f"Split: `{d_demog.get('split')}`")
    with col_r2:
        st.metric("Annotated Cells", d_morph["total_annotated_cells"])
        st.caption(f"Ground Truth: `{d_demog['ground_truth']['primary_4class']}`")
    with col_r3:
        st.metric("Annotated Blast %", f"{d_morph['annotated_blast_fraction']:.1%}")
    with col_r4:
        st.metric("Model Prediction", ctx["vision_predictions"].get("predicted_class", "Pending"))

    st.markdown("#### Annotated Morphology Breakdown")
    cat_df = pd.DataFrame([
        {"Category": k, "Count": d_morph["category_counts"].get(k, 0), "Percentage": f"{v:.1%}"}
        for k, v in d_morph["category_distribution"].items()
    ])
    if not cat_df.empty:
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.markdown("#### Research Cohort Agent Assessment")
    st.markdown(ctx.get("agentic_assessment", {}).get("final_report", "No report available."))

    st.info(
        "**Note:** The above assessment was generated from annotated ground-truth morphology data, "
        "not from a live uploaded image. It is provided for research evaluation only."
    )
