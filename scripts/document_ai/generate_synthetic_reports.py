"""
Synthetic Hematology Report Corpus Generator
==============================================
Stage 3 — Medical Document AI Training Pipeline

Generates 300+ synthetic medical reports (15 categories x 20 reports each)
in PDF and TXT formats with ground-truth JSON annotations.

ALL reports are fictional:
- Patient names, DOBs, MRNs are procedurally generated (not real individuals)
- is_synthetic = true in every annotation
- Institutions are fictional hospital names
- Clinical values are parameterized from published normal/abnormal reference ranges

Usage:
    python scripts/document_ai/generate_synthetic_reports.py
    python scripts/document_ai/generate_synthetic_reports.py --output-dir data/document_ai --count 20
"""

import json
import hashlib
import random
import os
import sys
import argparse
import textwrap
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Try reportlab — required for PDF generation
# ---------------------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Table, TableStyle,
        Spacer, HRFlowable, PageBreak,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("WARNING: reportlab not installed — PDF generation skipped, TXT only.")

# ---------------------------------------------------------------------------
# Seed & reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

INSTITUTIONS = [
    {"id": "INST_01", "name": "St. Helena Medical Center", "city": "Riverton", "state": "CA"},
    {"id": "INST_02", "name": "Northgate University Hospital", "city": "Ashford", "state": "NY"},
    {"id": "INST_03", "name": "Meridian Hematology Laboratories", "city": "Lakewood", "state": "TX"},
    {"id": "INST_04", "name": "Crescent Oncology & Pathology Institute", "city": "Millfield", "state": "FL"},
    {"id": "INST_05", "name": "Pacific Rim Blood Disorders Center", "city": "Harborview", "state": "WA"},
]

TEMPLATES = ["TEMPLATE_A", "TEMPLATE_B", "TEMPLATE_C"]

FIRST_NAMES = [
    "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Barbara",
    "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen",
    "Charles", "Lisa", "Christopher", "Nancy", "Daniel", "Betty", "Matthew", "Margaret",
    "Anthony", "Sandra", "Mark", "Ashley", "Donald", "Dorothy", "Steven", "Kimberly",
    "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Carol",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]

CLINICIANS = [
    "Dr. A. Patel", "Dr. M. Chen", "Dr. R. Sullivan", "Dr. S. Nakamura",
    "Dr. L. Torres", "Dr. J. Williams", "Dr. K. Okonkwo", "Dr. F. Reeves",
]

PATHOLOGISTS = [
    "Dr. E. Schwartz, MD, PhD", "Dr. H. Obi, MD", "Dr. T. Lindqvist, MD",
    "Dr. V. Kapoor, MBBCh", "Dr. N. Ferreira, MD", "Dr. C. Baudin, MD, PhD",
]


# ---------------------------------------------------------------------------
# Patient identity generator
# ---------------------------------------------------------------------------
def generate_patient(seed_int: int) -> dict:
    rng = random.Random(seed_int)
    fname = rng.choice(FIRST_NAMES)
    lname = rng.choice(LAST_NAMES)
    dob = date(
        rng.randint(1940, 2000),
        rng.randint(1, 12),
        rng.randint(1, 28),
    )
    mrn_num = rng.randint(1000000, 9999999)
    return {
        "name": f"{fname} {lname}",
        "dob": dob.strftime("%d %b %Y"),
        "mrn": f"MRN-{mrn_num}",
        "sex": rng.choice(["Male", "Female"]),
        "age": (date(2026, 11, 11) - dob).days // 365,
    }


def generate_collection_date(seed_int: int) -> str:
    rng = random.Random(seed_int + 1000)
    base = date(2024, 1, 1)
    offset = rng.randint(0, 680)
    return (base + timedelta(days=offset)).strftime("%d %b %Y")


# ---------------------------------------------------------------------------
# CATEGORY DEFINITIONS — 15 report types
# ---------------------------------------------------------------------------

CATEGORY_SPECS = {
    "CAT_A_MORPHOLOGY": {
        "label": "Bone Marrow Aspirate Report",
        "subtype": "bone_marrow_aspirate",
        "code": "BMA",
        "parameters": [
            {"name": "Myeloblasts", "unit": "%", "normal_low": 0.0, "normal_high": 5.0,
             "abnormal_low": 5.0, "abnormal_high": 30.0},
            {"name": "Promyelocytes", "unit": "%", "normal_low": 1.0, "normal_high": 5.0,
             "abnormal_low": 5.0, "abnormal_high": 20.0},
            {"name": "Myelocytes", "unit": "%", "normal_low": 5.0, "normal_high": 15.0,
             "abnormal_low": 15.0, "abnormal_high": 35.0},
            {"name": "Metamyelocytes", "unit": "%", "normal_low": 5.0, "normal_high": 15.0,
             "abnormal_low": 1.0, "abnormal_high": 5.0},
            {"name": "Band Neutrophils", "unit": "%", "normal_low": 10.0, "normal_high": 30.0,
             "abnormal_low": 1.0, "abnormal_high": 10.0},
            {"name": "Segmented Neutrophils", "unit": "%", "normal_low": 10.0, "normal_high": 20.0,
             "abnormal_low": 20.0, "abnormal_high": 50.0},
            {"name": "Erythroblasts", "unit": "%", "normal_low": 15.0, "normal_high": 35.0,
             "abnormal_low": 35.0, "abnormal_high": 70.0},
            {"name": "Lymphocytes", "unit": "%", "normal_low": 5.0, "normal_high": 20.0,
             "abnormal_low": 20.0, "abnormal_high": 80.0},
            {"name": "Plasma Cells", "unit": "%", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 60.0},
            {"name": "Monocytes", "unit": "%", "normal_low": 0.5, "normal_high": 3.0,
             "abnormal_low": 3.0, "abnormal_high": 15.0},
        ],
        "extra_fields": ["Cellularity", "M:E Ratio", "Iron Stores", "Dyserythropoiesis",
                         "Dysgranulopoiesis", "Dysmegakaryopoiesis"],
        "conclusion_normal": "Normocellular marrow with trilineage haematopoiesis. No morphological evidence of malignancy.",
        "conclusion_abnormal": "Hypercellular marrow with increased blast percentage. Features consistent with myeloid neoplasm. WHO 2022 classification criteria reviewed.",
    },
    "CAT_B_BIOPSY": {
        "label": "Bone Marrow Trephine Biopsy Report",
        "subtype": "bone_marrow_biopsy",
        "code": "BMB",
        "parameters": [
            {"name": "Cellularity", "unit": "%", "normal_low": 30.0, "normal_high": 70.0,
             "abnormal_low": 70.0, "abnormal_high": 100.0},
            {"name": "Fibrosis Grade (MF)", "unit": "MF-0 to MF-3", "normal_low": 0, "normal_high": 1,
             "abnormal_low": 2, "abnormal_high": 3},
            {"name": "CD34+ Blasts", "unit": "%", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 20.0},
        ],
        "extra_fields": ["Architecture", "Vascular Pattern", "Osteoblasts", "Osteoclasts",
                         "Fat Cell Proportion", "Paratrabecular Infiltration"],
        "conclusion_normal": "Normocellular trephine with no abnormal infiltrate. MF-0 reticulin.",
        "conclusion_abnormal": "Hypercellular biopsy with MF-2 reticulin fibrosis and paratrabecular blast aggregates.",
    },
    "CAT_C_CBC": {
        "label": "Complete Blood Count Report",
        "subtype": "complete_blood_count",
        "code": "CBC",
        "parameters": [
            {"name": "Haemoglobin", "unit": "g/dL", "normal_low": 12.0, "normal_high": 16.5,
             "abnormal_low": 6.0, "abnormal_high": 12.0},
            {"name": "Haematocrit", "unit": "%", "normal_low": 36.0, "normal_high": 48.0,
             "abnormal_low": 20.0, "abnormal_high": 36.0},
            {"name": "MCV", "unit": "fL", "normal_low": 80.0, "normal_high": 100.0,
             "abnormal_low": 60.0, "abnormal_high": 80.0},
            {"name": "MCH", "unit": "pg", "normal_low": 27.0, "normal_high": 33.0,
             "abnormal_low": 18.0, "abnormal_high": 27.0},
            {"name": "MCHC", "unit": "g/dL", "normal_low": 32.0, "normal_high": 36.0,
             "abnormal_low": 28.0, "abnormal_high": 32.0},
            {"name": "RBC Count", "unit": "x10^12/L", "normal_low": 3.8, "normal_high": 5.2,
             "abnormal_low": 1.5, "abnormal_high": 3.8},
            {"name": "WBC Count", "unit": "x10^9/L", "normal_low": 4.0, "normal_high": 11.0,
             "abnormal_low": 11.0, "abnormal_high": 100.0},
            {"name": "Platelets", "unit": "x10^9/L", "normal_low": 150.0, "normal_high": 400.0,
             "abnormal_low": 20.0, "abnormal_high": 150.0},
            {"name": "Neutrophils", "unit": "x10^9/L", "normal_low": 1.8, "normal_high": 7.5,
             "abnormal_low": 7.5, "abnormal_high": 80.0},
            {"name": "Lymphocytes", "unit": "x10^9/L", "normal_low": 1.0, "normal_high": 4.0,
             "abnormal_low": 4.0, "abnormal_high": 150.0},
            {"name": "RDW", "unit": "%", "normal_low": 11.5, "normal_high": 14.5,
             "abnormal_low": 14.5, "abnormal_high": 25.0},
        ],
        "extra_fields": ["Instrument", "Sample Quality", "Blood Film Comment"],
        "conclusion_normal": "Complete blood count within normal reference intervals.",
        "conclusion_abnormal": "Pancytopenia with hypochromic microcytic red cells. Differential diagnosis includes iron deficiency, haemoglobinopathy, or bone marrow failure.",
    },
    "CAT_D_SMEAR": {
        "label": "Peripheral Blood Smear Report",
        "subtype": "peripheral_blood_smear",
        "code": "PBS",
        "parameters": [
            {"name": "Neutrophils", "unit": "%", "normal_low": 50.0, "normal_high": 70.0,
             "abnormal_low": 70.0, "abnormal_high": 95.0},
            {"name": "Lymphocytes", "unit": "%", "normal_low": 20.0, "normal_high": 40.0,
             "abnormal_low": 40.0, "abnormal_high": 95.0},
            {"name": "Monocytes", "unit": "%", "normal_low": 2.0, "normal_high": 10.0,
             "abnormal_low": 10.0, "abnormal_high": 30.0},
            {"name": "Eosinophils", "unit": "%", "normal_low": 1.0, "normal_high": 4.0,
             "abnormal_low": 4.0, "abnormal_high": 20.0},
            {"name": "Basophils", "unit": "%", "normal_low": 0.0, "normal_high": 1.0,
             "abnormal_low": 1.0, "abnormal_high": 5.0},
            {"name": "Blasts", "unit": "%", "normal_low": 0.0, "normal_high": 0.0,
             "abnormal_low": 0.1, "abnormal_high": 30.0},
        ],
        "extra_fields": ["RBC Morphology", "Platelet Morphology", "Toxic Granulation",
                         "Hypersegmentation", "Atypical Lymphocytes"],
        "conclusion_normal": "Peripheral blood smear shows normal differential. No blast forms identified. Normal red cell and platelet morphology.",
        "conclusion_abnormal": "Peripheral blood smear shows circulating blast forms. Neutrophil hypersegmentation noted. Urgent haematology review recommended.",
    },
    "CAT_E_FLOW": {
        "label": "Flow Cytometry Immunophenotyping Report",
        "subtype": "flow_cytometry",
        "code": "FC",
        "parameters": [
            {"name": "CD19+ B cells", "unit": "%", "normal_low": 5.0, "normal_high": 20.0,
             "abnormal_low": 50.0, "abnormal_high": 95.0},
            {"name": "CD3+ T cells", "unit": "%", "normal_low": 55.0, "normal_high": 80.0,
             "abnormal_low": 5.0, "abnormal_high": 55.0},
            {"name": "CD4+ Helper T", "unit": "%", "normal_low": 30.0, "normal_high": 60.0,
             "abnormal_low": 5.0, "abnormal_high": 30.0},
            {"name": "CD8+ Cytotoxic T", "unit": "%", "normal_low": 15.0, "normal_high": 35.0,
             "abnormal_low": 35.0, "abnormal_high": 80.0},
            {"name": "CD4:CD8 Ratio", "unit": "ratio", "normal_low": 1.5, "normal_high": 3.0,
             "abnormal_low": 0.1, "abnormal_high": 1.5},
            {"name": "CD34+ Blasts", "unit": "%", "normal_low": 0.0, "normal_high": 1.0,
             "abnormal_low": 1.0, "abnormal_high": 95.0},
            {"name": "NK Cells (CD16+CD56+)", "unit": "%", "normal_low": 5.0, "normal_high": 20.0,
             "abnormal_low": 1.0, "abnormal_high": 5.0},
        ],
        "extra_fields": ["Instrument", "Events Acquired", "Gating Strategy", "Antibody Panel",
                         "Light Scatter Pattern", "Clonal Population Comment"],
        "conclusion_normal": "No immunophenotypic evidence of lymphoproliferative disorder. Normal lymphocyte subset distribution.",
        "conclusion_abnormal": "Immunophenotypic findings consistent with B-cell lymphoproliferative disorder. CD19+/CD5+/CD23+ co-expression. Recommend correlation with morphology and molecular studies.",
    },
    "CAT_F_IHC": {
        "label": "Immunohistochemistry Report",
        "subtype": "immunohistochemistry",
        "code": "IHC",
        "parameters": [
            {"name": "CD3", "unit": "positive/negative", "normal_low": "N/A", "normal_high": "N/A",
             "abnormal_low": "N/A", "abnormal_high": "N/A"},
            {"name": "CD20", "unit": "positive/negative", "normal_low": "N/A", "normal_high": "N/A",
             "abnormal_low": "N/A", "abnormal_high": "N/A"},
            {"name": "CD34", "unit": "positive/negative", "normal_low": "N/A", "normal_high": "N/A",
             "abnormal_low": "N/A", "abnormal_high": "N/A"},
            {"name": "MPO", "unit": "positive/negative", "normal_low": "N/A", "normal_high": "N/A",
             "abnormal_low": "N/A", "abnormal_high": "N/A"},
            {"name": "Ki-67", "unit": "%", "normal_low": 0.0, "normal_high": 10.0,
             "abnormal_low": 10.0, "abnormal_high": 95.0},
        ],
        "extra_fields": ["Tissue Block ID", "Fixation Method", "Antigen Retrieval",
                         "Primary Antibody Clone", "Staining Pattern"],
        "conclusion_normal": "IHC staining profile within expected limits for reactive tissue.",
        "conclusion_abnormal": "IHC profile supports high-grade B-cell lymphoma. Ki-67 proliferation index elevated.",
    },
    "CAT_G_FISH": {
        "label": "FISH (Fluorescence In Situ Hybridisation) Report",
        "subtype": "fish_cytogenetics",
        "code": "FISH",
        "parameters": [
            {"name": "BCR-ABL1 Fusion", "unit": "% positive nuclei", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 95.0},
            {"name": "MYC Rearrangement", "unit": "% positive nuclei", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 95.0},
            {"name": "del(17p) TP53", "unit": "% positive nuclei", "normal_low": 0.0, "normal_high": 10.0,
             "abnormal_low": 10.0, "abnormal_high": 95.0},
            {"name": "EGR1/del(5q)", "unit": "% positive nuclei", "normal_low": 0.0, "normal_high": 10.0,
             "abnormal_low": 10.0, "abnormal_high": 95.0},
        ],
        "extra_fields": ["Probe Set", "Cells Counted", "Instrument", "Hybridisation Efficiency",
                         "Cytogenetic Result"],
        "conclusion_normal": "No pathological FISH signal pattern detected in the assessed probes.",
        "conclusion_abnormal": "BCR-ABL1 fusion signal detected in greater than 95% of nuclei counted. Consistent with CML or Ph+ ALL.",
    },
    "CAT_H_KARYOTYPE": {
        "label": "Conventional Cytogenetics / Karyotype Report",
        "subtype": "karyotyping",
        "code": "KARY",
        "parameters": [
            {"name": "Mitoses Analysed", "unit": "count", "normal_low": 20, "normal_high": 25,
             "abnormal_low": 5, "abnormal_high": 20},
            {"name": "Clonal Abnormality %", "unit": "%", "normal_low": 0.0, "normal_high": 0.0,
             "abnormal_low": 1.0, "abnormal_high": 100.0},
        ],
        "extra_fields": ["Karyotype Formula (ISCN 2020)", "Banding Resolution", "Culture Conditions",
                         "Tissue Type", "Clonal Population Comment"],
        "conclusion_normal": "46,XX (or 46,XY). No clonal cytogenetic abnormality identified in 25 mitoses analysed.",
        "conclusion_abnormal": "Complex karyotype identified: del(5q), monosomy 7, +8. High-risk cytogenetic profile per IPSS-R classification.",
    },
    "CAT_I_NGS": {
        "label": "Next-Generation Sequencing (Myeloid Panel) Report",
        "subtype": "ngs_myeloid_panel",
        "code": "NGS",
        "parameters": [
            {"name": "DNMT3A p.R882H", "unit": "VAF %", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 55.0},
            {"name": "FLT3-ITD", "unit": "AR ratio", "normal_low": 0.0, "normal_high": 0.05,
             "abnormal_low": 0.05, "abnormal_high": 5.0},
            {"name": "NPM1 Type A", "unit": "VAF %", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 60.0},
            {"name": "IDH1 p.R132H", "unit": "VAF %", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 50.0},
            {"name": "TP53 Hotspot", "unit": "VAF %", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 80.0},
            {"name": "ASXL1 p.G646fs", "unit": "VAF %", "normal_low": 0.0, "normal_high": 2.0,
             "abnormal_low": 2.0, "abnormal_high": 50.0},
        ],
        "extra_fields": ["Panel Name", "Genes Covered", "Mean Coverage Depth",
                         "Minimum Coverage Threshold", "Pipeline Version", "TMB"],
        "conclusion_normal": "No pathogenic somatic variants identified in the assessed myeloid panel genes at a VAF threshold > 2%.",
        "conclusion_abnormal": "Pathogenic somatic variants identified: NPM1 Type A mutation (VAF 46%) with concurrent FLT3-ITD (AR 0.8). AML-associated driver mutations confirmed.",
    },
    "CAT_J_PCR": {
        "label": "Molecular PCR / RT-PCR Report",
        "subtype": "molecular_pcr",
        "code": "PCR",
        "parameters": [
            {"name": "BCR-ABL1 IS", "unit": "%IS", "normal_low": 0.0, "normal_high": 0.01,
             "abnormal_low": 0.01, "abnormal_high": 100.0},
            {"name": "PML-RARA", "unit": "copies/ABL1", "normal_low": 0.0, "normal_high": 0.0,
             "abnormal_low": 0.01, "abnormal_high": 1000.0},
            {"name": "JAK2 V617F", "unit": "VAF %", "normal_low": 0.0, "normal_high": 1.0,
             "abnormal_low": 1.0, "abnormal_high": 100.0},
        ],
        "extra_fields": ["Assay Method", "Sensitivity Threshold", "Reference Gene",
                         "EAC Protocol Version", "Transcript Type"],
        "conclusion_normal": "No BCR-ABL1 transcript detected. MRD negative at current assay sensitivity.",
        "conclusion_abnormal": "BCR-ABL1 IS = 2.3% (BCR-ABL/ABL). Above major molecular response threshold. Treatment response review indicated.",
    },
    "CAT_K_IRON": {
        "label": "Iron Studies and Reticulocyte Report",
        "subtype": "iron_reticulocyte",
        "code": "IRON",
        "parameters": [
            {"name": "Serum Iron", "unit": "umol/L", "normal_low": 10.0, "normal_high": 30.0,
             "abnormal_low": 2.0, "abnormal_high": 10.0},
            {"name": "TIBC", "unit": "umol/L", "normal_low": 45.0, "normal_high": 72.0,
             "abnormal_low": 72.0, "abnormal_high": 120.0},
            {"name": "Transferrin Saturation", "unit": "%", "normal_low": 15.0, "normal_high": 50.0,
             "abnormal_low": 1.0, "abnormal_high": 15.0},
            {"name": "Serum Ferritin", "unit": "ug/L", "normal_low": 15.0, "normal_high": 150.0,
             "abnormal_low": 1.0, "abnormal_high": 15.0},
            {"name": "Reticulocyte Count", "unit": "%", "normal_low": 0.5, "normal_high": 2.5,
             "abnormal_low": 0.0, "abnormal_high": 0.5},
            {"name": "Reticulocyte Hb (CHr)", "unit": "pg", "normal_low": 28.0, "normal_high": 36.0,
             "abnormal_low": 20.0, "abnormal_high": 28.0},
        ],
        "extra_fields": ["Hepcidin Level", "Soluble Transferrin Receptor", "Erythropoietin"],
        "conclusion_normal": "Iron studies within normal limits. Adequate iron stores.",
        "conclusion_abnormal": "Iron deficiency anaemia: low ferritin, elevated TIBC, reduced transferrin saturation. Reticulocyte response appropriately reduced.",
    },
    "CAT_L_COAG": {
        "label": "Coagulation Studies Report",
        "subtype": "coagulation",
        "code": "COAG",
        "parameters": [
            {"name": "PT", "unit": "seconds", "normal_low": 11.0, "normal_high": 13.5,
             "abnormal_low": 13.5, "abnormal_high": 60.0},
            {"name": "INR", "unit": "ratio", "normal_low": 0.9, "normal_high": 1.2,
             "abnormal_low": 1.2, "abnormal_high": 10.0},
            {"name": "APTT", "unit": "seconds", "normal_low": 25.0, "normal_high": 38.0,
             "abnormal_low": 38.0, "abnormal_high": 120.0},
            {"name": "Fibrinogen", "unit": "g/L", "normal_low": 1.5, "normal_high": 4.0,
             "abnormal_low": 0.1, "abnormal_high": 1.5},
            {"name": "D-Dimer", "unit": "ug/mL FEU", "normal_low": 0.0, "normal_high": 0.5,
             "abnormal_low": 0.5, "abnormal_high": 20.0},
        ],
        "extra_fields": ["Thrombin Time", "Mixing Study Result", "Factor Activity"],
        "conclusion_normal": "Coagulation parameters within normal reference intervals.",
        "conclusion_abnormal": "Elevated D-dimer with prolonged PT and low fibrinogen. Features consistent with disseminated intravascular coagulation (DIC).",
    },
    "CAT_M_MRD": {
        "label": "Minimal Residual Disease (MRD) Assessment",
        "subtype": "mrd_assessment",
        "code": "MRD",
        "parameters": [
            {"name": "MRD Level (flow)", "unit": "% residual blasts", "normal_low": 0.0, "normal_high": 0.01,
             "abnormal_low": 0.01, "abnormal_high": 1.0},
            {"name": "MRD Level (molecular)", "unit": "copies/10^5 ABL1", "normal_low": 0.0, "normal_high": 1.0,
             "abnormal_low": 1.0, "abnormal_high": 10000.0},
        ],
        "extra_fields": ["Method (flow/PCR/NGS)", "Sensitivity Threshold", "Pre-treatment MRD",
                         "Time-point (cycle/induction)", "ELN Response Category"],
        "conclusion_normal": "MRD negative. No detectable residual disease at current assay sensitivity.",
        "conclusion_abnormal": "MRD positive. Residual disease above threshold. Treatment escalation or clinical trial consideration recommended.",
    },
    "CAT_N_TRANSPLANT": {
        "label": "Bone Marrow Transplant Engraftment Report",
        "subtype": "transplant_engraftment",
        "code": "TX",
        "parameters": [
            {"name": "Donor Chimerism", "unit": "%", "normal_low": 95.0, "normal_high": 100.0,
             "abnormal_low": 50.0, "abnormal_high": 95.0},
            {"name": "Host Residual", "unit": "%", "normal_low": 0.0, "normal_high": 5.0,
             "abnormal_low": 5.0, "abnormal_high": 50.0},
            {"name": "Neutrophil Engraftment Day", "unit": "days post-transplant", "normal_low": 10, "normal_high": 20,
             "abnormal_low": 20, "abnormal_high": 60},
        ],
        "extra_fields": ["Donor Type", "Conditioning Regimen", "GVHD Prophylaxis",
                         "STR Marker Panel", "Lineage-Specific Chimerism"],
        "conclusion_normal": "Full donor chimerism achieved. No evidence of graft failure or relapse.",
        "conclusion_abnormal": "Declining donor chimerism. Increasing mixed chimerism. Relapse or graft failure risk. Urgent clinical review indicated.",
    },
    "CAT_O_SPEP": {
        "label": "Serum Protein Electrophoresis / Immunoglobulin Report",
        "subtype": "spep_immunoglobulin",
        "code": "SPEP",
        "parameters": [
            {"name": "Total Protein", "unit": "g/L", "normal_low": 60.0, "normal_high": 80.0,
             "abnormal_low": 80.0, "abnormal_high": 120.0},
            {"name": "Albumin", "unit": "g/L", "normal_low": 35.0, "normal_high": 50.0,
             "abnormal_low": 20.0, "abnormal_high": 35.0},
            {"name": "M-Protein", "unit": "g/L", "normal_low": 0.0, "normal_high": 0.0,
             "abnormal_low": 0.1, "abnormal_high": 80.0},
            {"name": "IgG", "unit": "g/L", "normal_low": 7.0, "normal_high": 16.0,
             "abnormal_low": 16.0, "abnormal_high": 100.0},
            {"name": "IgA", "unit": "g/L", "normal_low": 0.7, "normal_high": 4.0,
             "abnormal_low": 4.0, "abnormal_high": 50.0},
            {"name": "IgM", "unit": "g/L", "normal_low": 0.4, "normal_high": 2.3,
             "abnormal_low": 2.3, "abnormal_high": 80.0},
            {"name": "Free Light Chain Kappa", "unit": "mg/L", "normal_low": 3.3, "normal_high": 19.4,
             "abnormal_low": 19.4, "abnormal_high": 5000.0},
            {"name": "Free Light Chain Lambda", "unit": "mg/L", "normal_low": 5.7, "normal_high": 26.3,
             "abnormal_low": 26.3, "abnormal_high": 2000.0},
            {"name": "Kappa:Lambda Ratio", "unit": "ratio", "normal_low": 0.26, "normal_high": 1.65,
             "abnormal_low": 1.65, "abnormal_high": 100.0},
        ],
        "extra_fields": ["Electrophoresis Pattern", "M-Protein Immunotype", "Bence-Jones Protein",
                         "Urine Protein Electrophoresis"],
        "conclusion_normal": "No paraprotein identified. Normal serum protein electrophoresis.",
        "conclusion_abnormal": "IgG kappa paraprotein detected at 42 g/L. Kappa:Lambda ratio markedly elevated. Features consistent with multiple myeloma. CRAB criteria assessment recommended.",
    },
}


# ---------------------------------------------------------------------------
# Value generator
# ---------------------------------------------------------------------------

def generate_value(param: dict, abnormal: bool, rng: random.Random) -> dict:
    """Generate a realistic parameter value, flagged if abnormal."""
    name = param["name"]
    unit = param["unit"]

    if unit in ("positive/negative",):
        # Qualitative
        if abnormal:
            value_str = rng.choice(["Positive", "Strongly Positive"])
            flag = "Positive"
        else:
            value_str = "Negative"
            flag = ""
        return {"name": name, "reported_value": value_str, "unit": unit,
                "reference_interval": "Negative", "flag": flag}

    if unit in ("MF-0 to MF-3",):
        val = int(param["abnormal_low"]) if abnormal else 0
        return {"name": name, "reported_value": val, "unit": unit,
                "reference_interval": "MF-0 to MF-1", "flag": "High" if abnormal else ""}

    # Numeric
    n_low = float(param["normal_low"])
    n_high = float(param["normal_high"])
    a_low = float(param["abnormal_low"])
    a_high = float(param["abnormal_high"])

    if abnormal:
        raw = rng.uniform(a_low, a_high)
        flag = "H" if raw > n_high else "L"
    else:
        raw = rng.uniform(n_low, n_high)
        flag = ""

    # Round sensibly
    if abs(raw) > 100:
        val_str = f"{raw:.0f}"
    elif abs(raw) > 10:
        val_str = f"{raw:.1f}"
    else:
        val_str = f"{raw:.2f}"

    ref_str = f"{n_low}–{n_high}" if isinstance(n_low, float) else f"{n_low}–{n_high}"
    return {"name": name, "reported_value": float(val_str), "unit": unit,
            "reference_interval": ref_str, "flag": flag}


# ---------------------------------------------------------------------------
# Annotation generator
# ---------------------------------------------------------------------------

def generate_annotation(
    report_id: str,
    category_key: str,
    spec: dict,
    template_id: str,
    institution: dict,
    patient: dict,
    collection_date: str,
    is_abnormal: bool,
    is_preliminary: bool,
    rng: random.Random,
) -> dict:
    parameters = []
    for param in spec["parameters"]:
        abnormal_this = is_abnormal and rng.random() < 0.6
        parameters.append(generate_value(param, abnormal_this, rng))

    return {
        "report_id": report_id,
        "is_synthetic": True,
        "category": category_key,
        "subtype": spec["subtype"],
        "template_id": template_id,
        "institution_id": institution["id"],
        "institution_name": institution["name"],
        "patient_name": patient["name"],
        "patient_dob": patient["dob"],
        "patient_mrn": patient["mrn"],
        "patient_sex": patient["sex"],
        "patient_age": patient["age"],
        "collection_date": collection_date,
        "report_status": "Preliminary" if is_preliminary else "Final",
        "is_abnormal": is_abnormal,
        "requesting_clinician": rng.choice(CLINICIANS),
        "reporting_pathologist": rng.choice(PATHOLOGISTS),
        "parameters": parameters,
        "conclusion": spec["conclusion_abnormal"] if is_abnormal else spec["conclusion_normal"],
    }


# ---------------------------------------------------------------------------
# TXT report generator
# ---------------------------------------------------------------------------

def generate_txt_report(annotation: dict, spec: dict) -> str:
    sep = "=" * 72
    thin = "-" * 72
    lines = [
        sep,
        f"  {annotation['institution_name'].upper()}",
        f"  LABORATORY REPORT — {spec['label'].upper()}",
        sep,
        "",
        f"Patient Name  : {annotation['patient_name']}",
        f"Date of Birth : {annotation['patient_dob']}   Sex: {annotation['patient_sex']}   Age: {annotation['patient_age']} years",
        f"MRN           : {annotation['patient_mrn']}",
        f"Collection    : {annotation['collection_date']}",
        f"Report Status : {annotation['report_status']}",
        f"Requesting    : {annotation['requesting_clinician']}",
        "",
        thin,
        "RESULTS",
        thin,
        "",
    ]

    # Parameters table
    lines.append(f"{'Parameter':<35} {'Value':>12}  {'Unit':<15} {'Ref Range':<20} {'Flag'}")
    lines.append(thin)
    for p in annotation["parameters"]:
        val = str(p["reported_value"])
        lines.append(
            f"{p['name']:<35} {val:>12}  {p['unit']:<15} {p['reference_interval']:<20} {p['flag']}"
        )

    lines += [
        "",
        thin,
        "CONCLUSION",
        thin,
        "",
        textwrap.fill(annotation["conclusion"], width=72),
        "",
        thin,
        f"Reported by: {annotation['reporting_pathologist']}",
        f"Report ID  : {annotation['report_id']}",
        "",
        "*** SYNTHETIC REPORT — FOR RESEARCH USE ONLY — NOT REAL CLINICAL DATA ***",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF report generator (Template A, B, C variants)
# ---------------------------------------------------------------------------

def build_pdf_template_a(annotation: dict, spec: dict, output_path: Path, rng: random.Random):
    """Template A: clean professional layout, standard fonts."""
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2.5*cm, rightMargin=2.5*cm)
    styles = getSampleStyleSheet()
    story = []

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14, spaceAfter=4,
                         textColor=colors.HexColor("#1a3a5c"), alignment=TA_CENTER)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceAfter=3,
                         textColor=colors.HexColor("#2c5f8a"))
    normal = ParagraphStyle("n", parent=styles["Normal"], fontSize=9, leading=13)
    small = ParagraphStyle("sm", parent=styles["Normal"], fontSize=8, leading=11,
                            textColor=colors.grey)

    # Header
    story.append(Paragraph(annotation["institution_name"], h1))
    story.append(Paragraph(spec["label"], h2))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 6))

    # Patient demographics
    dem_data = [
        ["Patient:", annotation["patient_name"], "MRN:", annotation["patient_mrn"]],
        ["DOB:", annotation["patient_dob"], "Sex:", annotation["patient_sex"]],
        ["Collection:", annotation["collection_date"], "Status:", annotation["report_status"]],
        ["Requesting:", annotation["requesting_clinician"], "Report ID:", annotation["report_id"]],
    ]
    dem_table = Table(dem_data, colWidths=[3*cm, 7*cm, 3*cm, 5*cm])
    dem_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#1a3a5c")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(dem_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    # Results heading
    story.append(Paragraph("RESULTS", h2))

    # Results table
    res_header = ["Parameter", "Value", "Unit", "Reference Range", "Flag"]
    res_data = [res_header]
    for p in annotation["parameters"]:
        flag_txt = p["flag"]
        res_data.append([
            p["name"],
            str(p["reported_value"]),
            p["unit"],
            p["reference_interval"],
            flag_txt,
        ])

    col_w = [5.5*cm, 2.5*cm, 2.5*cm, 4*cm, 1.5*cm]
    res_table = Table(res_data, colWidths=col_w, repeatRows=1)
    res_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Highlight abnormal flags
    for i, p in enumerate(annotation["parameters"], start=1):
        if p["flag"] in ("H", "L", "High", "Positive", "Strongly Positive"):
            res_style.append(("TEXTCOLOR", (4, i), (4, i), colors.red))
            res_style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
    res_table.setStyle(TableStyle(res_style))
    story.append(res_table)
    story.append(Spacer(1, 10))

    # Conclusion
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    story.append(Paragraph("CONCLUSION", h2))
    story.append(Paragraph(annotation["conclusion"], normal))
    story.append(Spacer(1, 10))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Reported by: {annotation['reporting_pathologist']}", normal))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "SYNTHETIC REPORT — FOR RESEARCH USE ONLY — NOT REAL CLINICAL DATA",
        ParagraphStyle("warn", parent=styles["Normal"], fontSize=7,
                        textColor=colors.red, alignment=TA_CENTER)))

    doc.build(story)


def build_pdf_template_b(annotation: dict, spec: dict, output_path: Path, rng: random.Random):
    """Template B: compact multi-column header, slightly different styling."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    h1 = ParagraphStyle("h1b", parent=styles["Heading1"], fontSize=12, spaceAfter=2,
                         textColor=colors.HexColor("#5c0a1a"), alignment=TA_LEFT)
    h2 = ParagraphStyle("h2b", parent=styles["Heading2"], fontSize=10,
                         textColor=colors.HexColor("#8a2c2c"))
    normal = ParagraphStyle("nb", parent=styles["Normal"], fontSize=8.5, leading=12)

    story.append(Paragraph(f"{annotation['institution_name']} | {spec['label']}", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#5c0a1a")))
    story.append(Spacer(1, 4))

    info_data = [
        [f"Patient: {annotation['patient_name']}", f"MRN: {annotation['patient_mrn']}",
         f"DOB: {annotation['patient_dob']}", f"Sex: {annotation['patient_sex']}"],
        [f"Collection: {annotation['collection_date']}", f"Status: {annotation['report_status']}",
         f"Clinician: {annotation['requesting_clinician']}", f"ID: {annotation['report_id']}"],
    ]
    info_table = Table(info_data, colWidths=["25%", "25%", "30%", "20%"])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf0f0")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Laboratory Results", h2))

    res_header = ["Test", "Result", "Units", "Reference", "Interp."]
    res_data = [res_header]
    for p in annotation["parameters"]:
        res_data.append([p["name"], str(p["reported_value"]), p["unit"],
                         p["reference_interval"], p["flag"]])

    res_table = Table(res_data, colWidths=[5*cm, 2.5*cm, 2.5*cm, 4*cm, 1.5*cm], repeatRows=1)
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5c0a1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Conclusion:", h2))
    story.append(Paragraph(annotation["conclusion"], normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Authorised: {annotation['reporting_pathologist']}", normal))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "SYNTHETIC REPORT — RESEARCH USE ONLY — NOT REAL DATA",
        ParagraphStyle("warnb", parent=styles["Normal"], fontSize=6.5,
                        textColor=colors.red, alignment=TA_CENTER)))
    doc.build(story)


def build_pdf_template_c(annotation: dict, spec: dict, output_path: Path, rng: random.Random):
    """Template C: header-heavy formal layout — held-out test template."""
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            topMargin=3*cm, bottomMargin=2*cm,
                            leftMargin=3*cm, rightMargin=3*cm)
    styles = getSampleStyleSheet()
    story = []

    h1 = ParagraphStyle("h1c", parent=styles["Heading1"], fontSize=16, spaceAfter=6,
                         textColor=colors.HexColor("#0d4f1a"), alignment=TA_CENTER)
    h2 = ParagraphStyle("h2c", parent=styles["Heading2"], fontSize=10, spaceAfter=4,
                         textColor=colors.HexColor("#0d4f1a"))
    normal = ParagraphStyle("nc", parent=styles["Normal"], fontSize=9, leading=14)
    italic = ParagraphStyle("ic", parent=styles["Normal"], fontSize=8, leading=12,
                             textColor=colors.HexColor("#555555"))

    story.append(Paragraph("HAEMATOLOGY AND PATHOLOGY SERVICES", h1))
    story.append(Paragraph(annotation["institution_name"].upper(), h1))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor("#0d4f1a")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(spec["label"].upper(), h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#0d4f1a")))
    story.append(Spacer(1, 8))

    # Formal patient info box
    dem_data = [
        ["PATIENT IDENTIFICATION", "", "", ""],
        ["Full Name:", annotation["patient_name"], "MRN:", annotation["patient_mrn"]],
        ["Date of Birth:", annotation["patient_dob"], "Gender:", annotation["patient_sex"]],
        ["Age:", f"{annotation['patient_age']} years", "", ""],
        ["SPECIMEN DETAILS", "", "", ""],
        ["Collection Date:", annotation["collection_date"], "Report Status:", annotation["report_status"]],
        ["Requesting Physician:", annotation["requesting_clinician"], "Report No.:", annotation["report_id"]],
        ["Reporting Pathologist:", annotation["reporting_pathologist"], "", ""],
    ]
    dem_table = Table(dem_data, colWidths=[4*cm, 6.5*cm, 3*cm, 4*cm])
    dem_style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d4f1a")),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#0d4f1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 4), (-1, 4), colors.white),
        ("SPAN", (0, 0), (-1, 0)),
        ("SPAN", (0, 4), (-1, 4)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 1), (-1, 3), 0.3, colors.HexColor("#ccddcc")),
        ("GRID", (0, 5), (-1, -1), 0.3, colors.HexColor("#ccddcc")),
    ]
    dem_table.setStyle(TableStyle(dem_style))
    story.append(dem_table)
    story.append(Spacer(1, 10))

    # Results
    story.append(Paragraph("ANALYTICAL RESULTS", h2))
    res_header = ["Parameter", "Reported Value", "Unit", "Reference Interval", "Interpretation"]
    res_data = [res_header]
    for p in annotation["parameters"]:
        res_data.append([p["name"], str(p["reported_value"]), p["unit"],
                         p["reference_interval"], p["flag"] or "Within range"])

    col_w = [5*cm, 2.5*cm, 2.5*cm, 4.5*cm, 2*cm]
    res_table = Table(res_data, colWidths=col_w, repeatRows=1)
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d4f1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccddcc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdf4")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#0d4f1a")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("CONCLUSIONS AND COMMENTS", h2))
    story.append(Paragraph(annotation["conclusion"], normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report is issued in accordance with laboratory accreditation standards.",
        italic))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "SYNTHETIC REPORT — FOR AI RESEARCH AND TRAINING ONLY — CONTAINS NO REAL PATIENT DATA",
        ParagraphStyle("warnc", parent=styles["Normal"], fontSize=7,
                        textColor=colors.red, alignment=TA_CENTER)))
    doc.build(story)


PDF_BUILDERS = {
    "TEMPLATE_A": build_pdf_template_a,
    "TEMPLATE_B": build_pdf_template_b,
    "TEMPLATE_C": build_pdf_template_c,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_corpus(output_dir: Path, reports_per_category: int = 20):
    rng_master = random.Random(GLOBAL_SEED)

    corpus_dir = output_dir / "synthetic_corpus"
    annot_dir = output_dir / "annotations"
    for d in [corpus_dir, annot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    report_counter = 1

    categories = list(CATEGORY_SPECS.keys())
    templates = TEMPLATES
    institutions = INSTITUTIONS

    total_planned = len(categories) * reports_per_category
    print(f"Generating {total_planned} synthetic reports "
          f"({len(categories)} categories x {reports_per_category} each)...")

    for cat_key in categories:
        spec = CATEGORY_SPECS[cat_key]
        code = spec["code"]
        cat_reports = 0

        # Distribute reports across templates evenly
        # For reports_per_category=20: ~7 TEMPLATE_A, ~7 TEMPLATE_B, ~6 TEMPLATE_C
        template_distribution = []
        for i in range(reports_per_category):
            template_distribution.append(templates[i % len(templates)])
        rng_master.shuffle(template_distribution)

        for idx in range(reports_per_category):
            report_id = f"SYNTH_{code}_{report_counter:04d}"
            template_id = template_distribution[idx]
            institution = institutions[idx % len(institutions)]

            patient_seed = GLOBAL_SEED + report_counter * 7
            patient = generate_patient(patient_seed)
            collection_date = generate_collection_date(patient_seed)

            # Mix normal and abnormal
            is_abnormal = (idx % 3 != 0)  # ~2/3 abnormal
            is_preliminary = (idx % 7 == 0)  # ~1/7 preliminary

            rng = random.Random(GLOBAL_SEED + report_counter * 13)
            annotation = generate_annotation(
                report_id=report_id,
                category_key=cat_key,
                spec=spec,
                template_id=template_id,
                institution=institution,
                patient=patient,
                collection_date=collection_date,
                is_abnormal=is_abnormal,
                is_preliminary=is_preliminary,
                rng=rng,
            )
            all_annotations.append(annotation)

            # Save annotation JSON
            annot_path = annot_dir / f"{report_id}_annotation.json"
            with open(annot_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, default=str)

            # Save TXT report
            txt_path = corpus_dir / f"{report_id}.txt"
            txt_content = generate_txt_report(annotation, spec)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)

            # Save PDF report
            if HAS_REPORTLAB:
                pdf_path = corpus_dir / f"{report_id}.pdf"
                try:
                    builder_fn = PDF_BUILDERS[template_id]
                    builder_fn(annotation, spec, pdf_path, rng)
                except Exception as e:
                    print(f"  WARNING: PDF generation failed for {report_id}: {e}")

            report_counter += 1
            cat_reports += 1

        print(f"  [{cat_key}] {cat_reports} reports generated.")

    print(f"\nTotal reports generated: {report_counter - 1}")
    print(f"Annotations: {len(all_annotations)}")

    # Save master annotation list
    master_path = output_dir / "manifests" / "synthetic_annotations_master.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(all_annotations),
            "categories": len(categories),
            "reports_per_category": reports_per_category,
            "generated_at": "2026-11-11",
            "is_synthetic": True,
            "reports": [a["report_id"] for a in all_annotations],
        }, f, indent=2)
    print(f"Master manifest saved: {master_path}")

    return all_annotations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic hematology report corpus")
    parser.add_argument("--output-dir", type=str,
                        default="data/document_ai",
                        help="Root output directory (default: data/document_ai)")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of reports per category (default: 20, minimum 20)")
    args = parser.parse_args()

    count = max(20, args.count)
    out_dir = Path(args.output_dir)

    annotations = generate_corpus(output_dir=out_dir, reports_per_category=count)
    print("\nDone. Corpus generation complete.")
    print(f"  Corpus:      {out_dir / 'synthetic_corpus'}")
    print(f"  Annotations: {out_dir / 'annotations'}")
    print(f"  Manifest:    {out_dir / 'manifests' / 'synthetic_annotations_master.json'}")

