"""
synthetic/templates.py
─────────────────────────────────────────────────────────────────────────────
Rule-based templates for synthetic AML and MDS cases based on the
WHO Classification of Haematolymphoid Tumours (5th edition, 2022).

Defines:
- 21 cell class codes and full names from Matek et al. 2021 TCIA dataset
- Diagnostic conditions with characteristic blast ranges, differentials,
  lab panels (CBC), cytogenetic abnormalities, and clinical narratives.
"""

from typing import Dict, List, Any
import numpy as np

# 21 classes from Matek et al. 2021 (TCIA)
CELL_CLASSES = [
    "BAS",  # Basophil
    "EBO",  # Erythroblast
    "EOS",  # Eosinophil
    "FGC",  # Faggott cell (promyelocyte with Auer rod bundles)
    "HAC",  # Hairy cell
    "KSC",  # Smudge / damaged cell
    "LYI",  # Immature lymphocyte
    "LYT",  # Lymphocyte
    "MMZ",  # Metamyelocyte
    "MOB",  # Monoblast
    "MON",  # Monocyte
    "MYB",  # Myelocyte
    "MYO",  # Myeloblast
    "NGB",  # Band neutrophil
    "NGS",  # Segmented neutrophil
    "NIF",  # Not identified
    "OTH",  # Other
    "PEB",  # Proerythroblast
    "PLM",  # Plasma cell
    "PMO",  # Promyelocyte
    "ART",  # Artefact
]

# Blast classes contributing to blast % under WHO 5th
PRIMARY_BLAST_CLASSES = ["MYO", "MOB"]

CASE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "AML_t8_21": {
        "condition_id": "AML_t8_21",
        "condition_name": "AML with t(8;21)(q22;q22.1); RUNX1::RUNX1T1",
        "who5_category": "Acute myeloid leukaemia with defining genetic abnormalities",
        "blast_pct_range": (0.22, 0.65),
        "primary_blast_class": "MYO",
        "cytogenetics": ["t(8;21)(q22;q22.1)", "RUNX1::RUNX1T1 positive by RT-PCR"],
        "lab_ranges": {
            "wbc_10e9_l": (15.0, 60.0),
            "hb_g_dl": (6.5, 9.5),
            "platelets_10e9_l": (20.0, 65.0),
            "neutrophil_abs_10e9_l": (0.3, 1.8),
            "ldh_u_l": (450, 1200),
        },
        "morphology_bias": {
            "MYO": 0.40,
            "PMO": 0.15,
            "MYB": 0.10,
            "MMZ": 0.08,
            "NGS": 0.05,
            "EBO": 0.08,
            "LYT": 0.06,
            "MON": 0.04,
            "EOS": 0.04,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} presenting with progressive fatigue, exertional dyspnea, and lower extremity petechiae. Peripheral smear noted Auer rods in granular myeloblasts.",
            "{age}-year-old {sex} admitted with epistaxis and mucosal bleeding. Bone marrow aspirate reveals hypercellularity with prominent myeloid maturation and large blasts containing prominent nucleoli.",
        ],
    },
    "APL_PML_RARA": {
        "condition_id": "APL_PML_RARA",
        "condition_name": "Acute promyelocytic leukaemia with PML::RARA fusion",
        "who5_category": "Acute myeloid leukaemia with defining genetic abnormalities",
        "blast_pct_range": (0.30, 0.85),
        "primary_blast_class": "PMO",  # In APL, abnormal promyelocytes are blast equivalents
        "cytogenetics": ["t(15;17)(q24.1;q21.2)", "PML::RARA fusion positive"],
        "lab_ranges": {
            "wbc_10e9_l": (1.2, 25.0),
            "hb_g_dl": (6.0, 9.0),
            "platelets_10e9_l": (10.0, 35.0),
            "fibrinogen_mg_dl": (60, 140),
            "d_dimer_ug_ml": (5.0, 45.0),
            "ldh_u_l": (500, 1600),
        },
        "morphology_bias": {
            "PMO": 0.55,
            "FGC": 0.18,  # Faggott cells characteristic of APL
            "MYO": 0.10,
            "EBO": 0.05,
            "LYT": 0.04,
            "NGS": 0.03,
            "MMZ": 0.03,
            "MON": 0.02,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} presenting acutely with severe coagulopathy, spontaneous ecchymoses, and gingival bleeding. Bone marrow aspirate demonstrates heavily granulated promyelocytes with bundles of Auer rods (Faggott cells).",
            "{age}-year-old {sex} referred urgently following findings of severe thrombocytopenia and hypofibrinogenemia with hypergranular promyelocyte proliferation.",
        ],
    },
    "AML_inv16": {
        "condition_id": "AML_inv16",
        "condition_name": "AML with inv(16)(p13.1q22) or t(16;16)(p13.1;q22); CBFB::MYH11",
        "who5_category": "Acute myeloid leukaemia with defining genetic abnormalities",
        "blast_pct_range": (0.25, 0.70),
        "primary_blast_class": "MYO",
        "cytogenetics": ["inv(16)(p13.1q22)", "CBFB::MYH11 rearrangement detected"],
        "lab_ranges": {
            "wbc_10e9_l": (25.0, 90.0),
            "hb_g_dl": (7.0, 10.0),
            "platelets_10e9_l": (25.0, 80.0),
            "ldh_u_l": (600, 1800),
        },
        "morphology_bias": {
            "MYO": 0.35,
            "MOB": 0.15,
            "MON": 0.12,
            "EOS": 0.15,  # Dysplastic eosinophils with basophilic granules
            "PMO": 0.08,
            "EBO": 0.05,
            "NGS": 0.05,
            "LYT": 0.05,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} presenting with constitutional symptoms and hepatosplenomegaly. BM aspirate demonstrates myelomonocytic differentiation and morphologically aberrant eosinophil precursors with purple-violet granules.",
        ],
    },
    "AML_MR": {
        "condition_id": "AML_MR",
        "condition_name": "Acute myeloid leukaemia, myelodysplasia-related (AML-MR)",
        "who5_category": "Acute myeloid leukaemia, myelodysplasia-related",
        "blast_pct_range": (0.20, 0.50),
        "primary_blast_class": "MYO",
        "cytogenetics": ["Complex karyotype (>=3 abnormalities)", "del(5q)", "del(7q)"],
        "lab_ranges": {
            "wbc_10e9_l": (2.0, 18.0),
            "hb_g_dl": (6.0, 8.5),
            "platelets_10e9_l": (15.0, 45.0),
            "ldh_u_l": (350, 900),
        },
        "morphology_bias": {
            "MYO": 0.30,
            "EBO": 0.20,  # Prominent multilineage dysplasia
            "PEB": 0.06,
            "PMO": 0.10,
            "MMZ": 0.08,
            "NGB": 0.08,
            "NGS": 0.06,
            "LYT": 0.07,
            "MON": 0.05,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} with a history of refractory cytopenias presenting with progressive pancytopenia. BM examination exhibits multilineage dysplasia (>50% dysplastic erythroid and granulocytic lineages) and 28% myeloblasts.",
        ],
    },
    "MDS_IB1": {
        "condition_id": "MDS_IB1",
        "condition_name": "Myelodysplastic neoplasm with increased blasts 1 (MDS-IB1)",
        "who5_category": "Myelodysplastic neoplasms",
        "blast_pct_range": (0.05, 0.09),  # 5-9% BM blasts under WHO 5th
        "primary_blast_class": "MYO",
        "cytogenetics": ["Isolated del(20q) or normal karyotype"],
        "lab_ranges": {
            "wbc_10e9_l": (1.8, 4.5),
            "hb_g_dl": (7.5, 9.8),
            "platelets_10e9_l": (40.0, 110.0),
            "ldh_u_l": (180, 400),
        },
        "morphology_bias": {
            "MYO": 0.07,
            "EBO": 0.22,
            "PEB": 0.04,
            "MMZ": 0.15,
            "MYB": 0.12,
            "PMO": 0.05,
            "NGB": 0.12,
            "NGS": 0.10,
            "LYT": 0.08,
            "MON": 0.05,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} evaluated for macrocytic anemia and mild thrombocytopenia. Aspirate shows dysgranulopoiesis and dyserythropoiesis with 7% bone marrow myeloblasts.",
        ],
    },
    "MDS_IB2": {
        "condition_id": "MDS_IB2",
        "condition_name": "Myelodysplastic neoplasm with increased blasts 2 (MDS-IB2)",
        "who5_category": "Myelodysplastic neoplasms",
        "blast_pct_range": (0.10, 0.19),  # 10-19% BM blasts under WHO 5th
        "primary_blast_class": "MYO",
        "cytogenetics": ["Trisomy 8 [+8] or del(7q)"],
        "lab_ranges": {
            "wbc_10e9_l": (1.5, 4.0),
            "hb_g_dl": (6.8, 9.0),
            "platelets_10e9_l": (25.0, 75.0),
            "ldh_u_l": (250, 600),
        },
        "morphology_bias": {
            "MYO": 0.15,
            "EBO": 0.20,
            "PEB": 0.05,
            "PMO": 0.08,
            "MYB": 0.10,
            "MMZ": 0.12,
            "NGB": 0.10,
            "NGS": 0.08,
            "LYT": 0.07,
            "MON": 0.05,
        },
        "clinical_presentation_templates": [
            "{age}-year-old {sex} presenting with worsening transfusion dependency and recurrent sinopulmonary infections. BM aspirate reveals 14% myeloblasts with nuclear hypolobation (pseudo-Pelger-Huët) in neutrophils.",
        ],
    },
    "MDS_del5q": {
        "condition_id": "MDS_del5q",
        "condition_name": "Myelodysplastic neoplasm with isolated del(5q)",
        "who5_category": "Myelodysplastic neoplasms with defining genetic abnormalities",
        "blast_pct_range": (0.01, 0.04),  # < 5% blasts
        "primary_blast_class": "MYO",
        "cytogenetics": ["del(5)(q13q33) as sole abnormality"],
        "lab_ranges": {
            "wbc_10e9_l": (2.5, 5.5),
            "hb_g_dl": (6.5, 8.8),
            "platelets_10e9_l": (220.0, 650.0),  # Typically normal or elevated platelets!
            "ldh_u_l": (150, 300),
        },
        "morphology_bias": {
            "MYO": 0.02,
            "EBO": 0.25,
            "PEB": 0.05,
            "MMZ": 0.14,
            "MYB": 0.12,
            "PMO": 0.03,
            "NGB": 0.12,
            "NGS": 0.12,
            "LYT": 0.10,
            "PLM": 0.03,
            "MON": 0.02,
        },
        "clinical_presentation_templates": [
            "{age}-year-old female presenting with severe transfusion-dependent macrocytic anemia and thrombocytosis. Marrow aspirate exhibits monolobated megakaryocytes with hypolobated nuclei and <5% myeloblasts.",
        ],
    },
}
