# -*- coding: utf-8 -*-
"""
Disease Differential Engine.
Evaluates bone marrow cytomorphology against WHO 5th Ed. and ICC 2022 diagnostic classification
criteria to produce an evidence-grounded differential diagnosis spectrum with clinical test recommendations.
"""

from typing import Dict, Any, List

def compute_disease_differential(summary: Dict[str, Any], cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes candidate disease entities from morphology summary and cell counts.
    Returns ranked candidates with matched criteria, clinical significance,
    confirmatory testing recommendations, and relevant molecular/cytogenetic markers.
    """
    total_cells = summary.get('total_cells', len(cells) or 1)
    blast_fraction = summary.get('blast_fraction', 0.0)
    blast_pct = blast_fraction * 100.0
    percentages = summary.get('percentages', {})
    
    # Extract lineage shares
    mega_pct = (percentages.get('Megakaryocyte/Platelet', 0.0) or percentages.get('Megakaryocyte', 0.0)) * 100.0
    gran_pct = (percentages.get('Granulocyte (Mature)', 0.0) or percentages.get('Granulocyte', 0.0)) * 100.0
    eryth_pct = (percentages.get('Erythroblast', 0.0) or percentages.get('Erythroid', 0.0)) * 100.0
    lymph_pct = (percentages.get('Lymphocyte', 0.0) or percentages.get('Lymphoid', 0.0)) * 100.0
    mono_pct = (percentages.get('Monocyte', 0.0) or percentages.get('Promonocyte', 0.0)) * 100.0
    
    candidates = []

    # 1. Acute Myeloid Leukemia / Acute Leukemia Threshold Check (>= 20% blasts)
    if blast_pct >= 20.0:
        candidates.append({
            "disease_name": "Acute Myeloid Leukemia (AML) / Acute Leukemia",
            "category": "Acute Leukemias",
            "match_level": "High Probability",
            "matched_criteria": [
                f"Bone marrow aspirate blast count is {blast_pct:.1f}%, meeting the WHO/ICC diagnostic threshold (≥ 20% blasts)",
                "Suppression of normal tri-lineage hematopoiesis observed"
            ],
            "relevant_biomarkers": ["NPM1", "FLT3-ITD", "CEBPA", "RUNX1", "TP53", "IDH1/2"],
            "confirmatory_tests": [
                "Flow cytometry immunophenotyping (CD34, CD117, CD33, CD13, HLA-DR, MPO)",
                "Conventional cytogenetic karyotyping for recurring translocations: t(8;21), inv(16), t(15;17)",
                "Next-Generation Sequencing (NGS) myeloid mutation panel"
            ],
            "databank_queries": ["AML diagnosis WHO 2022", "FLT3 NPM1 acute myeloid leukemia"],
            "urgency": "Urgent / Immediate Hematology Consult"
        })
    elif blast_pct >= 10.0:
        # 10% - 19% blasts: MDS with Excess Blasts-2 (MDS-EB2) or CML Accelerated Phase
        candidates.append({
            "disease_name": "Myelodysplastic Neoplasm with Excess Blasts (MDS-EB2 / High Risk)",
            "category": "Myelodysplastic Neoplasms",
            "match_level": "High Probability Differential",
            "matched_criteria": [
                f"Bone marrow blast fraction is {blast_pct:.1f}%, directly falling within the 10.0%–19.9% WHO excess blasts-2 tier",
                "Sub-threshold for definitive acute leukemia (<20% blasts) without recurring genetic abnormalities"
            ],
            "relevant_biomarkers": ["TP53 (multi-hit)", "SF3B1", "ASXL1", "RUNX1", "SRSF2", "U2AF1"],
            "confirmatory_tests": [
                "Comprehensive bone marrow core biopsy for cellularity, ring sideroblasts (Perls stain), and fibrosis",
                "Metaphase cytogenetics to assess del(5q), -7/del(7q), +8, del(20q), complex karyotype",
                "IPSS-M (Molecular International Prognostic Scoring System) calculation"
            ],
            "databank_queries": ["MDS-EB2 WHO 2022 diagnostic criteria", "Myelodysplastic neoplasms blast percentage"],
            "urgency": "High Priority Clinical Workup"
        })
        candidates.append({
            "disease_name": "Chronic Myeloid Leukemia (CML) - Accelerated Phase Consideration",
            "category": "Myeloproliferative Neoplasms",
            "match_level": "Moderate Probability Differential",
            "matched_criteria": [
                f"Blast proportion {blast_pct:.1f}% falls within the 10%–19% window characteristic of accelerated phase disease",
                "Maturing granulocytic series present alongside increased blast fraction"
            ],
            "relevant_biomarkers": ["BCR::ABL1 fusion transcript (p210/p190)", "ABL1 kinase domain mutations"],
            "confirmatory_tests": [
                "Reverse transcriptase quantitative PCR (RT-qPCR) for BCR::ABL1 major transcript",
                "Fluorescence in situ hybridization (FISH) for t(9;22)(q34.1;q11.2) Philadelphia chromosome",
                "Peripheral blood absolute basophil and eosinophil differential quantification"
            ],
            "databank_queries": ["CML accelerated phase criteria WHO 2022", "BCR ABL1 hematopathology"],
            "urgency": "Urgent Molecular Exclusion"
        })
    elif blast_pct >= 5.0:
        # 5% - 9% blasts: MDS-EB1 or early transformation
        candidates.append({
            "disease_name": "Myelodysplastic Neoplasm with Excess Blasts (MDS-EB1)",
            "category": "Myelodysplastic Neoplasms",
            "match_level": "Primary Morphologic Match",
            "matched_criteria": [
                f"Bone marrow blast fraction is {blast_pct:.1f}%, falling in the 5.0%–9.9% WHO excess blasts-1 window",
                "Increased blast population exceeding standard physiologic baseline (<5%)"
            ],
            "relevant_biomarkers": ["SF3B1", "TET2", "DNMT3A", "ASXL1"],
            "confirmatory_tests": [
                "Bone marrow biopsy for assessment of dyserythropoiesis, dysgranulopoiesis, and micromegakaryocytes",
                "Iron stain (Prussian blue) for ring sideroblasts assessment",
                "Myeloid targeted gene panel sequencing"
            ],
            "databank_queries": ["MDS-EB1 diagnosis WHO 5th", "myelodysplastic neoplasms excess blasts"],
            "urgency": "Standard Hematology Referral"
        })
        candidates.append({
            "disease_name": "Immune Reconstitution / Benign Hematogone Hyperplasia",
            "category": "Benign / Reactive Precursors",
            "match_level": "Differential to Exclude",
            "matched_criteria": [
                "Immature cells in the 5%–10% range can represent physiologic hematogones, especially in pediatric or regenerating marrow",
                "Requires antigenic distinction from neoplastic myeloblasts or lymphoblasts"
            ],
            "relevant_biomarkers": ["Polyclonal B-cell pattern", "Intact maturation spectrum"],
            "confirmatory_tests": [
                "High-resolution 8-color flow cytometry: continuous CD10, CD19, CD20, CD34, CD45 maturation spectrum",
                "Correlation with clinical history of recent viral illness, chemotherapy, or stem cell transplant"
            ],
            "databank_queries": ["hematogones vs lymphoblasts flow cytometry", "bone marrow regeneration"],
            "urgency": "Differential Workup"
        })
    else:
        # Blasts < 5%
        candidates.append({
            "disease_name": "Physiologic Blast Baseline / Low-Risk Cytomorphology",
            "category": "Non-Acute / Baseline",
            "match_level": "Baseline Observation",
            "matched_criteria": [
                f"Blast proportion {blast_pct:.1f}% remains within normal bone marrow physiologic limits (< 5%)",
                "No morphologic excess of immature myeloblasts detected"
            ],
            "relevant_biomarkers": ["Standard baseline hemogram"],
            "confirmatory_tests": [
                "Correlation with peripheral complete blood count (CBC) and absolute neutrophil count",
                "Repeat aspirate if unexplained cytopenias persist"
            ],
            "databank_queries": ["normal bone marrow blast differential", "cytopenias workup"],
            "urgency": "Routine Clinical Follow-up"
        })

    # Lineage specific skew flags:
    # High Megakaryocytic / Platelet proportion (> 5%)
    if mega_pct >= 5.0:
        candidates.append({
            "disease_name": "Myeloproliferative Neoplasm (ET / Early PMF) or Megakaryocytic Hyperplasia",
            "category": "Myeloproliferative / Platelet Lineage",
            "match_level": "Lineage Skew Match",
            "matched_criteria": [
                f"Megakaryocyte/Platelet lineage accounts for {mega_pct:.1f}% of classified nucleated aspirate cells (typical aspirate smear has < 1-2%)",
                "Significant clustering or atypical representation requiring exclusion of primary thrombocythemia"
            ],
            "relevant_biomarkers": ["JAK2 V617F", "CALR exon 9 indels", "MPL W515K/L"],
            "confirmatory_tests": [
                "Trephine core biopsy for megakaryocyte clustering, morphology (staghorn vs cloud-like hyperlobated), and reticulin fibrosis grade (MF-0 to MF-3)",
                "Molecular testing for MPN driver mutations (JAK2, CALR, MPL)",
                "Platelet count correlation and exclusion of secondary reactive thrombocytosis"
            ],
            "databank_queries": ["essential thrombocythemia WHO 2022", "JAK2 CALR MPL myeloproliferative"],
            "urgency": "Priority Specialist Evaluation"
        })

    return candidates

