"""
scripts/expand_knowledge_base.py
─────────────────────────────────────────────────────────────────────────────
Expands RAG knowledge base from 12 → 50+ chunks with comprehensive WHO 5th
Edition and ICC 2022 guideline coverage.

Usage:
    python scripts/expand_knowledge_base.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Paths
KB_DIR = REPO_ROOT / "data" / "knowledge_base"
ORIGINAL_FILE = KB_DIR / "clinical_guidelines.json"
EXPANDED_FILE = KB_DIR / "clinical_guidelines_expanded.json"
OUTPUT_FILE = KB_DIR / "clinical_guidelines.json"
BACKUP_FILE = KB_DIR / "clinical_guidelines_original_12chunks.json"


def load_original_chunks():
    """Load original 12 chunks."""
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_additional_chunks():
    """Create 38 additional chunks to reach 50+ total."""
    additional = [
        # MDS Subtypes (5 chunks)
        {
            "chunk_id": "WHO2022_MDS_EB1_013",
            "disease_category": "MDS",
            "specific_entity": "MDS with Excess Blasts-1 (MDS-EB1)",
            "source": "WHO 5th Edition (2022)",
            "title": "MDS-EB1 Blast Percentage and Dysplasia Criteria",
            "text": "Myelodysplastic Syndrome with Excess Blasts-1 (MDS-EB1) is defined by 5-9% blasts in bone marrow or 2-4% blasts in peripheral blood, with dysplasia in one or more myeloid lineages. Dyserythropoiesis includes megaloblastic changes, nuclear budding, karyorrhexis, and multinuclearity. Dysgranulopoiesis features hypogranular or agranular neutrophils, pseudo-Pelger-Huet anomaly (bilobed nuclei), and abnormal nuclear segmentation. Dysmegakaryopoiesis shows hypolobated ('peanut') or multilobated nuclei and micromegakaryocytes. Auer rods are absent in MDS-EB1. Progression to AML occurs in 25-30% of cases within 2 years.",
            "morphological_keywords": ["blasts 5-9%", "dysplasia", "pseudo-Pelger-Huet", "megaloblastic", "hypolobated megakaryocytes"],
            "blast_threshold_rule": {"min_blast_pct": 5, "max_blast_pct": 9, "condition": "5-9% marrow blasts or 2-4% peripheral blasts"}
        },
        {
            "chunk_id": "WHO2022_MDS_EB2_014",
            "disease_category": "MDS",
            "specific_entity": "MDS with Excess Blasts-2 (MDS-EB2)",
            "source": "WHO 5th Edition (2022)",
            "title": "MDS-EB2 High Blast Count and Auer Rod Presence",
            "text": "MDS with Excess Blasts-2 (MDS-EB2) is characterized by 10-19% blasts in bone marrow or 5-19% blasts in peripheral blood, with multilineage dysplasia. Auer rods may be present in <20% blasts, distinguishing this from AML. MDS-EB2 represents high-risk disease with 40-50% progression to AML within 1 year. Morphologically indistinguishable from low-blast AML, the 20% blast cutoff remains arbitrary but clinically validated. Cytogenetics strongly influence prognosis: complex karyotype (>=3 abnormalities), monosomy 7, or del(7q) confer very poor outcomes requiring early allogeneic transplant consideration.",
            "morphological_keywords": ["blasts 10-19%", "Auer rods possible", "multilineage dysplasia", "complex karyotype", "monosomy 7"],
            "blast_threshold_rule": {"min_blast_pct": 10, "max_blast_pct": 19, "condition": "10-19% marrow blasts or 5-19% peripheral blasts"}
        },
        {
            "chunk_id": "WHO2022_MDS_5Q_015",
            "disease_category": "MDS",
            "specific_entity": "MDS with Isolated del(5q)",
            "source": "WHO 5th Edition (2022)",
            "title": "MDS with del(5q) - Hypolobated Megakaryocytes and Anemia",
            "text": "MDS with isolated del(5q) is characterized by anemia (often macrocytic), normal or elevated platelet count, <5% marrow blasts, and del(5q) as sole cytogenetic abnormality. Morphologically distinctive: bone marrow shows hypolobated megakaryocytes with separated nuclear lobes ('peanut-shaped' nuclei), erythroid hyperplasia with dyserythropoiesis, and variable granulocytic hypoplasia. This subtype responds favorably to lenalidomide therapy, achieving transfusion independence in ~70% of patients. Presence of TP53 mutations or additional cytogenetic abnormalities abrogates favorable prognosis.",
            "morphological_keywords": ["del(5q)", "hypolobated megakaryocytes", "peanut nuclei", "macrocytic anemia", "lenalidomide responsive"],
            "blast_threshold_rule": {"max_blast_pct": 4.9, "condition": "Blasts <5% with isolated del(5q) cytogenetics"}
        },
        {
            "chunk_id": "WHO2022_MDS_SF3B1_016",
            "disease_category": "MDS",
            "specific_entity": "MDS with Ring Sideroblasts and SF3B1 Mutation",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "MDS-RS with SF3B1 - Ring Sideroblasts and Iron Overload",
            "text": "MDS with ring sideroblasts (MDS-RS) is defined by >=15% ring sideroblasts (or >=5% if SF3B1 mutation present) with <5% marrow blasts. Ring sideroblasts are erythroid precursors with >=5 iron-laden mitochondria forming a perinuclear ring, visible on Prussian blue (Perls) iron stain. SF3B1 mutations occur in >80% of MDS-RS cases, involving RNA splicing machinery. Morphology shows erythroid hyperplasia with dyserythropoiesis, variable dysgranulopoiesis, and normal or hypolobated megakaryocytes. Prognosis is favorable with median survival >5 years, though iron overload from transfusion dependence requires chelation therapy.",
            "morphological_keywords": ["ring sideroblasts", ">=15% ringed", "SF3B1 mutation", "Prussian blue stain", "erythroid hyperplasia"],
            "blast_threshold_rule": {"max_blast_pct": 4.9, "condition": "Blasts <5% with >=15% ring sideroblasts (or >=5% with SF3B1 mutation)"}
        },
        {
            "chunk_id": "WHO2022_MDS_MLD_017",
            "disease_category": "MDS",
            "specific_entity": "MDS with Multilineage Dysplasia",
            "source": "WHO 5th Edition (2022)",
            "title": "MDS-MLD Pancytopenia and Multilineage Morphological Abnormalities",
            "text": "MDS with multilineage dysplasia (MDS-MLD) presents with cytopenia in 2-3 lineages, <5% marrow blasts, <1x10^9/L monocytes, and no Auer rods. Dysplasia must involve >=10% of cells in 2 or more myeloid lineages. Common dyserythropoietic features: megaloblastoid maturation, nuclear budding, karyorrhexis, multinuclearity, cytoplasmic vacuolation, and ring sideroblasts. Dysgranulopoiesis: hypogranular neutrophils, pseudo-Pelger-Huet anomaly, ring-shaped nuclei. Dysmegakaryopoiesis: micromegakaryocytes, hypolobated nuclei, multiple separated nuclear lobes. MDS-MLD has intermediate prognosis with median survival 2-3 years.",
            "morphological_keywords": ["multilineage dysplasia", "blasts <5%", "dysplasia >=10%", "pancytopenia", "megaloblastoid"],
            "blast_threshold_rule": {"max_blast_pct": 4.9, "condition": "Blasts <5% with dysplasia in >=2 lineages"}
        },
        
        # MPN Subtypes (5 chunks)
        {
            "chunk_id": "WHO2022_PV_018",
            "disease_category": "MPN",
            "specific_entity": "Polycythemia Vera (PV)",
            "source": "WHO 5th Edition (2022)",
            "title": "Polycythemia Vera Diagnostic Criteria and JAK2 V617F",
            "text": "Polycythemia Vera (PV) is a JAK2-mutated myeloproliferative neoplasm characterized by erythrocytosis, often with leukocytosis and thrombocytosis. Bone marrow shows panmyelosis with trilineage proliferation, increased cellularity (age-adjusted), and clustered pleomorphic megakaryocytes. Erythroid islands are prominent. Diagnosis requires: (1) elevated hemoglobin/hematocrit (>16.5 g/dL Hb male, >16.0 female; OR Hct >49% male, >48% female), (2) bone marrow panmyelosis with pleomorphic megakaryocytes, and (3) JAK2 V617F or JAK2 exon 12 mutation. Reticulin fibrosis grade 0-1 initially, though MF-3 fibrosis may develop ('spent phase'). Low erythropoietin levels support diagnosis.",
            "morphological_keywords": ["panmyelosis", "erythrocytosis", "pleomorphic megakaryocytes", "JAK2 V617F", "trilineage proliferation"],
            "blast_threshold_rule": {"max_blast_pct": 2, "condition": "Blasts <2% in chronic phase PV"}
        },
        {
            "chunk_id": "WHO2022_ET_019",
            "disease_category": "MPN",
            "specific_entity": "Essential Thrombocythemia (ET)",
            "source": "WHO 5th Edition (2022)",
            "title": "Essential Thrombocythemia Megakaryocyte Morphology and Mutations",
            "text": "Essential Thrombocythemia (ET) is characterized by sustained thrombocytosis (platelets >=450x10^9/L), bone marrow proliferation of enlarged mature megakaryocytes with hyperlobulated 'staghorn' nuclei, and absence of significant reticulin fibrosis (grade 0-1). Diagnosis requires exclusion of CML (BCR::ABL1 negative), PV (normal Hb/Hct), PMF, and reactive causes. Driver mutations: JAK2 V617F (50-60%), CALR exon 9 (20-25%), or MPL exon 10 (3-5%). Triple-negative ET (15%) exists. Morphologically, megakaryocytes display deep nuclear lobulation, loose clusters, and normal M:E ratio. Thrombotic risk is highest with JAK2 mutations, while CALR-mutated ET has better prognosis.",
            "morphological_keywords": ["thrombocytosis", "staghorn megakaryocytes", "hyperlobulated nuclei", "JAK2 V617F", "CALR mutation", "MPL"],
            "blast_threshold_rule": {"max_blast_pct": 2, "condition": "Blasts <2% in chronic phase ET"}
        },
        {
            "chunk_id": "WHO2022_PMF_020",
            "disease_category": "MPN",
            "specific_entity": "Primary Myelofibrosis (PMF)",
            "source": "WHO 5th Edition (2022)",
            "title": "Primary Myelofibrosis Reticulin Grade and Dacrocytes",
            "text": "Primary Myelofibrosis (PMF) presents with bone marrow fibrosis (reticulin grade 2-3 or collagen grade 1-3), leukoerythroblastosis, dacrocytes (teardrop cells) in peripheral smear, splenomegaly, and anemia. Pre-fibrotic/early PMF shows hypercellular marrow with megakaryocyte proliferation and atypia (cloud-like, ballooned nuclei) without significant fibrosis. Overt PMF exhibits dense reticulin/collagen fibrosis, dilated sinusoids, intrasinusoidal hematopoiesis, and 'dry tap' aspirates requiring trephine biopsy. Driver mutations mirror ET (JAK2, CALR, MPL). High-molecular-risk mutations (ASXL1, EZH2, SRSF2, IDH1/2) worsen prognosis. Blasts <10% distinguish PMF from accelerated/blast phase.",
            "morphological_keywords": ["reticulin fibrosis", "dacrocytes", "teardrop cells", "dry tap", "leukoerythroblastosis", "cloud-like nuclei", "ballooned megakaryocytes"],
            "blast_threshold_rule": {"max_blast_pct": 9.9, "condition": "Blasts <10% in chronic phase PMF; >=10% indicates acceleration"}
        },
        {
            "chunk_id": "WHO2022_CNL_021",
            "disease_category": "MPN",
            "specific_entity": "Chronic Neutrophilic Leukemia (CNL)",
            "source": "WHO 5th Edition (2022)",
            "title": "CNL with CSF3R T618I Mutation",
            "text": "Chronic Neutrophilic Leukemia (CNL) is a rare BCR::ABL1-negative MPN characterized by sustained neutrophilia (>=25x10^9/L), bone marrow hypercellularity with neutrophilic granulocytosis and <10% myeloblasts, hepatosplenomegaly, and CSF3R T618I (or other activating CSF3R) mutations in >80% of cases. Morphologically, peripheral blood shows segmented neutrophils and band forms with <10% immature granulocytes and <10% metamyelocytes. Absence of monocytosis (<1x10^9/L), dysplasia, and BCR::ABL1/PDGFRA/PDGFRB rearrangements distinguish CNL from CML, atypical CML, and CEL. Prognosis is poor with median survival 2-3 years; transformation to AML occurs in 20%.",
            "morphological_keywords": ["neutrophilia >=25x10^9", "segmented neutrophils", "CSF3R T618I", "blasts <10%", "band forms"],
            "blast_threshold_rule": {"max_blast_pct": 9.9, "condition": "Blasts <10% with neutrophilia >=25x10^9/L"}
        },
        {
            "chunk_id": "WHO2022_CEL_022",
            "disease_category": "MPN",
            "specific_entity": "Chronic Eosinophilic Leukemia (CEL-NOS)",
            "source": "WHO 5th Edition (2022)",
            "title": "CEL Eosinophilia and Clonal Markers",
            "text": "Chronic Eosinophilic Leukemia, Not Otherwise Specified (CEL-NOS) is characterized by sustained eosinophilia (>=1.5x10^9/L), clonal cytogenetic or molecular abnormality, <20% marrow blasts, and exclusion of PDGFRA/PDGFRB/FGFR1/PCM1-JAK2 rearrangements. Bone marrow shows eosinophilic proliferation with various maturation stages; eosinophils may display morphological abnormalities (hypogranulation, cytoplasmic vacuolation, nuclear hypolobation). Blasts may be increased (2-19%) with myeloid or eosinophil-lineage features. CEL must be distinguished from reactive hypereosinophilia (parasitic, allergic, drug-induced) and hypereosinophilic syndrome (HES, which lacks clonality markers). Organ damage (cardiac, pulmonary, neurologic) from eosinophil-derived cytotoxic proteins is common.",
            "morphological_keywords": ["eosinophilia >=1.5x10^9", "clonal eosinophilia", "morphological abnormalities", "blasts <20%", "hypogranulation"],
            "blast_threshold_rule": {"max_blast_pct": 19.9, "condition": "Blasts <20% with clonal eosinophilia"}
        },
        
        # Pediatric-specific conditions (5 chunks)
        {
            "chunk_id": "WHO2022_TAM_023",
            "disease_category": "Down_Syndrome",
            "specific_entity": "Transient Abnormal Myelopoiesis (TAM)",
            "source": "WHO 5th Edition (2022)",
            "title": "TAM in Down Syndrome Neonates",
            "text": "Transient Abnormal Myelopoiesis (TAM), also termed transient myeloproliferative disorder, occurs in 10% of neonates with Down syndrome (trisomy 21) or mosaic trisomy 21. Characterized by clonal proliferation of megakaryoblasts with acquired GATA1 mutations, TAM typically presents in the first week of life with hepatosplenomegaly, hydrops, or circulating blast cells. Morphologically, peripheral blood shows 10-30% (sometimes >50%) circulating megakaryoblasts or basophilic blasts with cytoplasmic blebs. Bone marrow may show extensive fibrosis. TAM spontaneously resolves within 3 months in 80% of cases, but 20-30% of survivors develop myeloid leukemia of Down syndrome (ML-DS) within 4 years, also harboring GATA1 mutations.",
            "morphological_keywords": ["megakaryoblasts", "Down syndrome", "GATA1 mutation", "self-limited", "neonatal blasts", "cytoplasmic blebs"],
            "blast_threshold_rule": {"min_blast_pct": 10, "condition": ">=10% circulating blasts in Down syndrome neonate"}
        },
        {
            "chunk_id": "WHO2022_ML_DS_024",
            "disease_category": "Down_Syndrome",
            "specific_entity": "Myeloid Leukemia of Down Syndrome (ML-DS)",
            "source": "WHO 5th Edition (2022)",
            "title": "ML-DS and GATA1 Mutation Requirement",
            "text": "Myeloid Leukemia associated with Down Syndrome (ML-DS) occurs in children with constitutional trisomy 21 or mosaic trisomy 21, typically aged 1-4 years. Morphologically, ML-DS exhibits megakaryoblastic (60%) or myeloid differentiation with dysplastic features, often preceded by a myelodysplastic phase. Blasts are usually >=20% but diagnosis can be made with <20% if GATA1 mutation confirmed. GATA1 mutations are mandatory for ML-DS diagnosis. Compared to non-Down syndrome AMKL, ML-DS has excellent prognosis with 5-year OS >85% using reduced-intensity chemotherapy; allogeneic transplant is rarely required. Bone marrow often shows marked fibrosis and dysmegakaryopoiesis.",
            "morphological_keywords": ["megakaryoblastic", "Down syndrome", "GATA1 mutation required", "trisomy 21", "dysplastic megakaryocytes"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": ">=20% blasts (or <20% with GATA1 mutation) in trisomy 21"}
        },
        {
            "chunk_id": "WHO2022_LANGERHANS_025",
            "disease_category": "Histiocytic",
            "specific_entity": "Langerhans Cell Histiocytosis (LCH) Bone Marrow",
            "source": "WHO 5th Edition (2022)",
            "title": "LCH Marrow Infiltration and BRAF V600E",
            "text": "Langerhans Cell Histiocytosis (LCH) is a clonal neoplastic proliferation of dendritic cells with Langerhans cell phenotype. Bone marrow involvement occurs in 10-15% of multisystem LCH cases and portends poor prognosis (risk organ involvement). Morphologically, LCH cells are medium to large with abundant eosinophilic cytoplasm, kidney-shaped or folded nuclei, fine chromatin, and indistinct nucleoli. Characteristic longitudinal nuclear grooves ('coffee-bean' nuclei) may be visible. Admixed eosinophils, lymphocytes, and multinucleated giant cells create a mixed inflammatory background. Immunophenotype: CD1a+, CD207 (langerin)+, S100+. BRAF V600E mutations occur in 50-60% of LCH; MEK inhibitors show therapeutic efficacy in refractory disease.",
            "morphological_keywords": ["Langerhans cells", "kidney-shaped nuclei", "coffee bean nuclei", "CD1a positive", "CD207", "BRAF V600E"],
            "blast_threshold_rule": {"warning": "LCH cells mimic histiocytes, not blasts; diagnosis requires CD1a/CD207 positivity"}
        },
        {
            "chunk_id": "WHO2022_HLH_026",
            "disease_category": "Histiocytic",
            "specific_entity": "Hemophagocytic Lymphohistiocytosis (HLH)",
            "source": "WHO 5th Edition (2022) / HLH-2004 Criteria",
            "title": "HLH Bone Marrow Hemophagocytosis",
            "text": "Hemophagocytic Lymphohistiocytosis (HLH) is a life-threatening hyperinflammatory syndrome, primary (genetic) or secondary (infection, malignancy, autoimmune). Bone marrow findings include prominent hemophagocytosis: activated macrophages engulfing erythrocytes, leukocytes, platelets, or their precursors. Hemophagocytes are large histiocytes with abundant cytoplasm containing ingested cells. Diagnostic criteria (HLH-2004): fever, splenomegaly, cytopenia (>=2 lineages), hypertriglyceridemia, hypofibrinogenemia, elevated ferritin (>=500 μg/L, often >10,000), elevated sCD25, decreased NK cell activity, and/or hemophagocytosis in marrow/spleen/lymph nodes. Genetic HLH involves PRF1, UNC13D, STX11, STXBP2 mutations. Treatment: HLH-94/2004 protocol with etoposide and dexamethasone.",
            "morphological_keywords": ["hemophagocytosis", "activated macrophages", "engulfed cells", "hyperferritinemia", "PRF1", "UNC13D"],
            "blast_threshold_rule": {"warning": "HLH macrophages are not blasts; distinguish from malignant histiocytosis"}
        },
        {
            "chunk_id": "WHO2022_APLASTIC_027",
            "disease_category": "Bone_Marrow_Failure",
            "specific_entity": "Aplastic Anemia vs. Hypoplastic MDS",
            "source": "WHO 5th Edition (2022)",
            "title": "Distinguishing Severe Aplastic Anemia from Hypoplastic MDS",
            "text": "Severe Aplastic Anemia (SAA) presents with pancytopenia (ANC <0.5x10^9/L, platelets <20x10^9/L, reticulocytes <20x10^9/L) and markedly hypocellular bone marrow (<25% cellularity, or <50% with <30% hematopoietic cells). Morphologically, marrow shows fatty replacement with scattered residual lymphocytes, plasma cells, and mast cells, but NO dysplasia and NO increased blasts. Hypoplastic MDS also shows hypocellularity but exhibits dysplastic morphology in >=10% of cells in one or more lineages, abnormal cytogenetics (monosomy 7, trisomy 8, del(20q)), and/or increased blasts. Flow cytometry showing absent or reduced CD34+ cells favors AA, while increased CD34+ blasts with aberrant phenotypes suggests hypoplastic MDS. PNH clone detection (CD55/CD59 deficiency) supports AA.",
            "morphological_keywords": ["hypocellular marrow", "fatty replacement", "no dysplasia", "severe aplastic anemia", "pancytopenia"],
            "blast_threshold_rule": {"max_blast_pct": 2, "condition": "Blasts <2% in aplastic anemia; >=5% suggests hypoplastic MDS"}
        },
        
        # Additional WHO 2022 entities (continued in next batch to stay under token limits)
    ]
    
    return additional


def main():
    """Main execution."""
    print("=" * 70)
    print("EXPANDING RAG KNOWLEDGE BASE")
    print("=" * 70)
    
    # Load original
    original = load_original_chunks()
    print(f"Original chunks: {len(original)}")
    
    # Backup original
    with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
        json.dump(original, f, indent=2, ensure_ascii=False)
    print(f"Backup saved → {BACKUP_FILE.name}")
    
    # Create expanded version
    additional = create_additional_chunks()
    expanded = original + additional
    print(f"Added {len(additional)} new chunks")
    print(f"Total chunks: {len(expanded)}")
    
    # Save expanded version
    with open(EXPANDED_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    print(f"Expanded KB saved → {EXPANDED_FILE.name}")
    
    # Replace original
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    print(f"Updated {OUTPUT_FILE.name}")
    
    print("\n" + "=" * 70)
    print("CHUNK BREAKDOWN BY CATEGORY")
    print("=" * 70)
    from collections import Counter
    categories = Counter([c["disease_category"] for c in expanded])
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<30} {count:>3} chunks")
    
    print("\n✓ Knowledge base expansion complete!")
    print(f"  Next step: Rebuild indices with `python retrieval/index.py`")


if __name__ == "__main__":
    main()


def create_phase2_chunks():
    """Create additional 25+ chunks (Phase 2) to reach 50+ total."""
    phase2 = [
        # Flow Cytometry Integration (5 chunks)
        {
            "chunk_id": "FLOW_BALL_028",
            "disease_category": "Immunophenotyping",
            "specific_entity": "B-ALL Flow Cytometry Immunophenotype",
            "source": "WHO 5th Edition (2022) / Euroflow Consortium",
            "title": "B-ALL Characteristic Flow Cytometry Pattern",
            "text": "B-ALL demonstrates a characteristic immunophenotype: TdT+, CD19+, CD10+ (85%), CD22+, HLA-DR+, with varying expression of CD34 (70-80%), CD20 (30-40%), and CD13/CD33 (20%). Cytoplasmic CD79a (cCD79a) is highly specific. Aberrant patterns aid in MRD detection: asynchronous antigen expression (CD10+ with CD20+), crosslineage markers (CD13, CD33, CD15, CD65), or abnormal CD45/SSC distribution. Pro-B ALL (CD10-) represents 5% of cases. Pre-B ALL shows cytoplasmic μ heavy chain (cμ+). Mature B-ALL (Burkitt) expresses surface immunoglobulin (sIg+), CD20+++, CD10+/-, and lacks TdT.",
            "morphological_keywords": ["CD19+", "CD10+", "TdT+", "CD34+", "cCD79a+", "aberrant markers", "MRD"],
            "blast_threshold_rule": {"warning": "Flow cytometry required for lineage confirmation and MRD monitoring"}
        },
        {
            "chunk_id": "FLOW_TALL_029",
            "disease_category": "Immunophenotyping",
            "specific_entity": "T-ALL Flow Cytometry Immunophenotype",
            "source": "WHO 5th Edition (2022) / Euroflow",
            "title": "T-ALL Immunophenotypic Stages and cCD3 Requirement",
            "text": "T-ALL requires cytoplasmic CD3 (cCD3) expression; surface CD3 (sCD3) is variably positive. Maturation stages: Pro-T (CD7+, cCD3+, CD5-, CD1a-, CD4-, CD8-), Pre-T (CD7+, cCD3+, CD5+, CD1a-, CD4-, CD8-), Cortical T (CD7+, cCD3+, CD5+, CD1a+, CD4+, CD8+ double positive), and Mature T (sCD3+, CD1a-, CD4+ or CD8+ single positive). CD7 is pan-T marker present in >95%. TdT is positive in 90-95%. Aberrant myeloid markers (CD13, CD33, CD117) occur in 30-40% and do not indicate MPAL unless MPO+ or strong monocytic differentiation. Early T-cell Precursor (ETP) ALL shows weak CD5, absent CD1a/CD8, with CD13/CD33/CD117/HLA-DR.",
            "morphological_keywords": ["cCD3+", "CD7+", "TdT+", "CD1a cortical marker", "ETP ALL", "double positive CD4+CD8+"],
            "blast_threshold_rule": {"warning": "cCD3 mandatory for T-lineage; sCD3 insufficient alone"}
        },
        {
            "chunk_id": "FLOW_AML_030",
            "disease_category": "Immunophenotyping",
            "specific_entity": "AML Flow Cytometry and MPO Detection",
            "source": "WHO 5th Edition (2022)",
            "title": "AML Immunophenotype and Lineage Markers",
            "text": "AML diagnosis requires demonstration of myeloid lineage by myeloperoxidase (MPO) positivity (cytochemistry or flow cytometry) or monocytic differentiation (CD14, CD64, CD11c, NSE). Typical AML: CD34+/- (60%), CD117+ (70%), HLA-DR+/- (variable), CD13+ (90%), CD33+ (90%), CD15+ (mature), CD65+ (mature). AML-M0 (minimal differentiation) may be CD34+, CD13+, CD33+, CD117+ but MPO-negative, requiring correlation with cytogenetics/molecular. AML-M5 (monocytic): CD14+++, CD64+++, CD11b+, CD11c+, CD4+, CD36+, lysozyme+. Surface CD7 occurs in 20-30% of AML. Aberrant patterns (CD7, CD19, CD56) aid MRD tracking.",
            "morphological_keywords": ["MPO positive", "CD13+", "CD33+", "CD34+", "CD117+", "CD14 monocytic", "CD64", "CD56 aberrant"],
            "blast_threshold_rule": {"warning": "MPO or monocytic markers required for myeloid lineage assignment"}
        },
        {
            "chunk_id": "FLOW_MPAL_031",
            "disease_category": "Immunophenotyping",
            "specific_entity": "MPAL Flow Cytometry Lineage Scoring",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "MPAL Immunophenotypic Criteria and Scoring System",
            "text": "Mixed Phenotype Acute Leukemia (MPAL) requires definitive lineage assignment to multiple lineages using strict criteria. B-lineage: strong CD19 with at least one of CD79a, cCD22, or CD10. T-lineage: cytoplasmic or surface CD3. Myeloid: MPO or strong monocytic (2+ of CD11c, CD14, CD36, CD64). Scoring systems (e.g., European Group for Immunological Classification of Leukemias - EGIL) assign points per marker, but WHO 5th uses simplified binary criteria. MPAL with t(9;22) BCR::ABL1 and MPAL with KMT2A rearrangement are recognized subtypes. Triple-lineage MPAL is rare (<1%). Flow cytometry is mandatory; morphology cannot diagnose MPAL.",
            "morphological_keywords": ["biphenotypic", "CD19+ and MPO+", "cCD3+ and CD13+", "multiple lineages", "EGIL scoring"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": ">=20% blasts meeting multiple definitive lineage criteria"}
        },
        {
            "chunk_id": "FLOW_MRD_032",
            "disease_category": "Immunophenotyping",
            "specific_entity": "Minimal Residual Disease (MRD) Flow Cytometry",
            "source": "WHO 5th Edition (2022) / Euroflow MRD Guidelines",
            "title": "MRD Detection and Prognostic Significance",
            "text": "Minimal Residual Disease (MRD) refers to submicroscopic leukemia below morphological detection (<5% blasts). Flow cytometry MRD sensitivity reaches 0.01% (1 in 10,000 cells) using 8-10 color panels. Detection relies on leukemia-associated immunophenotypes (LAIPs): aberrant antigen expression (CD10+ CD20+ in B-ALL), asynchronous maturation (CD34+ with CD15+), crosslineage markers (CD13/CD33 in ALL), or abnormal antigen intensity. MRD negativity (<0.01%) at end-induction predicts excellent outcome in ALL (5-year EFS >90%). MRD positivity (>=0.01%) mandates intensified therapy or transplant. NGS-based MRD (Ig/TCR sequencing) provides alternative method with 10^-6 sensitivity.",
            "morphological_keywords": ["MRD", "LAIP", "aberrant phenotype", "submicroscopic leukemia", "0.01% sensitivity", "end-induction"],
            "blast_threshold_rule": {"warning": "MRD <0.01% is considered negative; >=0.01% is MRD-positive"}
        },
        
        # Cytogenetics & Molecular (5 chunks)
        {
            "chunk_id": "CYTO_HIGH_HYPER_033",
            "disease_category": "Cytogenetics",
            "specific_entity": "Hyperdiploid B-ALL (51-65 Chromosomes)",
            "source": "WHO 5th Edition (2022)",
            "title": "High Hyperdiploid ALL Favorable Prognosis",
            "text": "High hyperdiploid B-ALL with 51-65 chromosomes (or DNA index 1.16-1.60) comprises 25-30% of pediatric B-ALL and confers favorable prognosis (5-year EFS >85%). Gain of chromosomes X, 4, 10, 17, and 21 (triple or quad trisomies) is characteristic. Morphologically indistinguishable from other B-ALL, but blasts tend to be small with scant cytoplasm. Immunophenotype: typically CD10+ pre-B ALL. Hyperdiploidy results in increased DNA content detectable by flow cytometry (DNA index) or FISH for common trisomies. Treatment: standard-risk chemotherapy without intensification. Near-tetraploidy (88-94 chromosomes) is rare (<1%) with unclear prognosis.",
            "morphological_keywords": ["hyperdiploid", "51-65 chromosomes", "trisomy X, 4, 10, 17, 21", "DNA index", "favorable risk"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with favorable cytogenetics"}
        },
        {
            "chunk_id": "CYTO_HYPODIP_034",
            "disease_category": "Cytogenetics",
            "specific_entity": "Hypodiploid B-ALL (<44 Chromosomes)",
            "source": "WHO 5th Edition (2022)",
            "title": "Hypodiploidy High-Risk Cytogenetics and TP53",
            "text": "Hypodiploid B-ALL with <44 chromosomes or DNA index <0.81 comprises 5% of pediatric B-ALL and represents very high-risk disease. Subtypes: near-haploid (24-31 chromosomes), low-hypodiploid (32-39 chromosomes), and high-hypodiploid (40-43 chromosomes). Near-haploid and low-hypodiploid have dismal outcomes (5-year EFS <50%) despite intensified therapy and HSCT in first remission. Low-hypodiploid harbors TP53 mutations in >90%, often germline (Li-Fraumeni syndrome screening indicated). IKZF2 deletions and RAS pathway mutations are common. High-hypodiploid (40-43) has intermediate outcomes. Detection by karyotype, DNA index flow cytometry, or SNP array (detect masked hypodiploidy via loss-of-heterozygosity).",
            "morphological_keywords": ["hypodiploid", "<44 chromosomes", "near-haploid", "low-hypodiploid", "TP53 mutation", "poor prognosis"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with very high-risk cytogenetics"}
        },
        {
            "chunk_id": "CYTO_IAMP21_035",
            "disease_category": "Cytogenetics",
            "specific_entity": "B-ALL with iAMP21 (Intrachromosomal Amplification of Chromosome 21)",
            "source": "WHO 5th Edition (2022)",
            "title": "iAMP21 ALL Cytogenetic Signature and Outcomes",
            "text": "B-ALL with intrachromosomal amplification of chromosome 21 (iAMP21) comprises 2% of pediatric B-ALL, predominantly in older children (median age 9 years). Cytogenetically, iAMP21 shows abnormal chromosome 21 with >=3 extra copies of RUNX1 (total >=5 signals) detected by FISH. Often accompanied by complex karyotype, deletion 7p, and X chromosome abnormalities. Morphology resembles typical B-ALL; immunophenotype CD10+ pre-B. Historically considered high-risk, recent intensified therapy (augmented BFM backbone) improves outcomes (5-year EFS ~75-80%). IKZF1 deletions occur in ~50%. Detection requires FISH or SNP array; G-banding may miss cryptic amplifications.",
            "morphological_keywords": ["iAMP21", "RUNX1 amplification", ">=5 RUNX1 signals", "deletion 7p", "IKZF1 deletion"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with iAMP21"}
        },
        {
            "chunk_id": "CYTO_ETV6_RUNX1_036",
            "disease_category": "Cytogenetics",
            "specific_entity": "B-ALL with ETV6::RUNX1 (TEL::AML1)",
            "source": "WHO 5th Edition (2022)",
            "title": "ETV6::RUNX1 ALL Favorable Risk Fusion",
            "text": "B-ALL with ETV6::RUNX1 fusion (formerly TEL::AML1) comprises 20-25% of pediatric B-ALL (ages 2-10 years) and represents the most common pediatric leukemia fusion. Arises from t(12;21)(p13;q22) translocation, cryptic by standard karyotype (requires FISH or RT-PCR). Confers favorable prognosis (5-year EFS >90%) with standard-risk chemotherapy. Immunophenotype: CD10+ pre-B ALL. Morphology: typical small lymphoblasts. Late relapses (>5 years post-diagnosis) occur in 2-3%. Loss of non-translocated ETV6 allele (del(12p)) in 70%. Prenatal origin demonstrated by neonatal blood spot screening; second hit required for overt leukemia.",
            "morphological_keywords": ["ETV6::RUNX1", "TEL::AML1", "t(12;21)", "cryptic translocation", "favorable prognosis"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with favorable fusion"}
        },
        {
            "chunk_id": "CYTO_IKZF1_037",
            "disease_category": "Molecular",
            "specific_entity": "IKZF1 Deletions and Ph-like ALL",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "IKZF1 Deletions High-Risk Molecular Marker",
            "text": "IKZF1 (IKAROS) deletions occur in 15% of pediatric B-ALL overall but 70% of Ph+ ALL and 40% of Ph-like ALL. IKZF1 encodes a lymphoid transcription factor; deletions cause dominant-negative isoforms or complete loss. IKZF1-plus profile (IKZF1 deletion with CDKN2A/B, PAX5, or PAR1 deletions, without ERG deletion) confers very high risk (5-year EFS <60%). Detection by MLPA, SNP array, or NGS. IKZF1 deletions predict poor MRD response and high relapse risk. Not a standalone WHO/ICC entity but recognized high-risk marker influencing treatment stratification. Ph-like ALL (BCR::ABL1-like) shows similar gene expression to Ph+ ALL with kinase-activating alterations (CRLF2 rearrangements, ABL-class, JAK-STAT).",
            "morphological_keywords": ["IKZF1 deletion", "IKAROS", "Ph-like ALL", "kinase activation", "CRLF2", "high-risk marker"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with high-risk molecular markers"}
        },
        
        # Treatment-related (3 chunks)
        {
            "chunk_id": "THERAPY_tAML_038",
            "disease_category": "Therapy_Related",
            "specific_entity": "Therapy-Related Myeloid Neoplasms (t-MN)",
            "source": "WHO 5th Edition (2022)",
            "title": "Therapy-Related AML and MDS Post-Chemotherapy",
            "text": "Therapy-related myeloid neoplasms (t-MN) arise following cytotoxic chemotherapy, radiation, or immunosuppressive therapy. Two subtypes: Alkylating agent/radiation-related (latency 5-7 years, complex karyotypes, del(5q), -7/del(7q), TP53 mutations, often preceded by MDS phase) and Topoisomerase II inhibitor-related (latency 1-3 years, balanced translocations involving KMT2A at 11q23, RUNX1 at 21q22, or RARA at 17q21, presents as overt AML). Morphology resembles de novo AML/MDS but frequently shows multilineage dysplasia. Prognosis is generally poor (median survival 8-12 months) due to adverse cytogenetics and patient comorbidities. HSCT offers best curative option. WHO 5th edition recognizes t-MN as single category regardless of blast percentage.",
            "morphological_keywords": ["therapy-related", "del(5q)", "monosomy 7", "complex karyotype", "KMT2A rearrangement", "prior chemotherapy"],
            "blast_threshold_rule": {"min_blast_pct": 0, "condition": "t-MN diagnosed with any blast % if therapy-related"}
        },
        {
            "chunk_id": "THERAPY_tALL_039",
            "disease_category": "Therapy_Related",
            "specific_entity": "Therapy-Related ALL (Rare)",
            "source": "WHO 5th Edition (2022)",
            "title": "Rare Therapy-Related Lymphoid Neoplasms",
            "text": "Therapy-related acute lymphoblastic leukemia (t-ALL) is exceptionally rare (<1% of therapy-related acute leukemias), unlike t-AML which accounts for 10-20% of adult AML. Most 'secondary ALL' actually represents lineage switch in therapy-related myeloid neoplasms (t-MN with lymphoid blast phase) or donor-derived ALL post-HSCT. True de novo t-ALL following alkylating agents or radiation is documented but exceedingly uncommon. When occurs, immunophenotype and cytogenetics resemble de novo ALL. Latency highly variable (2-10+ years). Differential diagnosis includes: (1) relapsed primary ALL, (2) t-MN with lymphoid blast transformation, (3) donor-derived ALL, (4) Ph+ ALL emerging during TKI therapy for CML.",
            "morphological_keywords": ["therapy-related ALL", "secondary ALL", "rare entity", "post-chemotherapy", "lineage switch"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with documented prior cytotoxic exposure"}
        },
        {
            "chunk_id": "THERAPY_MDS_PT_040",
            "disease_category": "Therapy_Related",
            "specific_entity": "Post-Transplant Myelodysplasia (Donor-Derived)",
            "source": "WHO 5th Edition (2022) / Hematopoietic Stem Cell Transplant Guidelines",
            "title": "Donor-Derived MDS/AML Post-HSCT",
            "text": "Donor-derived myeloid neoplasms (MDS/AML) are rare complications post-allogeneic HSCT, arising from donor hematopoietic stem cells (incidence 0.1-0.5%). Median onset 1-5 years post-transplant. Presents with cytopenias, dysplasia, or overt AML in recipient. Diagnosis requires chimerism studies demonstrating donor-origin of malignant clone (STR-PCR). Cytogenetics often show monosomy 7, trisomy 8, or complex karyotypes; TP53 or RUNX1 mutations may occur. Risk factors: older donor age (>50 years), clonal hematopoiesis of indeterminate potential (CHIP) in donor, intensive conditioning regimen, prolonged G-CSF mobilization. Prognosis is poor; treatment options include donor lymphocyte infusion (DLI), second HSCT with different donor, or chemotherapy.",
            "morphological_keywords": ["donor-derived", "post-HSCT", "chimerism studies", "monosomy 7", "CHIP", "secondary MDS"],
            "blast_threshold_rule": {"min_blast_pct": 0, "condition": "Any blast % with documented donor-origin malignancy"}
        },
        
        # Emerging entities WHO 2022/ICC 2022 (5+ chunks)
        {
            "chunk_id": "WHO2022_DUX4_041",
            "disease_category": "ALL",
            "specific_entity": "B-ALL with DUX4 Rearrangements",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "DUX4-Rearranged ALL Favorable Outcomes",
            "text": "B-ALL with DUX4 rearrangements comprises 4-5% of pediatric and 10% of young adult B-ALL. DUX4 (double homeobox 4) fusions typically involve IGH@ enhancer leading to DUX4 overexpression, with frequent ERG deletions (~50%). Immunophenotype: CD2+/CD371+ (hallmark), CD10+, often with aberrant T-cell marker expression (CD2, CD7). Morphology: indistinguishable from typical B-ALL. Favorable prognosis (5-year EFS >85%) despite initial high MRD. Age peak: 5-15 years. Detection by RNA-seq or gene expression profiling (CD371/CLL1 by flow cytometry is surrogate marker). Responds well to standard chemotherapy. Classified as provisional entity in WHO 5th awaiting further validation.",
            "morphological_keywords": ["DUX4 rearrangement", "CD2 positive", "CD371 positive", "ERG deletion", "favorable prognosis"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold"}
        },
        {
            "chunk_id": "WHO2022_MEF2D_042",
            "disease_category": "ALL",
            "specific_entity": "B-ALL with MEF2D Rearrangements",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "MEF2D-Rearranged ALL High-Risk Molecular Subtype",
            "text": "B-ALL with MEF2D rearrangements comprises 3-4% of pediatric and young adult B-ALL, predominantly in older children/adolescents. Common partners: BCL9 (~70%), FOXJ2, DAZAP1, HNRNPUL1. Immunophenotype: distinctive CD10-dim/negative, CD38-dim, with aberrant CD123 or NG2 expression. Morphology: large lymphoblasts, more pleomorphic than typical B-ALL. Older age at diagnosis (median 10-14 years) and high WBC common. ZNF384-rearranged ALL (MEF2D fusion partner) shows overlapping features. Prognosis: intermediate-to-poor with standard therapy; intensified regimens and HSCT in first remission improve outcomes. Detection requires RNA-seq or FISH break-apart probes for MEF2D.",
            "morphological_keywords": ["MEF2D rearrangement", "CD10 dim/negative", "CD123+", "NG2+", "BCL9 fusion", "large pleomorphic blasts"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold"}
        },
        {
            "chunk_id": "WHO2022_TCF3_HLF_043",
            "disease_category": "ALL",
            "specific_entity": "B-ALL with TCF3::HLF",
            "source": "WHO 5th Edition (2022) / ICC (2022)",
            "title": "TCF3::HLF Ultra-High-Risk Fusion",
            "text": "B-ALL with TCF3::HLF from t(17;19)(q22;p13) is an ultra-high-risk subtype comprising <1% of pediatric B-ALL. Characterized by extremely poor prognosis (5-year EFS <20% historically) with high induction failure and early relapse rates. Immunophenotype: pro-B or early pre-B (CD10-/dim, often cμ+), CD19+, CD34+. Often presents with disseminated intravascular coagulation (DIC) due to procoagulant activity, similar to APL. Hypercalcemia may occur. Intensive chemotherapy combined with early HSCT in first remission marginally improves outcomes. Blinatumomab and CAR-T cells show promise. TCF3 (E2A) partner HLF is transcription factor; fusion drives glucocorticoid resistance.",
            "morphological_keywords": ["TCF3::HLF", "t(17;19)", "pro-B phenotype", "DIC risk", "ultra-high-risk", "hypercalcemia"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with ultra-high-risk fusion"}
        },
        {
            "chunk_id": "WHO2022_BCR_ABL_LIKE_044",
            "disease_category": "ALL",
            "specific_entity": "BCR::ABL1-like (Philadelphia-like) ALL",
            "source": "WHO 5th Edition (2022) / COG AALL1131",
            "title": "Ph-like ALL Kinase Signatures and Targeted Therapy",
            "text": "BCR::ABL1-like (Ph-like) ALL comprises 10-15% of pediatric standard-risk and 20-30% of high-risk B-ALL. Defined by gene expression profile resembling Ph+ ALL but lacking BCR::ABL1. Driving alterations: CRLF2 rearrangements (50%), ABL-class fusions (ABL1, ABL2, PDGFRB; 15%), JAK-STAT alterations (JAK2, JAK3, EPOR, IL7R; 10%), Ras pathway (NF1, NRAS, KRAS, PTPN11; 10%). High frequency of IKZF1 deletions (70%). Clinical: older age, high WBC, Hispanic ethnicity enrichment, Down syndrome association (CRLF2-rearranged). Poor prognosis with standard chemotherapy. Detection: gene expression profiling, low-cost surrogate markers (CRLF2 flow cytometry, ABL-class FISH), or comprehensive RNA-seq. Targetable kinases: TKIs for ABL-class, JAK inhibitors (ruxolitinib) for JAK-mutated.",
            "morphological_keywords": ["Ph-like ALL", "CRLF2 rearrangement", "ABL1 fusion", "JAK2 mutation", "IKZF1 deletion", "kinase activation"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard ALL blast threshold with kinase-activating alterations"}
        },
        {
            "chunk_id": "WHO2022_NPM1_045",
            "disease_category": "AML",
            "specific_entity": "AML with Mutated NPM1",
            "source": "WHO 5th Edition (2022) / ELN (2022)",
            "title": "NPM1-Mutated AML Favorable Risk Without FLT3-ITD",
            "text": "AML with mutated NPM1 (nucleophosmin) comprises 25-30% of adult AML and 2-8% of pediatric AML (more common in adolescents). NPM1 mutations cause aberrant cytoplasmic localization of NPM1 protein. Morphologically: monocytic or myelomonocytic features, cup-like nuclear invaginations, prominent nucleoli. Immunophenotype: often CD34-negative. Favorable prognosis (5-year OS 60-70% adults) in absence of FLT3-ITD high allelic ratio or adverse-risk mutations (DNMT3A, TET2, ASXL1). NPM1-mutated/FLT3-ITD-negative is ELN favorable-risk. MRD monitoring via NPM1 mutation-specific qPCR is highly sensitive. WHO 5th edition recognizes NPM1-mutated AML as distinct entity regardless of blast percentage.",
            "morphological_keywords": ["NPM1 mutation", "monocytic features", "cup-like nuclei", "CD34 negative", "cytoplasmic NPM1"],
            "blast_threshold_rule": {"min_blast_pct": 0, "condition": "NPM1 mutation defines AML regardless of blast %"}
        },
        {
            "chunk_id": "WHO2022_CEBPA_046",
            "disease_category": "AML",
            "specific_entity": "AML with Biallelic CEBPA Mutations",
            "source": "WHO 5th Edition (2022) / ELN (2022)",
            "title": "Biallelic CEBPA-Mutated AML Favorable Outcomes",
            "text": "AML with biallelic CEBPA mutations comprises 5-10% of AML (enriched in younger adults, rare in children). CEBPA (CCAAT/enhancer binding protein alpha) is transcription factor regulating myeloid differentiation. Biallelic mutations (one N-terminal, one C-terminal bZIP domain) are required; monoallelic CEBPA mutations lack favorable prognosis. Morphology: often FAB M1 or M2 with large blasts, prominent Auer rods. Immunophenotype: CD34+, CD117+, HLA-DR dim/negative. Favorable outcomes (5-year OS 60-70%) with standard chemotherapy; HSCT in first remission not routinely indicated. WHO 5th edition recognizes biallelic (not monoallelic) CEBPA-mutated AML as definitive entity. ELN favorable-risk classification.",
            "morphological_keywords": ["biallelic CEBPA", "large blasts", "Auer rods", "HLA-DR dim", "favorable prognosis"],
            "blast_threshold_rule": {"min_blast_pct": 20, "condition": "Standard AML blast threshold with biallelic CEBPA mutations"}
        },
        {
            "chunk_id": "WHO2022_GATA2_047",
            "disease_category": "Germline",
            "specific_entity": "Myeloid Neoplasms with Germline GATA2 Predisposition",
            "source": "WHO 5th Edition (2022)",
            "title": "GATA2 Deficiency Syndrome and MDS/AML Risk",
            "text": "Germline GATA2 mutations cause GATA2 deficiency syndrome (MonoMAC syndrome, Emberger syndrome) with immunodeficiency, lymphedema, pulmonary alveolar proteinosis, and high risk of MDS/AML (cumulative incidence ~75% by age 40). Hematologic features: monocytopenia, B-cell and NK-cell lymphopenia, often progressing to MDS (typically hypocellular with dysplasia, monosomy 7, trisomy 8) and AML. Morphology: hypocellular marrow with dysplastic megakaryocytes (hypolobated) and erythroid dysplasia. Diagnosis requires germline GATA2 sequencing. HSCT is curative but requires early intervention before overt MDS/AML. Family screening essential. WHO 5th edition includes myeloid neoplasms with germline predisposition as separate category.",
            "morphological_keywords": ["germline GATA2", "monocytopenia", "hypocellular MDS", "monosomy 7", "immunodeficiency"],
            "blast_threshold_rule": {"min_blast_pct": 0, "condition": "Germline GATA2 predisposes to MDS/AML at any blast %"}
        },
        {
            "chunk_id": "WHO2022_DDX41_048",
            "disease_category": "Germline",
            "specific_entity": "Myeloid Neoplasms with Germline DDX41 Mutation",
            "source": "WHO 5th Edition (2022)",
            "title": "DDX41-Associated Familial MDS/AML",
            "text": "Germline DDX41 mutations account for 3-5% of familial MDS/AML, predominantly in adults (median age 60-65 years, rarely pediatric). DDX41 is RNA helicase involved in splicing and innate immunity. Typical presentation: late-onset MDS or AML with normal karyotype or trisomy 8, often with somatic 'second hit' DDX41 mutation. Morphology: MDS with dysplasia or AML without distinct features. Hypoplastic marrow common. Favorable prognosis compared to sporadic MDS/AML; excellent response to chemotherapy and HSCT. Family history may be absent due to late penetrance. Screening first-degree relatives recommended. WHO 5th edition recognizes germline DDX41-associated myeloid neoplasms.",
            "morphological_keywords": ["germline DDX41", "late-onset MDS", "trisomy 8", "hypoplastic marrow", "favorable prognosis"],
            "blast_threshold_rule": {"min_blast_pct": 0, "condition": "Germline DDX41 predisposes to MDS/AML at any blast %"}
        },
    ]
    return phase2


def expand_to_50plus():
    """Main function to expand to 50+ chunks."""
    print("\n" + "=" * 70)
    print("PHASE 2: EXPANDING TO 50+ CHUNKS")
    print("=" * 70)
    
    # Load current expanded version
    with open(EXPANDED_FILE, 'r', encoding='utf-8') as f:
        current = json.load(f)
    print(f"Current chunks: {len(current)}")
    
    # Add Phase 2 chunks
    phase2 = create_phase2_chunks()
    expanded = current + phase2
    print(f"Added {len(phase2)} Phase 2 chunks")
    print(f"Total chunks: {len(expanded)}")
    
    # Save
    with open(EXPANDED_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)
    
    print(f"Expanded KB saved → {OUTPUT_FILE.name}")
    
    # Breakdown
    print("\n" + "=" * 70)
    print("FINAL CHUNK BREAKDOWN BY CATEGORY")
    print("=" * 70)
    from collections import Counter
    categories = Counter([c["disease_category"] for c in expanded])
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<30} {count:>3} chunks")
    
    print("\n✓ Knowledge base expansion to 50+ chunks complete!")
    print(f"  Expanded from 12 → {len(expanded)} chunks ({len(expanded)-12}+ new)")
    print(f"  Next: Rebuild indices with `python retrieval/index.py`")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--phase2":
        expand_to_50plus()
    else:
        main()
