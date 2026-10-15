# RAG Clinical Knowledge Sources

## Overview
The Retrieval-Augmented Generation (RAG) system uses clinical knowledge from the **WHO Classification of Haematolymphoid Tumours, 5th Edition (2022)** and **International Consensus Classification (ICC) 2022** guidelines.

## Source Document
Location: `data/knowledge_base/clinical_guidelines.json`

## WHO 2022 Clinical Chunks (12 Total)

### 1. B-Lymphoblastic Leukemia/Lymphoma (B-ALL)
- **Chunk ID**: `WHO2022_BALL_001`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: Diagnostic criteria for B-ALL including cytomorphology (high N:C ratio, scant agranular cytoplasm), >=20% blast threshold, accounts for 80-85% of pediatric acute leukemias

### 2. T-Lymphoblastic Leukemia/Lymphoma (T-ALL)
- **Chunk ID**: `WHO2022_TALL_002`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: T-ALL morphology (pleomorphic blasts, convoluted nuclei), 15% of pediatric leukemias, mediastinal mass association, cytoplasmic CD3 requirement

### 3. Burkitt Leukemia (Mature B-cell ALL)
- **Chunk ID**: `WHO2022_BURKITT_003`
- **Source**: WHO 5th Edition (2022) / FAB Classification (L3)
- **Content**: Cytomorphological triad (intensely basophilic cytoplasm, lipid vacuoles, starry-sky pattern), MYC translocation t(8;14), oncologic emergency requiring pulse chemotherapy

### 4. Acute Myeloid Leukemia (AML) - General Criteria
- **Chunk ID**: `WHO2022_AML_DEF_004`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: AML definition (>=20% myeloid blasts), exceptions for recurrent genetic abnormalities (t(8;21), inv(16), PML::RARA), myeloblast morphology

### 5. Acute Promyelocytic Leukemia (APL)
- **Chunk ID**: `WHO2022_AML_SUB_005`
- **Source**: WHO 5th Edition (2022) / ELN (2022)
- **Content**: APL with PML::RARA, Auer rods/faggot cells, hypergranular promyelocytes, DIC risk, ATRA/ATO emergency treatment

### 6. AML with Maturation
- **Chunk ID**: `WHO2022_AML_MAT_006`
- **Source**: WHO 5th Edition (2022)
- **Content**: AML with maturation (FAB M2) and acute myelomonocytic leukemia (FAB M4), granulocytic maturation >10%, monoblast criteria

### 7. CML Chronic Phase
- **Chunk ID**: `WHO2022_CML_CP_007`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: CML-CP cytomorphology, "myelocyte bulge," high M:E ratio (10:1 to 30:1), blasts <10%, BCR::ABL1 positive, basophilia

### 8. CML Accelerated & Blast Phases
- **Chunk ID**: `WHO2022_CML_ADV_008`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: CML-AP (10-19% blasts), CML-BP (>=20% blasts), clonal evolution, myeloid vs lymphoid transformation

### 9. Haematogones vs. B-ALL
- **Chunk ID**: `WHO2022_HAEMATOGONES_009`
- **Source**: Clinical Pediatric Haematology Guidelines / WHO 2022
- **Content**: Distinguishing benign B-cell precursors (haematogones) from leukemic lymphoblasts, maturational spectrum vs. maturational arrest, flow cytometry requirement

### 10. Reactive Bone Marrow
- **Chunk ID**: `WHO2022_REACTIVE_010`
- **Source**: Pediatric Bone Marrow Pathology Handbook (WHO 2022)
- **Content**: Normal pediatric cellularity (80-90% infants, 60-70% older children), reactive granulocytic hyperplasia, infection markers (toxic granulation, Dohle bodies), blasts <5%

### 11. Mixed Phenotype Acute Leukemia (MPAL)
- **Chunk ID**: `WHO2022_MPAL_011`
- **Source**: WHO 5th Edition (2022) / ICC (2022)
- **Content**: MPAL diagnostic criteria (2-3% of pediatric leukemias), biphenotypic/bilineal blasts, lineage assignment criteria (MPO, CD3, CD19), flow cytometry essential

### 12. Monolayer Sampling Disclaimer
- **Chunk ID**: `DISCLAIMER_MONOLAYER_012`
- **Source**: Computational Pathology Methodology Standards
- **Content**: Scientific limitation explaining that annotated blast fractions from monolayer ROIs differ from exhaustive 500-cell clinical differential counts

## RAG Retrieval System Architecture

### Dense Retrieval
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384-dimensional vectors)
- **Index**: FAISS cosine similarity search
- **Purpose**: Semantic matching of clinical queries to WHO guidelines

### Sparse Retrieval  
- **Method**: BM25 (Best Matching 25) algorithm
- **Purpose**: Keyword-based matching for clinical terminology

### Hybrid Retrieval
- **Combination**: Reciprocal Rank Fusion (RRF) of dense + sparse results
- **Top-K**: Returns 2-4 most relevant chunks per query
- **Performance**: Evaluated on 10 clinical benchmarks (see `retrieval/evaluate_retrieval.py`)

## Official WHO References

### Primary Source
**WHO Classification of Haematolymphoid Tumours, 5th Edition (2022)**
- Publisher: International Agency for Research on Cancer (IARC)
- ISBN: 978-92-832-4522-9
- Official Link: https://publications.iarc.fr/615
- Citation: Alaggio R, Amador C, Anagnostopoulos I, et al. The 5th edition of the World Health Organization Classification of Haematolymphoid Tumours: Lymphoid Neoplasms. Leukemia. 2022;36(7):1720-1748. DOI: 10.1038/s41375-022-01620-2

### Complementary Source
**International Consensus Classification (ICC) 2022**
- Publisher: Nature (Blood Cancer Discovery)
- Official Link: https://ashpublications.org/bloodadvances/article/6/16/4850/485179/International-Consensus-Classification-of-Myeloid
- Citation: Arber DA, Orazi A, Hasserjian RP, et al. International Consensus Classification of Myeloid Neoplasms and Acute Leukemias: integrating morphologic, clinical, and genomic data. Blood. 2022;140(11):1200-1228. DOI: 10.1182/blood.2022015850

## How RAG is Used in Reports

1. **Query Formation**: Each report section generates queries like:
   - "Bone marrow cytomorphology blast {blast_fraction}%"
   - "Blast cells acute leukemia WHO threshold"
   - "Differential diagnosis {blast_fraction}% blasts"

2. **Retrieval**: System retrieves 2 most relevant WHO chunks

3. **Context Injection**: Retrieved WHO guidelines are provided to the LLM or used in fallback templates

4. **Citation**: Chunks are cited by ID (e.g., WHO2022_BALL_001) in synthesized reports

## Validation & Benchmarking

The retrieval system has been evaluated on 10 clinical queries covering:
- B-ALL diagnostic criteria
- Haematogones vs. leukemic blasts
- Burkitt leukemia features
- APL emergency recognition
- CML chronic vs. blast phase
- T-ALL characteristics
- Reactive marrow patterns
- MPAL lineage criteria

Results: `data/processed/retrieval_index/retrieval_benchmark_results.json`

## Future Enhancements

Potential additions to the knowledge base:
1. WHO 2022 Myelodysplastic Syndromes (MDS) chapters
2. WHO 2022 Myeloproliferative Neoplasms (MPN) detailed criteria
3. ELN 2022 AML risk stratification guidelines
4. COG (Children's Oncology Group) pediatric treatment protocols
5. Flow cytometry immunophenotyping patterns
6. Cytogenetic and molecular marker clinical correlations
