# data/dataset_cards/matek2021_bone_marrow.md

# Dataset Card — Matek et al. 2021 Bone Marrow Cytomorphology

## Identity

| Field | Value |
|---|---|
| **Full title** | An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies |
| **Authors** | Christian Matek, Sayedali Salehi, Karsten Spiekermann, Carsten Marr |
| **Year** | 2021 |
| **Repository** | The Cancer Imaging Archive (TCIA) |
| **DOI** | https://doi.org/10.7937/TCIA.AXH3-T579 |
| **License** | **CC BY 4.0** |
| **Collection name** | `Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer` |

## Modality

**Bone marrow aspirate** — single-cell microscopy images from Wright-Giemsa-stained BM aspirate slides.  
*Not* peripheral blood smear. This is the correct modality for a bone marrow morphology project.

## Scale

| Metric | Value |
|---|---|
| Total cell images | ~171,375 |
| Patients | 945 |
| Cell classes | **21** |
| Image format | PNG, 400×400 px (approx.) |

## 21 Cell Classes

| Code | Class |
|---|---|
| BAS | Basophil |
| EBO | Erythroblast |
| EOS | Eosinophil |
| FGC | Faggott cell |
| HAC | Hairy cell |
| KSC | Smudge cell |
| LYI | Immature lymphocyte |
| LYT | Lymphocyte |
| MMZ | Metamyelocyte |
| MOB | Monoblast |
| MON | Monocyte |
| MYB | Myelocyte |
| MYO | Myeloblast |
| NGB | Band neutrophil |
| NGS | Segmented neutrophil |
| NIF | Not identified |
| OTH | Other |
| PEB | Proerythroblast |
| PLM | Plasma cell |
| PMO | Promyelocyte |
| ART | Artefact |

## Blast % Computation (WHO 5th Edition)

Per WHO 5th edition:
- **AML threshold:** ≥ 20% blasts (myeloblasts + monoblasts in AML-M5)
- **MDS threshold:** < 20% blasts, ≥ 5% in MDS-EB1 (5–9%), ≥ 10% in MDS-EB2 (10–19%)

In this project, blast % is computed as:

```
blast_pct = differential["MYO"] + differential["MOB"]  # per case; extend for AML subtypes
```

## License & Attribution

This dataset is released under **CC BY 4.0**. You are free to use, share, and adapt it for any purpose, provided you give appropriate credit.

**Required citation:**

```
Matek C, Salehi S, Spiekermann K, Marr C.
"Human-level recognition of blast cells in aggressive haematological malignancies."
Nature Machine Intelligence, 2019.
Dataset DOI: https://doi.org/10.7937/TCIA.AXH3-T579
```

Also cite TCIA:

```
Clark K, Vendt B, Smith K, et al.
"The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
Information Repository."
Journal of Digital Imaging, 2013; 26(6): 1045–1057.
```

## Access

1. Register (free) at https://www.cancerimagingarchive.net/
2. Run `python data/download_tcia.py --out data/raw`
3. No DUA required beyond standard citation.

## Provenance Verification Date

Verified: 2026-10-05. License and modality confirmed against TCIA collection page.
