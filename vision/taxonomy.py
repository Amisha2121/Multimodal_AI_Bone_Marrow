"""
vision/taxonomy.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for the BoneMarrowWSI-PediatricLeukemia morphology
taxonomy used by all training and inference code.

Design rules:
  1. MORPH_CLASSES is the ordered list of 8 categories that the ViT classifier
     predicts.  The order is fixed (determines integer label indices).
  2. EXCLUDED_CATEGORIES lists the mapped_cell_category values that are DROPPED
     before training / inference.  These are non-cell or low-reliability labels.
  3. BLAST_CATEGORIES lists the MORPH_CLASSES that count towards the
     "annotated-cell blast fraction" metric.  This is deliberately conservative:
     only cells the dataset labels as a blast type are included.

NOTE: "annotated-cell blast fraction" ≠ clinical blast percentage.
  The consensus annotations are not guaranteed to be a complete, representative
  bone-marrow differential.  Do NOT make clinical claims from this value.
"""

# ── Ordered morphology class list (fixes label indices 0-7) ──────────────────
MORPH_CLASSES: list[str] = [
    "Blast",                    # 0 – lymphocytic_blast, myeloid_blast, monocytic_blast, …
    "Granulocyte (Mature/Band)",  # 1 – segmented_neutrophil, band_granulocyte, …
    "Megakaryocyte / Platelet", # 2 – giant_platelet, thrombocyte, megakaryocyte, …
    "Myeloid (Maturing)",       # 3 – neutrophilic_myelocyte, metamyelocyte, …
    "Promyelocyte",             # 4 – promyelocyte
    "Lymphoid Series",          # 5 – lymphocyte, plasma_cell, prolymphocyte, …
    "Other Morphological",      # 6 – erythroblast (when not blast-labelled), other
    "Monocyte Series",          # 7 – monocyte, promonocyte
]

# Integer index lookup helpers
MORPH_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(MORPH_CLASSES)}
IDX_TO_MORPH: dict[int, str] = {i: c for i, c in enumerate(MORPH_CLASSES)}

# ── Categories excluded from the training corpus ─────────────────────────────
EXCLUDED_CATEGORIES: set[str] = {
    "Artefact (Excluded)",
    "Spicule (Tissue Fragment - Excluded)",
    "Ambiguous / No Consensus (Excluded)",
}

# ── Blast-fraction numerator classes ─────────────────────────────────────────
# Used only for reporting the annotated-cell blast fraction; NOT used as target.
BLAST_CATEGORIES: list[str] = ["Blast"]

# ── Rare classes that receive extra monitoring during evaluation ───────────────
RARE_CLASSES: list[str] = [
    "Monocyte Series",      # n=265 in full corpus (smallest valid class)
    "Lymphoid Series",      # n=515
    "Other Morphological",  # n=389
]

# ─────────────────────────────────────────────────────────────────────────────
# Legacy: keep CELL_CLASSES for any module that still imports from aggregate.py
# This alias will be removed once all legacy code is migrated.
CELL_CLASSES = MORPH_CLASSES
