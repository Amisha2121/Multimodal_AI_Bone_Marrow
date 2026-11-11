"""
agents/self_consistency.py
─────────────────────────────────────────────────────────────────────────────
Self-Consistency Checking for LLM-Generated Clinical Reports.

Literature Foundation:
    - Manakul et al. (2023): "SelfCheckGPT: Zero-Resource Black-Box Hallucination
      Detection for Generative Large Language Models"
      https://arxiv.org/abs/2303.08896
      
    - Wang et al. (2023): "Self-Consistency Improves Chain of Thought Reasoning
      in Language Models"
      https://arxiv.org/abs/2203.11171
      
    - Dhuliawala et al. (2024): "Chain-of-Verification Reduces Hallucination
      in Large Language Models"
      https://arxiv.org/abs/2309.11495

Clinical Rationale:
    Real clinicians don't trust a single interpretation. Multiple pathologists
    review complex cases and reach consensus. Self-consistency emulates this
    by generating multiple candidate reports and checking for contradictions.
    
    Key principle: If a model is uncertain or hallucinating, multiple samples
    at higher temperature will produce inconsistent outputs. High agreement
    across samples = high confidence. Low agreement = uncertainty or hallucination.

Implementation:
    1. Generate N candidate reports (default: N=3) with temperature=0.3
    2. Extract factual claims from each report
    3. Compute pairwise consistency using Natural Language Inference (NLI)
    4. Select most consistent report or flag disagreements
    5. Add consistency score to Critic evaluation

Author: AI Assistant
Date: November 8, 2026
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("self_consistency")


@dataclass
class FactualClaim:
    """
    A factual claim extracted from a clinical report.
    
    Attributes:
        text: The claim text
        sentence_id: Index of sentence in report
        claim_type: Type of claim ("blast_fraction", "morphology", "diagnosis", "guideline_citation", "other")
    """
    text: str
    sentence_id: int
    claim_type: str


@dataclass
class ConsistencyResult:
    """
    Result of self-consistency checking across multiple report candidates.
    
    Attributes:
        num_samples: Number of candidate reports generated
        selected_report: The most consistent report (or consensus if tied)
        consistency_score: Overall consistency score (0-1, higher = more consistent)
        pairwise_scores: Matrix of pairwise NLI scores
        disagreement_flags: List of claims with high disagreement
        recommendation: Clinical recommendation based on consistency
    """
    num_samples: int
    selected_report: str
    consistency_score: float
    pairwise_scores: np.ndarray = field(repr=False)
    disagreement_flags: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    # Per-sample metrics
    sample_lengths: List[int] = field(default_factory=list)
    sample_num_claims: List[int] = field(default_factory=list)


def extract_factual_claims(report: str) -> List[FactualClaim]:
    """
    Extract factual claims from clinical report.
    
    Focus on:
    - Blast fraction statements
    - Morphology descriptions
    - Guideline citations
    - Clinical recommendations
    
    Args:
        report: Clinical report text
    
    Returns:
        List of FactualClaim objects
    """
    claims = []
    
    # Split into sentences (simple regex-based)
    sentences = re.split(r'(?<=[.!?])\s+', report)
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short fragments
            continue
        
        # Classify claim type based on keywords
        claim_type = "other"
        
        sentence_lower = sentence.lower()
        
        if any(keyword in sentence_lower for keyword in ["blast", "blasts", "%", "fraction"]):
            claim_type = "blast_fraction"
        elif any(keyword in sentence_lower for keyword in ["morphology", "cells", "lymphoblasts", "promyelocyte", "granulocyte", "monocyte"]):
            claim_type = "morphology"
        elif any(keyword in sentence_lower for keyword in ["diagnosis", "differential", "aml", "all", "mds"]):
            claim_type = "diagnosis"
        elif re.search(r'\[WHO\d+_[A-Za-z0-9_]+\]', sentence):
            claim_type = "guideline_citation"
        
        claims.append(FactualClaim(
            text=sentence,
            sentence_id=i,
            claim_type=claim_type
        ))
    
    return claims


def compute_nli_consistency(
    claims1: List[FactualClaim],
    claims2: List[FactualClaim],
    use_simple_overlap: bool = True,
) -> float:
    """
    Compute consistency score between two sets of claims.
    
    In production, this would use a proper NLI model (e.g., microsoft/deberta-v3-large-mnli).
    For simplicity, we use text overlap as a proxy.
    
    Args:
        claims1: Claims from report 1
        claims2: Claims from report 2
        use_simple_overlap: If True, use simple text overlap (no NLI model required)
    
    Returns:
        Consistency score (0-1, higher = more consistent)
    """
    if len(claims1) == 0 or len(claims2) == 0:
        return 0.0
    
    if use_simple_overlap:
        # Simple implementation: compute Jaccard similarity of claim texts
        # In production, replace with actual NLI model
        
        # Normalize claim texts for comparison
        texts1 = set(normalize_claim_text(c.text) for c in claims1)
        texts2 = set(normalize_claim_text(c.text) for c in claims2)
        
        # Jaccard similarity
        intersection = len(texts1 & texts2)
        union = len(texts1 | texts2)
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    else:
        # TODO: Implement proper NLI-based consistency
        # Use model like microsoft/deberta-v3-large-mnli
        # For each claim in claims1, check if it's entailed/contradicted by claims2
        # Average entailment scores
        raise NotImplementedError("NLI-based consistency not yet implemented. Use use_simple_overlap=True.")


def normalize_claim_text(text: str) -> str:
    """Normalize claim text for comparison."""
    # Remove citations (both WHO2022 and general WHO patterns)
    text = re.sub(r'\[WHO\d*_?[A-Za-z0-9_]+\]', '', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def detect_factual_disagreements(
    claims_per_sample: List[List[FactualClaim]],
    disagreement_threshold: float = 0.5,
) -> List[str]:
    """
    Detect factual claims that show high disagreement across samples.
    
    Args:
        claims_per_sample: List of claim lists (one per sample)
        disagreement_threshold: Threshold for flagging disagreement
    
    Returns:
        List of disagreement descriptions
    """
    if len(claims_per_sample) < 2:
        return []
    
    disagreements = []
    
    # Group claims by type
    by_type: Dict[str, List[List[FactualClaim]]] = {}
    for claims in claims_per_sample:
        for claim in claims:
            if claim.claim_type not in by_type:
                by_type[claim.claim_type] = [[] for _ in range(len(claims_per_sample))]
    
    for sample_idx, claims in enumerate(claims_per_sample):
        for claim in claims:
            by_type[claim.claim_type][sample_idx].append(claim)
    
    # Check for disagreements within each type
    for claim_type, claims_lists in by_type.items():
        # Compute pairwise consistency for this type
        n_samples = len(claims_lists)
        scores = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                score = compute_nli_consistency(claims_lists[i], claims_lists[j], use_simple_overlap=True)
                scores.append(score)
        
        if scores:
            mean_score = np.mean(scores)
            
            if mean_score < disagreement_threshold:
                # High disagreement detected
                example_claims = [
                    claims_lists[0][0].text if claims_lists[0] else "N/A",
                    claims_lists[1][0].text if len(claims_lists) > 1 and claims_lists[1] else "N/A",
                ]
                
                disagreements.append(
                    f"{claim_type.upper()} claims show low consistency ({mean_score:.2f}). "
                    f"Sample variations: '{example_claims[0][:80]}...' vs '{example_claims[1][:80]}...'"
                )
    
    return disagreements


def select_most_consistent_report(
    reports: List[str],
    pairwise_scores: np.ndarray,
) -> Tuple[int, str]:
    """
    Select the report that is most consistent with all others.
    
    Uses average consistency score with all other reports as selection criterion.
    
    Args:
        reports: List of candidate reports
        pairwise_scores: N x N matrix of pairwise consistency scores
    
    Returns:
        (selected_index, selected_report)
    """
    if len(reports) == 1:
        return 0, reports[0]
    
    # Compute average consistency for each report
    avg_scores = []
    for i in range(len(reports)):
        # Average consistency with all other reports
        scores_with_others = [pairwise_scores[i, j] for j in range(len(reports)) if i != j]
        avg_scores.append(np.mean(scores_with_others) if scores_with_others else 0.0)
    
    # Select report with highest average consistency
    best_idx = int(np.argmax(avg_scores))
    
    return best_idx, reports[best_idx]


def check_self_consistency(
    reports: List[str],
    consistency_threshold: float = 0.70,
) -> ConsistencyResult:
    """
    Check self-consistency across multiple LLM-generated reports.
    
    Args:
        reports: List of candidate reports (typically 3)
        consistency_threshold: Threshold for high consistency
    
    Returns:
        ConsistencyResult with selected report and consistency metrics
    """
    num_samples = len(reports)
    
    if num_samples == 1:
        # Single sample - no consistency check needed
        return ConsistencyResult(
            num_samples=1,
            selected_report=reports[0],
            consistency_score=1.0,
            pairwise_scores=np.array([[1.0]]),
            disagreement_flags=[],
            recommendation="SINGLE_SAMPLE",
            sample_lengths=[len(reports[0])],
            sample_num_claims=[len(extract_factual_claims(reports[0]))],
        )
    
    # Extract claims from each report
    claims_per_sample = [extract_factual_claims(report) for report in reports]
    
    # Compute pairwise consistency matrix
    pairwise_scores = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        pairwise_scores[i, i] = 1.0  # Perfect self-consistency
        for j in range(i + 1, num_samples):
            score = compute_nli_consistency(
                claims_per_sample[i],
                claims_per_sample[j],
                use_simple_overlap=True
            )
            pairwise_scores[i, j] = score
            pairwise_scores[j, i] = score  # Symmetric
    
    # Overall consistency score (mean of off-diagonal elements)
    off_diagonal = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            off_diagonal.append(pairwise_scores[i, j])
    
    overall_consistency = float(np.mean(off_diagonal)) if off_diagonal else 1.0
    
    # Detect disagreements
    disagreements = detect_factual_disagreements(claims_per_sample, disagreement_threshold=0.5)
    
    # Select most consistent report
    selected_idx, selected_report = select_most_consistent_report(reports, pairwise_scores)
    
    # Generate recommendation
    if overall_consistency >= consistency_threshold:
        recommendation = "HIGH_CONSISTENCY"
    elif overall_consistency >= 0.50:
        recommendation = "MODERATE_CONSISTENCY"
    else:
        recommendation = "LOW_CONSISTENCY"
    
    if len(disagreements) > 0:
        recommendation = "INCONSISTENT"
    
    return ConsistencyResult(
        num_samples=num_samples,
        selected_report=selected_report,
        consistency_score=overall_consistency,
        pairwise_scores=pairwise_scores,
        disagreement_flags=disagreements,
        recommendation=recommendation,
        sample_lengths=[len(r) for r in reports],
        sample_num_claims=[len(claims) for claims in claims_per_sample],
    )


def format_consistency_report(result: ConsistencyResult) -> str:
    """
    Format self-consistency results as human-readable report.
    
    Args:
        result: ConsistencyResult object
    
    Returns:
        Formatted markdown report
    """
    report = f"""
## Self-Consistency Check

**Samples Generated**: {result.num_samples}  
**Overall Consistency Score**: {result.consistency_score:.3f}  
**Recommendation**: {result.recommendation}

"""
    
    if result.num_samples > 1:
        report += "### Pairwise Consistency Matrix\n\n"
        report += "| Sample | "
        for i in range(result.num_samples):
            report += f"Sample {i+1} | "
        report += "\n|--------|"
        for _ in range(result.num_samples):
            report += "---------|"
        report += "\n"
        
        for i in range(result.num_samples):
            report += f"| Sample {i+1} | "
            for j in range(result.num_samples):
                report += f"{result.pairwise_scores[i, j]:.3f} | "
            report += "\n"
        
        report += "\n"
    
    if result.disagreement_flags:
        report += "### Detected Disagreements\n\n"
        for flag in result.disagreement_flags:
            report += f"- {flag}\n"
        report += "\n"
    
    # Clinical interpretation
    report += "### Clinical Interpretation\n\n"
    
    if result.consistency_score >= 0.70:
        report += "**HIGH CONSISTENCY**: Multiple samples show strong agreement. The model is confident in its assessment.\n"
    elif result.consistency_score >= 0.50:
        report += "**MODERATE CONSISTENCY**: Some variation across samples. The model shows moderate uncertainty.\n"
    else:
        report += "**LOW CONSISTENCY**: Significant disagreement across samples. The model is uncertain or hallucinating. Manual review strongly recommended.\n"
    
    if len(result.disagreement_flags) > 0:
        report += "\n**FACTUAL INCONSISTENCIES DETECTED**: Different samples made contradictory claims. This may indicate hallucination or high model uncertainty. Expert review is required.\n"
    
    report += "\n---\n\n"
    report += "**Literature**: Manakul et al. (2023) SelfCheckGPT, Wang et al. (2023) Self-Consistency\n"
    
    return report


def integrate_consistency_into_critic_score(
    original_grounding_score: float,
    consistency_score: float,
    weight: float = 0.3,
) -> float:
    """
    Integrate self-consistency score into Critic grounding score.
    
    Args:
        original_grounding_score: Critic's original evidence grounding score (0-1)
        consistency_score: Self-consistency score (0-1)
        weight: Weight for consistency component (default: 0.3)
    
    Returns:
        Combined score (0-1)
    """
    # Weighted combination
    combined = (1 - weight) * original_grounding_score + weight * consistency_score
    
    # If consistency is very low (<0.5), apply penalty
    if consistency_score < 0.5:
        penalty = (0.5 - consistency_score) * 0.5  # Up to 0.25 penalty
        combined = max(0.0, combined - penalty)
    
    return float(np.clip(combined, 0.0, 1.0))


# ── Diagnostic utilities ──────────────────────────────────────────────────────

def compute_report_statistics(reports: List[str]) -> Dict[str, Any]:
    """Compute statistics about report variations."""
    lengths = [len(r) for r in reports]
    num_claims = [len(extract_factual_claims(r)) for r in reports]
    
    return {
        "num_reports": len(reports),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
        "length_range": (min(lengths), max(lengths)),
        "claims_mean": np.mean(num_claims),
        "claims_std": np.std(num_claims),
        "claims_range": (min(num_claims), max(num_claims)),
    }


def visualize_consistency_matrix(pairwise_scores: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize pairwise consistency matrix as heatmap.
    
    Requires matplotlib. Optional visualization for debugging.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pairwise_scores, cmap=cm.RdYlGn, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Consistency Score', rotation=270, labelpad=15)
        
        # Labels
        n = pairwise_scores.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"Sample {i+1}" for i in range(n)])
        ax.set_yticklabels([f"Sample {i+1}" for i in range(n)])
        
        # Add values to cells
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f"{pairwise_scores[i, j]:.2f}",
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title("Self-Consistency Matrix")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Consistency matrix saved to {save_path}")
        
        return fig
    
    except ImportError:
        logger.warning("Matplotlib not available for visualization")
        return None
