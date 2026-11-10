"""
agents/synthesizer_with_consistency.py
─────────────────────────────────────────────────────────────────────────────
Enhanced Synthesizer Node with Self-Consistency Checking.

Wraps the original synthesizer to generate multiple candidate reports and
select the most consistent one using SelfCheckGPT methodology.

Usage:
    Replace `synthesizer_node` with `synthesizer_with_consistency_node` in graph.py
    to enable self-consistency checking.

Author: AI Assistant
Date: November 8, 2026
"""

import logging
from typing import Dict, Any

from agents.state import AgentState
from agents.synthesizer import build_llm_powered_synthesis, build_deterministic_synthesis
from agents.self_consistency import (
    check_self_consistency,
    format_consistency_report,
    integrate_consistency_into_critic_score,
)

logger = logging.getLogger("synthesizer_with_consistency")


def generate_multiple_synthesis_candidates(
    state: AgentState,
    num_samples: int = 3,
    temperature: float = 0.3,
) -> list[str]:
    """
    Generate multiple candidate reports using LLM with higher temperature.
    
    Args:
        state: Agent state with case data
        num_samples: Number of candidates to generate (default: 3)
        temperature: Sampling temperature for diversity (default: 0.3)
    
    Returns:
        List of candidate report strings
    """
    from agents.llm_provider import get_llm_provider
    
    pid = state.get("patient_id", "UNKNOWN")
    logger.info(f"[{pid}] Generating {num_samples} synthesis candidates (T={temperature})...")
    
    candidates = []
    
    try:
        # Generate multiple samples
        for i in range(num_samples):
            logger.debug(f"[{pid}] Generating candidate {i+1}/{num_samples}...")
            
            # Call build_llm_powered_synthesis but extract just the text
            # We need to temporarily modify temperature in the LLM call
            result = build_llm_powered_synthesis(state, temperature=temperature)
            draft = result.get("draft_differential", "")
            
            if draft:
                candidates.append(draft)
            else:
                logger.warning(f"[{pid}] Candidate {i+1} generation failed, skipping")
        
        if len(candidates) == 0:
            logger.warning(f"[{pid}] All candidates failed, falling back to deterministic")
            # Fallback to deterministic
            result = build_deterministic_synthesis(state)
            candidates.append(result.get("draft_differential", ""))
    
    except Exception as e:
        logger.error(f"[{pid}] Multi-sample generation failed: {e}")
        # Fallback to deterministic
        result = build_deterministic_synthesis(state)
        candidates.append(result.get("draft_differential", ""))
    
    logger.info(f"[{pid}] Generated {len(candidates)} candidate reports")
    return candidates


def synthesizer_with_consistency_node(
    state: AgentState,
    num_samples: int = 3,
    temperature: float = 0.3,
    enable_consistency_check: bool = True,
) -> AgentState:
    """
    Enhanced synthesizer that generates multiple candidates and selects the most consistent.
    
    Workflow:
        1. Generate N candidate reports (T=0.3 for diversity)
        2. Check self-consistency across candidates
        3. Select most consistent report
        4. Add consistency score to state for Critic integration
    
    Args:
        state: Agent state
        num_samples: Number of candidate reports (default: 3)
        temperature: Sampling temperature (default: 0.3)
        enable_consistency_check: If False, falls back to original single-sample mode
    
    Returns:
        Updated agent state with selected draft and consistency metadata
    """
    pid = state.get("patient_id", "UNKNOWN")
    
    # Check if consistency checking is disabled
    if not enable_consistency_check or num_samples <= 1:
        logger.info(f"[{pid}] Self-consistency disabled, using single-sample synthesis")
        # Fallback to original synthesizer
        result = build_llm_powered_synthesis(state, temperature=0.1)
        state["draft_differential"] = result["draft_differential"]
        state["differential_diagnosis_hierarchy"] = result["differential_diagnosis_hierarchy"]
        state["morphological_justification"] = result["morphological_justification"]
        state["consistency_check_enabled"] = False
        return state
    
    # Generate multiple candidates
    candidates = generate_multiple_synthesis_candidates(state, num_samples, temperature)
    
    if len(candidates) == 0:
        logger.error(f"[{pid}] No candidates generated, returning empty state")
        state["draft_differential"] = "(Generation failed)"
        state["consistency_check_enabled"] = False
        return state
    
    # Perform self-consistency check
    logger.info(f"[{pid}] Checking self-consistency across {len(candidates)} candidates...")
    consistency_result = check_self_consistency(candidates, consistency_threshold=0.70)
    
    # Select the most consistent report
    selected_report = consistency_result.selected_report
    
    # Update state with selected draft
    state["draft_differential"] = selected_report
    state["consistency_check_enabled"] = True
    state["consistency_score"] = consistency_result.consistency_score
    state["consistency_recommendation"] = consistency_result.recommendation
    state["consistency_disagreement_flags"] = consistency_result.disagreement_flags
    state["consistency_num_samples"] = consistency_result.num_samples
    state["consistency_pairwise_scores"] = consistency_result.pairwise_scores.tolist()  # For JSON serialization
    
    # Generate consistency report for logging/debugging
    consistency_report_text = format_consistency_report(consistency_result)
    state["consistency_report"] = consistency_report_text
    
    # Log results
    logger.info(
        f"[{pid}] Self-consistency: {consistency_result.consistency_score:.3f} "
        f"({consistency_result.recommendation})"
    )
    
    if consistency_result.disagreement_flags:
        logger.warning(
            f"[{pid}] Disagreements detected: {len(consistency_result.disagreement_flags)} issues"
        )
        for flag in consistency_result.disagreement_flags[:3]:  # Log first 3
            logger.warning(f"  - {flag}")
    
    # Extract differential hierarchy from selected report
    # (Simple heuristic: look for disease mentions)
    import re
    disease_mentions = []
    for disease in ["ALL", "AML", "MDS", "MPN", "CML", "CLL"]:
        if disease in selected_report:
            disease_mentions.append(disease)
    
    state["differential_diagnosis_hierarchy"] = disease_mentions if disease_mentions else ["Unknown"]
    state["morphological_justification"] = (
        f"LLM-generated assessment with self-consistency checking "
        f"({consistency_result.num_samples} samples, consistency={consistency_result.consistency_score:.3f})"
    )
    
    return state


def integrate_consistency_into_critic(state: AgentState) -> AgentState:
    """
    Modify Critic's grounding score to incorporate self-consistency.
    
    Call this after Critic node but before final approval decision.
    
    Args:
        state: Agent state with both critic_grounding_score and consistency_score
    
    Returns:
        Updated state with combined grounding score
    """
    if not state.get("consistency_check_enabled", False):
        # No consistency check was performed
        return state
    
    original_score = state.get("critic_grounding_score", 1.0)
    consistency_score = state.get("consistency_score", 1.0)
    
    # Combine scores (70% grounding, 30% consistency)
    combined_score = integrate_consistency_into_critic_score(
        original_grounding_score=original_score,
        consistency_score=consistency_score,
        weight=0.3
    )
    
    pid = state.get("patient_id", "UNKNOWN")
    logger.info(
        f"[{pid}] Combined grounding score: {original_score:.3f} (grounding) + "
        f"{consistency_score:.3f} (consistency) = {combined_score:.3f}"
    )
    
    # Store both scores for transparency
    state["critic_grounding_score_original"] = original_score
    state["critic_grounding_score"] = combined_score  # Override with combined
    state["consistency_weight_applied"] = 0.3
    
    return state


# ── Backward compatibility ────────────────────────────────────────────────────

# Alias for drop-in replacement in graph.py
def synthesizer_node(state: AgentState) -> AgentState:
    """
    Drop-in replacement for original synthesizer_node with self-consistency enabled.
    
    To disable self-consistency, set state["disable_self_consistency"] = True
    """
    disable = state.get("disable_self_consistency", False)
    
    if disable:
        # Use original single-sample synthesizer
        from agents.synthesizer import synthesizer_node as original_synthesizer
        return original_synthesizer(state)
    
    # Use enhanced synthesizer with consistency checking
    return synthesizer_with_consistency_node(
        state,
        num_samples=3,
        temperature=0.3,
        enable_consistency_check=True
    )
