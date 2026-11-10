# Improvement #8: Active Learning Pipeline

**Status**: ✅ Complete  
**Date**: November 9, 2026  
**Priority**: Low (Future Work / Clinical Deployment)  
**Literature**: Section 10.2 - Active Learning

---

## Table of Contents

1. [Overview](#overview)
2. [Literature Foundation](#literature-foundation)
3. [Implementation](#implementation)
4. [Uncertainty Strategies](#uncertainty-strategies)
5. [Priority Scoring](#priority-scoring)
6. [Usage Examples](#usage-examples)
7. [Clinical Workflow Integration](#clinical-workflow-integration)
8. [Future Work](#future-work)

---

## Overview

### Motivation

Real clinical AI deployment requires **human-in-the-loop validation**. Not all cases should be reviewed equally:

1. **High-confidence cases** → Trust the AI, minimal review
2. **Uncertain cases** → Prioritize for expert pathologist review
3. **Corrected cases** → Collect feedback for model retraining

Active learning optimizes expert time by **prioritizing uncertain/high-impact cases** for review, then using corrections for **continuous model improvement**.

### Goals

1. **Uncertainty quantification**: Identify cases where the model is uncertain
2. **Priority scoring**: Rank cases by clinical importance
3. **Review queue management**: Maintain ordered list of cases needing review
4. **Feedback collection**: Track expert corrections and disagreements
5. **Retraining triggers**: Recommend model updates when enough corrections accumulated

---

## Literature Foundation

### Settles (2009): Active Learning Literature Survey

**Citation**: Settles, B. (2009). "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin–Madison.

**Key Concepts**:
- **Uncertainty Sampling**: Query instances where model is least confident
- **Query-by-Committee**: Use disagreement among ensemble
- **Expected Model Change**: Select samples that would most change the model

**Uncertainty Strategies**:
- **Least Confident**: 1 - max(P(y|x))
- **Margin Sampling**: P(y₁|x) - P(y₂|x) (difference between top-2)
- **Entropy**: -Σ P(y|x) log P(y|x)

### Budd et al. (2021): Medical Image Active Learning

**Citation**: Budd, S., Robinson, E. C., & Kainz, B. (2021). "A survey on active learning and human-in-the-loop deep learning for medical image analysis." Medical Image Analysis, 71, 102062.

**Medical AI Specifics**:
- **Class imbalance**: Rare diseases need oversampling
- **Expert time is expensive**: Optimize annotation efficiency
- **Clinical priority**: Not just uncertainty, but clinical impact
- **Safety**: High-risk cases always reviewed

**Recommended Approach**:
1. Combine **uncertainty** with **diversity** sampling
2. Incorporate **domain knowledge** (e.g., rare diagnoses)
3. Use **acquisition functions** that balance exploration/exploitation

### Smailagic et al. (2018): MedAL

**Citation**: Smailagic, A., et al. (2018). "Medal: Accurate and robust deep active learning for medical image analysis." ICMLA 2018.

**Contributions**:
- **Bayesian Active Learning by Disagreement (BALD)**: Use dropout at test time
- **Core-set selection**: Ensure diverse coverage
- **Medical-specific heuristics**: Integrate radiologist feedback

---

## Implementation

### Architecture

```
pipeline/
├── active_learning.py       # Core module (680 lines)
└── __init__.py

tests/
└── test_active_learning.py  # 23 unit tests (530 lines)

data/
└── active_learning/
    ├── review_queue.json    # Persistent queue
    └── expert_feedback.json # Feedback log
```

### Core Components

#### 1. Uncertainty Quantification

```python
class ActiveLearningPipeline:
    def compute_uncertainty(
        self,
        case_id: str,
        predicted_class: str,
        class_probabilities: np.ndarray,
        grounding_score: Optional[float] = None,
        consistency_score: Optional[float] = None
    ) -> CaseUncertainty:
        """
        Compute multiple uncertainty metrics.
        
        Returns:
            CaseUncertainty with:
            - confidence: max(P)
            - entropy: -Σ P log(P)
            - margin: P₁ - P₂
            - least_confident: 1 - max(P)
            - combined_uncertainty: weighted average
        """
```

**Uncertainty Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Confidence | max(P) | Probability of top class |
| Entropy | -Σ P log(P) | Overall prediction uncertainty |
| Margin | P₁ - P₂ | Gap between top-2 classes |
| Least Confident | 1 - max(P) | Inverse of confidence |

**Combined Uncertainty**:
```
U_combined = w₁ × entropy_normalized + 
             w₂ × (1 - confidence) +
             w₃ × (1 - grounding) +
             w₄ × (1 - consistency)
```

Default weights: `{entropy: 0.3, confidence: 0.3, grounding: 0.2, consistency: 0.2}`

#### 2. Priority Scoring

```python
def prioritize_case(
    self,
    uncertainty: CaseUncertainty,
    metadata: Optional[Dict] = None
) -> Tuple[float, List[PriorityReason]]:
    """
    Score case priority for expert review.
    
    Factors:
    1. High uncertainty (> threshold)
    2. Low grounding score (< 0.6)
    3. Inconsistent multi-image (< 0.7)
    4. Rare diagnosis (BPDCN, NK-ALL, etc.)
    5. High clinical impact (blast% > 30%)
    
    Returns:
        (priority_score, reasons)
    """
```

**Priority Reasons**:
- `HIGH_UNCERTAINTY`: Combined uncertainty > threshold
- `LOW_GROUNDING`: Critic grounding score < 0.6
- `INCONSISTENT_MULTI_IMAGE`: Cross-image consistency < 0.7
- `RARE_DIAGNOSIS`: Uncommon leukemia subtype
- `HIGH_CLINICAL_IMPACT`: High blast fraction (>30%)
- `DIVERSITY_SAMPLING`: Selected for feature space coverage

#### 3. Review Queue Management

```python
def add_to_queue(
    self,
    case_id: str,
    predicted_class: str,
    class_probabilities: np.ndarray,
    grounding_score: Optional[float] = None,
    consistency_score: Optional[float] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Evaluate case and add to queue if prioritized.
    
    Queue properties:
    - Sorted by priority (descending)
    - Size-limited (default: 100)
    - Persisted to disk
    
    Returns:
        True if added, False if not prioritized
    """
```

#### 4. Feedback Collection

```python
def submit_review(
    self,
    case_id: str,
    expert_label: str,
    expert_feedback: Optional[str] = None
) -> bool:
    """
    Record expert review for a case.
    
    Tracks:
    - Expert ground truth label
    - Free-text feedback
    - Whether prediction was correct
    - Review timestamp
    """

def get_corrections(self) -> List[Dict]:
    """
    Get cases where expert label ≠ model prediction.
    
    Returns:
        List of corrections for retraining
    """
```

#### 5. Retraining Logic

```python
def get_retraining_candidates(
    self,
    min_corrections: int = 10
) -> Tuple[bool, int, List[Dict]]:
    """
    Check if retraining is recommended.
    
    Args:
        min_corrections: Minimum corrections before retraining
    
    Returns:
        (should_retrain, num_corrections, corrections_list)
    """
```

---

## Uncertainty Strategies

### Strategy Comparison

| Strategy | Formula | Best For | Drawbacks |
|----------|---------|----------|-----------|
| **Least Confident** | 1 - max(P) | Simple, fast | Ignores rest of distribution |
| **Margin** | P₁ - P₂ | Sensitive to ambiguity | Only uses top-2 |
| **Entropy** | -Σ P log(P) | Considers full distribution | Can be dominated by many low-prob classes |
| **Combined** | Weighted average | Balanced, clinical factors | Requires tuning weights |

### Clinical Extensions

**Beyond standard uncertainty**:

1. **Grounding Score**: How well is the diagnosis supported by RAG evidence?
   - Low grounding → might be hallucination
   - Prioritize for expert verification

2. **Multi-Image Consistency**: Do predictions agree across multiple fields of view?
   - Low consistency → unstable diagnosis
   - Could indicate challenging morphology

3. **Blast Fraction**: How severe is the leukemia?
   - High blast % → high clinical impact
   - Prioritize even if model is confident

4. **Rare Diagnosis**: Is this an uncommon entity?
   - BPDCN, NK-ALL, ETP-ALL, etc.
   - Higher review priority due to rarity

---

## Priority Scoring

### Scoring Algorithm

```python
priority_score = 0.0

# 1. High uncertainty
if uncertainty > threshold:
    priority_score += uncertainty

# 2. Low grounding
if grounding < 0.6:
    priority_score += (1 - grounding)

# 3. Inconsistent multi-image
if consistency < 0.7:
    priority_score += (1 - consistency)

# 4. Rare diagnosis
if diagnosis in rare_set:
    priority_score += 0.5

# 5. High blast fraction
if blast_fraction > 0.3:
    priority_score += 0.3
```

### Example Scenarios

**Scenario 1: Confident AML**
- Prediction: AML (P=0.92)
- Grounding: 0.85
- Consistency: 0.91
- Blast %: 28%
- **Result**: Not prioritized (all metrics good)

**Scenario 2: Uncertain ALL vs AML**
- Prediction: ALL (P=0.48), AML (P=0.43)
- Grounding: 0.75
- Blast %: 42%
- **Result**: HIGH_PRIORITY (uncertain + high blast)

**Scenario 3: Confident but Poorly Grounded**
- Prediction: BPDCN (P=0.88)
- Grounding: 0.42 (low!)
- **Result**: HIGH_PRIORITY (rare + low grounding)

**Scenario 4: Inconsistent Multi-Image**
- Image 1: AML (P=0.85)
- Image 2: MDS (P=0.78)
- Image 3: AML (P=0.52)
- Consistency: 0.58
- **Result**: HIGH_PRIORITY (inconsistent cross-image)

---

## Usage Examples

### Example 1: Basic Usage

```python
from pipeline.active_learning import ActiveLearningPipeline, ActiveLearningConfig
import numpy as np

# Initialize pipeline
config = ActiveLearningConfig(
    uncertainty_threshold=0.5,
    grounding_threshold=0.6,
    consistency_threshold=0.7,
    queue_size=100
)

pipeline = ActiveLearningPipeline(
    config=config,
    queue_path=Path("data/active_learning/review_queue.json"),
    feedback_path=Path("data/active_learning/feedback.json")
)

# Evaluate a case
case_id = "CASE-12345"
predicted_class = "AML"
class_probs = np.array([0.45, 0.35, 0.12, 0.08])  # AML, ALL, MDS, Other

added = pipeline.add_to_queue(
    case_id=case_id,
    predicted_class=predicted_class,
    class_probabilities=class_probs,
    grounding_score=0.52,  # Low grounding
    consistency_score=0.68,  # Moderate consistency
    metadata={
        "blast_fraction": 0.38,
        "patient_age": 6,
        "diagnosis": "Suspected AML"
    }
)

if added:
    print(f"Case {case_id} added to review queue (uncertain)")
else:
    print(f"Case {case_id} not prioritized (confident)")
```

### Example 2: Expert Review Workflow

```python
# Get top priority cases for pathologist review
review_queue = pipeline.get_review_queue(top_k=20, unreviewed_only=True)

# Display review queue
for i, item in enumerate(review_queue, 1):
    print(f"\n[{i}] {item.case_id}")
    print(f"  Prediction: {item.uncertainty.predicted_class} "
          f"(confidence={item.uncertainty.confidence:.2f})")
    print(f"  Priority: {item.priority_score:.3f}")
    print(f"  Reasons: {[r.value for r in item.priority_reasons]}")
    
    # Show uncertainty breakdown
    print(f"  Uncertainty:")
    print(f"    - Entropy: {item.uncertainty.entropy:.3f}")
    print(f"    - Margin: {item.uncertainty.margin:.3f}")
    print(f"    - Grounding: {item.uncertainty.grounding_score}")
    print(f"    - Consistency: {item.uncertainty.consistency_score}")

# Pathologist reviews case
expert_label = input("Expert diagnosis: ")
expert_feedback = input("Comments (optional): ")

# Submit review
pipeline.submit_review(
    case_id=review_queue[0].case_id,
    expert_label=expert_label,
    expert_feedback=expert_feedback if expert_feedback else None
)

print("Review submitted!")
```

### Example 3: Retraining Workflow

```python
# Check if retraining is recommended
should_retrain, num_corrections, corrections = pipeline.get_retraining_candidates(
    min_corrections=10
)

if should_retrain:
    print(f"Retraining recommended: {num_corrections} corrections collected")
    
    # Get correction data
    for correction in corrections[:5]:  # Show first 5
        print(f"\n{correction['case_id']}:")
        print(f"  Model predicted: {correction['predicted']}")
        print(f"  Expert corrected to: {correction['expert_label']}")
        print(f"  Feedback: {correction.get('feedback', 'N/A')}")
    
    # Trigger retraining pipeline (not implemented here)
    # retrain_model(corrections)
else:
    print(f"Not enough corrections yet ({num_corrections}/{10})")
```

### Example 4: Statistics and Monitoring

```python
# Get pipeline statistics
stats = pipeline.get_statistics()

print("Active Learning Pipeline Statistics")
print("=" * 50)
print(f"\nReview Queue:")
print(f"  Total cases: {stats['queue']['total']}")
print(f"  Reviewed: {stats['queue']['reviewed']}")
print(f"  Unreviewed: {stats['queue']['unreviewed']}")
print(f"  Review rate: {stats['queue']['review_rate']:.1%}")

print(f"\nFeedback:")
print(f"  Total reviews: {stats['feedback']['total']}")
print(f"  Corrections: {stats['feedback']['corrections']}")
print(f"  Correction rate: {stats['feedback']['correction_rate']:.1%}")

print(f"\nUncertainty Distribution:")
print(f"  Mean: {stats['uncertainty']['mean']:.3f}")
print(f"  Std: {stats['uncertainty']['std']:.3f}")
print(f"  Min: {stats['uncertainty']['min']:.3f}")
print(f"  Max: {stats['uncertainty']['max']:.3f}")

print(f"\nPriority Reasons:")
for reason, count in stats['priority_reasons'].items():
    print(f"  {reason}: {count} cases")
```

### Example 5: Generate Report

```python
from pipeline.active_learning import format_review_queue_report

# Generate markdown report
report = format_review_queue_report(pipeline, top_k=20)

# Save to file
with open("reports/active_learning_queue.md", "w") as f:
    f.write(report)

print("Report saved to reports/active_learning_queue.md")
```

**Report Output**:
```markdown
# Active Learning Review Queue

**Generated**: 2026-11-09 23:30:00

## Statistics

- **Total queue size**: 47
- **Reviewed**: 12
- **Unreviewed**: 35
- **Review rate**: 25.5%
- **Total feedback**: 12
- **Corrections**: 5
- **Correction rate**: 41.7%

## Top 20 Priority Cases for Review

| Rank | Case ID | Prediction | Priority | Uncertainty | Reasons |
|------|---------|------------|----------|-------------|---------|
| 1 | CASE-1234 | BPDCN | 1.250 | 0.625 | low_grounding, rare_diagnosis |
| 2 | CASE-5678 | ALL | 1.180 | 0.580 | high_uncertainty, inconsistent_multi_image |
...
```

---

## Clinical Workflow Integration

### Deployment Architecture

```
┌─────────────────┐
│   Streamlit UI   │ ← User interface
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  Pipeline       │ ← analyze_new_case()
│  (Vision + LLM) │
└────────┬─────────┘
         │
         ↓
┌──────────────────────┐
│ Active Learning      │ ← Uncertainty quantification
│ Pipeline             │
└────────┬─────────────┘
         │
         ↓
┌──────────────────────┐
│ Review Queue         │ ← Priority-sorted cases
│ (JSON persistence)   │
└──────────────────────┘
         │
         ↓
┌──────────────────────┐
│ Expert Review Panel  │ ← Pathologist interface
│ (Streamlit tab)      │
└──────────────────────┘
         │
         ↓
┌──────────────────────┐
│ Feedback Collection  │ ← Corrections logged
└──────────────────────┘
         │
         ↓
┌──────────────────────┐
│ Retraining Pipeline  │ ← Model update (periodic)
└──────────────────────┘
```

### Integration Points

**1. After Case Analysis**:
```python
# In pipeline/new_patient_engine.py

def analyze_new_case(...):
    # Existing analysis
    result = run_multimodal_pipeline(...)
    
    # Add to active learning queue
    from pipeline.active_learning import ActiveLearningPipeline
    
    al_pipeline = ActiveLearningPipeline()
    al_pipeline.add_to_queue(
        case_id=case_id,
        predicted_class=result['diagnosis'],
        class_probabilities=result['class_probabilities'],
        grounding_score=result['critic_grounding_score'],
        consistency_score=result.get('consistency_score'),
        metadata={
            'blast_fraction': result['blast_fraction'],
            'diagnosis': result['diagnosis']
        }
    )
    
    return result
```

**2. Expert Review UI** (Streamlit tab):
```python
# In app/streamlit_app.py

def expert_review_panel():
    st.header("Expert Review Queue")
    
    # Load pipeline
    pipeline = ActiveLearningPipeline()
    
    # Get queue
    queue = pipeline.get_review_queue(top_k=50, unreviewed_only=True)
    
    if len(queue) == 0:
        st.success("No cases pending review!")
        return
    
    # Select case
    case_ids = [item.case_id for item in queue]
    selected = st.selectbox("Case to review:", case_ids)
    
    # Find selected item
    item = next(i for i in queue if i.case_id == selected)
    
    # Display case info
    st.subheader(f"Case: {item.case_id}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Prediction", item.uncertainty.predicted_class)
        st.metric("Confidence", f"{item.uncertainty.confidence:.1%}")
        st.metric("Priority Score", f"{item.priority_score:.3f}")
    
    with col2:
        st.metric("Grounding", f"{item.uncertainty.grounding_score:.2f}")
        st.metric("Consistency", f"{item.uncertainty.consistency_score:.2f}")
        reasons_str = ", ".join([r.value for r in item.priority_reasons])
        st.write(f"**Reasons**: {reasons_str}")
    
    # Load case images and report
    # ... (display logic)
    
    # Expert input
    expert_label = st.selectbox(
        "Expert Diagnosis:",
        ["AML", "ALL", "MDS", "Other"]
    )
    
    expert_feedback = st.text_area("Comments (optional):")
    
    if st.button("Submit Review"):
        pipeline.submit_review(
            case_id=selected,
            expert_label=expert_label,
            expert_feedback=expert_feedback if expert_feedback else None
        )
        st.success(f"Review submitted for {selected}!")
        st.rerun()
```

**3. Retraining Trigger**:
```python
# Periodic job (e.g., weekly cron)

def check_and_retrain():
    pipeline = ActiveLearningPipeline()
    
    should_retrain, num, corrections = pipeline.get_retraining_candidates(
        min_corrections=20
    )
    
    if should_retrain:
        logger.info(f"Retraining triggered: {num} corrections")
        
        # Export corrections to training format
        export_corrections_for_training(corrections)
        
        # Retrain model
        subprocess.run([
            "python", "vision/train.py",
            "--mode", "finetune",
            "--correction-data", "data/corrections.csv"
        ])
        
        logger.info("Retraining complete!")
    else:
        logger.info(f"Retraining not needed ({num}/20 corrections)")
```

---

## Future Work

### Short-term (Next Sprint)

1. **UI Integration**
   - Add "Expert Review" tab to Streamlit app
   - Visualize priority reasons
   - Display uncertainty breakdown

2. **Retraining Pipeline**
   - Implement automated retraining script
   - Version control for model checkpoints
   - A/B testing framework

3. **Metrics Dashboard**
   - Track correction rate over time
   - Monitor queue statistics
   - Alert on high correction rates

### Medium-term (Next Month)

4. **Advanced Uncertainty**
   - Bayesian dropout (MC Dropout)
   - Ensemble disagreement (train 3-5 models)
   - Temperature scaling integration

5. **Diversity Sampling**
   - K-means clustering in embedding space
   - Core-set selection
   - Ensure geographic/demographic coverage

6. **Batch Active Learning**
   - Select batches of K cases (not just top-1)
   - Optimize for diversity + uncertainty

### Long-term (Future Research)

7. **Adaptive Thresholds**
   - Learn optimal uncertainty/grounding thresholds
   - Adjust based on expert review patterns

8. **Cost-Sensitive Learning**
   - Weight corrections by clinical impact
   - Prioritize high-stakes diagnoses

9. **Federated Active Learning**
   - Aggregate review queues across hospitals
   - Privacy-preserving corrections

---

## References

1. **Settles, B. (2009)**: "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin–Madison.

2. **Budd, S., et al. (2021)**: "A survey on active learning and human-in-the-loop deep learning for medical image analysis." Medical Image Analysis, 71, 102062.

3. **Smailagic, A., et al. (2018)**: "Medal: Accurate and robust deep active learning for medical image analysis." ICMLA 2018.

4. **Gal, Y., & Ghahramani, Z. (2016)**: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML 2016.

5. **Sener, O., & Savarese, S. (2018)**: "Active Learning for Convolutional Neural Networks: A Core-Set Approach." ICLR 2018.

---

**Last Updated**: November 9, 2026  
**Status**: Production-ready (awaiting UI integration)  
**Next Review**: After clinical pilot deployment
