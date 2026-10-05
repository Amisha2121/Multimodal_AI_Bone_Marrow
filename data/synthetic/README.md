# ⚠️ NOTICE: SYNTHETIC DATA IS STRICTLY ISOLATED FOR SOFTWARE TESTING ONLY

The files in this directory (`data/synthetic/`) are generated programmatically for:
- CI / Automated Unit Testing
- Interface verification and pipeline dry-runs

### Strict Research Guidelines:
1. **NEVER use data in this directory for training, validating, or testing machine learning models.**
2. **NEVER report performance metrics, confusion matrices, or RAG accuracy based on these synthetic cases.**
3. **NEVER present these cases as real patient clinical records.**
4. All real experimental evaluations MUST use validated, publicly available biomedical datasets documented in [`docs/DATA_PROVENANCE.md`](../../docs/DATA_PROVENANCE.md).
