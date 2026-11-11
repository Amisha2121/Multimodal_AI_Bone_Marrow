import pickle
from pipeline.summary import build_morphology_summary
from pipeline.patient_report_engine import build_patient_specific_report

with open('data/app_storage/analyses.pkl', 'rb') as f:
    analyses = pickle.load(f)

for aid, entry in analyses.items():
    case_id = entry.get("case_id", aid)
    print(f"Testing report engine on {aid} ({case_id})...")
    summary = build_morphology_summary(aid, case_dict=entry)
    rep = build_patient_specific_report(
        case_dict=entry,
        summary=summary,
        retrieved_chunks=[{'chunk_id': 'WHO2022_AML_001', 'text': 'WHO acute threshold >=20% blasts.', 'score': 0.95}],
        multibank_evidence={},
        disease_differential=[]
    )
    print(f"  Generated sections: {len(rep['sections'])}")
    print(f"  Visualizations manifest: {len(rep['visualization_manifest']['visualizations'])}")
    print(f"  Validation audit passed: {rep['validation_audit']['is_valid']}")
    print(f"  Total claims extracted: {rep['validation_audit']['audited_claims_count']}")
    print(f"  Grounding ratio: {rep['validation_audit']['grounding_ratio']}")

