import tempfile
from pathlib import Path
from scripts.document_ai.inference import DocumentIntelligenceEngine

engine = DocumentIntelligenceEngine(threshold=0.70)

with tempfile.TemporaryDirectory() as tmpdir:
    # 1. Blank file
    blank_p = Path(tmpdir) / 'blank.txt'
    blank_p.write_text('')
    res_blank = engine.process_document(str(blank_p))
    print('Blank test:', res_blank['document_prediction']['category'], '| Rejected:', res_blank['document_prediction']['is_rejected'])

    # 2. Unrelated text (Cardiology note)
    unrel_p = Path(tmpdir) / 'cardio.txt'
    unrel_p.write_text('Patient diagnosed with severe aortic stenosis. Left ventricular ejection fraction 45%. S1 and S2 present with systolic crescendo-decrescendo murmur at right upper sternal border. Cardiac catheterization advised.')
    res_unrel = engine.process_document(str(unrel_p))
    conf_unrel = res_unrel['document_prediction']['confidence'] * 100
    print('Unrelated test:', res_unrel['document_prediction']['category'], f'| Confidence: {conf_unrel:.2f}% | Rejected:', res_unrel['document_prediction']['is_rejected'])

    # 3. Valid CBC report
    cbc_p = Path(tmpdir) / 'cbc.txt'
    cbc_p.write_text('LABORATORY REPORT: COMPLETE BLOOD COUNT\nHaemoglobin: 13.5 g/dL (12-16)\nWBC: 6.8 x10^9/L (4-11)\nPlatelets: 240 x10^9/L (150-400)\nNeutrophils: 4.2 x10^9/L\nLymphocytes: 2.0 x10^9/L\nConclusion: Normal blood count.')
    res_cbc = engine.process_document(str(cbc_p))
    conf_cbc = res_cbc['document_prediction']['confidence'] * 100
    print('Valid CBC test:', res_cbc['document_prediction']['category'], f'| Confidence: {conf_cbc:.2f}% | Rejected:', res_cbc['document_prediction']['is_rejected'])

