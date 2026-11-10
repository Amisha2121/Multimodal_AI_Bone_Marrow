import sys
sys.path.insert(0, '.')
from pathlib import Path
from pipeline.new_patient_engine import analyze_new_case

IMG = Path(r"data\new_patient_cache\CASE-0765B93E\originals\IMAGE-01_debug_filtered.png")
print("Image:", IMG.name, "exists:", IMG.exists())
result = analyze_new_case(uploaded_files=[IMG], age_years=None, sex=None, sample_site=None)
print("status:", result.status)
print("cells:", result.total_cells_detected)
print("msg:", result.status_message[:120] if result.status_message else "")
