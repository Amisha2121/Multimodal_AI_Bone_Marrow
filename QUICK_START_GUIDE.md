# 🚀 Quick Start Guide — Bone Marrow AI Web Application

## 🎯 Getting Started

### 1. **Start the Application**

```bash
python app/flask_app.py
```

**Server starts on:**
- Local: http://127.0.0.1:5000
- Network: http://192.168.1.101:5000

You should see:
```
RAG system READY
LLM initialized (gemini/gemini-flash-lite-latest)
HybridRetriever initialized successfully with 63 clinical chunks.
```

### 2. **Open in Browser**

Navigate to: **http://127.0.0.1:5000**

---

## 📱 User Interface Tour

### **Sidebar Navigation**

#### Main Section:
- **🏠 Upload & Analyze** — Upload bone marrow images for analysis
- **📊 Results & XAI** — View detection, classification, and attention maps
- **💬 Ask Questions** — Chat with RAG-powered AI assistant

#### Advanced Section:
- **📈 System Metrics** — Real-time monitoring of all improvements
- **🎓 Active Learning** — Expert review queue for uncertain predictions

#### Header Badges:
- ✅ **Calibrated** — Temperature scaling active
- ✅ **RAG (63)** — 63 clinical guideline chunks
- ✅ **Multi-Image** — Aggregation with uncertainty
- ✅ **XAI** — Quantitative explainability metrics

---

## 🔬 Workflow: Analyzing a Sample

### **Step 1: Upload Images**

1. Click **"Upload & Analyze"** in sidebar
2. Drag & drop bone marrow images OR click "Choose Files"
3. Supported formats: JPG, PNG, TIFF
4. Wait for detection and classification (~5 seconds per image)

**What happens:**
- Adaptive thresholding detects nucleated cells
- Watershed segmentation isolates individual cells
- ViT-Small/16 classifies each cell
- GradCAM generates attention maps
- Results stored with unique analysis ID

### **Step 2: Review Results**

1. Click **"Results & XAI"** in sidebar
2. Top metrics cards show:
   - Images Processed
   - Cells Detected
   - Cells Classified
   - **Blast Fraction** (key clinical metric)

3. **Detection Visualization** — 3 views:
   - **Raw Candidates** — All detected objects
   - **Filtered Detection** — After quality filtering
   - **Contact Sheet** — Grid of all cells

4. **Explainability (XAI)** section:
   - Attention rollout overlays
   - Shows where model focused
   - Warmer colors = higher attention
   - Cell ID, class, confidence for each

5. **Morphology Distribution** (pie chart):
   - Breakdown by cell type
   - Count and percentage
   - Color-coded categories

### **Step 3: Ask Questions**

1. Click **"Ask Questions"** in sidebar
2. Type clinical questions like:
   - "What does this blast percentage suggest?"
   - "Are there concerning morphological features?"
   - "What additional tests should I order?"
   - "Is this consistent with AML?"

**How it works:**
- Hybrid retrieval (FAISS + BM25) searches 63 guideline chunks
- LLM synthesizes answer with case context
- Citations from WHO 2022 / ICC 2022 provided
- Chat history persisted per analysis

### **Step 4: Export Results**

1. On **Results** tab, click **"Export Analysis (JSON)"**
2. Downloads complete analysis as JSON:
   - All metrics and cell predictions
   - Chat history
   - Clinical report sections
   - Timestamps and metadata

---

## 📈 Advanced Features

### **System Metrics Dashboard**

Click **"System Metrics"** to monitor:

#### Knowledge Base
- **Total Chunks:** 63 (expanded from 12)
- **Retrieval MRR:** 0.050 (baseline)
- **Index Type:** Hybrid (FAISS + BM25)

#### Calibration
- **Status:** Active/Inactive
- **Pre-Cal ECE:** Expected Calibration Error
- **Method:** Temperature Scaling (LBFGS)

#### XAI Metrics
- **Available:** Pointing Game, Deletion AUC, Insertion AUC
- **Visualization:** GradCAM attention maps
- **Purpose:** Quantitative evaluation of attention quality

#### Self-Consistency
- **Status:** Active
- **Samples:** 3 per query
- **Temperature:** 0.3
- **Method:** Text overlap similarity

#### Active Learning
- **Queue Size:** Samples awaiting expert review
- **Threshold:** 0.45 (uncertainty trigger)
- **Methods:** Entropy, confidence, grounding, consistency

#### Analysis Statistics
- **Total Cases:** All analyses to date
- **Total Cells:** All classified cells
- **Avg Confidence:** Model certainty across predictions

#### RAG Status
- **Online/Offline:** System availability
- **LLM Provider:** gemini
- **Model:** gemini-flash-lite-latest

**Refresh Rate:** Auto-updates every 5 seconds

---

### **Active Learning — Expert Review**

Click **"Active Learning"** to review uncertain predictions:

#### Review Queue Table
- **Sample ID** — Unique identifier (truncated)
- **Predicted Class** — Model's classification
- **Uncertainty** — Combined score (0-100%)
- **Priority** — Urgency for review (red > yellow > gray)
- **Actions** — "Review" button

#### How Uncertainty is Calculated:
```
uncertainty = 0.3×entropy + 0.3×(1-confidence) + 0.2×(1-grounding) + 0.2×(1-consistency)
```

#### Expert Review Form:
1. Click **"Review"** on a sample
2. View uncertainty breakdown metrics
3. Enter **Expert Label** (correct cell type)
4. Set **Expert Confidence** slider (0-100%)
5. Add optional **Notes**
6. Click **"Submit Feedback"**

**Feedback is used to:**
- Build training dataset for model retraining
- Validate model performance
- Identify systematic errors
- Prioritize future improvements

---

## 🔍 Interpreting Results

### **Blast Fraction (Key Clinical Metric)**

| Blast % | Clinical Significance |
|---------|----------------------|
| < 5% | Normal or non-leukemic |
| 5-19% | Possible MDS, close monitoring needed |
| ≥ 20% | **WHO threshold for AML diagnosis** |
| ≥ 30% | Strongly suggestive of acute leukemia |

**Important:** Automated counts supplement, not replace, manual differential by trained hematopathologist.

### **Morphology Categories**

Common cell types detected:
- **Blast cells** — Immature precursors
- **Myeloblasts** — Early myeloid lineage
- **Promyelocytes** — Maturing myeloid cells
- **Myelocytes** — Mid-stage myeloid
- **Metamyelocytes** — Late-stage myeloid
- **Neutrophils** — Mature granulocytes (bands, segs)
- **Eosinophils** — Mature eosinophilic granulocytes
- **Basophils** — Mature basophilic granulocytes
- **Lymphocytes** — T/B/NK cells
- **Monocytes** — Mature monocytic lineage
- **Erythroid precursors** — Red cell lineage

### **XAI Attention Maps**

**What to look for:**
- **Nucleus focus** — Attention should center on nucleus
- **Cytoplasm awareness** — Some attention to cytoplasm edges
- **Granule detection** — For neutrophils, eosinophils
- **Chromatin pattern** — Nuclear texture matters for blast vs mature

**Red flags:**
- Attention on background (poor grounding)
- Attention on image artifacts
- Uniform attention (model uncertainty)

---

## 🛠️ Troubleshooting

### **Problem: "RAG system unavailable"**

**Solution:**
1. Check console for LLM initialization errors
2. Verify `.env` has correct API keys:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
3. Test health endpoint: `http://127.0.0.1:5000/api/health`
4. If using Gemini, check API quota limits

### **Problem: Images not displaying**

**Solution:**
1. Check `data/outputs/` directory exists
2. Verify image paths in analysis object
3. Check browser console for 404 errors
4. Ensure Flask `debug=True` for file serving

### **Problem: Upload fails**

**Solution:**
1. Check file size < 50MB (configurable in `flask_app.py`)
2. Ensure image format is supported (JPG, PNG, TIFF)
3. Check disk space in `data/uploads/`
4. Verify detector model weights exist

### **Problem: Slow report generation**

**Solution:**
- First generation: ~10-12 seconds (normal)
- Subsequent: Instant (cached)
- Force regeneration: Add `?regenerate=1` to API call
- Check LLM rate limits if timing out

### **Problem: Active Learning queue empty**

**Solution:**
- This is good! Means all predictions are confident
- Lower threshold in `pipeline/active_learning.py`:
  ```python
  uncertainty_threshold: float = 0.3  # was 0.45
  ```
- Or analyze more diverse/challenging samples

---

## 🧪 Testing the System

### **Test 1: Simple Upload**
1. Use a single bone marrow image
2. Should detect 10-50 cells
3. Classification completes < 10 seconds
4. Results page loads immediately

### **Test 2: Chat Query**
1. Ask: "What is the WHO threshold for AML?"
2. Should return: "20% blasts"
3. Response time: 2-3 seconds
4. Check sources include guideline citations

### **Test 3: System Metrics**
1. Navigate to Metrics tab
2. All cards should populate
3. RAG status should show "Online"
4. Values update every 5 seconds

### **Test 4: Export**
1. Click "Export Analysis (JSON)"
2. JSON file should download
3. Open in text editor
4. Verify contains: cells, morphology, chat_history, report

### **Test 5: Active Learning**
1. Navigate to Active Learning tab
2. If queue empty, lower threshold
3. Review a sample
4. Submit feedback
5. Check feedback count increases

---

## 📊 Performance Benchmarks

**Expected Performance:**
- **Image Upload:** 500ms - 1s (network transfer)
- **Cell Detection:** 2-4s per image (CPU-bound)
- **Cell Classification:** 1-2s per image (ViT inference)
- **XAI Generation:** 0.5-1s per cell (GradCAM)
- **RAG Retrieval:** 100-300ms (FAISS + BM25)
- **LLM Response:** 2-3s (Gemini API latency)
- **Report Generation:** 8-12s (5-7 LLM calls)

**Optimization Opportunities:**
- GPU acceleration for ViT inference
- Batch processing for multiple images
- Cache frequently retrieved chunks
- Async LLM calls for report sections
- Redis for session storage

---

## 🎓 Educational Use

### **For Students:**
1. Upload sample images (ask instructor)
2. Review morphology distribution
3. Compare blast % with manual count
4. Ask AI: "Describe the morphology of a myeloblast"
5. Explore XAI attention maps to understand model

### **For Instructors:**
1. Upload teaching cases with known diagnoses
2. Use Active Learning to validate model
3. Export results as teaching materials
4. Compare model predictions with expert consensus
5. Discuss sources of error and model limitations

### **For Researchers:**
1. System Metrics tab shows all improvements
2. Export JSON for further analysis
3. Active Learning provides labeled dataset
4. Retrieval evaluation metrics in `/data/retrieval_test_queries.json`
5. All code is documented in `/docs/`

---

## 🚀 Next Steps

### **Production Deployment:**
1. Disable Flask debug mode
2. Use production WSGI server (Gunicorn, uWSGI)
3. Add authentication (JWT tokens)
4. Configure HTTPS with SSL certificate
5. Set up PostgreSQL/MySQL database
6. Deploy retrieval index to Redis
7. Configure CORS for specific domains
8. Set up monitoring (Prometheus, Grafana)
9. Implement rate limiting
10. Add audit logging

### **Feature Expansion:**
1. PDF export with clinical report formatting
2. Multi-user support with role-based access
3. Case management (folders, tags, search)
4. Integration with hospital EMR (HL7/FHIR)
5. Mobile app for tablet-based review
6. Real-time collaboration (WebSocket)
7. Advanced visualizations (Plotly, D3.js)
8. Model versioning and A/B testing

### **Clinical Validation:**
1. Prospective study with expert pathologists
2. Inter-rater reliability analysis
3. Comparison with manual differential
4. Sensitivity/specificity for key diagnoses
5. FDA regulatory pathway (if applicable)
6. CAP/CLIA laboratory certification

---

## 📚 Additional Resources

- **Full Documentation:** `/docs/` directory
- **Improvement Details:** `IMPROVEMENTS_CHECKLIST.md`
- **Wow Factor Summary:** `WOW_FACTOR_SUMMARY.md`
- **Flask Enhancements:** `FLASK_ENHANCEMENTS.md`
- **Literature Survey:** `LITERATURE_SURVEY_CONTEXT.md`
- **Architecture:** `README.md`

---

## 🆘 Support

**Issues?**
1. Check Flask console for errors
2. Review browser console (F12)
3. Test API endpoints directly
4. Check `.env` configuration
5. Verify model weights exist
6. Ensure dependencies installed

**Contact:**
- Check project README for contact information
- Submit GitHub issue with:
  - Error message
  - Steps to reproduce
  - Flask console output
  - Browser console output

---

## ✅ System Status Checklist

Before starting, verify:

- ✅ Flask server running (port 5000)
- ✅ RAG system initialized (63 chunks)
- ✅ LLM connected (Gemini)
- ✅ Detector model loaded
- ✅ Classifier model loaded
- ✅ FAISS index loaded
- ✅ BM25 index loaded
- ✅ Health endpoint returns `"status": "ok"`
- ✅ Browser can access http://127.0.0.1:5000

**Console should show:**
```
HybridRetriever initialized successfully with 63 clinical chunks.
Retriever OK
LLM initialized (gemini/gemini-flash-lite-latest)
LLM smoke-test OK  -> 'OK'
RAG system READY
```

**Now you're ready to analyze bone marrow samples!** 🎉
