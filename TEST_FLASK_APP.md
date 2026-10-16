# 🧪 Flask App Testing Guide

## Quick Start Testing

### 1. Start the Server

```bash
# Make sure virtual environment is activated
python app/flask_app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

### 2. Open in Browser

Navigate to: **http://localhost:5000**

You should see the beautiful interface with:
- 🔬 Bone Marrow Analysis header
- Four tabs in the sidebar:
  - Upload & Analyze
  - Ask Questions
  - RAG Report
  - Cell Explorer
- History sidebar on the right

---

## 🧪 Feature Testing Checklist

### ✅ Test 1: Image Upload & Analysis

1. **Click on "Upload & Analyze" tab** (should be selected by default)
2. **Click the upload zone** (large dashed border area)
3. **Select one or more bone marrow images** from your test data
4. **Watch for**:
   - [ ] Loading spinner appears
   - [ ] "Analyzing images..." message shows
   - [ ] Metrics cards appear with numbers:
     - Images Processed
     - Cells Detected
     - Cells Classified
     - Blast Percentage

### ✅ Test 2: Three Image View Types

After upload completes:

1. **Look for three tabs above the detection image**:
   - [ ] "Raw Candidates" tab
   - [ ] "Filtered Detection" tab  
   - [ ] "Contact Sheet" tab

2. **Click each tab and verify**:
   - [ ] Raw Candidates: Shows all initial detections (before filtering)
   - [ ] Filtered Detection: Shows refined, high-quality detections (DEFAULT)
   - [ ] Contact Sheet: Shows grid of all individual cells

3. **Check that**:
   - [ ] Images load correctly
   - [ ] Active tab is highlighted in blue
   - [ ] Smooth transition between views

### ✅ Test 3: Cell Distribution Table

Scroll down after analysis:

- [ ] Table showing all cell types
- [ ] Count for each cell type
- [ ] Percentage for each cell type
- [ ] Proper formatting and borders

### ✅ Test 4: Chat Interface

1. **Click "Ask Questions" tab**
2. **Verify initial state**:
   - [ ] Microscope icon and welcome message if no analysis
   - [ ] Or empty chat ready state if analysis exists

3. **Type a question**, for example:
   - "What is the clinical significance of this blast percentage?"
   - "What additional testing would you recommend?"
   - "Explain the cell distribution findings"

4. **Verify chat behavior**:
   - [ ] Your message appears with blue "You" avatar
   - [ ] "Thinking..." message appears with green microscope avatar
   - [ ] AI response appears below your message
   - [ ] Response is relevant to your analysis
   - [ ] Chat scrolls to bottom automatically
   - [ ] Can send multiple messages
   - [ ] History is preserved

5. **Test Enter key**: Press Enter to send (Shift+Enter for new line)

### ✅ Test 5: Wikipedia-Style RAG Report

1. **Click "RAG Report" tab**
2. **Watch for**:
   - [ ] Loading spinner with "Generating comprehensive report..."
   - [ ] Report generation (may take 10-30 seconds)

3. **Verify report sections**:
   - [ ] Header with case ID and timestamp
   - [ ] **Overview section** with clinical summary
   - [ ] **Multiple cell-type sections**, each with:
     - Cell type name with count and percentage
     - Detailed clinical explanation (3-4 sentences)
     - **Up to 4 representative cell images** below the text
   - [ ] **Clinical Interpretation section** with recommendations
   
4. **Check formatting**:
   - [ ] Professional medical document style
   - [ ] Blue left border on each section
   - [ ] Cell images in grid layout
   - [ ] Images load correctly
   - [ ] Readable font sizes and spacing

### ✅ Test 6: Cell Explorer

1. **Click "Cell Explorer" tab**
2. **Verify display**:
   - [ ] Shows total count of cells at top
   - [ ] Filter dropdown with "All" selected by default
   - [ ] Grid of cell cards

3. **Check each cell card shows**:
   - [ ] Cell image (cropped from contact sheet)
   - [ ] **Cell type name** (e.g., "Blast", "Lymphoid Series")
   - [ ] **Cell description** (e.g., "Immature precursor cell")
   - [ ] **Confidence score** as percentage

4. **Test filtering**:
   - [ ] Click dropdown and select different cell types
   - [ ] Grid updates to show only that type
   - [ ] Cell count updates
   - [ ] Select "All" to see all cells again

### ✅ Test 7: Analysis History

Located in the **right sidebar**:

1. **Verify history list shows**:
   - [ ] Thumbnail image of detection
   - [ ] Case ID (truncated)
   - [ ] Cell count and blast percentage
   - [ ] Timestamp

2. **Test switching analyses**:
   - [ ] Upload a second analysis
   - [ ] Click on first analysis in history
   - [ ] All tabs update to show first analysis
   - [ ] Click on second analysis
   - [ ] All tabs update to show second analysis
   - [ ] Active analysis highlighted in light blue

### ✅ Test 8: Navigation Flow

1. **Test complete workflow**:
   - [ ] Upload → View metrics → Check images → Ask question → View report → Explore cells
   - [ ] All features work seamlessly
   - [ ] Can jump to any tab at any time
   - [ ] Data persists across tab switches

2. **Test with multiple analyses**:
   - [ ] Upload analysis #1
   - [ ] Go to chat, ask a question
   - [ ] Upload analysis #2
   - [ ] Switch back to analysis #1 in history
   - [ ] Chat history for #1 is preserved
   - [ ] Switch to analysis #2
   - [ ] Chat history for #2 is separate

---

## 🐛 Common Issues & Solutions

### Issue: Images Don't Load
**Symptoms**: Gray boxes or broken image icons

**Solutions**:
1. Check browser console (F12) for 404 errors
2. Verify `data/outputs/` directory exists
3. Check that analysis pipeline is generating debug images
4. Ensure `debug_filtered_path`, `debug_raw_path`, `debug_contact_sheet_path` are set

### Issue: Chat Returns Errors
**Symptoms**: "Error: Failed to get response"

**Solutions**:
1. Check that RAG system initialized (look for "RAG init failed" in server logs)
2. Verify LLM provider is configured in `.env`
3. Check retriever data is available
4. Test with simpler questions first

### Issue: Report Takes Too Long
**Symptoms**: Loading spinner for >60 seconds

**Solutions**:
1. Check server logs for errors
2. Verify RAG retrieval is working
3. Test with smaller cell counts first
4. Check LLM API rate limits

### Issue: Upload Fails
**Symptoms**: Alert "Upload failed"

**Solutions**:
1. Check file size (<50MB)
2. Verify file format (PNG, JPG, JPEG, TIFF)
3. Look at server logs for Python errors
4. Ensure `analyze_new_case` function works

### Issue: Cell Images Missing in Report
**Symptoms**: Report sections but no images

**Solutions**:
1. Check that `cell.image_path` is set in cell predictions
2. Verify cell crop images are being generated
3. Check image paths are relative to repo root
4. Test cell explorer first (uses same image paths)

---

## 📊 Expected Results

### After Successful Upload:

**Metrics (example values)**:
- Images Processed: 1
- Cells Detected: 150-300 (depends on image)
- Cells Classified: 150-300
- Blast Percentage: 5-30% (varies)

**Cell Distribution Table**:
- Should show 5-7 different cell types
- Counts should sum to total cells
- Percentages should sum to ~100%

**Detection Images**:
- Raw: May show many overlapping boxes
- Filtered: Clean, non-overlapping detections
- Contact: Grid of individual cell crops

### After Chat Question:

**Response should**:
- Be 2-5 sentences
- Reference the specific analysis data
- Cite clinical guidelines when relevant
- Be medically appropriate
- Answer in context

### After Report Generation:

**Report should have**:
- 5-8 sections total
- 1 Overview section
- 3-5 cell-type sections (one per major cell type)
- 1 Clinical Interpretation section
- 10-20 cell images total across all sections
- Professional medical language

### In Cell Explorer:

**Should display**:
- 50-300 individual cells (full dataset)
- 7-8 unique cell type options in dropdown
- Each cell properly labeled with name
- Confidence scores 0.5-1.0 (50-100%)
- Images properly cropped and displayed

---

## 🎯 Success Criteria

✅ **All features working** if you can:
1. Upload multiple images successfully
2. See all three image analysis views
3. Chat and get relevant responses
4. Generate report with cell images
5. Browse and filter cells by type
6. Switch between multiple analyses in history

✅ **UI is responsive** if:
- Loading indicators appear during processing
- Error messages are clear
- Tab switches are instant
- Images load progressively
- No console errors in browser

✅ **Data is accurate** if:
- Cell counts match across views
- Blast percentage is consistent
- Cell type labels are correct
- Images correspond to cell types
- Report data matches metrics

---

## 📸 Screenshots to Take

Document your testing with screenshots of:
1. Initial upload page
2. All three image views (raw, filtered, contact)
3. Cell distribution table
4. Chat conversation (2-3 exchanges)
5. Complete RAG report (full page)
6. Cell explorer with filter applied
7. Analysis history with multiple entries

---

## 🔍 Advanced Testing

### Performance Testing
- Upload 3-5 images at once
- Generate report for 300+ cells
- Create 10+ analyses in history
- Ask 10+ chat questions

### Edge Cases
- Upload very large image (>20MB)
- Upload unsupported format (.bmp, .gif)
- Ask very complex medical question
- Filter to cell type with only 1-2 cells
- Switch analyses rapidly

### Browser Compatibility
- Test in Chrome ✅
- Test in Firefox ✅
- Test in Edge ✅
- Test on mobile (responsive design)

---

**Good luck with testing! Report any issues you find.** 🧪✨
