# Final UI Requirements Summary

## Critical Requirements

### General
- **NO EMOJIS** anywhere in the UI
- **Collapsible sidebar** - Add toggle button to hide/show left sidebar for more screen space
- Remove all icon/emoji usage, use text labels only

### Tab 1: Detection View
**Primary Focus**: Show latest analyzed image with detection overlay

**Layout**:
- Small upload button at top-right corner
- Main area shows **3 detection views in tabs**:
  1. **Raw Candidates** - All initially detected regions before filtering
  2. **Filtered Detections** - Genuine cells after quality filtering (cyan/orange boxes)
  3. **Contact Sheet** - Grid of 50 cell crops with IDs and dimensions
- By default, show the **Filtered Detections** view for latest analysis
- Below image: Quick metrics (cells detected, blast %, classification counts)
- Below metrics: Cell distribution bar chart

**Behavior**:
- On page load, automatically show latest analysis if exists
- Upload button triggers analysis, then immediately shows new results in same tab
- No need to switch tabs after upload

### Tab 2: Chat Interface
**Layout**: Messaging app style

**Structure**:
```
[Chat messages area - scrollable, messages flow upward]
├── Message 1 (oldest)
├── Message 2
└── Message N (newest - at bottom)

[Input area - fixed at bottom]
├── Text input box
└── Send button
```

**Message Style**:
- User questions: Light blue background, aligned left
- Assistant answers: Gray background, aligned left  
- Show timestamp for each message
- Messages stored in persistent history per analysis
- Load history when switching between analyses

**NO** suggested question buttons
**NO** emojis or icons

### Tab 3: Analysis Details  
**Wikipedia-style report with RAG**

**Structure**:
```
[Case Overview Section]
├── Generated paragraph using RAG
└── No images

[Cell Type 1 Section (e.g., "Blast - 5 cells, 10.2%")]
├── Generated paragraph about significance using RAG
└── 4 sample cell images in row

[Cell Type 2 Section]
├── Generated paragraph
└── 4 sample cell images

[Cell Type 3 Section]
└── (repeat)

[Clinical Interpretation Section]
├── Generated paragraph with recommendations using RAG
└── No images
```

**Content Generation**:
- Each section uses RAG to retrieve WHO guidelines
- LLM generates clinical paragraphs (3-5 sentences each)
- Must include:
  - Overview of findings
  - Section per cell type found (top 5 by count)
  - Clinical interpretation with recommendations
- Show actual detected cell images inline with text
- Professional medical documentation style

### Tab 4: Cell Explorer
- Keep current implementation
- Grid of individual cells
- Filter dropdown by cell type
- Show cell name, brief description, confidence %
- Remove emoji from title

### History Panel (Right Side)
- Show all previous analyses
- Display: Case ID, timestamp, cell count, blast %
- Show thumbnail of detection image
- Click to switch analysis context
- Active analysis highlighted
- History persists across all tabs

## Technical Implementation Notes

### Image Serving
- Serve 3 detection images per analysis:
  - `debug_raw_path` - Raw detections
  - `debug_filtered_path` - Filtered detections (primary view)
  - `debug_contact_sheet_path` - Contact sheet
- URL encode paths properly
- Handle Windows file paths correctly

### RAG Integration for Tab 3
```python
# For each section:
1. Build query based on cell type and counts
2. retriever.retrieve(query, top_k=2)
3. Format evidence into context
4. llm_provider.generate() with clinical prompt
5. Return structured sections with paragraphs + cell images
```

### Chat Persistence
- Store messages per analysis_id
- Load on tab switch
- Display in chronological order (oldest at top)
- Scroll to bottom on new message

### Sidebar Toggle
- Add hamburger menu button
- Clicking toggles sidebar visibility
- Main content area expands when sidebar hidden
- State persists during session

## API Endpoints Needed

### Existing (modify):
- `GET /api/analysis/<id>` - Add `detection_images` object with 3 paths
- `POST /api/chat` - Ensure history is stored
- `GET /api/chat/<id>` - Return full chat history

### New:
- `GET /api/analysis-report/<id>` - Generate RAG-based Wikipedia-style report
  Returns: `{sections: [{title, content, cell_images}]}`

## Priority Order
1. Remove all emojis
2. Fix Tab 1 to show detection by default with 3 view tabs
3. Fix Tab 2 chat interface (messages up, input bottom, persist history)
4. Implement Tab 3 RAG-based analysis report
5. Add sidebar toggle functionality
