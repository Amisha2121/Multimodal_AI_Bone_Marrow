# 🚀 Quick Start: Groq Llama RAG

## 3-Step Setup (5 minutes)

### Step 1: Get Free API Key
```
1. Visit: https://console.groq.com
2. Sign up (free)
3. Go to "API Keys"
4. Click "Create API Key"
5. Copy the key (starts with gsk_...)
```

### Step 2: Configure
```bash
# Create .env file
copy .env.example .env

# Edit .env and add:
GROQ_API_KEY=gsk_your_actual_key_here

# Install Groq
pip install groq
```

### Step 3: Verify
```bash
python setup_groq_rag.py
```

✅ You should see: **"✅ ALL CHECKS PASSED - RAG IS READY TO USE!"**

---

## Run Your App

```bash
streamlit run app/streamlit_app.py
```

Your bone marrow analyzer now uses **state-of-the-art Llama 3.3 70B** for intelligent RAG! 🎉

---

## What Changed?

### Before (Deterministic)
```
[Cell Detection] → [Template Rules] → [Report]
     ❌ Rigid templates
     ❌ Limited context understanding
```

### Now (Groq Llama RAG)
```
[Cell Detection] → [WHO Guidelines Retrieval] → [Llama 3.3 70B] → [Report]
     ✅ Dynamic synthesis
     ✅ Evidence-based reasoning
     ✅ Natural language
```

---

## Example Output Difference

### Old (Template):
> "The cytological image distribution falls within Non-malignant / Reactive / Other morphology..."

### New (Groq Llama):
> "Based on automated image analysis detecting 154 cells with 5.2% blast fraction and predominant megakaryocytic morphology, this case demonstrates features consistent with reactive marrow. The blast proportion is well below the WHO 2022 threshold of ≥20% for acute leukemia [WHO2022_CML_CP_007], suggesting..."

**Much more contextual and evidence-based!** ✨

---

## Troubleshooting

### ❌ "GROQ_API_KEY not found"
**Fix**: Make sure `.env` file exists with `GROQ_API_KEY=gsk_...`

### ❌ "groq library not installed"
**Fix**: Run `pip install groq`

### ❌ "Rate limit exceeded"
**Fix**: Wait 1 minute (free tier: 30 requests/minute)

---

## Toggle Modes

### Use LLM (Recommended)
```bash
set SYNTHESIZER_USE_LLM=true
streamlit run app/streamlit_app.py
```

### Use Templates (Old)
```bash
set SYNTHESIZER_USE_LLM=false
streamlit run app/streamlit_app.py
```

---

## Free Tier Limits

- ✅ **14,400 requests/day** (600 patients/day)
- ✅ **30 requests/minute**
- ✅ **Llama 3.3 70B** (state-of-the-art)
- ✅ **$0 cost**

More than enough for research and clinical use!

---

## Need Help?

- 📖 **Full Setup Guide**: `GROQ_LLAMA_RAG_SETUP.md`
- 🔧 **Technical Details**: `GROQ_RAG_CHANGES.md`
- 🌐 **Groq Docs**: https://console.groq.com/docs

---

**That's it! You're now using cutting-edge AI for bone marrow analysis.** 🚀
