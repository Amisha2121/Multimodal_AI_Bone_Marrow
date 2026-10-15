# Groq Llama RAG Integration Setup Guide

## Overview

The system now supports **true RAG (Retrieval-Augmented Generation)** using **Groq's Llama models** for dynamic, LLM-powered clinical assessments instead of deterministic templates.

### Benefits of Groq Llama

✅ **Ultra-fast inference** (up to 10x faster than standard APIs)  
✅ **Free tier available** (14,400 requests/day on free plan)  
✅ **State-of-the-art Llama 3.3 70B** model  
✅ **No local GPU required** (cloud-based inference)  
✅ **Better medical reasoning** than rule-based templates  

---

## Step 1: Get Your Groq API Key (FREE)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy your key (starts with `gsk_...`)

---

## Step 2: Configure Environment

### Create `.env` file

Copy the example file and add your API key:

```bash
copy .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

**IMPORTANT**: Never commit `.env` to version control! It's already in `.gitignore`.

---

## Step 3: Install Dependencies

Install the Groq library:

```bash
pip install groq
```

Or reinstall all requirements to get the latest:

```bash
pip install -r requirements.txt
```

---

## Step 4: Verify Configuration

Check that your setup is working:

```bash
python agents/llm_provider.py
```

Expected output:
```
======================================================================
LLM PROVIDER TEST
======================================================================

Provider: groq
Model: llama-3.3-70b-versatile

Generating response...

Response:
[LLM-generated answer about WHO blast threshold]

======================================================================
✓ LLM PROVIDER TEST SUCCESSFUL
======================================================================
```

---

## Step 5: Test RAG Pipeline

Run the full agent graph with RAG:

```bash
python agents/graph.py
```

This will execute a sample patient case through the full pipeline:
1. **Retriever**: Searches WHO guidelines
2. **Synthesizer**: Uses **Groq Llama** to generate clinical assessment
3. **Critic**: Validates evidence grounding

---

## Configuration Options

### Available Models

Edit `configs/default.yaml` to change models:

```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile  # RECOMMENDED (best reasoning)
  # Other options:
  # model: llama-3.1-70b-versatile  (slightly older)
  # model: mixtral-8x7b-32768       (faster, less capable)
  max_tokens: 2048
  temperature: 0.1
```

### Switch Between LLM and Deterministic Mode

**Option 1: Environment Variable** (temporary)
```bash
# Use LLM-powered synthesis (default)
set SYNTHESIZER_USE_LLM=true
python app/streamlit_app.py

# Use deterministic template (legacy)
set SYNTHESIZER_USE_LLM=false
python app/streamlit_app.py
```

**Option 2: Code Change** (permanent)

Edit `agents/synthesizer.py`:
```python
# Line 34
USE_LLM = True   # Set to False for deterministic mode
```

---

## Troubleshooting

### Error: "GROQ_API_KEY not found in environment"

**Solution**: Make sure you created `.env` file with your API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

Then restart your terminal/IDE to reload environment variables.

### Error: "groq library not installed"

**Solution**: 
```bash
pip install groq
```

### Error: "Rate limit exceeded"

**Solution**: Groq free tier has limits:
- 14,400 requests/day
- 30 requests/minute

Wait a minute and try again, or upgrade to Groq Pro.

### Error: "Model not found"

**Solution**: Check available models at [https://console.groq.com/docs/models](https://console.groq.com/docs/models)

Currently supported:
- `llama-3.3-70b-versatile` (RECOMMENDED)
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

---

## Performance Comparison

| Mode | Speed | Quality | Cost |
|------|-------|---------|------|
| **Groq Llama 3.3 70B** | ⚡ Ultra-fast (~1-2s) | 🏆 Excellent medical reasoning | 💰 FREE tier |
| Deterministic Templates | ⚡⚡ Instant | ⚠️ Rigid, limited context | FREE |
| Local Mistral-7B | 🐢 Slow (10-30s) | ⚠️ Moderate quality | FREE (requires GPU) |

**RECOMMENDATION**: Use Groq Llama 3.3 70B for production - best balance of speed, quality, and cost.

---

## Architecture

### Before (Deterministic)
```
Retriever → Rule-Based Templates → Critic
```

### After (LLM-Powered RAG)
```
Retriever → Groq Llama 3.3 70B → Critic
            (synthesizes from      (validates
             retrieved evidence)    citations)
```

---

## Example Output Comparison

### Deterministic Mode (Old)
```
The cytological image distribution falls within Non-malignant / Reactive / 
Other morphology. Detected blast proportions are low or demonstrate ambiguous 
lineage features.

Evaluation for Mixed Phenotype Acute Leukemia (MPAL) or reactive marrow 
expansion may be appropriate...
```

### LLM-Powered Mode (New)
```
### Cytomorphology Assessment — Case CASE-36D44A03

Based on the automated image analysis detecting 154 cells with a 5.2% blast 
fraction and predominant Megakaryocyte/Platelet morphology, this case shows 
features of a reactive or non-malignant marrow process.

The blast proportion falls well below the WHO 2022 threshold of ≥20% for 
acute leukemia diagnosis [WHO2022_CML_CP_007]. The predominance of mature 
megakaryocytic lineage cells suggests reactive thrombopoiesis, though 
correlation with CBC platelet count is essential...

[More detailed, contextual, evidence-based analysis]
```

---

## Cost Estimates

### Groq Free Tier
- **14,400 requests/day** = ~600 patients/day
- **30 requests/minute** = sufficient for clinical use
- **Cost**: $0

### Groq Pro (if needed)
- **No rate limits**
- **Cost**: ~$0.27 per million input tokens, ~$1.10 per million output tokens
- **Typical patient assessment**: ~1,500 input + 800 output tokens = **$0.001** per patient

---

## Next Steps

1. ✅ Set up Groq API key in `.env`
2. ✅ Run verification: `python agents/llm_provider.py`
3. ✅ Test full pipeline: `python agents/graph.py`
4. ✅ Launch UI: `streamlit run app/streamlit_app.py`
5. 🎯 Upload a bone marrow image and see LLM-powered RAG in action!

---

## Support

- **Groq Documentation**: [https://console.groq.com/docs](https://console.groq.com/docs)
- **Groq Discord**: [https://groq.com/discord](https://groq.com/discord)
- **Rate Limits**: [https://console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)

---

**Congratulations! Your system now uses state-of-the-art Groq Llama 3.3 70B for intelligent RAG-powered clinical assessments.** 🚀
