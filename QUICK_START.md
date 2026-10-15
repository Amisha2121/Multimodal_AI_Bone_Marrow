# 🚀 Quick Start Guide

## Installation

```bash
# Install required package
pip install google-generativeai
```

## Test LLM Connection

```bash
python agents/llm_provider.py
```

Expected output:
```
Provider: gemini
Model: gemini-1.5-flash
✓ LLM PROVIDER TEST SUCCESSFUL
```

## Start Application

```bash
python app/app.py
```

Then open: http://localhost:5000

## What You'll Get

### Enhanced Reports with:
- ✅ 8-10 sentence comprehensive sections
- ✅ Unique content for each case
- ✅ Doctor-friendly language
- ✅ WHO 2022 criteria
- ✅ Specific test recommendations
- ✅ Clinical urgency flags

### Powered by:
- 🤖 Google Gemini 1.5 Flash
- 📊 Large 1M token context
- ⚡ Fast generation (~15-30 seconds)

## Quick Configuration

### Switch LLM Provider
Edit `configs/default.yaml`:

```yaml
llm:
  provider: gemini          # or "groq" or "openai"
  model: gemini-1.5-flash   # or "gemini-1.5-pro"
```

### Adjust Report Detail
Edit `app/app.py` temperature values:
- `0.1-0.2` - Consistent, factual
- `0.3-0.5` - More varied

## Troubleshooting

### Authentication Error?
1. Get new key: https://makersuite.google.com/app/apikey
2. Update `.env`: `GOOGLE_API_KEY=your_key_here`
3. Restart app

### Reports Too Generic?
- Check LLM connection
- Verify Gemini API key is valid
- Look for "LLM generation failed" in console

## Documentation

- `SETUP_COMPLETE.md` - Full overview
- `GEMINI_LLM_SETUP.md` - Gemini details
- `REPORT_IMPROVEMENTS.md` - Report enhancements
- `API_KEY_NOTE.md` - API key help

## Ready to Go! 🎉
