# Google Gemini LLM Integration

## Overview
The system has been successfully configured to use Google Gemini as the LLM provider instead of Groq. Gemini offers powerful language models with large context windows and excellent performance.

## Changes Made

### 1. **Environment Configuration (.env)**
Added Google Gemini API key:
```env
GOOGLE_API_KEY=AIzaSyAb8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA64d4GRh7pw
```

### 2. **LLM Provider (agents/llm_provider.py)**
Added complete Gemini support:
- `_initialize_gemini()` - Initializes Google Generative AI client
- `_generate_gemini()` - Handles text generation with Gemini API
- Message format conversion (system/user/assistant → Gemini format)
- Error handling and logging

**Key Features:**
- Automatic message format conversion
- System instructions are prepended to prompts
- Support for multi-turn conversations
- Configurable temperature and max_tokens
- Comprehensive error handling

### 3. **Configuration (configs/default.yaml)**
Updated LLM configuration:
```yaml
llm:
  provider: gemini                    # Switched from "groq" to "gemini"
  model: gemini-1.5-flash             # Fast & efficient model
  max_tokens: 8192                    # Increased from 2048 (Gemini supports large outputs)
  temperature: 0.1                    # Consistent, deterministic responses
```

### 4. **Example Configuration (.env.example)**
Updated template with Gemini API key placeholder for future reference.

## Available Gemini Models

You can switch between these models in `configs/default.yaml`:

### **gemini-1.5-flash** (Currently Selected)
- **Best for:** Fast responses, high throughput
- **Context window:** 1M tokens
- **Speed:** Very fast
- **Cost:** Low
- **Use case:** Production deployments, real-time applications

### **gemini-1.5-pro**
- **Best for:** Complex reasoning, highest quality
- **Context window:** 2M tokens
- **Speed:** Moderate
- **Cost:** Higher
- **Use case:** Complex medical analysis, detailed reports

### **gemini-1.5-pro-002**
- **Best for:** Latest improvements
- **Context window:** 2M tokens
- **Features:** Enhanced reasoning, better instruction following

## How It Works

### Message Flow
```
User Request
    ↓
Flask App (app.py)
    ↓
LLM Provider (get_llm_provider)
    ↓
Gemini Client (google.generativeai)
    ↓
Generate Content
    ↓
Return Response
```

### Message Format Conversion
The system automatically converts between OpenAI-style messages and Gemini format:

**Input (OpenAI format):**
```python
[
    {"role": "system", "content": "You are a hematopathologist..."},
    {"role": "user", "content": "Analyze this bone marrow sample..."}
]
```

**Converted to Gemini:**
```
System instruction: You are a hematopathologist...

User prompt: Analyze this bone marrow sample...
```

## Installation Requirements

Install the Google Generative AI library:
```bash
pip install google-generativeai
```

Or add to your requirements.txt:
```
google-generativeai>=0.3.0
```

## Testing the Setup

Run the built-in test:
```bash
python agents/llm_provider.py
```

Expected output:
```
======================================================================
LLM PROVIDER TEST
======================================================================

Provider: gemini
Model: gemini-1.5-flash

Generating response...

Response:
[Gemini's response about WHO blast threshold]

======================================================================
✓ LLM PROVIDER TEST SUCCESSFUL
======================================================================
```

## Switching Between Providers

You can easily switch between LLM providers by editing `configs/default.yaml`:

### Use Gemini (Current)
```yaml
llm:
  provider: gemini
  model: gemini-1.5-flash
```

### Switch to Groq
```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
```

### Switch to OpenAI
```yaml
llm:
  provider: openai
  model: gpt-4
```

No code changes needed - just update the config!

## Configuration Options

### Gemini-Specific Settings
```yaml
llm:
  provider: gemini
  model: gemini-1.5-flash        # or gemini-1.5-pro
  max_tokens: 8192               # Max output tokens (Gemini supports up to 8192)
  temperature: 0.1               # 0.0 = deterministic, 1.0 = creative
```

### Temperature Guidelines
- **0.0 - 0.2:** Deterministic, factual responses (medical reports)
- **0.3 - 0.5:** Balanced creativity and consistency
- **0.6 - 1.0:** More creative, varied responses

### Token Limits
- **gemini-1.5-flash:** Up to 8,192 output tokens
- **gemini-1.5-pro:** Up to 8,192 output tokens
- **Context window:** Both models support 1M-2M token context

## Benefits of Gemini

### 1. **Large Context Windows**
- gemini-1.5-flash: 1M tokens
- gemini-1.5-pro: 2M tokens
- Can handle entire medical documents, knowledge bases

### 2. **High Quality**
- State-of-the-art reasoning
- Excellent medical/scientific knowledge
- Strong instruction following

### 3. **Cost Effective**
- gemini-1.5-flash is very affordable
- Free tier available for testing
- Pay-as-you-go pricing

### 4. **Speed**
- gemini-1.5-flash is extremely fast
- Suitable for real-time applications
- Lower latency than many alternatives

### 5. **Safety**
- Built-in safety filters
- Responsible AI features
- Medical content handling

## API Key Security

### ✅ Best Practices
- Store API key in `.env` file (already done)
- `.env` is in `.gitignore` (never commit it)
- Use environment variables in production
- Rotate keys periodically

### ❌ Never Do This
- Hardcode API keys in source code
- Commit `.env` to version control
- Share API keys in public repositories
- Use production keys in development

## Performance Comparison

| Provider | Model | Speed | Context | Cost | Quality |
|----------|-------|-------|---------|------|---------|
| **Gemini** | 1.5-flash | ⚡⚡⚡ | 1M | $ | ⭐⭐⭐⭐ |
| Gemini | 1.5-pro | ⚡⚡ | 2M | $$ | ⭐⭐⭐⭐⭐ |
| Groq | Llama 3.3 70B | ⚡⚡⚡ | 32K | $ | ⭐⭐⭐⭐ |
| OpenAI | GPT-4 | ⚡ | 128K | $$$ | ⭐⭐⭐⭐⭐ |

## Troubleshooting

### Error: "GOOGLE_API_KEY not found"
**Solution:** Check that `.env` file exists and contains:
```env
GOOGLE_API_KEY=AIzaSyAb8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA64d4GRh7pw
```

### Error: "google-generativeai not installed"
**Solution:** Install the library:
```bash
pip install google-generativeai
```

### Error: "API key invalid"
**Solution:** 
1. Verify the API key is correct
2. Check it hasn't expired
3. Get a new key at https://makersuite.google.com/app/apikey

### Slow responses
**Solution:** 
- Switch to `gemini-1.5-flash` for faster responses
- Reduce `max_tokens` if you don't need long outputs
- Check your internet connection

### Rate limiting
**Solution:**
- Gemini has generous rate limits
- Add retry logic with exponential backoff
- Consider upgrading to paid tier if needed

## Next Steps

### 1. **Test Report Generation**
Upload a bone marrow sample and generate a report to verify Gemini integration:
```bash
# Start the Flask app
python app/app.py

# Or Streamlit interface
streamlit run app/streamlit_app.py
```

### 2. **Monitor Performance**
- Check response times
- Verify report quality
- Compare with previous Groq outputs

### 3. **Optimize Settings**
- Experiment with temperature (0.1 - 0.5)
- Adjust max_tokens based on report length
- Try gemini-1.5-pro for more complex cases

### 4. **Production Deployment**
- Set up proper API key management
- Implement rate limiting
- Add error handling and fallbacks
- Monitor API usage and costs

## Support

### Getting Help
- **Gemini Documentation:** https://ai.google.dev/docs
- **API Reference:** https://ai.google.dev/api
- **Pricing:** https://ai.google.dev/pricing
- **Support:** https://ai.google.dev/support

### API Key Management
- **Get API Key:** https://makersuite.google.com/app/apikey
- **Usage Dashboard:** https://console.cloud.google.com/

## Summary

✅ **Completed:**
- Added Gemini API key to `.env`
- Implemented Gemini support in `llm_provider.py`
- Updated configuration to use Gemini
- Set model to `gemini-1.5-flash` for optimal performance
- Updated documentation

🎯 **Current Configuration:**
- Provider: **Google Gemini**
- Model: **gemini-1.5-flash**
- Max Tokens: **8,192**
- Temperature: **0.1**

🚀 **Ready to Use:**
The system is now configured to use Google Gemini for all LLM operations, including report generation, chat responses, and clinical analysis. All existing functionality remains the same - only the underlying LLM provider has changed.
