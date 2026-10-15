# 🔑 How to Get a Valid Gemini API Key

## The Current Problem
Your API key `AQ.Ab8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA64d4GRh7pw` is not valid.

**Google API keys should:**
- Start with `AIzaSy` (not `AQ.`)
- Be exactly 39 characters long
- Only contain alphanumeric characters (no periods `.`)

## 📝 Step-by-Step: Get Your Gemini API Key

### Option 1: Google AI Studio (Recommended)
1. **Go to:** https://aistudio.google.com/app/apikey
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy the key** - it will look like: `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
5. **Paste it in your `.env` file**

### Option 2: Google Cloud Console
1. **Go to:** https://console.cloud.google.com/
2. **Create or select a project**
3. **Enable the "Generative Language API"**
4. **Go to APIs & Services > Credentials**
5. **Click "Create Credentials" > "API Key"**
6. **Copy the generated key**

## 💾 Update Your .env File

Open: `c:\Users\AMISHA\Desktop\Codes\Multimodal_AI_Bone_Marrow\.env`

Replace the line:
```env
GOOGLE_API_KEY=AQ.Ab8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA64d4GRh7pw
```

With your new key:
```env
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 🧪 Test Your Key

After updating, run:
```bash
python agents/llm_provider.py
```

You should see:
```
✓ LLM PROVIDER TEST SUCCESSFUL
```

## 🔄 Alternative: Use Groq (Works Right Now!)

If you want to use the system immediately without getting a new Gemini key, Groq is already configured and working! Just change the config:

**Edit `configs/default.yaml`:**
```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
```

Groq uses the key already in your `.env` and is very fast!

## ❓ Still Having Issues?

The key format matters:
- ✅ **VALID:** `AIzaSyAb8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA`
- ❌ **INVALID:** `AQ.Ab8RN6KCCP1Zsp7yoIifXUng4zCgmaIbYU0qiCbA` (has period)
- ❌ **INVALID:** Anything not starting with `AIzaSy`

If you're sure your key is correct but still getting errors, make sure:
1. You've saved the `.env` file
2. You've restarted the Flask app
3. The key has no extra spaces or line breaks
