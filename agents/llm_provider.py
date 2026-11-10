"""
agents/llm_provider.py
─────────────────────────────────────────────────────────────────────────────
LLM Provider Abstraction Layer.

Supports multiple LLM backends:
  - Groq (Llama 3.3 70B, Llama 3.1 70B, Mixtral 8x7B) - Fast with free tier
  - Google Gemini (Gemini 1.5 Pro, Gemini 1.5 Flash) - Powerful with large context
  - OpenAI (GPT-4, GPT-3.5)
  - Local (HuggingFace models via transformers)

Configuration is loaded from configs/default.yaml and .env
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.config_loader import load_config

logger = logging.getLogger("llm_provider")


class LLMProvider:
    """
    Unified LLM interface supporting multiple backends.
    Automatically selects provider based on config.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.llm_config = config.get("llm", {})
        self.provider = self.llm_config.get("provider", "groq")
        if self.provider == "gemini":
            self.model_name = self.llm_config.get("gemini_model") or self.llm_config.get("model", "gemini-1.5-flash")
            self.max_tokens = self.llm_config.get("gemini_max_tokens", self.llm_config.get("max_tokens", 8192))
        else:
            self.model_name = self.llm_config.get("model", "llama-3.3-70b-versatile")
            self.max_tokens = self.llm_config.get("max_tokens", 2048)
        self.temperature = self.llm_config.get("temperature", 0.1)
        self._use_new_genai = False  # Flag for Gemini API version

        self.client = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the selected LLM provider."""
        if self.provider == "groq":
            self._initialize_groq()
        elif self.provider == "gemini":
            self._initialize_gemini()
        elif self.provider == "openai":
            self._initialize_openai()
        elif self.provider == "local":
            self._initialize_local()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _initialize_groq(self):
        """Initialize Groq client with Llama models."""
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "Groq library not installed. Install with: pip install groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Get your free API key at https://console.groq.com and add it to .env"
            )

        self.client = Groq(api_key=api_key)
        logger.info(f"Groq LLM initialized with model: {self.model_name}")

    def _initialize_gemini(self):
        """Initialize Google Gemini client.

        Supports two credential formats:
          - Standard API key (starts with 'AIzaSy'): passed as api_key param
          - OAuth2 access token (starts with 'AQ.'): passed as Bearer credentials
        """
        try:
            # Try new package first
            try:
                from google import genai
                self._use_new_genai = True
            except ImportError:
                # Fall back to old package
                import google.generativeai as genai
                self._use_new_genai = False
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. Install with: pip install google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment. "
                "Get your API key at https://aistudio.google.com/app/apikey and add it to .env"
            )

        # Detect if this is an OAuth2 access token (AQ. prefix) vs a standard API key
        is_oauth_token = not api_key.startswith("AIzaSy")

        if self._use_new_genai:
            if is_oauth_token:
                # OAuth2 access token — wrap in google.oauth2.credentials.Credentials
                try:
                    from google.oauth2.credentials import Credentials as OAuthCredentials
                    creds = OAuthCredentials(token=api_key)
                    self.client = genai.Client(credentials=creds)
                    logger.info(f"Google Gemini LLM initialized (new API, OAuth2 token) with model: {self.model_name}")
                except ImportError:
                    raise ImportError(
                        "google-auth not installed. Install with: pip install google-auth"
                    )
            else:
                self.client = genai.Client(api_key=api_key)
                logger.info(f"Google Gemini LLM initialized (new API) with model: {self.model_name}")
        else:
            if is_oauth_token:
                try:
                    from google.oauth2.credentials import Credentials as OAuthCredentials
                    import google.auth.transport.requests
                    creds = OAuthCredentials(token=api_key)
                    genai.configure(credentials=creds)
                except ImportError:
                    raise ImportError(
                        "google-auth not installed. Install with: pip install google-auth"
                    )
            else:
                genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Google Gemini LLM initialized (legacy API, {'OAuth2' if is_oauth_token else 'API key'}) with model: {self.model_name}")

    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. Add it to .env"
            )

        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI LLM initialized with model: {self.model_name}")

    def _initialize_local(self):
        """Initialize local HuggingFace model."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Transformers library not installed. Install with: pip install transformers"
            )

        local_model = self.llm_config.get("local_model", "mistralai/Mistral-7B-Instruct-v0.3")
        logger.info(f"Loading local HuggingFace model: {local_model}")
        self.client = pipeline(
            "text-generation",
            model=local_model,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        logger.info(f"Local LLM initialized with model: {local_model}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text as string
        """
        if self.provider == "groq":
            return self._generate_groq(messages, **kwargs)
        elif self.provider == "gemini":
            return self._generate_gemini(messages, **kwargs)
        elif self.provider == "openai":
            return self._generate_openai(messages, **kwargs)
        elif self.provider == "local":
            return self._generate_local(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_groq(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate using Groq API with automatic retry on rate limits."""
        import time
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "quota" in err_str
                if is_rate_limit and attempt < max_retries - 1:
                    wait_sec = 2.0 * (attempt + 1)
                    logger.warning(f"Groq rate limit encountered. Retrying in {wait_sec}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    logger.error(f"Groq generation failed: {e}")
                    raise

    def _generate_gemini(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate using Google Gemini API with retry + model fallback on 503."""
        max_tokens  = kwargs.get("max_tokens",  self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        # Convert messages to a single prompt string
        prompt_parts       = []
        system_instruction = None
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                prompt_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Previous response: {msg['content']}")
        full_prompt = "\n\n".join(prompt_parts)
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{full_prompt}"

        # Ordered fallback model list — tried in sequence on 503 or 429
        model_candidates = [
            self.model_name,
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]
        # Preserve order while deduplicating
        FALLBACK_MODELS = list(dict.fromkeys(m for m in model_candidates if m))

        last_exc = None
        for model_attempt, model in enumerate(FALLBACK_MODELS):
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    if self._use_new_genai:
                        response = self.client.models.generate_content(
                            model=model,
                            contents=full_prompt,
                            config={
                                "temperature": temperature,
                                "max_output_tokens": max_tokens,
                            }
                        )
                    else:
                        generation_config = {
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        }
                        response = self.client.generate_content(
                            full_prompt,
                            generation_config=generation_config,
                        )

                    # Extract text safely
                    text = getattr(response, "text", None)
                    if not text and hasattr(response, "candidates") and response.candidates:
                        c = response.candidates[0]
                        if hasattr(c, "content") and hasattr(c.content, "parts"):
                            text = "".join(getattr(p, "text", "") or "" for p in c.content.parts)
                    final_text = (text or "").strip()

                    # Success — persist the working model so future calls use it directly
                    if model != self.model_name:
                        logger.warning(f"Switched to working Gemini model: {model}")
                        self.model_name = model
                    return final_text

                except Exception as e:
                    last_exc = e
                    err_str = str(e)
                    is_transient = (
                        "503" in err_str or "UNAVAILABLE" in err_str
                        or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                        or "quota" in err_str.lower()
                        or "high demand" in err_str.lower()
                        or "overload" in err_str.lower()
                    )
                    if is_transient:
                        if attempt < max_retries - 1:
                            wait_sec = 1.5 ** attempt   # 1s, 1.5s
                            logger.warning(
                                f"Gemini {err_str[:60]}... on {model!r} (attempt {attempt+1}/{max_retries}). "
                                f"Retrying in {wait_sec:.1f}s..."
                            )
                            time.sleep(wait_sec)
                        else:
                            logger.warning(
                                f"Gemini exhausted retries on {model!r}. Trying next fallback model..."
                            )
                        continue  # next retry or next model
                    else:
                        # Non-transient error — try next model if available, else raise
                        logger.warning(f"Gemini error on {model!r}: {e}. Trying fallback model...")
                        break

        # All models exhausted
        logger.error(f"All Gemini models exhausted. Last error: {last_exc}")
        raise last_exc


    def _generate_openai(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate using OpenAI API."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def _generate_local(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate using local HuggingFace model."""
        # Convert messages to single prompt string
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

        try:
            response = self.client(prompt, **kwargs)
            return response[0]["generated_text"]
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise


# ── Singleton Instance ───────────────────────────────────────────────────────

_GLOBAL_LLM_PROVIDER: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get singleton LLM provider instance."""
    global _GLOBAL_LLM_PROVIDER
    if _GLOBAL_LLM_PROVIDER is None:
        _GLOBAL_LLM_PROVIDER = LLMProvider()
    return _GLOBAL_LLM_PROVIDER


# ── Quick Test CLI ───────────────────────────────────────────────────────────

def main():
    """Test LLM provider with a simple query."""
    print("\n" + "=" * 70)
    print("LLM PROVIDER TEST")
    print("=" * 70)

    try:
        provider = get_llm_provider()
        print(f"\nProvider: {provider.provider}")
        print(f"Model: {provider.model_name}")

        messages = [
            {"role": "system", "content": "You are a helpful clinical pathology assistant."},
            {"role": "user", "content": "What is the WHO blast threshold for acute leukemia diagnosis?"},
        ]

        print("\nGenerating response...")
        response = provider.generate(messages)
        print(f"\nResponse:\n{response}")

        print("\n" + "=" * 70)
        print("✓ LLM PROVIDER TEST SUCCESSFUL")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure you have:")
        print("  1. Set GROQ_API_KEY in .env file")
        print("  2. Installed groq: pip install groq")
        print("  3. Configured llm.provider: groq in configs/default.yaml")


if __name__ == "__main__":
    main()
