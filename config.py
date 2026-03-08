import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; environment variables must be set manually

# --- vLLM (Docker) ---
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
VLLM_MODEL   = os.getenv("VLLM_MODEL",   "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")

# --- Ollama (Local) ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "qwen3.5:122b")

# --- Veritabanı ---
DB_NAME = os.getenv("DB_NAME", "maarif_gen.db")

# --- PDF Çıkarım & Kalite ---
LLM_EXTRACTION_FALLBACK = os.getenv("LLM_EXTRACTION_FALLBACK", "true").lower() == "true"
MIN_QUALITY_SCORE       = int(os.getenv("MIN_QUALITY_SCORE", "40"))
OCR_ENABLED             = os.getenv("OCR_ENABLED", "false").lower() == "true"
LLM_FALLBACK_MAX_PAGES  = int(os.getenv("LLM_FALLBACK_MAX_PAGES", "40"))
