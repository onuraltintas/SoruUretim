#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen3-72B-AWQ"
VLLM_PORT=8000
STREAMLIT_PORT=8501
LOG_DIR="logs"

mkdir -p "${LOG_DIR}"

echo "================================================"
echo "  Maarif-Gen  |  Qwen3-72B-AWQ + vLLM"
echo "================================================"

# ── vLLM kontrolü ─────────────────────────────────
if curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "✓ vLLM zaten çalışıyor (port ${VLLM_PORT})"
else
    echo "▶ vLLM başlatılıyor..."
    echo "  Model : ${MODEL}"
    echo "  Log   : ${LOG_DIR}/vllm.log"
    echo "  Not   : Model yoksa HuggingFace'den otomatik indirilir (~40 GB)"
    echo ""

    vllm serve "${MODEL}" \
        --dtype auto \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --port "${VLLM_PORT}" \
        --host 0.0.0.0 \
        > "${LOG_DIR}/vllm.log" 2>&1 &

    echo "  vLLM PID: $!"
    echo ""
    echo "  Hazır olana kadar bekleniyor..."
    echo "  (İlk çalıştırmada model indirme süresi ~40 GB)"

    MAX_WAIT=1800   # 30 dakika (ilk indirme için)
    WAITED=0
    while ! curl -s --max-time 2 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
        sleep 10
        WAITED=$((WAITED + 10))
        printf "  Bekleniyor... %ds\r" "${WAITED}"
        if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
            echo ""
            echo "✗ vLLM ${MAX_WAIT}s içinde başlamadı."
            echo "  Detay: tail -f ${LOG_DIR}/vllm.log"
            exit 1
        fi
    done

    echo ""
    echo "✓ vLLM hazır!"
fi

# ── Streamlit ──────────────────────────────────────
echo ""
echo "▶ Streamlit başlatılıyor..."
echo "  Adres: http://localhost:${STREAMLIT_PORT}"
echo ""

streamlit run app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.headless true
