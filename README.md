# Maarif-Gen: Otomatik Soru Üretim Sistemi

Türkiye Yüzyılı Maarif Modeli öğretim programlarını temel alarak yapılandırılmış bir SQLite veritabanı üzerinden açık uçlu sorular ve değerlendirme rubrikleri üreten yapay zekâ tabanlı bir araçtır.

## Özellikler

- **Müfredat Ağacı:** Maarif Modeli kazanımlarını (öğrenme çıktılarını) hiyerarşik olarak sunar.
- **Akıllı Bağlam Üretimi:** Seçilen kazanıma uygun, gerçek hayatla ilişkilendirilmiş senaryolar kurgular.
- **Süreç Odaklı Ölçme:** Kazanım bileşenleri bazında özgün açık uçlu sorular üretir.
- **Bilişsel Düzey:** Bloom taksonomisine göre bilişsel seviye etiketlemesi yapar.
- **Çıktı Formatları:** Üretilen soruları Word (.docx) soru bankası veya fine-tuning için JSONL veri seti olarak dışa aktarır.
- **vLLM & Blackwell Optimizasyonu:** NVIDIA Blackwell (GB10) mimarisi üzerinde Qwen 3.5 122B gibi devasa modellerle akıl yürütme (reasoning) modunda yüksek performanslı çalışma.

## Gereksinimler

- Python 3.9+
- Çalışan bir LLM sunucusu: [vLLM](https://github.com/vllm-project/vllm) veya [Ollama](https://ollama.com/)
- SQLite (Veritabanı için)

## Kurulum

```bash
# 1. Depoyu klonlayın
git clone https://github.com/onuraltintas/SoruUretim.git
cd SoruUretim

# 2. Sanal ortam oluşturun ve bağımlılıkları yükleyin
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Ortam değişkenlerini ayarlayın
cp .env.example .env
# .env dosyasını düzenleyerek kendi sunucu adreslerinizi ve model isimlerinizi girin
```

## Kullanım

### Web Arayüzü (Streamlit)

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini açarak sihirbazı takip edebilirsiniz:
1. Ders ve Sınıf seçimi
2. Ünite/Tema ve Öğrenme Çıktısı seçimi
3. Bağlam (Senaryo) oluşturma
4. Soru üretimi ve Soru Bankasına kayıt

## Ortam Değişkenleri (.env)

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `VLLM_API_URL` | `http://localhost:8000/v1` | vLLM sunucu adresi |
| `OLLAMA_API_URL` | `http://localhost:11434/v1` | Ollama sunucu adresi |
| `VLLM_MODEL` | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | vLLM üzerinde yüklü model adı |
| `OLLAMA_MODEL` | `qwen2.5:72b` | Ollama üzerinde yüklü model adı |
| `DB_NAME` | `maarif_gen.db` | SQLite veritabanı dosyası |

## Proje Yapısı

```
SoruUretim/
├── src/
│   └── generators/
│       ├── llm_client.py   # LLM API istemcisi ve filtreleme mantığı
│       └── prompts.py      # Müfredata uygun sistem ve kullanıcı promptları
├── app.py                  # Streamlit ana uygulama arayüzü
├── config.py               # Merkezi konfigürasyon yönetimi
├── maarif_gen.db           # Maarif Modeli müfredat veritabanı (SQLite)
├── md/                     # Ham müfredat verileri (Markdown)
├── SoruBankasi/            # Üretilen çıktıların (.docx, .jsonl) kaydedildiği dizin
├── .env                    # Aktif ortam değişkenleri
└── README.md               # Proje dökümantasyonu
```

## vLLM (Blackwell / Qwen 3.5) Çalıştırma Komutu

Aşağıdaki komut, Qwen 3.5 122B modelini "Akıl Yürütme" (Reasoning) modu aktif ve sunucu tarafında ayrıştırılmış (parsed) şekilde başlatır:

```bash
docker rm -f vllm_server ; docker run -d --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm_server \
  vllm/vllm-openai:cu130-nightly-aarch64 \
  "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" \
  --quantization gptq \
  --dtype float16 \
  --gpu-memory-utilization 0.70 \
  --trust-remote-code \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```
---
*Bu proje, Türkiye Yüzyılı Maarif Modeli'nin dijital dönüşümü ve ölçme-değerlendirme süreçlerinin yapay zekâ ile desteklenmesi amacıyla geliştirilmiştir.*