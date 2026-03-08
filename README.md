# Maarif-Gen: Otomatik Soru Üretim Sistemi

Türkiye Yüzyılı Maarif Modeli öğretim programı PDF'lerini okuyarak müfredat ağacı oluşturan ve yerel LLM sunucuları üzerinden açık uçlu sorular ile değerlendirme rubriği üreten bir araç.

## Özellikler

- PDF öğretim programlarını ayrıştırır ve yapılandırılmış bir SQLite veritabanına aktarır
- Streamlit tabanlı sihirbaz arayüzüyle ders → sınıf → ünite → kazanım seçimi
- Bağlam (senaryo) oluşturma ve süreç bileşeni bazında soru üretimi
- Bloom taksonomisine göre bilişsel düzey etiketleme
- Word (.docx) soru bankası ve fine-tuning JSONL veri seti çıktısı
- CLI arayüzü (`main.py`) ile hızlı toplu üretim

## Gereksinimler

- Python 3.9+
- `poppler-utils` (pdftotext komutu için)
- Çalışan bir LLM sunucusu: [vLLM](https://github.com/vllm-project/vllm) veya [Ollama](https://ollama.com/)

### poppler kurulumu

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

## Kurulum

```bash
# 1. Depoyu klonlayın
git clone <repo-url>
cd SoruUretim

# 2. Sanal ortam oluşturun ve bağımlılıkları yükleyin
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Ortam değişkenlerini ayarlayın
cp .env.example .env
# .env dosyasını düzenleyerek kendi sunucu adreslerinizi girin

# 4. Veritabanını oluşturun
python db_setup.py

# 5. PDF'leri içe aktarın (program/ klasörüne PDF'leri koyun)
python etl_pipeline.py
```

## Kullanım

### Web Arayüzü

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini açın.

**Sihirbaz adımları:**
1. Ders seçin
2. Sınıf seçin
3. Ünite/Tema seçin
4. Öğrenme Çıktısı seçin
5. Bağlam oluşturun
6. Süreç bileşenlerini seçip soru üretin
7. Soru bankasına kaydedin veya indirin

### CLI

```bash
# program/ klasöründeki PDF'ler listelenir; seçilen dosya işlenir
python main.py
```

Üretilen sorular `processed_data/` klasörüne JSON olarak kaydedilir.

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `VLLM_API_URL` | `http://localhost:8000/v1` | vLLM sunucu adresi (CLI için) |
| `OLLAMA_API_URL` | `http://localhost:11434/v1` | Ollama sunucu adresi (web arayüzü için) |
| `VLLM_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` | CLI'da kullanılacak model |
| `OLLAMA_MODEL` | `gemma3:27b` | Web arayüzünde kullanılacak model |
| `DB_NAME` | `maarif_gen.db` | SQLite veritabanı dosya adı |

## Proje Yapısı

```
SoruUretim/
├── program/              # Öğretim programı PDF dosyaları (git'e eklenmez)
├── processed_data/       # CLI çıktı JSON dosyaları
├── SoruBankasi/          # Kaydedilen .docx ve .jsonl dosyaları
├── src/
│   ├── extractors/
│   │   └── pdf_extractor.py   # PDF ayrıştırıcı
│   ├── generators/
│   │   └── llm_client.py      # LLM API istemcisi
│   └── models/
│       └── schemas.py         # TypedDict veri modelleri
├── app.py                # Streamlit web arayüzü
├── main.py               # CLI giriş noktası
├── db_setup.py           # Veritabanı şema kurulumu
├── etl_pipeline.py       # PDF → DB aktarım hattı
├── config.py             # Merkezi konfigürasyon
└── .env.example          # Ortam değişkenleri şablonu
```

## Desteklenen PDF Formatları

`PDFExtractor`, aşağıdaki öğretim programı formatlarını destekler:

- Türkçe: `N. TEMA: Ad` / `N. ÜNİTE: Ad`
- Türkçe büyük harf başlık (Tarih, TDE tarzı)
- İngilizce: `THEME N: Ad`
- Kazanım kodları: `XXX.10.1.1` veya `ENG.PREP.1.G1`


"""
docker rm -f vllm_server ; docker run -d --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm_server \
  vllm/vllm-openai:cu130-nightly-aarch64 \
  "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" \
  --max-model-len 4096 \
  --quantization gptq \
  --dtype float16 \
  --gpu-memory-utilization 0.70 \
  --trust-remote-code \
  --enforce-eager

"""