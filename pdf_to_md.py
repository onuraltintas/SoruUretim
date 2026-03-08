#!/usr/bin/env python3
"""
pdf_to_md.py — PDF → Markdown dönüştürücü (Qwen2.5-VL-72B, vLLM)
=================================================================
Kullanım:
    # Tüm PDF'ler (program/ klasöründen):
    python pdf_to_md.py --all

    # Belirli dosyalar:
    python pdf_to_md.py "program/Fizik.pdf" "program/Kimya.pdf"

    # Farklı vLLM endpoint:
    python pdf_to_md.py --all --api-url http://localhost:8000/v1

    # Zaten var olan md dosyalarını yeniden işle (üstüne yaz):
    python pdf_to_md.py --all --overwrite

Çıktı:
    md/<DersAdı>.md   (md/ klasörü yoksa otomatik oluşturulur)

Gereksinimler:
    pip install openai pdf2image pymupdf
"""

import argparse
import base64
import datetime
import glob
import os
import sys
import time
from io import BytesIO
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai paketi kurulu değil: pip install openai")

try:
    import pdf2image
except ImportError:
    sys.exit("pdf2image paketi kurulu değil: pip install pdf2image")

# ── Sabitler ────────────────────────────────────────────────────────────────
PDF_DIR     = "program"
OUT_DIR     = "md"
DEFAULT_URL = "http://localhost:8000/v1"
API_KEY     = "EMPTY"
MODEL       = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
DPI         = 150
MAX_TOKENS  = 3000
TEMPERATURE = 0.0

PROMPT = """You are an expert document digitizer specializing in Turkish educational curriculum documents. Your task is to convert the provided document image into Markdown format.

CRITICAL INSTRUCTIONS:
1. **IGNORE RECURRING HEADERS & FOOTERS:** Visually identify and STRICTLY IGNORE any recurring text in the top header (e.g., book/series name headers) or bottom footer (e.g., page numbers, watermarks).
2. **PRESERVE MAIN TITLES:** Do NOT ignore the MAIN TITLE or CHAPTER TITLE of the document. If text is large, bold, and clearly the subject, transcribe it as a Level 1 Header (# Title).
3. **TEXT FIDELITY:** Transcribe the MAIN content EXACTLY as it appears. Do not summarize, correct grammar, or change the wording. Use the exact same language (Turkish) and tone.
4. **LAYOUT & STRUCTURE:**
   * Detect and format headings using Markdown headers (#, ##, ###).
   * Preserve lists and bullet points accurately.
   * **Text Formatting:** Preserve bold (**text**) and italic (*text*) styling.
   * **Multi-column Text:** If multiple columns exist, read first column completely before moving to next.
5. **NON-TEXT ELEMENTS:** Replace any map, chart, graph, or illustration with a detailed description in TURKISH enclosed in square brackets:
   * Example: `[Görsel: Türkiye fiziki haritası, dağlık alanlar ve ovalar gösteriliyor]`
6. **TABLES:** Convert tables into proper Markdown table syntax.
7. **OUTPUT ONLY MARKDOWN:** Start directly with the content. No conversational filler.
"""

# ── Yardımcılar ─────────────────────────────────────────────────────────────

def encode_image_jpeg(image) -> str:
    """PIL Image → base64 JPEG string."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_server(api_url: str) -> bool:
    """vLLM sunucusunun hazır olup olmadığını kontrol eder."""
    try:
        client = OpenAI(base_url=api_url, api_key=API_KEY)
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"✅ Sunucu hazır — Mevcut modeller: {', '.join(model_ids)}")
        return True
    except Exception as e:
        print(f"❌ Sunucuya bağlanılamadı: {e}")
        return False


def process_page(client: OpenAI, image, page_num: int, model: str) -> str:
    """Tek bir PDF sayfasını VLM'e gönderir, Markdown metin döner."""
    b64 = encode_image_jpeg(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content


def convert_pdf(
    pdf_path: str,
    out_dir: str,
    api_url: str,
    model: str,
    dpi: int,
    overwrite: bool,
) -> bool:
    """PDF'i sayfa sayfa VLM'e gönderip .md dosyasına yazar."""
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{pdf_path.stem}.md"

    # Zaten varsa ve overwrite=False ise atla
    if out_file.exists() and not overwrite:
        print(f"⏭  Atlandı (zaten mevcut): {out_file}")
        return True

    print(f"\n{'='*60}")
    print(f"📄 {pdf_path.name}")
    print(f"   → {out_file}")
    print(f"   Başlangıç: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # PDF → görüntüler
    try:
        print("  PDF sayfalara dönüştürülüyor...")
        images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
        print(f"  {len(images)} sayfa yüklendi.")
    except Exception as e:
        print(f"❌ PDF açılamadı: {e}")
        return False

    client = OpenAI(base_url=api_url, api_key=API_KEY)
    full_md = ""
    n = len(images)

    for i, image in enumerate(images, start=1):
        t0 = time.time()
        sys.stdout.write(f"  Sayfa {i:>3}/{n} [{datetime.datetime.now().strftime('%H:%M:%S')}] … ")
        sys.stdout.flush()

        try:
            content = process_page(client, image, i, model)
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s, {len(content)} karakter)")
            full_md += f"\n\n{content}"
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ HATA ({elapsed:.1f}s): {e}")
            full_md += f"\n\n<!-- Sayfa {i}: hata — {e} -->\n\n"

    # Kaydet
    try:
        out_file.write_text(full_md.strip(), encoding="utf-8")
        size_kb = out_file.stat().st_size // 1024
        print(f"\n✅ Kaydedildi: {out_file} ({size_kb} KB)\n")
        return True
    except Exception as e:
        print(f"❌ Dosya kaydedilemedi: {e}")
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────

def list_pdfs() -> list[str]:
    return sorted(glob.glob(f"{PDF_DIR}/*.pdf"))


def pick_interactively() -> list[str]:
    pdfs = list_pdfs()
    if not pdfs:
        sys.exit(f"'{PDF_DIR}/' klasöründe PDF bulunamadı.")

    print("\nMevcut PDF dosyaları:")
    for idx, p in enumerate(pdfs, start=1):
        size_mb = Path(p).stat().st_size // (1024 * 1024)
        print(f"  [{idx:>2}] {Path(p).name}  ({size_mb} MB)")
    print(f"  [ 0] Hepsini seç")

    raw = input("\nNumara(lar)ı girin (virgülle ayırın, ör: 1,3,5): ").strip()
    if raw == "0":
        return pdfs

    selected = []
    for token in raw.split(","):
        token = token.strip()
        if not token.isdigit():
            print(f"  Geçersiz giriş atlandı: '{token}'")
            continue
        idx = int(token)
        if 1 <= idx <= len(pdfs):
            selected.append(pdfs[idx - 1])
        else:
            print(f"  Geçersiz numara atlandı: {idx}")

    if not selected:
        sys.exit("Geçerli seçim yapılmadı, çıkılıyor.")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="PDF → Markdown dönüştürücü (Qwen2.5-VL / vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdfs", nargs="*", metavar="PDF",
        help="Dönüştürülecek PDF dosyaları (boş=interaktif seçim)",
    )
    parser.add_argument("--all",       action="store_true", help="Tüm PDF'leri işle")
    parser.add_argument("--overwrite", action="store_true", help="Var olan md dosyalarının üstüne yaz")
    parser.add_argument("--api-url",   default=DEFAULT_URL, help=f"vLLM API URL (varsayılan: {DEFAULT_URL})")
    parser.add_argument("--model",     default=MODEL,       help=f"Model adı (varsayılan: {MODEL})")
    parser.add_argument("--out-dir",   default=OUT_DIR,     help=f"Çıktı klasörü (varsayılan: {OUT_DIR})")
    parser.add_argument("--dpi",       type=int, default=DPI, help=f"Render DPI (varsayılan: {DPI})")

    args = parser.parse_args()

    # Sunucu kontrolü
    print(f"🔗 vLLM sunucusu kontrol ediliyor: {args.api_url}")
    if not check_server(args.api_url):
        sys.exit(
            "   Sunucuyu şu komutla başlatın:\n"
            "   docker run --gpus all -p 8000:8000 ... "
            "vllm serve 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ'"
        )

    # PDF seçimi
    if args.all:
        selected = list_pdfs()
        if not selected:
            sys.exit(f"'{PDF_DIR}/' klasöründe PDF bulunamadı.")
    elif args.pdfs:
        selected = args.pdfs
    else:
        selected = pick_interactively()

    print(f"\n📋 Seçilen {len(selected)} PDF işlenecek:")
    for p in selected:
        print(f"   • {p}")

    # İşlem
    t_start = time.time()
    ok, fail = 0, 0

    for pdf in selected:
        success = convert_pdf(
            pdf_path=pdf,
            out_dir=args.out_dir,
            api_url=args.api_url,
            model=args.model,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
        if success:
            ok += 1
        else:
            fail += 1

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏁 Tamamlandı — {ok} başarılı, {fail} başarısız ({total/60:.1f} dakika)")
    print(f"   Çıktı klasörü: {args.out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
