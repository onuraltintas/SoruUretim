#!/usr/bin/env python3
"""
md_to_db.py — Markdown → SQLite ETL
=====================================
md/ klasöründeki Markdown müfredat dosyalarını ayrıştırıp
maarif_gen.db veritabanına yükler.

Kullanım:
    python md_to_db.py --all
    python md_to_db.py md/Kimya.md
    python md_to_db.py md/Kimya.md --dry-run

Hiyerarşi:
    subjects → grades → units → outcomes → components
"""

import argparse
import glob
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_NAME = config.DB_NAME
MD_DIR  = "md"

# ── Veri yapıları ────────────────────────────────────────────────────────────

@dataclass
class Component:
    code: str
    description: str

@dataclass
class Outcome:
    code: str
    description: str
    implementation_guide: str = ""
    components: list[Component] = field(default_factory=list)

@dataclass
class Unit:
    name: str
    unit_no: int = 0
    outcomes: list[Outcome] = field(default_factory=list)

@dataclass
class Grade:
    level_name: str
    units: list[Unit] = field(default_factory=list)

@dataclass
class Subject:
    name: str
    grades: list[Grade] = field(default_factory=list)


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

# Outcomes bloğunu bitiren anahtar kelimeler (küçük harf, substring kontrolü)
_STOP_KEYWORDS = [
    "içerik çerçevesi", "öğrenme kanıtları", "öğrenme-öğretme yaşantıları",
    "öğrenme–öğretme yaşantıları", "öğrenme öğretme yaşantıları",
    "öğrenme-öğretim uygulamaları", "öğrenme öğretim uygulamaları",
    "farklılaştırma", "öğretmen yansıtmaları",
    "programlar arası", "disiplinler arası", "beceriler arası",
    "temel kabuller", "ön değerlendirme", "köprü kurma",
    "zenginleştirme", "destekleme",
]

# Implementation guide bölümünü durduran anahtar kelimeler
_GUIDE_STOP_KEYWORDS = [
    "farklılaştırma", "öğretmen yansıtmaları", "içerik çerçevesi",
    "öğrenme kanıtları", "zenginleştirme", "destekleme",
]


def _is_heading(line: str) -> int:
    """Başlık satırı ise # sayısını döndürür, değilse 0."""
    if not line.startswith("#"):
        return 0
    cnt = 0
    for ch in line:
        if ch == "#":
            cnt += 1
        else:
            break
    return cnt if cnt <= 4 else 0


def _heading_text(line: str) -> str:
    """'## Başlık' → 'Başlık'"""
    return line.lstrip("#").strip()


def _is_stop(text_lower: str) -> bool:
    return any(kw in text_lower for kw in _STOP_KEYWORDS)


def _is_guide_stop(text_lower: str) -> bool:
    return any(kw in text_lower for kw in _GUIDE_STOP_KEYWORDS)


# Tema başlığı: "1. TEMA: ..." veya "1\. Tema: ..."
_RE_UNIT = re.compile(r"^(\d+)(?:\\)?\.?\s*TEMA\s*[:\-]?\s*(.+)", re.IGNORECASE)

# Kazanım kodu: "KİM.9.1.1." veya "KİM.9.1.1" (nokta opsiyonel)
# Güvenli regex: [^\s.]+ → harf bloğu (nokta veya boşluk içermez)
_RE_CODE = re.compile(r"^([^\s.]+)\.(\d+)\.(\d+)(?:\.(\d+))?\.?$")

# Bileşen: "a) ..."
_RE_COMP = re.compile(r"^([a-zçğışöü])\)\s+(.+)", re.UNICODE)

# Grade tespiti için kodu parse et: "KİM.9.1.1" → "9. SINIF"
def _grade_from_code(code: str) -> str | None:
    parts = code.split(".")
    # parts[0]=KİM, parts[1]=9, parts[2]=1, parts[3]=1
    if len(parts) >= 3 and parts[1].isdigit():
        return f"{parts[1]}. SINIF"
    return None


def _parse_outcome_heading(text: str):
    """
    'KİM.9.1.1. Açıklama' → ('KİM.9.1.1', 'Açıklama')
    'KiM.11.3.1. ...'    → ('KİM.11.3.1', '...')  (OCR hatası düzeltilir)
    Açıklama ZORUNLU.
    """
    parts = text.split(None, 1)
    if not parts:
        return None
    candidate = parts[0].rstrip(".")
    segs = candidate.split(".")
    # Minimum: PREFIX.SINIF.TEMA.NO veya PREFIX.SINIF.NO
    if len(segs) < 3:
        return None
    # İlk segment harf, geri kalanlar sayı olmalı
    if not segs[0].isalpha():
        return None
    if not all(s.isdigit() for s in segs[1:3]):
        return None
    # Açıklama ZORUNLU
    if len(parts) < 2 or not parts[1].strip():
        return None
    # Kodu normalize et (uppercase)
    normalized = segs[0].upper() + "." + ".".join(segs[1:])
    return normalized, parts[1].strip()


# ── Parser — 1. Geçiş ────────────────────────────────────────────────────────

def _peek_grade(lines: list[str], start_idx: int) -> str | None:
    """
    start_idx'ten (0-bazlı) sonra ilk kazanım kodunu bulup grade adını döndürür.
    Yeni bir TEMA başlığı gelince durur (metadata bölümlerini geçer).
    """
    for raw in lines[start_idx:]:
        line = raw.strip()
        if not line:
            continue
        level = _is_heading(line)
        text  = _heading_text(line) if level else line

        # Yeni TEMA → bu tema için bulamazdık, dur
        if level and _RE_UNIT.match(text):
            break

        # Kazanım kodu? (heading veya düz metin)
        parsed = _parse_outcome_heading(text)
        if parsed:
            return _grade_from_code(parsed[0])
    return None


def parse_md(md_path: str) -> Subject:
    path    = Path(md_path)
    subject = Subject(name=path.stem)
    all_lines = path.read_text(encoding="utf-8").splitlines()

    grade_map: dict[str, Grade] = {}
    in_curriculum   = False   # "X. SINIF TEMALARI" bölümü başladı mı?
    current_unit: Unit | None     = None
    current_grade: Grade | None   = None
    current_outcome: Outcome | None = None
    in_stop = False

    # Müfredat başlangıç marker: "X. SINIF TEMALARI"
    _RE_CURRICULUM_START = re.compile(r"^\d+\.\s*SINIF\s+TEMALARI", re.IGNORECASE)

    for idx, raw in enumerate(all_lines):
        line = raw.strip()
        if not line:
            continue

        level = _is_heading(line)

        # ── Müfredat başlangıcı tespiti ──────────────────────────────────
        if level and _RE_CURRICULUM_START.match(_heading_text(line)):
            in_curriculum = True
            in_stop       = False
            continue

        if not in_curriculum:
            continue

        # ── STOP bölümü ──────────────────────────────────────────────────
        if level and _is_stop(_heading_text(line).lower()):
            current_outcome = None
            in_stop = True
            continue

        # ── Tema/Ünite ───────────────────────────────────────────────────
        if level:
            text = _heading_text(line)
            m = _RE_UNIT.match(text)
            if m:
                no        = int(m.group(1))
                raw_name  = m.group(2).strip().rstrip("\\").strip()
                full_name = f"{no}. TEMA: {raw_name}"
                current_unit    = Unit(name=full_name, unit_no=no)
                current_outcome = None
                in_stop         = False

                # Grade'i peek-ahead ile bul (STOP'ları geç, sadece yeni TEMA de dur)
                grade_name = _peek_grade(all_lines, idx + 1)
                if grade_name:
                    if grade_name not in grade_map:
                        new_grade = Grade(level_name=grade_name)
                        grade_map[grade_name] = new_grade
                        subject.grades.append(new_grade)
                    current_grade = grade_map[grade_name]
                    current_grade.units.append(current_unit)

                log.debug("  Unit: %s (grade: %s)", full_name, grade_name)
                continue

        if in_stop:
            # in_stop olsa bile outcome başlıklarını yakala (heading veya düz metin)
            if current_unit is not None:
                text   = _heading_text(line) if level else line
                parsed = _parse_outcome_heading(text)
                if parsed:
                    code, desc = parsed
                    current_outcome = Outcome(code=code, description=desc)
                    current_unit.outcomes.append(current_outcome)
                    in_stop = False   # outcome bulununca stop sona erer
                    log.debug("    Outcome (in_stop): %s", code)
            continue

        # ── Kazanım başlığı (heading veya düz metin) ─────────────────────
        if current_unit is not None:
            text   = _heading_text(line) if level else line
            parsed = _parse_outcome_heading(text)
            if parsed:
                code, desc = parsed
                current_outcome = Outcome(code=code, description=desc)
                current_unit.outcomes.append(current_outcome)
                log.debug("    Outcome: %s", code)
                continue

        # ── Süreç bileşeni ───────────────────────────────────────────────
        if not level and current_outcome is not None:
            m = _RE_COMP.match(line)
            if m:
                current_outcome.components.append(
                    Component(code=m.group(1), description=m.group(2).strip())
                )

    # Grade'leri sırala
    subject.grades.sort(key=lambda g: int(g.level_name.split(".")[0]))
    return subject





# ── Parser — 2. Geçiş: Implementation Guide ──────────────────────────────────

def extract_implementation_guides(md_path: str, subject: Subject) -> None:
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()

    outcome_map: dict[str, Outcome] = {}
    for grade in subject.grades:
        for unit in grade.units:
            for outcome in unit.outcomes:
                outcome_map[outcome.code] = outcome

    in_section = False
    in_stop    = False
    current_oc: Outcome | None = None
    buffer: list[str] = []

    def flush():
        nonlocal buffer, current_oc
        if current_oc and buffer:
            current_oc.implementation_guide = "\n".join(buffer).strip()
        buffer = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        level = _is_heading(line)
        text  = _heading_text(line) if level else line

        if level and ("yaşantıları" in text.lower() or "uygulamaları" in text.lower()) and ("öğretme" in text.lower() or "öğrenme" in text.lower()):
            flush(); in_section = True; in_stop = False; current_oc = None
            continue

        if not in_section:
            continue

        # Stop
        if level and _is_guide_stop(text.lower()):
            flush(); in_stop = True; current_oc = None
            continue

        # Yeni tema → bölüm bitti
        if level and _RE_UNIT.match(text):
            flush(); in_section = False
            continue

        if in_stop:
            continue

        # Kazanım referans başlığı (SADECE KOD, açıklama yok)
        if level:
            parts = text.split(None, 1)
            if len(parts) == 1:  # sadece tek kelime = kod
                candidate = parts[0].rstrip(".")
                segs = candidate.split(".")
                if len(segs) >= 3 and all(s.isdigit() for s in segs[1:3]):
                    flush()
                    in_stop    = False
                    current_oc = outcome_map.get(candidate)
                    continue

        # Metin satırı
        if not level and current_oc:
            buffer.append(line)

    flush()


# ── Veritabanı ───────────────────────────────────────────────────────────────

def delete_subject(cur, name):
    cur.execute("SELECT id FROM subjects WHERE name = ?", (name,))
    row = cur.fetchone()
    if not row:
        return
    sid = row[0]
    cur.execute("PRAGMA foreign_keys = OFF")
    cur.execute("""DELETE FROM components WHERE outcome_id IN (
        SELECT o.id FROM outcomes o JOIN units u ON o.unit_id=u.id
        JOIN grades g ON u.grade_id=g.id WHERE g.subject_id=?)""", (sid,))
    cur.execute("""DELETE FROM outcomes WHERE unit_id IN (
        SELECT u.id FROM units u JOIN grades g ON u.grade_id=g.id
        WHERE g.subject_id=?)""", (sid,))
    cur.execute("DELETE FROM units WHERE grade_id IN (SELECT id FROM grades WHERE subject_id=?)", (sid,))
    cur.execute("DELETE FROM grades WHERE subject_id=?", (sid,))
    cur.execute("DELETE FROM subjects WHERE id=?", (sid,))
    cur.execute("PRAGMA foreign_keys = ON")


def insert_subject(cur, subject):
    cur.execute("INSERT INTO subjects (name) VALUES (?)", (subject.name,))
    subject_id = cur.lastrowid
    log.info("Subject '%s' (id=%d) yazılıyor...", subject.name, subject_id)
    total_g = total_u = total_o = total_c = 0

    for grade in subject.grades:
        cur.execute("INSERT INTO grades (level_name, subject_id) VALUES (?,?)",
                    (grade.level_name, subject_id))
        grade_id = cur.lastrowid; total_g += 1

        for unit in grade.units:
            cur.execute("INSERT INTO units (name, unit_no, grade_id) VALUES (?,?,?)",
                        (unit.name, unit.unit_no, grade_id))
            unit_id = cur.lastrowid; total_u += 1

            for outcome in unit.outcomes:
                cur.execute("""INSERT OR IGNORE INTO outcomes
                    (code, description, implementation_guide, unit_id)
                    VALUES (?,?,?,?)""",
                    (outcome.code, outcome.description,
                     outcome.implementation_guide, unit_id))
                oid = cur.lastrowid or None
                if not oid:
                    cur.execute("SELECT id FROM outcomes WHERE code=? AND unit_id=?",
                                (outcome.code, unit_id))
                    r = cur.fetchone()
                    oid = r[0] if r else None
                if oid:
                    total_o += 1
                    for comp in outcome.components:
                        cur.execute("INSERT INTO components (code, description, outcome_id) VALUES (?,?,?)",
                                    (comp.code, comp.description, oid))
                        total_c += 1

    log.info("  ✓ %d grade | %d unit | %d outcome | %d component",
             total_g, total_u, total_o, total_c)


# ── CLI ─────────────────────────────────────────────────────────────────────

def list_md_files():
    return sorted(glob.glob(f"{MD_DIR}/*.md"))


def main():
    parser = argparse.ArgumentParser(description="Markdown müfredat → SQLite ETL")
    parser.add_argument("files", nargs="*", metavar="MD")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        selected = list_md_files()
        if not selected:
            sys.exit(f"'{MD_DIR}/' klasöründe .md dosyası bulunamadı.")
    elif args.files:
        selected = args.files
    else:
        files = list_md_files()
        if not files:
            sys.exit(f"'{MD_DIR}/' klasöründe .md dosyası bulunamadı.")
        print("\nMevcut MD dosyaları:")
        for i, f in enumerate(files, 1):
            print(f"  [{i:>2}] {Path(f).name}")
        print("  [ 0] Hepsini seç")
        raw = input("\nNumara(lar): ").strip()
        selected = (files if raw == "0" else
                    [files[int(t.strip()) - 1] for t in raw.split(",")
                     if t.strip().isdigit() and 1 <= int(t.strip()) <= len(files)])

    if not selected:
        sys.exit("Geçerli seçim yok.")
    if not os.path.exists(DB_NAME):
        sys.exit(f"DB bulunamadı: {DB_NAME}. Önce: python db_setup.py")

    conn = sqlite3.connect(DB_NAME)
    cur  = conn.cursor()
    ok = fail = 0

    for md_path in selected:
        log.info("İşleniyor: %s", md_path)
        try:
            subject = parse_md(md_path)
            extract_implementation_guides(md_path, subject)

            if args.dry_run:
                g = len(subject.grades)
                u = sum(len(gr.units) for gr in subject.grades)
                o = sum(len(un.outcomes) for gr in subject.grades for un in gr.units)
                c = sum(len(oc.components) for gr in subject.grades
                        for un in gr.units for oc in un.outcomes)
                log.info("[DRY-RUN] %s → %d grade | %d unit | %d outcome | %d component",
                         subject.name, g, u, o, c)
                for gr in subject.grades:
                    gu = len(gr.units)
                    go = sum(len(u.outcomes) for u in gr.units)
                    gc = sum(len(o.components) for u in gr.units for o in u.outcomes)
                    log.info("         %s: %d unit | %d outcome | %d component",
                             gr.level_name, gu, go, gc)
            else:
                delete_subject(cur, subject.name)
                insert_subject(cur, subject)
                conn.commit()
            ok += 1
        except Exception as e:
            log.error("HATA [%s]: %s", md_path, e, exc_info=True)
            conn.rollback()
            fail += 1

    conn.close()
    print(f"\n{'='*50}")
    print(f"Tamamlandı: {ok} başarılı, {fail} başarısız")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
