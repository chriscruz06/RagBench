"""
Preprocess the Catechism of the Catholic Church.

What this does:
1. Parses the CCC text into numbered paragraphs (§1 through §2865)
2. Filters out footnotes (short citation lines that also start with numbers)
3. Extracts scripture cross-references embedded in each paragraph
4. Preserves section/chapter structure in metadata
5. Outputs clean JSON with one entry per paragraph

The key challenge: both CCC paragraphs and footnotes start with numbers.
We distinguish them by:
- Real paragraphs are longer (100+ chars of theological content)
- Footnotes are short and contain citation markers (Cf., AAS, vol., etc.)
- Each real paragraph number (1-2865) appears exactly once

Usage:
    python -m ingestion.preprocess_ccc

Input:  data/raw/catechism/ccc_raw.txt
Output: data/processed/catechism/ccc_paragraphs.json
        data/processed/catechism/ccc_sections.json (table of contents)
"""

import re
import json
from pathlib import Path


RAW_PATH = Path("data/raw/catechism/ccc_raw.txt")
OUTPUT_DIR = Path("data/processed/catechism")


# ── Scripture reference patterns ──────────────────────────────────
SCRIPTURE_REF_PATTERN = re.compile(
    r'\b('
    r'(?:1|2|3|I|II|III)?\s*'
    r'(?:Gen|Ex|Lev|Num|Deut|Josh|Judg|Ruth|Sam|Kgs|Chr|Ezra|Neh|'
    r'Tob|Jdt|Esth|Job|Ps|Prov|Eccl|Song|Wis|Sir|Isa|Jer|Lam|Bar|'
    r'Ezek|Dan|Hos|Joel|Am|Obad|Jon|Mic|Nah|Hab|Zeph|Hag|Zech|Mal|'
    r'Mac|Mt|Mk|Lk|Jn|Acts|Rom|Cor|Gal|Eph|Phil|Col|Thess|Tim|Tit|'
    r'Phlm|Heb|Jas|Pet|Jude|Rev)'
    r'\.?\s*'
    r'\d+(?::\d+(?:-\d+)?)?'
    r'(?:,\s*\d+(?:-\d+)?)*'
    r')',
    re.IGNORECASE
)

# ── Footnote detection patterns ───────────────────────────────────
# Footnotes typically contain these markers
FOOTNOTE_MARKERS = re.compile(
    r'\b(Cf\.|cf\.|AAS|vol\.|pp?\.\s*\d|ibid|op\.\s*cit|loc\.\s*cit'
    r'|Denzinger|PL\s+\d|PG\s+\d|CSEL|CCL|SC\s+\d'
    r'|Encyclical|Apostolic|Discourse|Homily|Catechesis'
    r'|L\'Osservatore|Enchiridion|DS\s+\d'
    r'|Vatican\s+Council|Council\s+of\s+Trent)',
    re.IGNORECASE
)

# ── CCC paragraph number pattern ─────────────────────────────────
PARAGRAPH_NUM_PATTERN = re.compile(r'^(\d{1,4})\s+(.+)', re.DOTALL)


def is_footnote(text: str) -> bool:
    """
    Determine if a text block is a footnote rather than a real CCC paragraph.

    Heuristics:
    - Footnotes are usually short (under 200 chars)
    - Footnotes contain citation markers (Cf., AAS, vol., pp., etc.)
    - Real paragraphs contain theological content and are longer
    """
    # Very short texts are almost certainly footnotes or fragments
    if len(text) < 80:
        return True

    # Check for footnote citation markers
    if FOOTNOTE_MARKERS.search(text):
        # If it's short AND has citation markers, definitely a footnote
        if len(text) < 300:
            return True
        # Longer texts with citation markers might be real paragraphs
        # that quote sources — check if the citation is just a small part
        marker_matches = len(FOOTNOTE_MARKERS.findall(text))
        words = len(text.split())
        # If more than 30% of content is citations, it's a footnote
        if marker_matches > 2 and words < 50:
            return True

    return False


def parse_paragraphs(text: str) -> list[dict]:
    """
    Parse CCC text into numbered paragraphs, filtering out footnotes.

    Strategy:
    1. Find all lines starting with a number (1-2865)
    2. Collect text until the next numbered line
    3. Filter out footnotes using heuristics
    4. Deduplicate: if a paragraph number appears multiple times,
       keep the longest one (real paragraphs are longer than footnotes)
    """
    lines = text.split("\n")
    raw_paragraphs = []
    current_para_num = None
    current_text_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for paragraph number at start of line
        match = PARAGRAPH_NUM_PATTERN.match(stripped)

        if match:
            num = int(match.group(1))
            rest = match.group(2).strip()

            # CCC paragraphs are 1-2865
            if 1 <= num <= 2865:
                # Save previous paragraph
                if current_para_num is not None:
                    full_text = " ".join(current_text_lines).strip()
                    if full_text:
                        raw_paragraphs.append({
                            "paragraph_number": current_para_num,
                            "text": full_text,
                        })

                current_para_num = num
                current_text_lines = [rest]
                continue

        # Otherwise, append to current paragraph
        if current_para_num is not None:
            current_text_lines.append(stripped)

    # Don't forget the last paragraph
    if current_para_num is not None:
        full_text = " ".join(current_text_lines).strip()
        if full_text:
            raw_paragraphs.append({
                "paragraph_number": current_para_num,
                "text": full_text,
            })

    print(f"  Raw parsed blocks: {len(raw_paragraphs)}")

    # ── Filter footnotes ─────────────────────────────────────
    filtered = [p for p in raw_paragraphs if not is_footnote(p["text"])]
    print(f"  After footnote filtering: {len(filtered)}")

    # ── Deduplicate: keep longest text for each paragraph number ─
    best = {}
    for para in filtered:
        num = para["paragraph_number"]
        if num not in best or len(para["text"]) > len(best[num]["text"]):
            best[num] = para

    paragraphs = sorted(best.values(), key=lambda p: p["paragraph_number"])
    print(f"  After deduplication: {len(paragraphs)}")

    # Add reference field
    for para in paragraphs:
        para["reference"] = f"CCC §{para['paragraph_number']}"

    return paragraphs


def extract_scripture_refs(text: str) -> list[str]:
    """Extract scripture cross-references from a paragraph."""
    matches = SCRIPTURE_REF_PATTERN.findall(text)
    seen = set()
    refs = []
    for ref in matches:
        ref_clean = ref.strip()
        if ref_clean not in seen:
            seen.add(ref_clean)
            refs.append(ref_clean)
    return refs


def enrich_paragraphs(paragraphs: list[dict]) -> list[dict]:
    """Add scripture cross-references and section info to each paragraph."""
    for para in paragraphs:
        para["scripture_refs"] = extract_scripture_refs(para["text"])
        para["has_scripture_refs"] = len(para["scripture_refs"]) > 0
    return paragraphs


def build_section_index(paragraphs: list[dict]) -> list[dict]:
    """
    Build a rough section index based on paragraph number ranges.
    These are the four main "Parts" of the CCC.
    """
    sections = [
        {"part": 1, "title": "The Profession of Faith", "start": 1, "end": 1065},
        {"part": 2, "title": "The Celebration of the Christian Mystery", "start": 1066, "end": 1690},
        {"part": 3, "title": "Life in Christ", "start": 1691, "end": 2557},
        {"part": 4, "title": "Christian Prayer", "start": 2558, "end": 2865},
    ]

    for section in sections:
        section["paragraph_count"] = len([
            p for p in paragraphs
            if section["start"] <= p["paragraph_number"] <= section["end"]
        ])

    return sections


def preprocess():
    """Main preprocessing pipeline."""
    if not RAW_PATH.exists():
        print(f"[ERROR] Raw CCC file not found at {RAW_PATH}")
        print()
        print("  To get the CCC text, you have a few options:")
        print("  1. Copy-paste from: https://www.vatican.va/archive/ENG0015/_INDEX.HTM")
        print("  2. Search for 'Catechism of the Catholic Church full text'")
        print("  3. Use the USCCB flipbook and copy text from there")
        print()
        print(f"  Save the full text as: {RAW_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ccc] Reading {RAW_PATH}...")
    raw_text = RAW_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"  Raw file: {len(raw_text):,} characters")

    # Step 1: Parse into paragraphs
    paragraphs = parse_paragraphs(raw_text)

    if not paragraphs:
        print("  [WARNING] No paragraphs found!")
        return

    # Step 2: Enrich with scripture references
    paragraphs = enrich_paragraphs(paragraphs)
    paras_with_refs = sum(1 for p in paragraphs if p["has_scripture_refs"])
    total_refs = sum(len(p["scripture_refs"]) for p in paragraphs)
    print(f"  Scripture cross-references: {total_refs} refs across {paras_with_refs} paragraphs")

    # Step 3: Save paragraphs
    para_path = OUTPUT_DIR / "ccc_paragraphs.json"
    with open(para_path, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {para_path}")

    # Step 4: Build and save section index
    sections = build_section_index(paragraphs)
    section_path = OUTPUT_DIR / "ccc_sections.json"
    with open(section_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    print(f"  Saved: {section_path}")

    # Summary
    print(f"\n[ccc] Done!")
    print(f"  Paragraphs: {len(paragraphs)}")
    print(f"  Coverage: §{paragraphs[0]['paragraph_number']} – §{paragraphs[-1]['paragraph_number']}")
    for section in sections:
        print(f"  Part {section['part']}: {section['title']} ({section['paragraph_count']} paragraphs)")


if __name__ == "__main__":
    preprocess()