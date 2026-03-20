"""
Preprocess the Catechism of the Catholic Church.

What this does:
1. Parses the CCC text into numbered paragraphs (§1 through §2865)
2. Extracts scripture cross-references embedded in each paragraph
3. Preserves section/chapter structure in metadata
4. Outputs clean JSON with one entry per paragraph

This is important because:
- Each CCC paragraph is a natural "chunk" for retrieval
- Paragraph numbers become citation references (CCC §1234)
- Scripture cross-refs become ground truth for eval triples

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
# Matches things like "Gen 1:1", "Mt 28:19", "1 Cor 13:4-7", "Jn 3:16"
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

# ── CCC paragraph number pattern ─────────────────────────────────
# Matches "123 " or "1234 " at the start of a line (paragraph numbers)
PARAGRAPH_NUM_PATTERN = re.compile(r'^(\d{1,4})\s+(.+)', re.DOTALL)

# Also matches "§123" or "¶123" style
PARAGRAPH_SYMBOL_PATTERN = re.compile(r'^[§¶]\s*(\d{1,4})\s+(.+)', re.DOTALL)


def parse_paragraphs(text: str) -> list[dict]:
    """
    Parse CCC text into numbered paragraphs.

    Tries multiple strategies since the raw text format varies:
    1. Look for paragraph numbers at line starts
    2. Look for § symbols
    3. Fall back to splitting on blank lines
    """
    paragraphs = []

    # Strategy 1: Split on paragraph numbers
    # CCC paragraphs start with their number (e.g., "1234 The Church teaches...")
    lines = text.split("\n")
    current_para_num = None
    current_text_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for paragraph number at start of line
        match = PARAGRAPH_NUM_PATTERN.match(stripped)
        if not match:
            match = PARAGRAPH_SYMBOL_PATTERN.match(stripped)

        if match:
            num = int(match.group(1))
            rest = match.group(2).strip()

            # Sanity check: CCC paragraphs are 1-2865
            if 1 <= num <= 2900:
                # Save previous paragraph
                if current_para_num is not None:
                    full_text = " ".join(current_text_lines).strip()
                    if full_text:
                        paragraphs.append({
                            "paragraph_number": current_para_num,
                            "text": full_text,
                            "reference": f"CCC §{current_para_num}",
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
            paragraphs.append({
                "paragraph_number": current_para_num,
                "text": full_text,
                "reference": f"CCC §{current_para_num}",
            })

    return paragraphs


def extract_scripture_refs(text: str) -> list[str]:
    """Extract scripture cross-references from a paragraph."""
    matches = SCRIPTURE_REF_PATTERN.findall(text)
    # Deduplicate while preserving order
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
        print()
        print("  Format tips:")
        print("  - Paragraph numbers should appear at the start of lines")
        print("  - e.g., '1234 The Church teaches that...'")
        print("  - The script handles both '1234 ...' and '§1234 ...' formats")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ccc] Reading {RAW_PATH}...")
    raw_text = RAW_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"  Raw file: {len(raw_text):,} characters")

    # Step 1: Parse into paragraphs
    paragraphs = parse_paragraphs(raw_text)
    print(f"  Parsed {len(paragraphs)} paragraphs")

    if not paragraphs:
        print("  [WARNING] No numbered paragraphs found!")
        print("  The raw text might not have paragraph numbers at line starts.")
        print("  Check the format of your ccc_raw.txt file.")
        print()
        print("  Expected format:")
        print("  1 God, infinitely perfect and blessed in himself...")
        print("  2 In order that this call should resound throughout the world...")
        print("  ...")

        # Fallback: split on double newlines and number them
        print("\n  Falling back to paragraph splitting on blank lines...")
        blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
        paragraphs = []
        for i, block in enumerate(blocks, 1):
            if len(block) > 50:  # skip very short blocks (likely headers)
                paragraphs.append({
                    "paragraph_number": i,
                    "text": block,
                    "reference": f"block_{i}",
                })
        print(f"  Created {len(paragraphs)} blocks from fallback splitting")

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
