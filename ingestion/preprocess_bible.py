"""
Preprocess the Douay-Rheims Bible from Project Gutenberg.

What this does:
1. Strips Gutenberg header/footer boilerplate
2. Splits into individual books
3. Preserves chapter:verse references in metadata
4. Outputs clean text files, one per book, into data/processed/bible/

Usage:
    python -m ingestion.preprocess_bible

Input:  data/raw/bible/douay_rheims_raw.txt
Output: data/processed/bible/<book_name>.txt
        data/processed/bible/manifest.json  (book list + stats)
"""

import re
import json
from pathlib import Path


RAW_PATH = Path("data/raw/bible/douay_rheims_raw.txt")
OUTPUT_DIR = Path("data/processed/bible")


# ── Known book names in the Douay-Rheims ──────────────────────────
# Gutenberg uses these as section headers
BOOK_PATTERNS = [
    # Old Testament
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Josue", "Judges", "Ruth",
    "1 Kings", "2 Kings", "3 Kings", "4 Kings",
    # DR uses "Kings" where modern translations use "Samuel" and "Kings"
    "1 Paralipomenon", "2 Paralipomenon",  # = 1-2 Chronicles
    "1 Esdras", "2 Esdras",  # = Ezra, Nehemiah
    "Tobias", "Judith", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Canticle of Canticles",
    "Wisdom", "Ecclesiasticus",  # = Sirach
    "Isaias", "Jeremias", "Lamentations", "Baruch",
    "Ezechiel", "Daniel", "Osee", "Joel", "Amos", "Abdias",
    "Jonas", "Micheas", "Nahum", "Habacuc", "Sophonias",
    "Aggeus", "Zacharias", "Malachias",
    "1 Machabees", "2 Machabees",
    # New Testament
    "The Holy Gospel Of Jesus Christ, According To Saint Matthew",
    "The Holy Gospel Of Jesus Christ, According To Saint Mark",
    "The Holy Gospel Of Jesus Christ, According To Saint Luke",
    "The Holy Gospel Of Jesus Christ, According To Saint John",
    "The Acts Of The Apostles",
    "The Epistle Of Saint Paul To The Romans",
    "The First Epistle Of Saint Paul To The Corinthians",
    "The Second Epistle Of Saint Paul To The Corinthians",
    "The Epistle Of Saint Paul To The Galatians",
    "The Epistle Of Saint Paul To The Ephesians",
    "The Epistle Of Saint Paul To The Philippians",
    "The Epistle Of Saint Paul To The Colossians",
    "The First Epistle Of Saint Paul To The Thessalonians",
    "The Second Epistle Of Saint Paul To The Thessalonians",
    "The First Epistle Of Saint Paul To Timothy",
    "The Second Epistle Of Saint Paul To Timothy",
    "The Epistle Of Saint Paul To Titus",
    "The Epistle Of Saint Paul To Philemon",
    "The Epistle Of Saint Paul To The Hebrews",
    "The Catholic Epistle Of Saint James The Apostle",
    "The First Epistle Of Saint Peter The Apostle",
    "The Second Epistle Of Saint Peter The Apostle",
    "The First Epistle Of Saint John The Apostle",
    "The Second Epistle Of Saint John The Apostle",
    "The Third Epistle Of Saint John The Apostle",
    "The Catholic Epistle Of Saint Jude The Apostle",
    "The Apocalypse Of Saint John The Apostle",
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find start of actual content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def split_into_books(text: str) -> dict[str, str]:
    """
    Split the Bible text into individual books.

    Returns a dict of {book_name: book_text}.
    This is a best-effort parser — Gutenberg formatting varies.
    """
    books = {}

    # Try to find book boundaries by looking for known book names
    # as standalone lines (usually in all caps or title case)
    lines = text.split("\n")
    current_book = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        # Check if this line is a book header
        matched_book = None
        for book in BOOK_PATTERNS:
            # Match case-insensitively, allow some variation
            if stripped.upper() == book.upper() or stripped.upper().startswith(book.upper()):
                matched_book = book
                break

        if matched_book:
            # Save previous book
            if current_book and current_lines:
                books[current_book] = "\n".join(current_lines).strip()

            current_book = matched_book
            current_lines = []
        elif current_book:
            current_lines.append(line)

    # Don't forget the last book
    if current_book and current_lines:
        books[current_book] = "\n".join(current_lines).strip()

    return books


def clean_book_name(name: str) -> str:
    """Convert a book name to a clean filename."""
    # Shorten long NT names
    short = name
    short = re.sub(r"The Holy Gospel.*According To Saint ", "Gospel of ", short)
    short = re.sub(r"The Acts Of The Apostles", "Acts", short)
    short = re.sub(r"The (?:First |Second |Third )?(?:Catholic )?Epistle Of Saint Paul To (?:The )?", "", short)
    short = re.sub(r"The (?:First |Second |Third )?(?:Catholic )?Epistle Of Saint ", "", short)
    short = re.sub(r" The Apostle", "", short)

    # Handle numbered epistles
    if "First" in name:
        short = "1 " + short
    elif "Second" in name:
        short = "2 " + short
    elif "Third" in name:
        short = "3 " + short

    short = re.sub(r"The Apocalypse.*", "Revelation", short)

    # Clean up
    short = short.strip()
    safe = re.sub(r"[^\w\s-]", "", short)
    safe = re.sub(r"\s+", "_", safe)
    return safe.lower()


def preprocess():
    """Main preprocessing pipeline."""
    if not RAW_PATH.exists():
        print(f"[ERROR] Raw Bible file not found at {RAW_PATH}")
        print(f"  Download from: https://www.gutenberg.org/cache/epub/8300/pg8300.txt")
        print(f"  Save as: {RAW_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[bible] Reading {RAW_PATH}...")
    raw_text = RAW_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"  Raw file: {len(raw_text):,} characters")

    # Step 1: Strip Gutenberg boilerplate
    clean_text = strip_gutenberg_boilerplate(raw_text)
    print(f"  After stripping boilerplate: {len(clean_text):,} characters")

    # Step 2: Split into books
    books = split_into_books(clean_text)
    print(f"  Found {len(books)} books")

    if not books:
        # Fallback: if book splitting fails, just save the whole thing
        print("  [WARNING] Book splitting failed — saving as single file")
        output_path = OUTPUT_DIR / "full_bible.txt"
        output_path.write_text(clean_text, encoding="utf-8")
        print(f"  Saved: {output_path}")
        return

    # Step 3: Save each book as a separate file
    manifest = []
    for book_name, book_text in books.items():
        filename = clean_book_name(book_name) + ".txt"
        output_path = OUTPUT_DIR / filename

        output_path.write_text(book_text, encoding="utf-8")

        entry = {
            "original_name": book_name,
            "filename": filename,
            "characters": len(book_text),
            "lines": book_text.count("\n") + 1,
        }
        manifest.append(entry)
        print(f"  {filename}: {entry['characters']:,} chars, {entry['lines']} lines")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[bible] Done! {len(books)} books saved to {OUTPUT_DIR}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    preprocess()
