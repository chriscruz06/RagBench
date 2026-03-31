import fitz  # pymupdf - much better PDF text extraction

doc = fitz.open("data/raw/catechism/ccc_raw.pdf")
print(f"Pages: {len(doc)}")

text_parts = []
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        text_parts.append(text)
    if i % 100 == 0:
        print(f"  Extracted page {i}...")

full_text = "\n".join(text_parts)
print(f"Total characters: {len(full_text):,}")
print(f"\nPreview:\n{full_text[5000:5500]}")

with open("data/raw/catechism/ccc_raw.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("\nSaved to data/raw/catechism/ccc_raw.txt")