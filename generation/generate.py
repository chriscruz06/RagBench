"""
Generation — take retrieved chunks + user query, produce a grounded answer.

This module enforces the "curator, not theologian" principle:
- System prompt constrains the LLM to only use retrieved context
- Citation enforcement requires specific references (CCC §, scripture verse)
- Abstention when context is insufficient
"""

from retrieval.search import RetrievedChunk
from config.settings import settings


# ── System Prompt ─────────────────────────────────────────────────
# This is "Layer 1" of the hallucination defense. The prompt is designed
# to make the LLM a careful curator of retrieved content, not a freelance
# theologian.

SYSTEM_PROMPT = """\
You are a Catholic theological study assistant. Your SOLE purpose is to help \
users understand Catholic teaching by presenting and organizing content retrieved \
from the Catechism of the Catholic Church (CCC) and Sacred Scripture.

STRICT RULES:
1. Answer ONLY based on the retrieved passages provided below. Do not use any \
   knowledge beyond what is in the provided context.
2. Every claim must cite a specific source: use "CCC §[number]" for Catechism \
   paragraphs and standard verse notation (e.g., "Romans 5:8") for Scripture.
3. If the retrieved passages do not contain sufficient information to answer \
   the question, say: "The retrieved sources do not directly address this \
   question. You may want to consult [suggest relevant CCC section or \
   Scripture book]."
4. Do NOT synthesize, speculate, or interpolate beyond what the sources state.
5. Do NOT offer personal theological opinions or interpretations.
6. Present the teaching as the sources present it — faithfully and accurately.
7. When sources contain cross-references, mention them to help the user study further.

You are a curator of these texts, not a theologian. Let the documents speak.\
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not chunks:
        return "(No relevant passages were retrieved.)"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "Unknown")
        doc_type = chunk.metadata.get("doc_type", "")
        ref = chunk.metadata.get("reference", "")

        header = f"[Source {i}: {doc_type.upper()}"
        if ref:
            header += f" — {ref}"
        header += f" | from {source} | relevance: {chunk.score:.2f}]"

        context_parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user-turn prompt with context and query."""
    context = format_context(chunks)

    return f"""\
RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}

Provide a clear, well-cited answer based ONLY on the retrieved context above.\
"""


def generate(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """
    Generate an answer using the LLM.

    Returns:
        {
            "answer": str,          — the generated response
            "chunks_used": int,     — number of chunks in context
            "abstained": bool,      — True if context was insufficient
            "sources": list[dict],  — metadata of chunks used
        }
    """
    # ── Abstention check (Layer 4) ───────────────────────────
    if not chunks:
        return {
            "answer": (
                "I don't have sufficient source material to answer this question "
                "confidently. Try rephrasing your question or asking about a topic "
                "covered in the Catechism of the Catholic Church."
            ),
            "chunks_used": 0,
            "abstained": True,
            "sources": [],
        }

    prompt = build_prompt(query, chunks)

    # ── Call the LLM ─────────────────────────────────────────
    if settings.llm_provider == "ollama":
        answer = _call_ollama(prompt)
    elif settings.llm_provider == "openai":
        answer = _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    return {
        "answer": answer,
        "chunks_used": len(chunks),
        "abstained": False,
        "sources": [c.metadata for c in chunks],
    }


def _call_ollama(prompt: str) -> str:
    """Call local Ollama instance."""
    import ollama

    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": settings.temperature,
            "num_predict": settings.max_response_tokens,
        },
    )
    return response["message"]["content"]


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_response_tokens,
    )
    return response.choices[0].message.content
