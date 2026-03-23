"""
Generation - take retrieved chunks + user query, produce a grounded answer.

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
#
# The prompt is structured to work well with smaller models like Mistral 7B:
# - Numbered rules are easier for small models to follow than prose
# - Explicit "DO" and "DO NOT" framing reduces ambiguity
# - The output format section gives the model a clear template

SYSTEM_PROMPT = """\
You are a Catholic theological study assistant. You answer questions using ONLY \
the retrieved passages provided in the user message. You are a curator of these \
texts, not a theologian; let the documents speak.

RULES:
1. Use ONLY information from the RETRIEVED CONTEXT below. Do not add outside knowledge.
2. Cite every claim with its source: "CCC §[number]" for Catechism, standard verse \
   notation (e.g., "Romans 5:8") for Scripture.
3. When multiple retrieved passages address the question, SYNTHESIZE them into a \
   coherent answer. Connect related teachings across sources.
4. If a passage contains a cross-reference to another CCC paragraph or Scripture \
   verse, mention it to help the user study further.
5. If the retrieved passages do not contain enough information, say so clearly and \
   suggest which CCC sections or Scripture books might help.
6. Do NOT offer personal opinions, speculate, or interpret beyond what the sources state.

OUTPUT FORMAT:
- Start with a direct answer to the question (1-2 sentences).
- Then provide supporting detail from the sources with citations.
- End with any relevant cross-references for further study.\
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
        para_num = chunk.metadata.get("paragraph_number", "")

        # Build a clear header so the LLM knows how to cite this chunk
        header_parts = [f"Source {i}"]
        if doc_type:
            header_parts.append(doc_type.upper())
        if ref:
            header_parts.append(ref)
        elif para_num:
            header_parts.append(f"CCC §{para_num}")
        header_parts.append(f"from {source}")

        # Include score info so the LLM can weight sources
        if chunk.rerank_score is not None:
            header_parts.append(f"relevance: {chunk.rerank_score:.2f}")
        else:
            header_parts.append(f"relevance: {chunk.score:.2f}")

        header = " | ".join(header_parts)
        context_parts.append(f"[{header}]\n{chunk.text}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the user-turn prompt with context and query."""
    context = format_context(chunks)

    return f"""\
RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}

Using ONLY the retrieved context above, provide a clear answer with specific citations \
(CCC § numbers and Scripture verses). If multiple sources are relevant, synthesize them.\
"""


def generate(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """
    Generate an answer using the LLM.

    Returns:
        {
            "answer": str,          - the generated response
            "chunks_used": int,     - number of chunks in context
            "abstained": bool,      - True if context was insufficient
            "sources": list[dict],  - metadata of chunks used
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