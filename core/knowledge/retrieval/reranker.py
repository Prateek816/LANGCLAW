"""
LLM-based re-ranker using LangChain + Groq.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """\
You are a relevance scoring assistant. Given a search query and a list of text passages, rank the passages by their relevance to the query.

Query: {query}

Passages:
{passages}

Return ONLY a valid JSON array of passage indices (0-based), ordered from most relevant to least relevant.
Example: [2, 0, 3, 1]

Your response (JSON array only):"""


class LLMReranker:
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        max_chars: int = 300,
    ) -> None:
        self._llm = ChatGroq(api_key=api_key, model=model_name)
        self._max_chars = max_chars

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        passages_text = "\n\n".join(
            f"[{i}] {c['content'][: self._max_chars]}"
            for i, c in enumerate(candidates)
        )
        prompt = _RERANK_PROMPT.format(query=query, passages=passages_text)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            match = re.search(r"\[[\d,\s]+\]", raw)
            if not match:
                raise ValueError(f"No JSON array found in: {raw!r}")
            indices: list[int] = json.loads(match.group())

            reranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            seen = set(indices)
            for i, c in enumerate(candidates):
                if i not in seen:
                    reranked.append(c)

            return reranked[:top_k]

        except Exception as exc:
            logger.warning("[LLMReranker] Reranking failed (%s), using original order.", exc)
            return candidates[:top_k]