"""Step 6: retrieve -> prompt -> LLM. This is the only class the backend needs."""
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

from . import config
from .embeddings import EmbeddingManager
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever
from .vector_store import VectorStore

import re
_FULLWIDTH_CITE = re.compile(r"【\s*(\d+)\s*】")
logger = logging.getLogger(__name__)

NO_ANSWER = (
    "I couldn't find anything relevant to that in the CCTS documents I have. "
    "Try rephrasing, or ask about a specific CCTS process such as registration, "
    "validation or monitoring."
)


def _format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape sent to the frontend. 'n' matches the [n] citations in the answer."""
    return [
        {
            "n": n,
            "source_file": c["metadata"]["source_file"],
            "page": c["metadata"]["page"],
            "score": round(c["score"], 3),
            "preview": c["content"][:200].replace("\n", " ") + "...",
        }
        for n, c in enumerate(chunks, start=1)
    ]


class RAGPipeline:
    """Create ONE instance at backend startup and reuse it (loading the
    embedding model on every request would be very slow).

    retriever / llm can be injected, which is how the tests run without
    downloading models or calling Groq.
    """

    def __init__(self, retriever: Optional[Retriever] = None, llm: Any = None):
        if retriever is None:
            retriever = Retriever(VectorStore(), EmbeddingManager())
        self.retriever = retriever
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY is not set. Put it in .env (see .env.example).")
            from langchain_groq import ChatGroq  # reads GROQ_API_KEY from the environment

            self._llm = ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
        return self._llm

    def _build_messages(self, question, chunks, history) -> list:
        messages = [("system", SYSTEM_PROMPT)]
        # history: [{"role": "user" | "assistant", "content": "..."}]
        for turn in (history or [])[-config.MAX_HISTORY_MESSAGES :]:
            role = "ai" if turn["role"] == "assistant" else "human"
            messages.append((role, turn["content"]))
        messages.append(("human", build_user_prompt(question, chunks)))
        return messages

    def query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
    ) -> Dict[str, Any]:
        """Returns {answer, sources, confidence, grounded}.
        grounded=False means nothing relevant was found and the LLM was NOT called."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")

        chunks = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        if not chunks:
            return {"answer": NO_ANSWER, "sources": [], "confidence": 0.0, "grounded": False}

        response = self.llm.invoke(self._build_messages(question, chunks, history))
        return {
            "answer":  _FULLWIDTH_CITE.sub(r"[\1]", response.content),
            "sources": _format_sources(chunks),
            "confidence": round(max(c["score"] for c in chunks), 3),
            "grounded": True,
        }

    def stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
    ) -> Iterator[Dict[str, Any]]:
        """Yields events: one {"type": "sources"}, many {"type": "token"}, one {"type": "done"}."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")

        chunks = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        if not chunks:
            yield {"type": "sources", "sources": [], "grounded": False}
            yield {"type": "token", "text": NO_ANSWER}
            yield {"type": "done"}
            return

        yield {"type": "sources", "sources": _format_sources(chunks), "grounded": True}
        for piece in self.llm.stream(self._build_messages(question, chunks, history)):
            if piece.content:
                yield {"type": "token", "text": piece.content}
        yield {"type": "done"}
