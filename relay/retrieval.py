"""Retrieval over the chunk index (BM25 first; swappable)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    paper_id: str
    version: str
    page: int
    section: str
    chunk_id: str
    text: str
    token_count: int | None = None

    def citation(self) -> str:
        return f"[{self.paper_id}, {self.page}, {self.chunk_id}]"


class BaseRetriever(Protocol):
    async def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class NoOpReranker:
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks


class OverlapReranker:
    """Simple term-overlap reranker."""

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        qterms = set(query.lower().split())
        def score(c: Chunk) -> float:
            cterms = set(c.text.lower().split())
            if not cterms:
                return 0.0
            return len(qterms & cterms) / len(qterms | cterms)
        return sorted(chunks, key=score, reverse=True)


class LLMReranker:
    """LLM-based reranker (uses model client)."""

    def __init__(self, client, scheduler=None):
        self.client = client
        self.scheduler = scheduler

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        # ask model to rank chunk ids by relevance
        ids = [c.chunk_id for c in chunks]
        passages = "\n".join(f"{c.chunk_id}: {c.text[:200]}" for c in chunks)
        messages = [
            {"role": "system", "content": "Rank the passages by relevance to the query. Return JSON: {\"ranking\": [chunk_id, ...]} (most relevant first)."},
            {"role": "user", "content": f"Query: {query}\n\nPassages:\n{passages}"},
        ]
        try:
            if self.scheduler is not None:
                from relay.trace import Trace
                step = None
                # use a dummy step? We'll call client directly to avoid trace requirement
                resp = await self.client.stream(messages, temperature=0.0, max_tokens=256)
            else:
                resp = await self.client.stream(messages, temperature=0.0, max_tokens=256)
            import json, re
            m = re.search(r"\{.*\}", resp.text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            ranking = data.get("ranking", [])
            id_to_chunk = {c.chunk_id: c for c in chunks}
            ordered = [id_to_chunk[cid] for cid in ranking if cid in id_to_chunk]
            # append any missing
            for c in chunks:
                if c not in ordered:
                    ordered.append(c)
            return ordered
        except Exception:
            return chunks


class BM25Retriever:
    def __init__(self, chunks_path: str | Path, reranker: Reranker | None = None) -> None:
        self.chunks_path = Path(chunks_path)
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.reranker = reranker or NoOpReranker()
        self._load()

    def _load(self) -> None:
        with open(self.chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.chunks.append(
                    Chunk(
                        paper_id=obj["paper_id"],
                        version=obj["version"],
                        page=obj["page"],
                        section=obj["section"],
                        chunk_id=obj["chunk_id"],
                        text=obj["text"],
                        token_count=obj.get("token_count"),
                    )
                )
        tokenized = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    async def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if not self.chunks or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        top = [self.chunks[i] for i in ranked[:k]]
        top = await self.reranker.rerank(query, top)
        return top
