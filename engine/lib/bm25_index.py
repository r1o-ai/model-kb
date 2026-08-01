"""
BM25 Sparse Retrieval Index

Provides keyword-based search using BM25 (Best Match 25) algorithm.
Complements semantic embeddings for hybrid retrieval.

Usage:
    from lib.bm25_index import BM25Index

    index = BM25Index()
    index.build(records, text_field="text")
    results = index.search("query text", top_k=10)
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 not available. Install: pip install rank-bm25")


@dataclass
class BM25Result:
    """Result from BM25 search"""
    record_id: Any
    text: str
    score: float
    rank: int
    record: Dict[str, Any]


class BM25Index:
    """
    BM25 index for sparse (keyword-based) retrieval.

    Provides fast keyword matching to complement dense (embedding) search.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize BM25 index.

        Args:
            tokenizer: Optional custom tokenizer function.
                       Default: lowercase + whitespace split + min 2 chars
        """
        if not HAS_BM25:
            raise ImportError("rank_bm25 required. Install: pip install rank-bm25")

        self.tokenizer = tokenizer or self._default_tokenize
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self.record_ids: List[Any] = []
        self.records: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenizer: lowercase, split, filter short tokens"""
        if not text:
            return []
        # Remove special characters, lowercase, split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) >= 2]

    def build(
        self,
        records: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        text_extractor: Optional[Callable[[Dict], str]] = None
    ) -> int:
        """
        Build BM25 index from records.

        Args:
            records: List of record dictionaries
            id_field: Field name for record ID
            text_field: Field name for text (if text_extractor not provided)
            text_extractor: Optional function to extract search text from record

        Returns:
            Number of records indexed
        """
        self.corpus = []
        self.record_ids = []
        self.records = []
        self.texts = []

        for record in records:
            record_id = record.get(id_field)
            if record_id is None:
                continue

            # Extract text
            if text_extractor:
                text = text_extractor(record)
            else:
                text = record.get(text_field, "")

            if not text:
                continue

            tokens = self.tokenizer(text)
            if not tokens:
                continue

            self.corpus.append(tokens)
            self.record_ids.append(record_id)
            self.records.append(record)
            self.texts.append(text)

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            logger.info(f"Built BM25 index with {len(self.corpus)} records")

        return len(self.corpus)

    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """
        Search for records using BM25.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of BM25Result sorted by score descending
        """
        if not self.bm25:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices by score
        indexed_scores = [(i, scores[i]) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
            if score > 0:
                results.append(BM25Result(
                    record_id=self.record_ids[idx],
                    text=self.texts[idx],
                    score=float(score),
                    rank=rank,
                    record=self.records[idx]
                ))

        return results

    def get_scores(self, query: str) -> Dict[Any, float]:
        """
        Get BM25 scores for all records.

        Args:
            query: Search query string

        Returns:
            Dict mapping record_id to BM25 score
        """
        if not self.bm25:
            return {}

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return {}

        scores = self.bm25.get_scores(query_tokens)

        return {
            self.record_ids[i]: float(scores[i])
            for i in range(len(scores))
            if scores[i] > 0
        }

    def get_ranked_list(
        self,
        query: str,
        top_k: int = 50
    ) -> List[Tuple[Any, str, float]]:
        """
        Get ranked list for RRF fusion.

        Args:
            query: Search query string
            top_k: Number of results

        Returns:
            List of (record_id, text, score) tuples sorted by score descending
        """
        results = self.search(query, top_k=top_k)
        return [(r.record_id, r.text, r.score) for r in results]

    def get_record(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """Get record by ID"""
        try:
            idx = self.record_ids.index(record_id)
            return self.records[idx]
        except ValueError:
            return None

    def __len__(self) -> int:
        """Return number of indexed records"""
        return len(self.corpus)
