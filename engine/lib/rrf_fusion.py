"""
Reciprocal Rank Fusion (RRF)

Combines rankings from multiple retrieval methods (BM25, semantic, etc.)
using the RRF formula which requires no parameter tuning.

Formula: score(d) = Σ 1 / (k + rank_i(d))

Reference:
    Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion Outperforms
    Condorcet and Individual Rank Learning Methods" (SIGIR 2009)

Usage:
    from lib.rrf_fusion import RRFFusion

    rrf = RRFFusion(k=60)
    fused = rrf.fuse({
        'bm25': [(id1, text1, score1), ...],
        'semantic': [(id2, text2, score2), ...]
    })
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Result after RRF fusion"""
    record_id: Any
    text: str
    rrf_score: float
    component_ranks: Dict[str, int] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)
    record: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'record_id': self.record_id,
            'text': self.text,
            'rrf_score': self.rrf_score,
            'component_ranks': self.component_ranks,
            'component_scores': self.component_scores
        }


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining multiple retrieval methods.

    RRF is preferred over weighted score combination because:
    1. No parameter tuning required (k=60 works well universally)
    2. Handles different score distributions (BM25 vs cosine similarity)
    3. Research shows 8-15% improvement over single methods
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Ranking constant (default 60 from original paper).
               Higher k reduces impact of high ranks.
               60 is well-tested across domains.
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[Any, str, float]]],
        top_k: Optional[int] = None,
        records: Optional[Dict[Any, Dict]] = None
    ) -> List[FusedResult]:
        """
        Fuse multiple ranked lists using RRF.

        Args:
            ranked_lists: Dict mapping source name to list of (record_id, text, score).
                         Each list should be sorted by score descending.
            top_k: Optional limit on number of results
            records: Optional dict mapping record_id to full record

        Returns:
            List of FusedResult sorted by RRF score descending
        """
        rrf_scores: Dict[Any, float] = defaultdict(float)
        texts: Dict[Any, str] = {}
        component_ranks: Dict[Any, Dict[str, int]] = defaultdict(dict)
        component_scores: Dict[Any, Dict[str, float]] = defaultdict(dict)

        # Process each ranked list
        for source, results in ranked_lists.items():
            for rank, (record_id, text, score) in enumerate(results, start=1):
                # RRF formula: 1 / (rank + k)
                rrf_scores[record_id] += 1.0 / (rank + self.k)
                texts[record_id] = text
                component_ranks[record_id][source] = rank
                component_scores[record_id][source] = score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        if top_k:
            sorted_ids = sorted_ids[:top_k]

        # Build results
        results = []
        for record_id in sorted_ids:
            record = records.get(record_id) if records else None
            results.append(FusedResult(
                record_id=record_id,
                text=texts[record_id],
                rrf_score=rrf_scores[record_id],
                component_ranks=dict(component_ranks[record_id]),
                component_scores=dict(component_scores[record_id]),
                record=record
            ))

        return results

    def explain(self, result: FusedResult) -> str:
        """
        Generate human-readable explanation of fusion result.

        Args:
            result: A FusedResult from fuse()

        Returns:
            Explanation string
        """
        parts = [f"RRF Score: {result.rrf_score:.4f}"]
        parts.append("Components:")

        for source in sorted(result.component_ranks.keys()):
            rank = result.component_ranks[source]
            score = result.component_scores.get(source, 0)
            contribution = 1.0 / (rank + self.k)
            parts.append(f"  {source}: rank={rank}, score={score:.3f}, rrf_contrib={contribution:.4f}")

        return "\n".join(parts)


def fuse_simple(
    bm25_results: List[Tuple[Any, str, float]],
    semantic_results: List[Tuple[Any, str, float]],
    k: int = 60,
    top_k: int = 20
) -> List[FusedResult]:
    """
    Simple fusion of BM25 and semantic results.

    Args:
        bm25_results: List of (record_id, text, score) from BM25
        semantic_results: List of (record_id, text, score) from semantic search
        k: RRF constant
        top_k: Number of results

    Returns:
        Fused results
    """
    rrf = RRFFusion(k=k)
    return rrf.fuse({
        'bm25': bm25_results,
        'semantic': semantic_results
    }, top_k=top_k)


def fuse_with_pagerank(
    bm25_results: List[Tuple[Any, str, float]],
    semantic_results: List[Tuple[Any, str, float]],
    pagerank_scores: Optional[Dict[Any, float]] = None,
    k: int = 60,
    top_k: int = 20
) -> List[FusedResult]:
    """
    Three-signal fusion: BM25 + semantic + PageRank.

    PageRank acts as a graph-structural authority signal — entities with
    more incoming edges (hubs) get boosted in search results.

    Args:
        bm25_results: List of (record_id, text, score) from BM25
        semantic_results: List of (record_id, text, score) from semantic search
        pagerank_scores: Dict mapping record_id -> pagerank score (0-1).
                        If None, falls back to two-signal fusion.
        k: RRF constant
        top_k: Number of results

    Returns:
        Fused results with three component scores
    """
    ranked_lists = {
        'bm25': bm25_results,
        'semantic': semantic_results,
    }

    if pagerank_scores:
        # Convert pagerank dict to a ranked list (sorted by score desc)
        # We need (record_id, text, score) tuples
        # Only include IDs that appear in either BM25 or semantic results
        all_ids = set()
        text_map = {}
        for rid, text, _ in bm25_results:
            all_ids.add(rid)
            text_map[rid] = text
        for rid, text, _ in semantic_results:
            all_ids.add(rid)
            text_map[rid] = text

        pr_list = []
        for rid in all_ids:
            if rid in pagerank_scores:
                pr_list.append((rid, text_map.get(rid, ""), pagerank_scores[rid]))

        # Sort by PageRank score descending
        pr_list.sort(key=lambda x: x[2], reverse=True)

        if pr_list:
            ranked_lists['pagerank'] = pr_list

    rrf = RRFFusion(k=k)
    return rrf.fuse(ranked_lists, top_k=top_k)
