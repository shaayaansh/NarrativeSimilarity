"""Structural similarity model ported from src/modeling.ipynb.

This module provides a `StructuralSimilarityModel` class that expects one
pair-level dictionary with the same key structure used in `alignment_df` rows.
Required keys are:
- `alignment` (dict with `matches`, each match has `a_indices` and `b_indices`)
- `EventsA_align` (list of event strings)
- `EventsB_align` (list of event strings)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. Install it with: pip install sentence-transformers"
    ) from exc


@dataclass(frozen=True)
class StructuralModelParams:
    """Learned joint-model parameters from src/modeling.ipynb output."""

    c_hat_joint: float = 1.5830
    alpha_hat_joint: float = 0.2021
    beta_hat_joint: float = 0.9339


class StructuralSimilarityModel:
    """Predict narrative similarity from structural and semantic alignment."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        params: StructuralModelParams | None = None,
    ) -> None:
        self.params = params or StructuralModelParams()
        self._embedder = SentenceTransformer(embedding_model_name)

    @staticmethod
    def _get_event_text(events: Sequence[str], idx: int) -> str | None:
        # Keep notebook behavior: try 1-indexed access, then 0-indexed access.
        if 1 <= idx <= len(events):
            return events[idx - 1]
        if 0 <= idx < len(events):
            return events[idx]
        return None

    @staticmethod
    def _cosine_distance_01(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 1.0
        cos_sim = float(np.dot(vec_a, vec_b) / denom)
        return (1.0 - cos_sim) / 2.0

    @staticmethod
    def _extract_aligned_pairs(row: Dict[str, Any]) -> Tuple[List[Tuple[int, int]], set[int], set[int]]:
        alignment = row.get("alignment") or {}
        matches = alignment.get("matches", []) or []

        aligned_pairs: List[Tuple[int, int]] = []
        aligned_a: set[int] = set()
        aligned_b: set[int] = set()

        for match in matches:
            a_inds = match.get("a_indices", []) or []
            b_inds = match.get("b_indices", []) or []

            aligned_a.update(a_inds)
            aligned_b.update(b_inds)

            for i in a_inds:
                for j in b_inds:
                    aligned_pairs.append((i, j))

        return aligned_pairs, aligned_a, aligned_b

    def compute_alignment_distance(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Match notebook `compute_alignment_distance` on one alignment_df-style row."""
        aligned_pairs, aligned_a, aligned_b = self._extract_aligned_pairs(row)

        events_a = row.get("EventsA_align") or []
        events_b = row.get("EventsB_align") or []
        e_a = len(events_a)
        e_b = len(events_b)

        cost_skip = (e_a - len(aligned_a)) + (e_b - len(aligned_b))

        aligned_pairs_sorted = sorted(aligned_pairs, key=lambda x: x[0])
        b_sequence = [j for _, j in aligned_pairs_sorted]

        cost_reorder = 0
        for i in range(len(b_sequence)):
            for j in range(i + 1, len(b_sequence)):
                if b_sequence[i] > b_sequence[j]:
                    cost_reorder += 1

        num_pairs = len(aligned_pairs_sorted)
        max_reorder = num_pairs * (num_pairs - 1) / 2
        z = max_reorder + (e_a + e_b)
        d_alignment = (cost_reorder + cost_skip) / z if z > 0 else 0.0

        return {
            "cost_reorder": float(cost_reorder),
            "cost_skip": float(cost_skip),
            "D_alignment": float(d_alignment),
            "alignment_similarity": float(1.0 - d_alignment),
        }

    def alignment_distance(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Alias for compatibility with notebook terminology."""
        return self.compute_alignment_distance(row)

    def compute_semantic_distance(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Match notebook `compute_semantic_distance` on one alignment_df-style row."""
        aligned_pairs, _, _ = self._extract_aligned_pairs(row)

        if len(aligned_pairs) == 0:
            return {"D_semantic": 1.0, "semantic_similarity": 0.0, "num_aligned_pairs": 0}

        events_a = row.get("EventsA_align") or []
        events_b = row.get("EventsB_align") or []

        texts_a: List[str] = []
        texts_b: List[str] = []
        for i, j in aligned_pairs:
            ta = self._get_event_text(events_a, i)
            tb = self._get_event_text(events_b, j)
            if ta is None or tb is None:
                continue
            texts_a.append(ta)
            texts_b.append(tb)

        if len(texts_a) == 0:
            return {"D_semantic": 1.0, "semantic_similarity": 0.0, "num_aligned_pairs": 0}

        emb_a = self._embedder.encode(texts_a, convert_to_numpy=True, normalize_embeddings=False)
        emb_b = self._embedder.encode(texts_b, convert_to_numpy=True, normalize_embeddings=False)

        dists = [self._cosine_distance_01(a, b) for a, b in zip(emb_a, emb_b)]
        d_semantic = float(np.mean(dists))

        return {
            "D_semantic": d_semantic,
            "semantic_similarity": float(1.0 - d_semantic),
            "num_aligned_pairs": int(len(dists)),
        }

    def semantic_distance(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Alias for compatibility with notebook terminology."""
        return self.compute_semantic_distance(row)

    def predict_similarity(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Compute distances and predict event-level similarity score for one pair."""
        align = self.compute_alignment_distance(row)
        sem = self.compute_semantic_distance(row)

        pred = (
            self.params.c_hat_joint
            + self.params.alpha_hat_joint * align["alignment_similarity"]
            + self.params.beta_hat_joint * sem["semantic_similarity"]
        )

        return {
            **align,
            **sem,
            "pred_event_rating_mean_joint": float(pred),
            "c_hat_joint": self.params.c_hat_joint,
            "alpha_hat_joint": self.params.alpha_hat_joint,
            "beta_hat_joint": self.params.beta_hat_joint,
        }
