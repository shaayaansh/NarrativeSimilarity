#!/usr/bin/env python3
"""CLI evaluation for structure-aware E5 checkpoints.

Usage:
  python scripts/eval_e5_structural.py --config configs/e5_structural_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


@dataclass
class EvalConfig:
    checkpoints_root: Path
    checkpoint_selector: str  # int-like or "all"
    checkpoint_pattern: str

    tasks: List[str]

    eval_pair_path: Path
    retrieval_path: Path
    ranking_path: Path
    semeval_2026_path: Path
    semeval_2022_path: Path
    synthetic_version: str
    synthetic_v1_path: Path
    synthetic_v2_path: Path

    model_max_length: int
    hf_cache_dir: Path
    use_bfloat16: bool

    output_dir: Path
    output_filename: str

    run_llm_baselines: bool
    llm_model: str
    openai_key_path: Path
    llm_sleep_seconds: float
    run_embedding_baselines: bool
    baseline_roberta_model: str
    baseline_bge_model: str
    baseline_untrained_e5_model: str
    baseline_storyemb_model: str


VALID_TASKS = {
    "eval_pair_task",
    "retrieval_task",
    "ranking_task",
    "synthetic_task",
    "semeval_2026_task",
    "semeval_2022_correlation_task",
}


def load_config(config_path: Path) -> EvalConfig:
    root = project_root()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tasks = raw["tasks"]["run"]
    if isinstance(tasks, str):
        # support both: "all" and comma-separated strings like "eval_pair_task, retrieval_task"
        tasks = [x.strip() for x in tasks.split(",") if x.strip()]
    tasks = list(tasks)
    if "all" in [t.lower() for t in tasks]:
        tasks = sorted(list(VALID_TASKS))

    cfg = EvalConfig(
        checkpoints_root=resolve_path(root, raw["checkpoints"]["root"]),
        checkpoint_selector=str(raw["checkpoints"]["which"]),
        checkpoint_pattern=str(raw["checkpoints"].get("pattern", "epoch_*")),
        tasks=tasks,
        eval_pair_path=resolve_path(root, raw["data"]["eval_pair_path"]),
        retrieval_path=resolve_path(root, raw["data"]["retrieval_path"]),
        ranking_path=resolve_path(root, raw["data"].get("ranking_path", "data/eval_data/ranking_eval_df.csv")),
        semeval_2026_path=resolve_path(
            root, raw["data"].get("semeval_2026_path", "data/eval_data/SemEval2026-Task_4-dev-v1/dev_track_a.jsonl")
        ),
        semeval_2022_path=resolve_path(
            root, raw["data"].get("semeval_2022_path", "data/eval_data/SemEval2022-Task8/semeval_2022_eval_data.csv")
        ),
        synthetic_version=str(raw["data"].get("synthetic_version", "v2")).lower(),
        synthetic_v1_path=resolve_path(root, raw["data"]["synthetic_v1_path"]),
        synthetic_v2_path=resolve_path(root, raw["data"]["synthetic_v2_path"]),
        model_max_length=int(raw["model"]["max_length"]),
        hf_cache_dir=Path(raw["model"]["hf_cache_dir"]),
        use_bfloat16=bool(raw["model"]["use_bfloat16"]),
        output_dir=resolve_path(root, raw["output"]["dir"]),
        output_filename=str(raw["output"]["filename"]),
        run_llm_baselines=bool(raw["baselines"].get("run_llm", False)),
        llm_model=str(raw["baselines"].get("llm_model", "gpt-5")),
        openai_key_path=resolve_path(root, raw["baselines"].get("openai_key_path", "openai_key.txt")),
        llm_sleep_seconds=float(raw["baselines"].get("llm_sleep_seconds", 0.05)),
        run_embedding_baselines=bool(raw["baselines"].get("run_embedding_baselines", False)),
        baseline_roberta_model=str(raw["baselines"].get("baseline_roberta_model", "roberta-base")),
        baseline_bge_model=str(raw["baselines"].get("baseline_bge_model", "BAAI/bge-base-en-v1.5")),
        baseline_untrained_e5_model=str(
            raw["baselines"].get("baseline_untrained_e5_model", "intfloat/e5-mistral-7b-instruct")
        ),
        baseline_storyemb_model=str(raw["baselines"].get("baseline_storyemb_model", "uhhlt/story-emb")),
    )

    bad = [t for t in cfg.tasks if t not in VALID_TASKS]
    if bad:
        raise ValueError(f"Unknown tasks in config: {bad}. Valid: {sorted(VALID_TASKS)}")
    if cfg.synthetic_version not in {"v1", "v2"}:
        raise ValueError("data.synthetic_version must be v1 or v2")

    return cfg


def list_checkpoints(cfg: EvalConfig) -> List[Path]:
    if not cfg.checkpoints_root.exists():
        raise FileNotFoundError(f"checkpoints root not found: {cfg.checkpoints_root}")

    all_dirs = [d for d in cfg.checkpoints_root.glob(cfg.checkpoint_pattern) if d.is_dir()]
    if not all_dirs:
        raise ValueError(f"No checkpoints found in {cfg.checkpoints_root} matching {cfg.checkpoint_pattern}")

    def epoch_num(path: Path) -> int:
        m = re.search(r"epoch_(\d+)", path.name)
        if not m:
            return 10**9
        return int(m.group(1))

    all_dirs = sorted(all_dirs, key=epoch_num)

    if cfg.checkpoint_selector.lower() == "all":
        return all_dirs

    epoch = int(cfg.checkpoint_selector)
    for d in all_dirs:
        if epoch_num(d) == epoch:
            return [d]
    raise ValueError(f"Requested epoch {epoch} not found under {cfg.checkpoints_root}")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class Embedder:
    def __init__(
        self,
        checkpoint_path: Path | str,
        max_length: int,
        hf_cache_dir: Path,
        use_bfloat16: bool,
        use_e5_query_prefix: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (self.device == "cuda" and use_bfloat16) else torch.float32
        self.use_e5_query_prefix = use_e5_query_prefix

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path), trust_remote_code=True, cache_dir=str(hf_cache_dir)
        )
        self.model = AutoModel.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True,
            cache_dir=str(hf_cache_dir),
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        # Use a safe per-model max length to avoid runtime shape errors
        # (e.g., roberta-base has 514 positional slots and cannot run at 1024).
        tokenizer_cap = getattr(self.tokenizer, "model_max_length", None)
        config_cap = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)

        caps = [max_length]
        if isinstance(tokenizer_cap, int) and 0 < tokenizer_cap < 10**6:
            caps.append(tokenizer_cap)
        if isinstance(config_cap, int) and 0 < config_cap < 10**6:
            caps.append(config_cap)
        self.max_length = int(min(caps))
        print(f"[Embedder] model={checkpoint_path} | max_length={self.max_length}")

    def prepare_text(self, text: str) -> str:
        text = str(text)
        if self.use_e5_query_prefix:
            return f"query: retrieve stories with a similar narrative to the given story. {text}"
        return text

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.model(**enc).last_hidden_state
            pooled = mean_pool(h, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu())
        return torch.cat(out, dim=0)


def run_eval_pair_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"story_text_a", "story_text_b", "label"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"eval_pair_task missing columns: {sorted(miss)}")

    rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="eval_pair_task", unit="pair"):
        a = emb.prepare_text(str(r.story_text_a))
        b = emb.prepare_text(str(r.story_text_b))
        ea, eb = emb.embed([a, b], batch_size=2)
        sim = float(torch.dot(ea, eb).item())
        label = int(r.label)
        rows.append({"label": label, "cosine": sim})

    rdf = pd.DataFrame(rows)
    pos = rdf[rdf["label"] == 1]["cosine"]
    neg = rdf[rdf["label"] == 0]["cosine"]
    result = {
        "num_pairs": int(len(rdf)),
        "mean_positive": float(pos.mean()) if len(pos) else None,
        "mean_negative": float(neg.mean()) if len(neg) else None,
        "std_positive": float(pos.std(ddof=1)) if len(pos) > 1 else None,
        "std_negative": float(neg.std(ddof=1)) if len(neg) > 1 else None,
        "mean_gap_pos_minus_neg": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else None,
    }
    return result


def run_retrieval_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"query_text", "correct_option_index", "option_0_text", "option_1_text", "option_2_text", "option_3_text", "option_4_text"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"retrieval_task missing columns: {sorted(miss)}")

    correct = 0
    n = len(df)
    for r in tqdm(df.itertuples(index=False), total=n, desc="retrieval_task", unit="query"):
        q = emb.prepare_text(str(r.query_text))
        opts = [emb.prepare_text(str(getattr(r, f"option_{i}_text"))) for i in range(5)]
        eq = emb.embed([q], batch_size=1).squeeze(0)
        eo = emb.embed(opts, batch_size=5)
        sims = (eo @ eq).numpy()
        pred = int(np.argmax(sims))
        gt = int(r.correct_option_index)
        correct += int(pred == gt)

    return {
        "num_queries": int(n),
        "num_correct": int(correct),
        "accuracy": float(correct / n) if n else 0.0,
    }


def run_ranking_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"query_text"} | {f"option_{i}_text" for i in range(5)} | {f"option_{i}_rank" for i in range(5)}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"ranking_task missing columns: {sorted(miss)}")

    ndcgs: List[float] = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="ranking_task", unit="query"):
        q = emb.prepare_text(str(r.query_text))
        opts = [emb.prepare_text(str(getattr(r, f"option_{i}_text"))) for i in range(5)]
        # Convert ranks (1=best..5=worst) to graded relevance (5..1)
        rel = np.array([6 - int(getattr(r, f"option_{i}_rank")) for i in range(5)], dtype=np.float64)

        eq = emb.embed([q], batch_size=1).squeeze(0)
        eo = emb.embed(opts, batch_size=5)
        sims = (eo @ eq).numpy()
        pred_order = np.argsort(-sims)  # descending similarity

        gains = rel[pred_order]
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(gains), dtype=np.float64))
        dcg = float(np.sum(gains * discounts))

        ideal_order = np.argsort(-rel)
        ideal_gains = rel[ideal_order]
        idcg = float(np.sum(ideal_gains * discounts))
        ndcg = float(dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        "num_queries": int(len(ndcgs)),
        "mean_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "std_ndcg": float(np.std(ndcgs, ddof=1)) if len(ndcgs) > 1 else None,
    }


def run_synthetic_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"story_1", "story_2", "story_3", "struct_score_12", "struct_score_13"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"synthetic_task missing columns: {sorted(miss)}")

    n = 0
    correct = 0
    ties = 0

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="synthetic_task", unit="theme"):
        s12 = float(r.struct_score_12)
        s13 = float(r.struct_score_13)
        if s12 == s13:
            ties += 1
            continue

        gt = "story_2" if s12 > s13 else "story_3"
        texts = [emb.prepare_text(str(r.story_1)), emb.prepare_text(str(r.story_2)), emb.prepare_text(str(r.story_3))]
        e = emb.embed(texts, batch_size=3)
        e1, e2, e3 = e[0], e[1], e[2]
        sim12 = float(torch.dot(e1, e2).item())
        sim13 = float(torch.dot(e1, e3).item())
        pred = "story_2" if sim12 > sim13 else "story_3"

        n += 1
        correct += int(pred == gt)

    return {
        "num_non_tie_rows": int(n),
        "num_ties_skipped": int(ties),
        "num_correct": int(correct),
        "accuracy": float(correct / n) if n else 0.0,
    }


def _parse_bool_label(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in {"true", "1", "yes", "y", "a", "text_a", "text_a_is_closer"}:
            return True
        if vv in {"false", "0", "no", "n", "b", "text_b"}:
            return False
    raise ValueError(f"Could not parse semeval label value: {v!r}")


def run_semeval_2026_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"anchor_text", "text_a", "text_b", "text_a_is_closer"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"semeval_2026_task missing columns: {sorted(miss)}")

    n = len(df)
    correct = 0
    for r in tqdm(df.itertuples(index=False), total=n, desc="semeval_2026_task", unit="triplet"):
        anchor = emb.prepare_text(str(r.anchor_text))
        ta = emb.prepare_text(str(r.text_a))
        tb = emb.prepare_text(str(r.text_b))
        e = emb.embed([anchor, ta, tb], batch_size=3)
        ea, e_a, e_b = e[0], e[1], e[2]
        sim_a = float(torch.dot(ea, e_a).item())
        sim_b = float(torch.dot(ea, e_b).item())
        pred_text_a_closer = sim_a > sim_b
        gt_text_a_closer = _parse_bool_label(r.text_a_is_closer)
        correct += int(pred_text_a_closer == gt_text_a_closer)

    return {
        "num_triplets": int(n),
        "num_correct": int(correct),
        "accuracy": float(correct / n) if n else 0.0,
    }


def run_semeval_2022_correlation_task(df: pd.DataFrame, emb: Embedder) -> Dict:
    req = {"pair_id", "article_1", "article_2", "NAR"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"semeval_2022_correlation_task missing columns: {sorted(miss)}")

    rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="semeval_2022_correlation_task", unit="pair"):
        a = emb.prepare_text(str(r.article_1))
        b = emb.prepare_text(str(r.article_2))
        ea, eb = emb.embed([a, b], batch_size=2)
        sim = float(torch.dot(ea, eb).item())
        nar = float(r.NAR)
        rows.append({"pair_id": str(r.pair_id), "cosine": sim, "NAR": nar})

    rdf = pd.DataFrame(rows)
    # Spearman is primary here because NAR is ordinal (1..4); Pearson is secondary.
    spearman = float(rdf["cosine"].corr(rdf["NAR"], method="spearman")) if len(rdf) > 1 else None
    pearson = float(rdf["cosine"].corr(rdf["NAR"], method="pearson")) if len(rdf) > 1 else None
    return {
        "num_pairs": int(len(rdf)),
        "spearman_corr_cosine_vs_NAR": spearman,
        "pearson_corr_cosine_vs_NAR": pearson,
        "nar_mean": float(rdf["NAR"].mean()) if len(rdf) else None,
        "nar_median": float(rdf["NAR"].median()) if len(rdf) else None,
    }


def get_openai_client(cfg: EvalConfig) -> OpenAI:
    key_path = cfg.openai_key_path
    if not key_path.exists():
        raise FileNotFoundError(f"OpenAI key file not found: {key_path}")
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"OpenAI key file is empty: {key_path}")
    return OpenAI(api_key=api_key)


def parse_choice(text: str, valid: List[str]) -> Optional[str]:
    t = (text or "").strip()
    try:
        obj = json.loads(t)
        ch = str(obj.get("choice", "")).strip().lower()
        if ch in valid:
            return ch
    except Exception:
        pass

    t_low = t.lower()
    for v in valid:
        if v in t_low:
            return v
    return None


def run_llm_eval_pair(df: pd.DataFrame, cfg: EvalConfig, client: OpenAI) -> Dict:
    req = {"story_text_a", "story_text_b", "label"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"eval_pair_task missing columns: {sorted(miss)}")

    system_prompt = (
        "You are a narrative structure judge. Given two stories, decide if they are structurally similar. "
        "Focus on event progression and what happens, not wording. "
        "Return strict JSON only: {\"choice\": \"positive\"} or {\"choice\": \"negative\"}."
    )

    correct = 0
    errors = 0
    n = len(df)
    for r in tqdm(df.itertuples(index=False), total=n, desc="llm_eval_pair", unit="pair"):
        user = (
            f"Story A:\n{str(r.story_text_a)}\n\n"
            f"Story B:\n{str(r.story_text_b)}\n\n"
            "Are these structurally similar?"
        )
        try:
            resp = client.responses.create(
                model=cfg.llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
            )
            pred = parse_choice(getattr(resp, "output_text", ""), ["positive", "negative"])
            gt = "positive" if int(r.label) == 1 else "negative"
            correct += int(pred == gt)
        except Exception:
            errors += 1
        time.sleep(cfg.llm_sleep_seconds)

    return {"num_pairs": int(n), "num_correct": int(correct), "num_errors": int(errors), "accuracy": float(correct / n) if n else 0.0}


def run_llm_retrieval(df: pd.DataFrame, cfg: EvalConfig, client: OpenAI) -> Dict:
    req = {"query_text", "correct_option_index", "option_0_text", "option_1_text", "option_2_text", "option_3_text", "option_4_text"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"retrieval_task missing columns: {sorted(miss)}")

    system_prompt = (
        "You are a narrative structure judge. For a query story and five candidate stories, choose the single option most structurally similar. "
        "Return strict JSON only: {\"choice\": <int>} where int is 0..4."
    )

    n = len(df)
    correct = 0
    errors = 0
    for r in tqdm(df.itertuples(index=False), total=n, desc="llm_retrieval", unit="query"):
        user = (
            f"Query story:\n{str(r.query_text)}\n\n"
            f"Option 0:\n{str(r.option_0_text)}\n\n"
            f"Option 1:\n{str(r.option_1_text)}\n\n"
            f"Option 2:\n{str(r.option_2_text)}\n\n"
            f"Option 3:\n{str(r.option_3_text)}\n\n"
            f"Option 4:\n{str(r.option_4_text)}\n\n"
            "Which option is structurally most similar to the query?"
        )
        try:
            resp = client.responses.create(
                model=cfg.llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
            )
            txt = getattr(resp, "output_text", "")
            pred = None
            try:
                obj = json.loads((txt or "").strip())
                pred = int(obj.get("choice"))
            except Exception:
                m = re.search(r"\b([0-4])\b", txt or "")
                pred = int(m.group(1)) if m else None

            correct += int(pred == int(r.correct_option_index))
        except Exception:
            errors += 1
        time.sleep(cfg.llm_sleep_seconds)

    return {"num_queries": int(n), "num_correct": int(correct), "num_errors": int(errors), "accuracy": float(correct / n) if n else 0.0}


def run_llm_synthetic(df: pd.DataFrame, cfg: EvalConfig, client: OpenAI) -> Dict:
    req = {"story_1", "story_2", "story_3", "struct_score_12", "struct_score_13"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"synthetic_task missing columns: {sorted(miss)}")

    system_prompt = (
        "You are a narrative structure judge. Given one anchor story and two candidates, choose which candidate is structurally closer. "
        "Return strict JSON only: {\"choice\": \"story_2\"} or {\"choice\": \"story_3\"}."
    )

    n = 0
    correct = 0
    ties = 0
    errors = 0
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="llm_synthetic", unit="theme"):
        s12 = float(r.struct_score_12)
        s13 = float(r.struct_score_13)
        if s12 == s13:
            ties += 1
            continue
        gt = "story_2" if s12 > s13 else "story_3"

        user = (
            f"Anchor (story_1):\n{str(r.story_1)}\n\n"
            f"Candidate (story_2):\n{str(r.story_2)}\n\n"
            f"Candidate (story_3):\n{str(r.story_3)}\n\n"
            "Which candidate is structurally closer to the anchor?"
        )

        try:
            resp = client.responses.create(
                model=cfg.llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
            )
            pred = parse_choice(getattr(resp, "output_text", ""), ["story_2", "story_3"])
            n += 1
            correct += int(pred == gt)
        except Exception:
            errors += 1
        time.sleep(cfg.llm_sleep_seconds)

    return {
        "num_non_tie_rows": int(n),
        "num_ties_skipped": int(ties),
        "num_correct": int(correct),
        "num_errors": int(errors),
        "accuracy": float(correct / n) if n else 0.0,
    }


def run_llm_semeval_2026(df: pd.DataFrame, cfg: EvalConfig, client: OpenAI) -> Dict:
    req = {"anchor_text", "text_a", "text_b", "text_a_is_closer"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"semeval_2026_task missing columns: {sorted(miss)}")

    system_prompt = (
        "You are a narrative structure judge. Given one anchor story and two candidate stories, "
        "choose which candidate is structurally closer to the anchor. "
        "Return strict JSON only: {\"choice\": \"text_a\"} or {\"choice\": \"text_b\"}."
    )

    n = len(df)
    correct = 0
    errors = 0
    for r in tqdm(df.itertuples(index=False), total=n, desc="llm_semeval_2026", unit="triplet"):
        user = (
            f"Anchor story:\n{str(r.anchor_text)}\n\n"
            f"Candidate text_a:\n{str(r.text_a)}\n\n"
            f"Candidate text_b:\n{str(r.text_b)}\n\n"
            "Which candidate is structurally closer to the anchor?"
        )
        try:
            resp = client.responses.create(
                model=cfg.llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
            )
            pred = parse_choice(getattr(resp, "output_text", ""), ["text_a", "text_b"])
            gt = "text_a" if _parse_bool_label(r.text_a_is_closer) else "text_b"
            correct += int(pred == gt)
        except Exception:
            errors += 1
        time.sleep(cfg.llm_sleep_seconds)

    return {"num_triplets": int(n), "num_correct": int(correct), "num_errors": int(errors), "accuracy": float(correct / n) if n else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI evaluation for structural embedding checkpoints")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list_checkpoints(cfg)
    print("checkpoints to evaluate:")
    for c in checkpoints:
        print(" -", c)

    pair_df = pd.read_csv(cfg.eval_pair_path) if "eval_pair_task" in cfg.tasks else None
    retrieval_df = pd.read_csv(cfg.retrieval_path) if "retrieval_task" in cfg.tasks else None
    ranking_df = pd.read_csv(cfg.ranking_path) if "ranking_task" in cfg.tasks else None
    semeval_2026_df = pd.read_json(cfg.semeval_2026_path, lines=True) if "semeval_2026_task" in cfg.tasks else None
    semeval_2022_df = (
        pd.read_csv(cfg.semeval_2022_path) if "semeval_2022_correlation_task" in cfg.tasks else None
    )
    if "synthetic_task" in cfg.tasks:
        syn_path = cfg.synthetic_v1_path if cfg.synthetic_version == "v1" else cfg.synthetic_v2_path
        synthetic_df = pd.read_csv(syn_path)
    else:
        synthetic_df = None

    results: Dict = {
        "config": {
            "checkpoint_selector": cfg.checkpoint_selector,
            "tasks": cfg.tasks,
            "synthetic_version": cfg.synthetic_version,
            "run_llm_baselines": cfg.run_llm_baselines,
            "llm_model": cfg.llm_model if cfg.run_llm_baselines else None,
            "run_embedding_baselines": cfg.run_embedding_baselines,
        },
        "embedding": {},
    }

    for ckpt in checkpoints:
        print(f"\nEvaluating embedding checkpoint: {ckpt.name}")
        emb = Embedder(
            checkpoint_path=ckpt,
            max_length=cfg.model_max_length,
            hf_cache_dir=cfg.hf_cache_dir,
            use_bfloat16=cfg.use_bfloat16,
        )

        ckpt_res = {}
        if "eval_pair_task" in cfg.tasks:
            ckpt_res["eval_pair_task"] = run_eval_pair_task(pair_df, emb)
        if "retrieval_task" in cfg.tasks:
            ckpt_res["retrieval_task"] = run_retrieval_task(retrieval_df, emb)
        if "ranking_task" in cfg.tasks:
            ckpt_res["ranking_task"] = run_ranking_task(ranking_df, emb)
        if "semeval_2026_task" in cfg.tasks:
            ckpt_res["semeval_2026_task"] = run_semeval_2026_task(semeval_2026_df, emb)
        if "semeval_2022_correlation_task" in cfg.tasks:
            ckpt_res["semeval_2022_correlation_task"] = run_semeval_2022_correlation_task(semeval_2022_df, emb)
        if "synthetic_task" in cfg.tasks:
            ckpt_res["synthetic_task"] = run_synthetic_task(synthetic_df, emb)

        results["embedding"][ckpt.name] = ckpt_res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cfg.run_embedding_baselines:
        print("\nRunning embedding baselines...")
        baseline_specs = [
            ("roberta_base", cfg.baseline_roberta_model, False),
            ("bge_base", cfg.baseline_bge_model, False),
            ("e5_untrained", cfg.baseline_untrained_e5_model, True),
            ("story_emb", cfg.baseline_storyemb_model, False),
        ]
        baseline_results: Dict = {}
        for baseline_name, model_id, use_e5_prefix in baseline_specs:
            print(f"\nEvaluating embedding baseline: {baseline_name} ({model_id})")
            emb = Embedder(
                checkpoint_path=model_id,
                max_length=cfg.model_max_length,
                hf_cache_dir=cfg.hf_cache_dir,
                use_bfloat16=cfg.use_bfloat16,
                use_e5_query_prefix=use_e5_prefix,
            )
            bres = {}
            if "eval_pair_task" in cfg.tasks:
                bres["eval_pair_task"] = run_eval_pair_task(pair_df, emb)
            if "retrieval_task" in cfg.tasks:
                bres["retrieval_task"] = run_retrieval_task(retrieval_df, emb)
            if "ranking_task" in cfg.tasks:
                bres["ranking_task"] = run_ranking_task(ranking_df, emb)
            if "semeval_2026_task" in cfg.tasks:
                bres["semeval_2026_task"] = run_semeval_2026_task(semeval_2026_df, emb)
            if "semeval_2022_correlation_task" in cfg.tasks:
                bres["semeval_2022_correlation_task"] = run_semeval_2022_correlation_task(semeval_2022_df, emb)
            if "synthetic_task" in cfg.tasks:
                bres["synthetic_task"] = run_synthetic_task(synthetic_df, emb)
            baseline_results[baseline_name] = bres
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["embedding_baselines"] = baseline_results

    if cfg.run_llm_baselines:
        print("\nRunning LLM baselines...")
        client = get_openai_client(cfg)
        llm_res = {}
        if "eval_pair_task" in cfg.tasks:
            llm_res["eval_pair_task"] = run_llm_eval_pair(pair_df, cfg, client)
        if "retrieval_task" in cfg.tasks:
            llm_res["retrieval_task"] = run_llm_retrieval(retrieval_df, cfg, client)
        if "ranking_task" in cfg.tasks:
            print("ranking_task for LLM baseline is not implemented in this script yet.")
        if "semeval_2026_task" in cfg.tasks:
            llm_res["semeval_2026_task"] = run_llm_semeval_2026(semeval_2026_df, cfg, client)
        if "synthetic_task" in cfg.tasks:
            llm_res["synthetic_task"] = run_llm_synthetic(synthetic_df, cfg, client)
        results["llm_baseline"] = llm_res

    out_path = cfg.output_dir / cfg.output_filename
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved eval results:", out_path)
    print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    main()
