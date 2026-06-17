#!/usr/bin/env python3
"""Train a structural embedding model from CLI.

Usage:
  python scripts/train_e5_structural.py --config configs/e5_structural_train.yaml

This script is extracted from src/e5_mistral_structural_finetune.ipynb and supports
both objectives:
- infonce (triplet-based)
- contrastive_mse (pair regression)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class TrainConfig:
    scores_path: Path
    alignments_path: Path
    output_dir: Path
    model_name: str
    model_family: str
    story_prefix: str
    max_length: int
    use_bfloat16: bool
    hf_cache_dir: Path
    loss_type: str
    seed: int
    train_frac: float
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    max_grad_norm: float
    temperature: float
    hard_negative_bin: int
    positive_bin: int
    easy_negative_bins: List[int]
    max_triplets_per_anchor: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    eval_every_steps: int
    checkpoint_every_epochs: int
    resume_training: bool


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def load_config(config_path: Path) -> TrainConfig:
    root = project_root()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_family = str(raw["model"].get("model_family", "e5")).lower()
    story_prefix = str(raw["model"].get("story_prefix", "passage: " if model_family == "e5" else ""))

    lora_raw = raw.get("lora", {})
    use_lora = bool(lora_raw.get("enabled", True))
    raw_target_modules = lora_raw.get("target_modules", "auto")
    if raw_target_modules == "auto":
        target_modules = default_lora_target_modules(model_family)
    else:
        target_modules = list(raw_target_modules)

    return TrainConfig(
        scores_path=resolve_path(root, raw["data"]["scores_path"]),
        alignments_path=resolve_path(root, raw["data"]["alignments_path"]),
        output_dir=resolve_path(root, raw["data"]["output_dir"]),
        model_name=raw["model"]["model_name"],
        model_family=model_family,
        story_prefix=story_prefix,
        max_length=int(raw["model"]["max_length"]),
        use_bfloat16=bool(raw["model"]["use_bfloat16"]),
        hf_cache_dir=Path(raw["model"]["hf_cache_dir"]),
        loss_type=str(raw["training"]["loss_type"]),
        seed=int(raw["training"]["seed"]),
        train_frac=float(raw["training"]["train_frac"]),
        batch_size=int(raw["training"]["batch_size"]),
        num_epochs=int(raw["training"]["num_epochs"]),
        lr=float(raw["training"]["lr"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        warmup_ratio=float(raw["training"]["warmup_ratio"]),
        grad_accum_steps=int(raw["training"]["grad_accum_steps"]),
        max_grad_norm=float(raw["training"]["max_grad_norm"]),
        temperature=float(raw["training"]["temperature"]),
        hard_negative_bin=int(raw["sampling"]["hard_negative_bin"]),
        positive_bin=int(raw["sampling"]["positive_bin"]),
        easy_negative_bins=list(raw["sampling"]["easy_negative_bins"]),
        max_triplets_per_anchor=int(raw["sampling"]["max_triplets_per_anchor"]),
        use_lora=use_lora,
        lora_r=int(lora_raw.get("lora_r", 16)),
        lora_alpha=int(lora_raw.get("lora_alpha", 32)),
        lora_dropout=float(lora_raw.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        eval_every_steps=int(raw["logging"]["eval_every_steps"]),
        checkpoint_every_epochs=int(raw["checkpointing"]["checkpoint_every_epochs"]),
        resume_training=bool(raw["checkpointing"].get("resume_training", False)),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_lora_target_modules(model_family: str) -> List[str]:
    if model_family == "e5":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_family == "bge":
        return ["query", "key", "value"]
    raise ValueError("model.model_family must be one of: e5, bge")


def format_story_for_model(text: str, cfg: TrainConfig) -> str:
    text = str(text).strip()
    return f"{cfg.story_prefix}{text}" if cfg.story_prefix else text


class PairRegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: TrainConfig):
        self.rows = [
            {
                "text_a": format_story_for_model(r["story_a_text"], cfg),
                "text_b": format_story_for_model(r["story_b_text"], cfg),
                "target": float(r["score_norm"]),
            }
            for _, r in df.iterrows()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class TripletDataset(Dataset):
    def __init__(self, triplets: List[Dict]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def build_triplets(df: pd.DataFrame, cfg: TrainConfig) -> List[Dict]:
    directed = []
    for _, r in df.iterrows():
        bin_id = int(r["sim_bin"])
        score = float(r["score_norm"])
        a_id, b_id = str(r["story_a_id"]), str(r["story_b_id"])
        a_text, b_text = r["story_a_text"], r["story_b_text"]

        directed.append(
            {
                "anchor_id": a_id,
                "anchor_text": a_text,
                "other_id": b_id,
                "other_text": b_text,
                "bin": bin_id,
                "score": score,
            }
        )
        directed.append(
            {
                "anchor_id": b_id,
                "anchor_text": b_text,
                "other_id": a_id,
                "other_text": a_text,
                "bin": bin_id,
                "score": score,
            }
        )

    by_anchor: Dict[str, List[Dict]] = {}
    for row in directed:
        by_anchor.setdefault(row["anchor_id"], []).append(row)

    rng = random.Random(cfg.seed)
    triplets: List[Dict] = []

    for _, rows in by_anchor.items():
        pos = [x for x in rows if x["bin"] == cfg.positive_bin]
        hard = [x for x in rows if x["bin"] == cfg.hard_negative_bin]
        easy = [x for x in rows if x["bin"] in cfg.easy_negative_bins]

        if not pos or not hard or not easy:
            continue

        rng.shuffle(pos)
        max_k = min(len(pos), cfg.max_triplets_per_anchor)

        for p in pos[:max_k]:
            h = rng.choice(hard)
            e = rng.choice(easy)
            triplets.append(
                {
                    "anchor": format_story_for_model(p["anchor_text"], cfg),
                    "positive": format_story_for_model(p["other_text"], cfg),
                    "hard_negative": format_story_for_model(h["other_text"], cfg),
                    "easy_negative": format_story_for_model(e["other_text"], cfg),
                }
            )

    return triplets


def collate_pair(batch):
    return {
        "text_a": [x["text_a"] for x in batch],
        "text_b": [x["text_b"] for x in batch],
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.float32),
    }


def collate_triplet(batch):
    return {
        "anchor": [x["anchor"] for x in batch],
        "positive": [x["positive"] for x in batch],
        "hard_negative": [x["hard_negative"] for x in batch],
        "easy_negative": [x["easy_negative"] for x in batch],
    }


def load_pair_df(cfg: TrainConfig) -> pd.DataFrame:
    with open(cfg.scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    with open(cfg.alignments_path, "r", encoding="utf-8") as f:
        alignments = json.load(f)

    scores_df = pd.DataFrame(scores)
    align_df = pd.DataFrame(
        [
            {
                "pair_id": x.get("pair_id"),
                "story_a_id": (x.get("story_a") or {}).get("id"),
                "story_b_id": (x.get("story_b") or {}).get("id"),
                "story_a_text": (x.get("story_a") or {}).get("narrative"),
                "story_b_text": (x.get("story_b") or {}).get("narrative"),
            }
            for x in alignments
        ]
    )

    pair_df = scores_df.merge(
        align_df[["pair_id", "story_a_text", "story_b_text"]],
        on="pair_id",
        how="left",
    )

    pair_df = pair_df.dropna(
        subset=["story_a_text", "story_b_text", "pred_event_rating_mean_joint"]
    ).copy()
    pair_df["structural_score"] = pair_df["pred_event_rating_mean_joint"].astype(float)

    min_s, max_s = pair_df["structural_score"].min(), pair_df["structural_score"].max()
    pair_df["score_norm"] = (pair_df["structural_score"] - min_s) / (max_s - min_s + 1e-12)

    try:
        pair_df["sim_bin"] = pd.qcut(
            pair_df["score_norm"], q=4, labels=[0, 1, 2, 3], duplicates="raise"
        ).astype(int)
    except ValueError:
        ranked = pair_df["score_norm"].rank(method="first", pct=True)
        pair_df["sim_bin"] = pd.qcut(ranked, q=4, labels=[0, 1, 2, 3]).astype(int)
        print("qcut fallback used: duplicate score edges detected; applied rank-based quartiles.")

    print(f"pairs: {len(pair_df)}")
    print(f"score range: ({float(min_s)}, {float(max_s)})")
    print("bin counts:")
    print(pair_df["sim_bin"].value_counts().sort_index().to_string())
    return pair_df


def latest_checkpoint(checkpoint_dir: Path, loss_type: str) -> Tuple[int, Path | None]:
    if loss_type == "contrastive_mse":
        pattern = "epoch_*_mse_loss"
        parser = lambda n: int(n.split("_")[1])
    else:
        pattern = "epoch_*"
        parser = lambda n: int(n.split("_")[1])

    candidates: List[Tuple[int, Path]] = []
    for d in checkpoint_dir.glob(pattern):
        if not d.is_dir():
            continue
        try:
            candidates.append((parser(d.name), d))
        except Exception:
            continue
    if not candidates:
        return 0, None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def to_serializable_config(cfg: TrainConfig) -> Dict:
    out = cfg.__dict__.copy()
    out["scores_path"] = str(cfg.scores_path)
    out["alignments_path"] = str(cfg.alignments_path)
    out["output_dir"] = str(cfg.output_dir)
    out["hf_cache_dir"] = str(cfg.hf_cache_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train structure-aware E5 embedding model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.loss_type not in {"infonce", "contrastive_mse"}:
        raise ValueError("training.loss_type must be one of: infonce, contrastive_mse")
    if cfg.model_family not in {"e5", "bge"}:
        raise ValueError("model.model_family must be one of: e5, bge")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Model family: {cfg.model_family}")
    print(f"Model name: {cfg.model_name}")
    print(f"Story prefix: {cfg.story_prefix!r}")
    print(f"LoRA enabled: {cfg.use_lora}")
    if cfg.use_lora:
        print(f"LoRA target modules: {cfg.target_modules}")

    # Keep HF cache off home folder by default.
    os.environ["HF_HOME"] = str(cfg.hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cfg.hf_cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cfg.hf_cache_dir / "transformers")

    pair_df = load_pair_df(cfg)
    pair_df = pair_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    cut = int(len(pair_df) * cfg.train_frac)
    train_df, val_df = pair_df.iloc[:cut].copy(), pair_df.iloc[cut:].copy()
    print(f"train pairs: {len(train_df)}")
    print(f"val pairs  : {len(val_df)}")

    if cfg.loss_type == "contrastive_mse":
        train_dataset = PairRegressionDataset(train_df, cfg)
        val_dataset = PairRegressionDataset(val_df, cfg)
        collate_fn = collate_pair
        print("Using contrastive_mse with pair dataset.")
    else:
        train_triplets = build_triplets(train_df, cfg)
        val_triplets = build_triplets(val_df, cfg)
        train_dataset = TripletDataset(train_triplets)
        val_dataset = TripletDataset(val_triplets)
        collate_fn = collate_triplet
        print("Using infonce with triplet dataset.")
        print(f"train triplets: {len(train_dataset)} | val triplets: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and cfg.use_bfloat16) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=str(cfg.hf_cache_dir))
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True

    model_load_path: str | Path = cfg.model_name
    checkpoint_dir = cfg.output_dir / ("checkpoints_mse_loss" if cfg.loss_type == "contrastive_mse" else "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_epoch = 0
    global_step = 0
    latest = None
    if cfg.resume_training:
        resume_epoch, latest = latest_checkpoint(checkpoint_dir, cfg.loss_type)
        if latest is not None:
            print(f"Resuming from checkpoint: {latest} (completed epoch={resume_epoch})")
            if not cfg.use_lora:
                model_load_path = latest
                tokenizer = AutoTokenizer.from_pretrained(str(latest))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        str(model_load_path),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=str(cfg.hf_cache_dir),
    )
    if added_pad_token and model_load_path == cfg.model_name:
        model.resize_token_embeddings(len(tokenizer))

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        if latest is not None and cfg.resume_training:
            model = PeftModel.from_pretrained(model, str(latest), is_trainable=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(latest))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"Warning: tokenizer not loaded from checkpoint: {e}")
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning enabled: trainable params {trainable:,} / {total:,}")

    def encode_texts(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )

    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def embed(texts: List[str]) -> torch.Tensor:
        toks = encode_texts(texts)
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)
        emb = mean_pool(out.last_hidden_state, toks["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def loss_contrastive_mse(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb_a = embed(batch["text_a"])
        emb_b = embed(batch["text_b"])
        pred = F.cosine_similarity(emb_a, emb_b, dim=-1)
        pred_01 = (pred + 1.0) / 2.0
        target = batch["target"].to(device=device, dtype=pred_01.dtype)
        return F.mse_loss(pred_01, target)

    def loss_infonce(batch: Dict[str, List[str]]) -> torch.Tensor:
        anc = embed(batch["anchor"])
        pos = embed(batch["positive"])
        hneg = embed(batch["hard_negative"])
        eneg = embed(batch["easy_negative"])

        pos_sim = F.cosine_similarity(anc, pos, dim=-1)
        hneg_sim = F.cosine_similarity(anc, hneg, dim=-1)
        eneg_sim = F.cosine_similarity(anc, eneg, dim=-1)

        logits = torch.stack([pos_sim, hneg_sim, eneg_sim], dim=1) / cfg.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="eval", leave=False):
                loss = loss_contrastive_mse(batch) if cfg.loss_type == "contrastive_mse" else loss_infonce(batch)
                losses.append(float(loss.item()))
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    num_train_steps = steps_per_epoch * cfg.num_epochs
    num_warmup_steps = int(num_train_steps * cfg.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    if cfg.resume_training:
        if latest is not None:
            global_step = resume_epoch * steps_per_epoch
            for _ in range(global_step):
                scheduler.step()

    print(f"num_train_steps: {num_train_steps} | warmup: {num_warmup_steps}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Start epoch: {resume_epoch + 1} | start global_step: {global_step}")

    if resume_epoch >= cfg.num_epochs:
        print("All configured epochs already completed. Exiting.")
        return

    model.train()
    running: List[float] = []

    for epoch in range(resume_epoch, cfg.num_epochs):
        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            loss = loss_contrastive_mse(batch) if cfg.loss_type == "contrastive_mse" else loss_infonce(batch)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            running.append(float(loss.item()) * cfg.grad_accum_steps)

            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                mean_loss = float(np.mean(running[-50:]))
                pbar.set_postfix({"loss": f"{mean_loss:.4f}", "step": global_step})

                if global_step % cfg.eval_every_steps == 0:
                    val_loss = evaluate(val_loader)
                    print(f"\nstep={global_step} train_loss={mean_loss:.4f} val_loss={val_loss:.4f}")

        val_loss = evaluate(val_loader)
        print(f"Epoch {epoch+1} done | val_loss={val_loss:.4f}")

        if (epoch + 1) % cfg.checkpoint_every_epochs == 0:
            if cfg.loss_type == "contrastive_mse":
                ckpt_path = checkpoint_dir / f"epoch_{epoch+1:02d}_mse_loss"
            else:
                ckpt_path = checkpoint_dir / f"epoch_{epoch+1:02d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            (ckpt_path / "train_config.json").write_text(
                json.dumps(to_serializable_config(cfg), indent=2), encoding="utf-8"
            )
            print(f"Saved checkpoint: {ckpt_path}")

    suffix = "mse_loss" if cfg.loss_type == "contrastive_mse" else cfg.loss_type
    save_dir = cfg.output_dir / f"{cfg.model_name.replace('/', '__')}_{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "train_config.json").write_text(
        json.dumps(to_serializable_config(cfg), indent=2), encoding="utf-8"
    )
    artifact_type = "adapter+tokenizer" if cfg.use_lora else "model+tokenizer"
    print(f"Saved final {artifact_type} to: {save_dir}")


if __name__ == "__main__":
    main()
