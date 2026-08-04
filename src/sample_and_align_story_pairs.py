from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SYSTEM_PROMPT = """You are a careful narrative event alignment assistant.

Your task is to identify structural correspondences between two ordered sequences of eventful sentences extracted from full stories.

Two sentences should be matched if they play a similar conceptual or thematic role in their respective sequence of sentences, even if the level of detail differs.

Important:
- One story may describe an episode using multiple detailed events, while the other may summarize it briefly.
- You may match one event to multiple events.
- You may match multiple events to one event.
- You may match groups of events to groups of events.
- Only match events if they represent the same underlying narrative episode or structural role.
- Be conservative and avoid vague matches.

Return STRICT JSON with this format:

{
  "matches": [
    {
      "a_indices": [<int>, ...],
      "b_indices": [<int>, ...]
    }
  ],
  "unmatched_a": [<int>, ...],
  "unmatched_b": [<int>, ...]
}

Indices are 1-based.
Do not include explanations.
Output JSON only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random story pairs and request event alignments from an LLM."
    )
    parser.add_argument(
        "--events-path",
        type=Path,
        default=Path("data/asq_event_extraction_full.json"),
        help="Path to extracted events JSON.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/Alignment/asq_random_2000_story_pair_alignments.json"),
        help="Output JSON path for alignment results.",
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV of preconstructed pairs. Uses story_id_a/story_id_b if present, "
            "otherwise falls back to i/j indices into --events-path."
        ),
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=2000,
        help="Number of random unique pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible pair sampling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omit for models that only support their default temperature.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save output every N newly processed pairs.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Optional cap on number of LLM calls for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the LLM; store sampled pairs with empty alignment payload.",
    )
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def load_events(events_path: Path) -> List[Dict[str, Any]]:
    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {events_path}, found {type(data).__name__}")

    cleaned: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        rec_id = rec.get("id")
        events = rec.get("events")
        if rec_id is None or not isinstance(events, list):
            continue
        events_str = [str(x) for x in events if str(x).strip()]
        if len(events_str) == 0:
            continue
        cleaned.append(
            {
                "id": str(rec_id),
                "label": rec.get("label"),
                "qn1": rec.get("qn1"),
                "qn2": rec.get("qn2"),
                "narrative": rec.get("narrative"),
                "events": events_str,
            }
        )
    return cleaned


def load_client(project_root: Path):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'openai'. Install it in your environment before running without --dry-run."
        ) from exc

    key_path = project_root / "openai_key.txt"
    if not key_path.exists():
        raise FileNotFoundError(f"OpenAI key file not found: {key_path}")
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"OpenAI key file is empty: {key_path}")
    return OpenAI(api_key=api_key)


def extract_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    except Exception:
        return ""


def parse_alignment_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{(?:.|\n|\r)*\}", text)
    if match:
        candidate = match.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model response did not contain a valid JSON object.")


def normalize_alignment(alignment: Dict[str, Any]) -> Dict[str, Any]:
    matches = alignment.get("matches", [])
    unmatched_a = alignment.get("unmatched_a", [])
    unmatched_b = alignment.get("unmatched_b", [])

    out_matches: List[Dict[str, List[int]]] = []
    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue
            a_indices = item.get("a_indices", [])
            b_indices = item.get("b_indices", [])
            if not isinstance(a_indices, list) or not isinstance(b_indices, list):
                continue
            try:
                out_matches.append(
                    {
                        "a_indices": [int(x) for x in a_indices],
                        "b_indices": [int(x) for x in b_indices],
                    }
                )
            except Exception:
                continue

    try:
        out_unmatched_a = [int(x) for x in unmatched_a] if isinstance(unmatched_a, list) else []
    except Exception:
        out_unmatched_a = []
    try:
        out_unmatched_b = [int(x) for x in unmatched_b] if isinstance(unmatched_b, list) else []
    except Exception:
        out_unmatched_b = []

    return {
        "matches": out_matches,
        "unmatched_a": out_unmatched_a,
        "unmatched_b": out_unmatched_b,
    }


def build_user_prompt(events_a: Sequence[str], events_b: Sequence[str]) -> str:
    payload = {
        "events_a": list(events_a),
        "events_b": list(events_b),
    }
    return (
        "Align these two event sequences.\n"
        "Return only the JSON object in the required schema.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def sample_unique_pairs(num_items: int, num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    if num_items < 2:
        raise ValueError("Need at least 2 stories to create pairs.")
    max_pairs = num_items * (num_items - 1) // 2
    if num_pairs > max_pairs:
        raise ValueError(f"Requested {num_pairs} pairs, but only {max_pairs} unique pairs exist.")

    rng = random.Random(seed)
    pair_set = set()
    while len(pair_set) < num_pairs:
        i, j = rng.sample(range(num_items), 2)
        if i > j:
            i, j = j, i
        pair_set.add((i, j))
    return list(pair_set)


def normalize_text_key(text: Any) -> str:
    return " ".join(str(text or "").split())


def text_to_events(text: Any) -> List[str]:
    text_norm = normalize_text_key(text)
    if not text_norm:
        return []
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text_norm)
        if len(s.strip()) > 0
    ]
    return sentences if sentences else [text_norm]


def make_csv_text_record(row_num: int, side: str, text: Any) -> Dict[str, Any]:
    narrative = normalize_text_key(text)
    if not narrative:
        raise ValueError(f"Missing story_text_{side} on row {row_num}")
    digest = hashlib.sha1(narrative.encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"csv_text_{side}_{digest}",
        "label": None,
        "qn1": None,
        "qn2": None,
        "narrative": narrative,
        "events": text_to_events(narrative),
    }


def load_pairs_from_csv(
    pairs_path: Path, events_records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    id_to_record = {rec["id"]: rec for rec in events_records}
    text_to_id: Dict[str, str] = {}
    for rec in events_records:
        key = normalize_text_key(rec.get("narrative"))
        if key and key not in text_to_id:
            text_to_id[key] = rec["id"]

    pairs: List[Dict[str, Any]] = []
    extra_records: Dict[str, Dict[str, Any]] = {}

    def id_from_text_or_create(row: Dict[str, str], row_num: int, side: str) -> str:
        key = normalize_text_key(row.get(f"story_text_{side}"))
        if key in text_to_id:
            return text_to_id[key]
        rec = make_csv_text_record(row_num, side, key)
        extra_records[rec["id"]] = rec
        text_to_id[key] = rec["id"]
        return rec["id"]

    with open(pairs_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_ids = {"story_id_a", "story_id_b"}.issubset(fieldnames)
        has_indices = {"i", "j"}.issubset(fieldnames)
        has_text = {"story_text_a", "story_text_b"}.issubset(fieldnames)
        if not has_ids and not has_indices and not has_text:
            raise ValueError(
                f"{pairs_path} must contain story_id_a/story_id_b, i/j, or story_text_a/story_text_b columns."
            )

        for row_num, row in enumerate(reader, start=2):
            try:
                if has_ids and row.get("story_id_a") and row.get("story_id_b"):
                    a_id = str(row["story_id_a"])
                    b_id = str(row["story_id_b"])
                elif has_indices:
                    i = int(row["i"])
                    j = int(row["j"])
                    if 0 <= i < len(events_records) and 0 <= j < len(events_records):
                        a_id = events_records[i]["id"]
                        b_id = events_records[j]["id"]
                    elif has_text:
                        a_id = id_from_text_or_create(row, row_num, "a")
                        b_id = id_from_text_or_create(row, row_num, "b")
                    else:
                        raise IndexError(
                            f"i/j out of range for events file with {len(events_records)} records: i={i}, j={j}"
                        )
                else:
                    a_id = id_from_text_or_create(row, row_num, "a")
                    b_id = id_from_text_or_create(row, row_num, "b")
            except Exception as exc:
                raise ValueError(f"Could not parse pair row {row_num} in {pairs_path}: {row}") from exc

            if a_id not in id_to_record and a_id not in extra_records:
                raise KeyError(f"story_id_a={a_id!r} from row {row_num} not found in events file")
            if b_id not in id_to_record and b_id not in extra_records:
                raise KeyError(f"story_id_b={b_id!r} from row {row_num} not found in events file")

            pair_id = row.get("pair_id") or f"{a_id}__{b_id}"
            pairs.append(
                {
                    "pair_id": str(pair_id),
                    "a_id": a_id,
                    "b_id": b_id,
                    "pair_metadata": {
                        k: v
                        for k, v in row.items()
                        if k not in {"story_text_a", "story_text_b"}
                    },
                }
            )

    return pairs, extra_records


def read_existing_results(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {out_path}, found {type(data).__name__}")
    return data


def save_results(out_path: Path, results: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    events_path = resolve_path(args.events_path, project_root)
    out_path = resolve_path(args.out_path, project_root)
    pairs_path = resolve_path(args.pairs_path, project_root) if args.pairs_path else None

    events_records = load_events(events_path)
    id_to_record = {rec["id"]: rec for rec in events_records}

    if pairs_path is not None:
        sampled_pairs, extra_records = load_pairs_from_csv(pairs_path, events_records)
        id_to_record.update(extra_records)
        pair_source = str(pairs_path)
    else:
        sampled_index_pairs = sample_unique_pairs(
            num_items=len(events_records), num_pairs=args.num_pairs, seed=args.seed
        )
        sampled_pairs = [
            {
                "pair_id": f"{events_records[i]['id']}__{events_records[j]['id']}",
                "a_id": events_records[i]["id"],
                "b_id": events_records[j]["id"],
                "pair_metadata": {},
            }
            for i, j in sampled_index_pairs
        ]
        pair_source = f"random(num_pairs={args.num_pairs}, seed={args.seed})"

    existing_results = read_existing_results(out_path)
    done_pair_ids = {
        str(item.get("pair_id"))
        for item in existing_results
        if isinstance(item, dict) and item.get("pair_id") is not None
    }

    worklist = []
    for pair in sampled_pairs:
        pair_id = pair["pair_id"]
        if pair_id in done_pair_ids:
            continue
        worklist.append(pair)

    print(f"Events loaded: {len(events_records)} from {events_path}")
    print(f"Pair source: {pair_source}")
    print(f"Target sampled pairs: {len(sampled_pairs)}")
    print(f"Existing results in file: {len(existing_results)}")
    print(f"Remaining pairs to process this run: {len(worklist)}")
    print(f"Output file: {out_path}")

    client = None if args.dry_run else load_client(project_root)
    results = existing_results.copy()

    processed = 0
    for pair in worklist:
        pair_id = pair["pair_id"]
        a_id = pair["a_id"]
        b_id = pair["b_id"]
        pair_metadata = pair.get("pair_metadata") or {}

        rec_a = id_to_record[a_id]
        rec_b = id_to_record[b_id]

        if args.dry_run:
            raw = ""
            alignment = {"matches": [], "unmatched_a": [], "unmatched_b": []}
            ok = True
            error = None
        else:
            user_prompt = build_user_prompt(rec_a["events"], rec_b["events"])
            try:
                request_kwargs = {
                    "model": args.model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if args.temperature is not None:
                    request_kwargs["temperature"] = args.temperature

                resp = client.responses.create(**request_kwargs)
                raw = extract_response_text(resp)
                parsed = parse_alignment_json(raw)
                alignment = normalize_alignment(parsed)
                ok = True
                error = None
            except Exception as exc:
                raw = ""
                alignment = {"matches": [], "unmatched_a": [], "unmatched_b": []}
                ok = False
                error = str(exc)

        results.append(
            {
                "pair_id": pair_id,
                "story_a": {
                    "id": rec_a["id"],
                    "label": rec_a.get("label"),
                    "qn1": rec_a.get("qn1"),
                    "qn2": rec_a.get("qn2"),
                    "narrative": rec_a.get("narrative"),
                    "events": rec_a["events"],
                },
                "story_b": {
                    "id": rec_b["id"],
                    "label": rec_b.get("label"),
                    "qn1": rec_b.get("qn1"),
                    "qn2": rec_b.get("qn2"),
                    "narrative": rec_b.get("narrative"),
                    "events": rec_b["events"],
                },
                "model_output_raw": raw,
                "alignment": alignment,
                "ok": ok,
                "error": error,
                "model": args.model,
                "temperature": args.temperature,
                "pair_metadata": pair_metadata,
            }
        )

        processed += 1
        if processed % args.checkpoint_every == 0:
            save_results(out_path, results)
            print(f"Checkpoint saved: +{processed} this run | total={len(results)}")

        if args.max_calls is not None and processed >= args.max_calls:
            print(f"Reached --max-calls={args.max_calls}; stopping early.")
            break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    save_results(out_path, results)
    num_ok = sum(1 for x in results if isinstance(x, dict) and x.get("ok") is True)
    num_err = sum(1 for x in results if isinstance(x, dict) and x.get("ok") is False)
    print(f"Done. Total saved records: {len(results)} | ok={num_ok} | failed={num_err}")


if __name__ == "__main__":
    main()
