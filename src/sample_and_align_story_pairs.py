from __future__ import annotations

import argparse
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
        default="gpt-oss-120b",
        help="SambaNova model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for chat completion.",
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
        from sambanova import SambaNova
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'sambanova'. Install it in your environment before running without --dry-run."
        ) from exc

    key_path = project_root / "sambanova_key.txt"
    if not key_path.exists():
        raise FileNotFoundError(f"SambaNova key file not found: {key_path}")
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"SambaNova key file is empty: {key_path}")
    return SambaNova(base_url="https://api.sambanova.ai/v1", api_key=api_key)


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

    events_records = load_events(events_path)
    id_to_record = {rec["id"]: rec for rec in events_records}

    sampled_index_pairs = sample_unique_pairs(
        num_items=len(events_records), num_pairs=args.num_pairs, seed=args.seed
    )
    sampled_pairs: List[Tuple[str, str]] = [
        (events_records[i]["id"], events_records[j]["id"]) for i, j in sampled_index_pairs
    ]

    existing_results = read_existing_results(out_path)
    done_pair_ids = {
        str(item.get("pair_id"))
        for item in existing_results
        if isinstance(item, dict) and item.get("pair_id") is not None
    }

    worklist = []
    for a_id, b_id in sampled_pairs:
        pair_id = f"{a_id}__{b_id}"
        if pair_id in done_pair_ids:
            continue
        worklist.append((pair_id, a_id, b_id))

    print(f"Events loaded: {len(events_records)} from {events_path}")
    print(f"Target sampled pairs: {len(sampled_pairs)} (seed={args.seed})")
    print(f"Existing results in file: {len(existing_results)}")
    print(f"Remaining pairs to process this run: {len(worklist)}")
    print(f"Output file: {out_path}")

    client = None if args.dry_run else load_client(project_root)
    results = existing_results.copy()

    processed = 0
    for pair_id, a_id, b_id in worklist:
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
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                )
                raw = resp.choices[0].message.content
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
