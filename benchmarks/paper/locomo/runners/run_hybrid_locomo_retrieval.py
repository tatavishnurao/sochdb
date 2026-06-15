#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_utils import Embedder, normalize_text
from lexical_search import TrigramIndex
from reranker_utils import Reranker
from sochdb_batch_client import search_batch as search_sochdb_batch


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def approx_tokens(text: str) -> int:
    return len(text.split())


def _format_date(dt: datetime) -> str:
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def _parse_timestamp_date(timestamp: str) -> Optional[datetime]:
    match = re.search(
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+\d{1,2}\s+[A-Za-z]+,\s+\d{4}\b",
        timestamp,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    timestamp_text = re.sub(
        r"\b(am|pm)\b",
        lambda m: m.group(1).upper(),
        match.group(0),
        flags=re.IGNORECASE,
    )
    try:
        return datetime.strptime(timestamp_text, "%I:%M %p on %d %B, %Y")
    except ValueError:
        return None


def _resolved_temporal_lines(text: str, timestamp: str) -> List[str]:
    base_dt = _parse_timestamp_date(timestamp)
    if base_dt is None:
        return []

    offsets = {
        "today": 0,
        "yesterday": -1,
        "tomorrow": 1,
    }
    lines = []
    lower_text = text.lower()
    for word, offset in offsets.items():
        if re.search(rf"\b{word}\b", lower_text):
            resolved_dt = base_dt + timedelta(days=offset)
            lines.append(f"resolved_time: {word} = {_format_date(resolved_dt)}")
    return lines


def render_memory_text(memory: dict, mode: str) -> str:
    text = str(memory.get("text", ""))
    if mode == "raw":
        return text
    if mode == "speaker":
        lines = []
        speaker = memory.get("speaker")
        if speaker is not None and speaker != "":
            lines.append(f"speaker: {speaker}")
        lines.append(f"text: {text}")
        return "\n".join(lines)
    if mode == "speaker_time":
        lines = []
        for field in ["speaker", "timestamp"]:
            value = memory.get(field)
            if value is not None and value != "":
                lines.append(f"{field}: {value}")
        timestamp = str(memory.get("timestamp", ""))
        if timestamp:
            lines.extend(_resolved_temporal_lines(text, timestamp))
        lines.append(f"text: {text}")
        return "\n".join(lines)
    if mode == "speaker_session":
        lines = []
        for field in ["speaker", "session", "dia_id"]:
            value = memory.get(field)
            if value is not None and value != "":
                lines.append(f"{field}: {value}")
        lines.append(f"text: {text}")
        return "\n".join(lines)
    if mode != "metadata":
        raise ValueError(f"Unknown memory render mode: {mode}")

    lines = []
    metadata_fields = [
        "speaker",
        "timestamp",
        "session",
        "session_num",
        "dia_id",
        "dia_num",
        "sample_id",
        "speaker_a",
        "speaker_b",
    ]
    for field in metadata_fields:
        value = memory.get(field)
        if value is not None and value != "":
            lines.append(f"{field}: {value}")

    lines.append(f"text: {text}")
    timestamp = str(memory.get("timestamp", ""))
    if timestamp:
        lines.extend(_resolved_temporal_lines(text, timestamp))
    return "\n".join(lines)


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def topk_local_vector(
    query_vec: List[float],
    memory_ids: List[int],
    memory_vecs: List[List[float]],
    k: int,
) -> List[int]:
    scored = []
    for mid, vec in zip(memory_ids, memory_vecs):
        scored.append((cosine(query_vec, vec), mid))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:k]]


def rrf_fuse(
    bm25_ranked: List[int],
    vector_ranked: List[int],
    final_k: int,
    rrf_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> List[int]:
    ranked = rrf_fuse_with_scores(
        bm25_ranked=bm25_ranked,
        vector_ranked=vector_ranked,
        final_k=final_k,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
    return [mid for mid, _ in ranked]


def rrf_fuse_with_scores(
    bm25_ranked: List[int],
    vector_ranked: List[int],
    final_k: int,
    rrf_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}

    for rank, mid in enumerate(bm25_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, mid in enumerate(vector_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + vector_weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:final_k]


def rrf_fuse_three_legs(
    bm25_ranked: List[int],
    vector_ranked: List[int],
    grep_ranked: List[int],
    final_k: int,
    rrf_k: int,
    bm25_weight: float,
    vector_weight: float,
    grep_weight: float,
) -> List[Tuple[int, float]]:
    """RRF fusion for three legs: BM25, vector, and grep."""
    scores: Dict[int, float] = {}

    for rank, mid in enumerate(bm25_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, mid in enumerate(vector_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + vector_weight / (rrf_k + rank)

    for rank, mid in enumerate(grep_ranked, start=1):
        scores[mid] = scores.get(mid, 0.0) + grep_weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:final_k]


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {mid: 1.0 for mid in scores}

    spread = max_score - min_score
    return {mid: (score - min_score) / spread for mid, score in scores.items()}


def group_by_sample(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["sample_id"], []).append(r)
    return out


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))[:80]


def import_sochdb_client():
    import sochdb
    if hasattr(sochdb, "SochDBClient"):
        return sochdb.SochDBClient
    from sochdb.grpc_client import SochDBClient
    return SochDBClient


def import_local_hnsw():
    from sochdb import HnswIndex
    return HnswIndex


def parse_port(value: Optional[str]) -> int:
    if value is None or str(value).strip() == "":
        return 50051
    return int(value)


def create_sochdb_index(client, index_name: str, dim: int):
    try:
        return client.create_index(name=index_name, dimension=dim)
    except Exception as e:
        print(f"[warn] create_index failed/exists for {index_name}: {e}")
        return None


def insert_sochdb_vectors(client, index_name: str, ids: List[int], vectors: List[List[float]]):
    return client.insert_vectors(index_name=index_name, ids=ids, vectors=vectors)


def search_sochdb(client, index_name: str, query_vec: List[float], k: int) -> List[int]:
    raw = client.search(index_name=index_name, query=query_vec, k=k)

    if hasattr(raw, "results"):
        items = raw.results
    elif isinstance(raw, dict) and "results" in raw:
        items = raw["results"]
    else:
        items = raw

    out = []
    for item in items:
        if isinstance(item, dict):
            raw_id = item.get("id") or item.get("vector_id")
        else:
            raw_id = getattr(item, "id", None) or getattr(item, "vector_id", None)

        if raw_id is None:
            continue
        try:
            out.append(int(raw_id))
        except Exception:
            pass

    return out


def search_sochdb_batch_ids(
    client,
    index_name: str,
    query_vecs: List[List[float]],
    k: int,
    ef: int = 0,
) -> List[List[int]]:
    raw_batches = search_sochdb_batch(
        client=client,
        index_name=index_name,
        queries=query_vecs,
        k=k,
        ef=ef,
    )
    out: List[List[int]] = []

    for items in raw_batches:
        batch_ids: List[int] = []
        for item in items:
            if isinstance(item, dict):
                raw_id = item.get("id") or item.get("vector_id")
            else:
                raw_id = getattr(item, "id", None) or getattr(item, "vector_id", None)

            if raw_id is None:
                continue
            try:
                batch_ids.append(int(raw_id))
            except Exception:
                pass
        out.append(batch_ids)

    return out


def build_context(
    memory_ids: List[int],
    id_to_memory: Dict[int, Dict[str, Any]],
    retrieved_id_mode: str = "memory",
) -> str:
    parts = []
    for mid in memory_ids:
        m = id_to_memory.get(mid)
        if not m:
            continue
        if retrieved_id_mode == "parent":
            parts.append(
                f"[view_memory_id={mid} parent_memory_id={m.get('parent_memory_id', mid)} memory_view={m.get('memory_view', 'raw_turn')} "
                f"sample_id={m.get('sample_id')} "
                f"type={m.get('memory_type', 'raw_turn')} "
                f"session={m.get('session')} dia_id={m.get('dia_id')} "
                f"speaker={m.get('speaker')}] {m.get('text', '')}"
            )
        else:
            parts.append(
                f"[memory_id={mid} sample_id={m.get('sample_id')} "
                f"type={m.get('memory_type', 'raw_turn')} "
                f"session={m.get('session')} dia_id={m.get('dia_id')} "
                f"speaker={m.get('speaker')}] {m.get('text', '')}"
            )
    return "\n".join(parts)


def resolve_parent_ids(
    memory_ids: List[int],
    id_to_memory: Dict[int, Dict[str, Any]],
    k: int,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Deduplicate ranked memory IDs by parent_memory_id, preserving rank.
    Returns (parent_ids, view_ids, memory_views).
    """
    seen_parents = set()
    parent_ids = []
    view_ids = []
    views = []
    for mid in memory_ids:
        m = id_to_memory.get(mid)
        if not m:
            continue
        parent_id = m.get("parent_memory_id", mid)
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        parent_ids.append(parent_id)
        view_ids.append(mid)
        views.append(m.get("memory_view", "raw_turn"))
        if len(parent_ids) >= k:
            break
    return parent_ids, view_ids, views


_EVENT_TYPE_RULES = [
    ("identity_support", [
        "identity", "support group", "lgbt", "lgbtq", "community", "belong",
        "pride", "coming out", "trans ", "transgender", "queer", "ally",
        "marginalized", "inclusive", "acceptance", "representation",
    ]),
    ("health_lifestyle", [
        "health", "doctor", "hospital", "clinic", "exercise", "diet", "weight",
        "therapy", "therapist", "counseling", "counselor", "workout", "fitness",
        "medicine", "medication", "symptoms", "illness", "sick", "wellness",
        "yoga", "meditat", "mental health", "mental", "anxiety", "depress", "stress",
        "gym", "runn", "swimm", "work out", "checkup", "appointment with dr",
    ]),
    ("career_goal", [
        "job", "career", "profession", "employment", "hiring", "fired", "promot",
        "salary", "office", "company", "boss", "interview", "occupation", "vocation",
        "work as", "working as", "position", "role at", "field of",
    ]),
    ("family_relationship", [
        "family", "mother", "father", "mom", "dad", "sibling", "brother", "sister",
        "parent", "child", "daughter", "son", "partner", "husband", "wife",
        "married", "marriage", "divorce", "engaged", "fiance", "baby", "pregnan",
        "grandparent", "aunt", "uncle", "cousin", "relative", "in-law",
    ]),
    ("food_recipe", [
        "food", "cook", "recipe", "meal", "restaurant", "dinner", "lunch",
        "breakfast", "bak", "chef", "cuisine", "dish", "ingredient", "flavor",
        "taste", "kitchen", "dining", "ate", "eat", "snack", "brunch", "cafe",
    ]),
    ("pet_animal", [
        "pet", "dog", "cat", "animal", "puppy", "kitten", "bird", "fish", "rabbit",
        "hamster", "horse", "vet", "breed", "adopt ", "kennel", "leash", "walk the",
    ]),
    ("travel_outdoors", [
        "travel", "trip", "vacation", "holiday", "outdoor", "hiking", "hike",
        "walk ", "park", "beach", "mountain", "camp", "flight", "road trip",
        "scenic", "nature", "trail", "fishing", "boat", "island", "backpack",
    ]),
    ("creative_work", [
        "art", "music", "write", "writing", "book", "creative", "painting", "poem",
        "song", "novel", "story", "compose", "craft", "design", "photograph",
        "draw", "sculpture", "instrument", "guitar", "piano", "sing", "dance",
        "theater", "theatre", "acting", "perform",
    ]),
    ("preference", [
        "like", "love", "prefer", "favorite", "favourite", "enjoy", "fond of",
        "keen on", "adore", "hate", "dislike", "can't stand", "best", "worst",
        "would rather", "into ", "really into", "big fan", "not a fan",
    ]),
    ("recommendation_advice", [
        "should", "recommend", "suggest", "advice", "advise", "try", "consider",
        "ought", "worth", "important", "must", "need to", "you can", "would be good",
        "tip", "suggestion", "have you tried", "you might want",
    ]),
    ("activity", [
        "activity", "event", "gathering", "concert", "game", "party", "celebration",
        "hang out", "hangout", "movie", "film", "show", "class", "course", "lesson",
        "club", "meetup", "went to", "going to", "attended", "visit", "trip to",
    ]),
]

_ENTITY_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "did", "does", "will", "would",
    "could", "should", "shall", "may", "might", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "over", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "about", "up",
    "out", "if", "it", "its", "this", "that", "these", "those", "what", "which",
    "who", "whom", "whose", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "him", "her", "his", "they", "them", "their", "am", "got", "went",
    "go", "going", "like", "really", "thing", "things", "know", "think", "thought",
    "said", "say", "yeah", "yes", "oh", "well", "actually", "basically",
    "that's", "it's", "i'm", "don't", "didn't", "doesn't", "can't", "won't",
    "couldn't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
})


def _classify_event_type(text: str, speaker: str) -> str:
    if not text:
        return "generic_event"
    lower = text.lower()
    words = set(re.findall(r"[a-z]+", lower))
    for event_type, keywords in _EVENT_TYPE_RULES:
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if " " in kw_lower:
                if kw_lower in lower:
                    return event_type
            else:
                if kw_lower in words:
                    return event_type
    return "generic_event"


def _extract_event_fact(text: str, speaker: str, event_type: str) -> str:
    if event_type == "generic_event":
        if speaker:
            return f"{speaker} said or indicated: {text}"
        return text
    speaker_tag = f"{speaker} " if speaker else ""
    lower = text.lower()
    if event_type == "career_goal":
        for phrase in ["want to be", "want to become", "dream of", "aspires to",
                        "thinking about", "interested in", "considering",
                        "looking into", "plan to", "plans to", "hoping to"]:
            if phrase in lower:
                return f"{speaker_tag}is interested in career direction related to: {text}"
        return f"{speaker_tag}discussed career-related topics: {text}"
    if event_type == "health_lifestyle":
        return f"{speaker_tag}discussed health or lifestyle matters: {text}"
    if event_type == "family_relationship":
        return f"{speaker_tag}mentioned family or relationships: {text}"
    if event_type == "activity":
        return f"{speaker_tag}described an activity or event: {text}"
    if event_type == "preference":
        return f"{speaker_tag}expressed a preference: {text}"
    if event_type == "recommendation_advice":
        return f"{speaker_tag}gave or received advice or a recommendation: {text}"
    if event_type == "travel_outdoors":
        return f"{speaker_tag}discussed travel or outdoor activities: {text}"
    if event_type == "creative_work":
        return f"{speaker_tag}discussed creative work or interests: {text}"
    if event_type == "identity_support":
        return f"{speaker_tag}discussed identity or community support: {text}"
    if event_type == "pet_animal":
        return f"{speaker_tag}mentioned pets or animals: {text}"
    if event_type == "food_recipe":
        return f"{speaker_tag}discussed food, cooking, or dining: {text}"
    return f"{speaker_tag}said or indicated: {text}"


def _build_event_view_text(memory: dict) -> str:
    speaker = str(memory.get("speaker", "") or "").strip()
    text = str(memory.get("text", "") or "").strip()
    timestamp = str(memory.get("timestamp", "") or "").strip()
    session = str(memory.get("session", "") or "").strip()
    dia_id = str(memory.get("dia_id", "") or "").strip()
    event_type = _classify_event_type(text, speaker)
    fact = _extract_event_fact(text, speaker, event_type)
    lines = []
    if speaker:
        lines.append(f"speaker: {speaker}")
    if timestamp:
        lines.append(f"timestamp: {timestamp}")
    if session:
        lines.append(f"session: {session}")
    if dia_id:
        lines.append(f"dia_id: {dia_id}")
    lines.append(f"event_type: {event_type}")
    lines.append(f"fact: {fact}")
    lines.append(f"text: {text}")
    return "\n".join(lines)


def _extract_entities(memory: dict) -> tuple:
    text = str(memory.get("text", "") or "")
    speaker = str(memory.get("speaker", "") or "").strip()
    entities = set()
    attributes = set()

    if speaker:
        entities.add(speaker)

    words = text.split()
    for i, w in enumerate(words):
        stripped = w.rstrip("?,.!;:'\")")
        if not stripped or len(stripped) <= 1:
            continue
        if stripped[0].isupper() and stripped.lower() not in _ENTITY_STOP_WORDS:
            entities.add(stripped)

    lower_words = [w.rstrip("?,.!;:'\")").lower() for w in words if len(w) > 2]
    content_words = [w for w in lower_words if w not in _ENTITY_STOP_WORDS and not w.isdigit()]

    for w in content_words[:10]:
        entities.add(w)

    if speaker:
        for w in content_words[:5]:
            attributes.add(f"{speaker.lower()} {w}")

    return sorted(entities), sorted(attributes)


def _build_entity_view_text(memory: dict) -> str:
    speaker = str(memory.get("speaker", "") or "").strip()
    text = str(memory.get("text", "") or "").strip()
    entities, attributes = _extract_entities(memory)
    lines = []
    if speaker:
        lines.append(f"speaker: {speaker}")
    lines.append(f"entities: {', '.join(entities)}")
    lines.append(f"attributes: {', '.join(attributes)}")
    lines.append(f"text: {text}")
    return "\n".join(lines)


def _group_memories_by_session(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for m in memories:
        session = str(m.get("session", "") or "").strip()
        if not session:
            continue
        groups.setdefault(session, []).append(m)
    for session in groups:
        groups[session].sort(key=lambda m: (_parse_int(m.get("dia_num", "")) or 0, str(m.get("dia_id", ""))))
    return groups


def _build_neighbor_window_view_text(
    memory: dict,
    session_groups: Dict[str, List[Dict[str, Any]]],
    view_window_radius: int,
) -> str:
    speaker = str(memory.get("speaker", "") or "").strip()
    text = str(memory.get("text", "") or "").strip()
    session = str(memory.get("session", "") or "").strip()
    dia_id = str(memory.get("dia_id", "") or "").strip()
    sample_id = str(memory.get("sample_id", "") or "").strip()

    lines = []
    if sample_id:
        lines.append(f"sample_id: {sample_id}")
    if session:
        lines.append(f"session: {session}")
    if dia_id:
        lines.append(f"dia_id: {dia_id}")
    if speaker:
        lines.append(f"speaker: {speaker}")
    lines.append(f"text: {text}")

    if session and session in session_groups:
        group = session_groups[session]
        target_idx = None
        for i, m in enumerate(group):
            if int(m.get("memory_id", -1)) == int(memory.get("memory_id", -1)):
                target_idx = i
                break
        if target_idx is not None:
            for offset in range(-view_window_radius, view_window_radius + 1):
                if offset == 0:
                    continue
                neighbor_idx = target_idx + offset
                if 0 <= neighbor_idx < len(group):
                    neighbor = group[neighbor_idx]
                    n_speaker = str(neighbor.get("speaker", "") or "").strip()
                    n_text = str(neighbor.get("text", "") or "").strip()
                    n_dia_id = str(neighbor.get("dia_id", "") or "").strip()
                    prefix = "prev" if offset < 0 else "next"
                    turn_line = f"{prefix}_turn(offset={offset:+d}"
                    if n_dia_id:
                        turn_line += f" dia_id={n_dia_id}"
                    turn_line += ")"
                    if n_speaker:
                        turn_line += f" {n_speaker}: {n_text}"
                    else:
                        turn_line += f" {n_text}"
                    lines.append(turn_line)

    return "\n".join(lines)


_MULTIVIEW_ID_FACTOR = 1000
_MULTIVIEW_VIEW_NAME_TO_TYPE = {
    "turn": "turn_view",
    "event": "event_view",
    "entity": "entity_view",
    "neighbor_window": "neighbor_window_view",
}
_MULTIVIEW_VIEW_OFFSETS = {
    "turn_view": 0,
    "event_view": 1,
    "entity_view": 2,
    "neighbor_window_view": 3,
}
_MULTIVIEW_DEFAULT_VIEW_NAMES = tuple(_MULTIVIEW_VIEW_NAME_TO_TYPE.keys())
_MULTIVIEW_VIEWS_PER_MEMORY = len(_MULTIVIEW_DEFAULT_VIEW_NAMES)


def parse_memory_view_types(value: str) -> List[str]:
    raw_types = [v.strip() for v in str(value).split(",") if v.strip()]
    invalid = [v for v in raw_types if v not in _MULTIVIEW_VIEW_NAME_TO_TYPE]
    if invalid:
        valid = ",".join(_MULTIVIEW_DEFAULT_VIEW_NAMES)
        raise ValueError(f"Invalid memory view type(s): {','.join(invalid)}. Valid values: {valid}")
    deduped = []
    seen = set()
    for view_name in raw_types:
        if view_name not in seen:
            deduped.append(view_name)
            seen.add(view_name)
    if not deduped:
        raise ValueError("At least one memory view type must be selected")
    return deduped


def compute_view_overfetch(
    memory_view_mode: str,
    candidate_k: int,
    source_memory_count: int,
    active_view_count: int = _MULTIVIEW_VIEWS_PER_MEMORY,
) -> Tuple[int, int]:
    if memory_view_mode != "multiview":
        return candidate_k, candidate_k
    active_view_count = max(1, active_view_count)
    max_views = source_memory_count * active_view_count
    view_candidate_k = min(candidate_k * active_view_count, max_views)
    return candidate_k, view_candidate_k


def build_memory_search_records(
    memories: List[Dict[str, Any]],
    memory_render_mode: str,
    memory_view_mode: str,
    view_window_radius: int,
    memory_view_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if memory_view_mode == "turn":
        records = []
        for m in memories:
            mid = int(m["memory_id"])
            records.append({
                "record_id": mid,
                "source_memory_id": mid,
                "view_type": "turn_view",
                "view_id_str": f"{mid}::turn",
                "rendered_text": render_memory_text(m, memory_render_mode),
            })
        return records

    session_groups = _group_memories_by_session(memories)
    records = []
    active_view_names = memory_view_types or list(_MULTIVIEW_DEFAULT_VIEW_NAMES)
    active_view_types = [_MULTIVIEW_VIEW_NAME_TO_TYPE[v] for v in active_view_names]

    for m in memories:
        mid = int(m["memory_id"])
        rid_base = mid * _MULTIVIEW_ID_FACTOR

        if "turn_view" in active_view_types:
            records.append({
                "record_id": rid_base + _MULTIVIEW_VIEW_OFFSETS["turn_view"],
                "source_memory_id": mid,
                "view_type": "turn_view",
                "view_id_str": f"{mid}::turn",
                "rendered_text": render_memory_text(m, memory_render_mode),
            })

        if "event_view" in active_view_types:
            records.append({
                "record_id": rid_base + _MULTIVIEW_VIEW_OFFSETS["event_view"],
                "source_memory_id": mid,
                "view_type": "event_view",
                "view_id_str": f"{mid}::event",
                "rendered_text": _build_event_view_text(m),
            })

        if "entity_view" in active_view_types:
            records.append({
                "record_id": rid_base + _MULTIVIEW_VIEW_OFFSETS["entity_view"],
                "source_memory_id": mid,
                "view_type": "entity_view",
                "view_id_str": f"{mid}::entity",
                "rendered_text": _build_entity_view_text(m),
            })

        if "neighbor_window_view" in active_view_types:
            records.append({
                "record_id": rid_base + _MULTIVIEW_VIEW_OFFSETS["neighbor_window_view"],
                "source_memory_id": mid,
                "view_type": "neighbor_window_view",
                "view_id_str": f"{mid}::neighbor_window",
                "rendered_text": _build_neighbor_window_view_text(m, session_groups, view_window_radius),
            })

    return records


def compute_multiview_diagnostics(
    candidate_ids: List[int],
    rrf_scores: Dict[int, float],
    record_to_source: Dict[int, int],
    record_to_view_type: Dict[int, str],
    memory_view_mode: str,
) -> Dict[str, Any]:
    if memory_view_mode != "multiview":
        return {}

    raw_view_candidate_count = len(candidate_ids)

    source_to_views: Dict[int, List[str]] = {}
    for rid in candidate_ids:
        sid = record_to_source.get(rid, rid)
        vt = record_to_view_type.get(rid, "turn_view")
        source_to_views.setdefault(sid, []).append(vt)

    unique_source_candidate_count = len(source_to_views)
    duplicate_view_candidate_count = raw_view_candidate_count - unique_source_candidate_count

    view_type_counts_before_dedup: Dict[str, int] = {}
    for rid in candidate_ids:
        vt = record_to_view_type.get(rid, "turn_view")
        view_type_counts_before_dedup[vt] = view_type_counts_before_dedup.get(vt, 0) + 1

    best_view_per_source: Dict[int, str] = {}
    best_score_per_source: Dict[int, float] = {}
    for rid in candidate_ids:
        sid = record_to_source.get(rid, rid)
        score = rrf_scores.get(rid, 0.0)
        if sid not in best_score_per_source or score > best_score_per_source[sid]:
            best_score_per_source[sid] = score
            best_view_per_source[sid] = record_to_view_type.get(rid, "turn_view")

    view_type_counts_after_dedup: Dict[str, int] = {}
    for vt in best_view_per_source.values():
        view_type_counts_after_dedup[vt] = view_type_counts_after_dedup.get(vt, 0) + 1

    sources_with_multiple_view_hits_count = sum(
        1 for views in source_to_views.values() if len(views) > 1
    )
    max_views_per_source_in_candidates = max(
        (len(views) for views in source_to_views.values()), default=0
    )

    return {
        "raw_view_candidate_count": raw_view_candidate_count,
        "unique_source_candidate_count": unique_source_candidate_count,
        "duplicate_view_candidate_count": duplicate_view_candidate_count,
        "view_type_counts_before_dedup": view_type_counts_before_dedup,
        "view_type_counts_after_dedup": view_type_counts_after_dedup,
        "sources_with_multiple_view_hits_count": sources_with_multiple_view_hits_count,
        "max_views_per_source_in_candidates": max_views_per_source_in_candidates,
    }


def dedup_view_hits_to_source_ids(
    ranked_record_ids: List[int],
    record_scores: Dict[int, float],
    record_to_source: Dict[int, int],
    k: int,
) -> Tuple[List[int], Dict[int, float]]:
    source_scores: Dict[int, float] = {}
    source_order: Dict[int, int] = {}
    order = 0
    for rid in ranked_record_ids:
        sid = record_to_source.get(rid, rid)
        score = record_scores.get(rid, 0.0)
        if sid not in source_scores:
            source_scores[sid] = score
            source_order[sid] = order
            order += 1
        else:
            if score > source_scores[sid]:
                source_scores[sid] = score

    ranked = sorted(source_scores.items(), key=lambda x: (-x[1], source_order.get(x[0], 0)))
    deduped_ids = [sid for sid, _ in ranked[:k]]
    deduped_scores = {sid: score for sid, score in ranked[:k]}
    return deduped_ids, deduped_scores


_QUESTION_WORDS = frozenset({
    "who", "what", "where", "when", "why", "how",
    "which", "whom", "whose", "does", "did", "is",
    "are", "was", "were", "can", "could", "would",
    "should", "will", "shall", "has", "have", "had",
    "do", "did", "the", "a", "an", "of", "in", "on",
    "to", "for", "with", "at", "by", "from", "and",
    "or", "but", "not", "that", "this", "these", "those",
    "it", "its", "be", "been", "being", "if", "then",
    "than", "so", "very", "just", "also", "about", "into",
})

_TEMPORAL_WORDS = frozenset({
    "when", "date", "time", "year", "month", "week", "day",
    "yesterday", "today", "tomorrow", "ago", "last", "next",
    "before", "after", "during", "while", "since", "until",
    "recently", "earlier", "later", "first", "last",
})

_ACTION_VERBS = frozenset({
    "go", "went", "going", "do", "did", "doing", "make", "made",
    "get", "got", "take", "took", "give", "gave", "say", "said",
    "tell", "told", "ask", "asked", "know", "knew", "think",
    "thought", "want", "wanted", "like", "liked", "work", "worked",
    "study", "studied", "play", "played", "live", "lived", "move",
    "moved", "buy", "bought", "sell", "sold", "join", "joined",
    "meet", "met", "call", "called", "visit", "visited", "leave",
    "left", "start", "started", "finish", "finished", "become",
    "became", "change", "changed", "attend", "attended",
})

_ANSWER_TYPE_HINTS = {
    "gift": ["gift", "present"],
    "advice": ["advice", "advise", "recommend", "recommendation", "suggest", "should"],
    "career": ["career", "job", "work", "profession", "education", "field", "pursue"],
    "hobby": ["hobby", "interest", "enjoy", "like"],
    "activity": ["activity", "activities", "do", "event", "outing"],
    "food": ["food", "meal", "snack", "eat", "dinner", "lunch"],
    "recipe": ["recipe", "cook", "cooking"],
    "health": ["health", "healthy", "wellness", "diet", "exercise", "fitness"],
    "lifestyle": ["lifestyle", "lifestyles", "habit", "routine"],
    "travel": ["travel", "trip", "vacation"],
    "place": ["place", "where", "location"],
    "event": ["event", "party", "concert", "class", "meeting"],
    "relationship": ["relationship", "partner", "friend", "married", "dating"],
    "pet": ["pet", "dog", "cat"],
    "family": ["family", "mother", "father", "sister", "brother", "parent"],
    "book": ["book", "read", "novel"],
    "movie": ["movie", "film", "watch"],
    "sport": ["sport", "game", "running", "fitness"],
    "class": ["class", "course", "lesson", "school"],
    "recommendation": ["recommendation", "recommend", "suggestion", "suggest"],
    "identity": ["identity", "who", "community", "lgbt", "lgbtq"],
    "support": ["support", "help", "group", "resource"],
    "temporal": sorted(_TEMPORAL_WORDS),
}

_ANSWER_TYPE_EXPANSIONS = {
    "gift": ["gift"],
    "advice": ["advice", "recommendation"],
    "career": ["career", "education", "work"],
    "hobby": ["hobby", "interest"],
    "activity": ["activity", "event"],
    "food": ["food", "snacks", "healthy eating"],
    "recipe": ["recipes", "cooking"],
    "health": ["health", "diet", "exercise", "healthy"],
    "lifestyle": ["lifestyle", "habits", "wellness"],
    "travel": ["travel", "place"],
    "place": ["place", "location"],
    "event": ["event", "session"],
    "relationship": ["relationship"],
    "pet": ["pet"],
    "family": ["family"],
    "book": ["book"],
    "movie": ["movie"],
    "sport": ["sport", "fitness"],
    "class": ["class", "course"],
    "recommendation": ["recommendation", "suggestion"],
    "identity": ["identity", "community"],
    "support": ["support", "resources"],
    "temporal": ["when", "date", "time"],
}


def generate_query_variants(question: str) -> List[str]:
    variants: List[str] = []

    variants.append(question)

    words = question.split()
    lower_words = [w.lower().rstrip("?,.!;:") for w in words]

    content_words = [w for w in lower_words if w not in _QUESTION_WORDS and len(w) > 1]
    if content_words:
        entity_query = " ".join(content_words)
        if entity_query != question.lower().rstrip("?,.!;:"):
            variants.append(entity_query)

    proper_names = []
    for w in words:
        stripped = w.rstrip("?,.!;:")
        if stripped and stripped[0].isupper() and len(stripped) > 1:
            lower_stripped = stripped.lower()
            if lower_stripped not in _QUESTION_WORDS:
                proper_names.append(stripped)
    if proper_names:
        variants.append(" ".join(proper_names))

    if lower_words and any(w in _TEMPORAL_WORDS for w in lower_words):
        temporal_parts = []
        for w_orig, w_lower in zip(words, lower_words):
            if w_lower in _TEMPORAL_WORDS or (w_lower not in _QUESTION_WORDS and len(w_lower) > 1):
                temporal_parts.append(w_orig.rstrip("?,.!;:"))
        if temporal_parts:
            temporal_query = " ".join(temporal_parts)
            if temporal_query not in variants and temporal_query.lower() != question.lower().rstrip("?,.!;:"):
                variants.append(temporal_query)

    action_parts = []
    for w_orig, w_lower in zip(words, lower_words):
        if w_lower in _ACTION_VERBS or (w_lower not in _QUESTION_WORDS and w_lower not in _TEMPORAL_WORDS and len(w_lower) > 1):
            action_parts.append(w_orig.rstrip("?,.!;:"))
    if action_parts:
        action_query = " ".join(action_parts)
        if action_query not in variants and action_query.lower() != question.lower().rstrip("?,.!;:"):
            variants.append(action_query)

    return variants


def _extract_question_entities(question: str) -> List[str]:
    entities = []
    seen = set()
    for match in re.finditer(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", question):
        name = match.group(0).strip()
        lower = name.lower()
        parts = lower.split()
        if not name or all(p in _QUESTION_WORDS for p in parts):
            continue
        if lower not in seen:
            entities.append(name)
            seen.add(lower)
    return entities


def _extract_answer_type_hints(question: str) -> List[str]:
    lower = question.lower()
    words = set(re.findall(r"[a-z]+", lower))
    hints = []
    for hint, triggers in _ANSWER_TYPE_HINTS.items():
        for trigger in triggers:
            trigger_lower = trigger.lower()
            if " " in trigger_lower:
                matched = trigger_lower in lower
            else:
                matched = trigger_lower in words
            if matched:
                hints.append(hint)
                break
    return hints


def _meaningful_tokens(text: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z']*", text):
        lower = token.lower()
        if lower not in _QUESTION_WORDS and lower not in _ENTITY_STOP_WORDS and len(lower) > 1:
            tokens.append(token)
    return tokens


def _valid_entity_probe(probe: str, entities: List[str]) -> bool:
    meaningful = _meaningful_tokens(probe)
    if len(meaningful) < 3:
        return False
    if len(probe.split()) <= 1:
        return False
    if not entities:
        return True
    probe_lower = probe.lower()
    return any(entity.lower() in probe_lower for entity in entities)


def generate_entity_constrained_probes(question: str, max_probes: int = 3) -> List[str]:
    probes = [question]
    entities = _extract_question_entities(question)
    hints = _extract_answer_type_hints(question)
    content_words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z']*", question.lower())
        if w not in _QUESTION_WORDS and w not in _ENTITY_STOP_WORDS and len(w) > 2
    ]
    hint_terms: List[str] = []
    for hint in hints:
        hint_terms.extend(_ANSWER_TYPE_EXPANSIONS.get(hint, [hint]))
    if not hint_terms:
        hint_terms = content_words[:4]

    candidate_probes: List[str] = []
    main_entities = entities or []
    if main_entities:
        for entity in main_entities[:2]:
            candidate_probes.append(" ".join([entity] + hint_terms[:4]))
        if len(main_entities) >= 2:
            candidate_probes.append(" ".join(main_entities[:2] + hint_terms[:5]))
        candidate_probes.append(" ".join(main_entities[:1] + content_words[:5]))
    else:
        candidate_probes.append(" ".join(content_words[:6]))
        if hints:
            candidate_probes.append(" ".join(hint_terms[:5]))

    seen = {question.lower()}
    for probe in candidate_probes:
        normalized_probe = " ".join(probe.split())
        lower = normalized_probe.lower()
        if lower in seen:
            continue
        if not _valid_entity_probe(normalized_probe, entities):
            continue
        probes.append(normalized_probe)
        seen.add(lower)
        if len(probes) >= max_probes + 1:
            break
    return probes


def _extract_question_anchors(question: str) -> Tuple[List[str], List[str]]:
    words = question.split()
    entities = []
    content_words = []
    for w in words:
        stripped = w.rstrip("?,.!;:")
        if not stripped or len(stripped) <= 1:
            continue
        lower = stripped.lower()
        if stripped[0].isupper() and lower not in _QUESTION_WORDS:
            entities.append(stripped)
        elif lower not in _QUESTION_WORDS and lower not in _TEMPORAL_WORDS and len(lower) > 2:
            content_words.append(lower)
    return entities, content_words


def _extract_candidate_anchors(
    candidate_ids: List[int],
    id_to_memory: Dict[int, Dict[str, Any]],
    top_n: int,
) -> Tuple[List[str], List[str], List[str]]:
    speakers = set()
    key_terms = set()
    session_clues = set()

    for mid in candidate_ids[:top_n]:
        m = id_to_memory.get(mid)
        if not m:
            continue
        speaker = str(m.get("speaker", "")).strip()
        if speaker:
            speakers.add(speaker)
        session = str(m.get("session", "")).strip()
        if session:
            session_clues.add(session)
        text = str(m.get("text", ""))
        for w in text.split():
            stripped = w.rstrip("?,.!;:").lower()
            if (len(stripped) > 2
                    and stripped not in _QUESTION_WORDS
                    and stripped not in _TEMPORAL_WORDS
                    and not stripped.isdigit()):
                key_terms.add(stripped)

    return sorted(speakers), sorted(key_terms)[:15], sorted(session_clues)


def generate_decomposed_queries(
    question: str,
    candidate_ids: List[int],
    id_to_memory: Dict[int, Dict[str, Any]],
    top_n: int = 10,
    max_queries: int = 6,
) -> List[str]:
    q_entities, q_content = _extract_question_anchors(question)
    c_speakers, c_key_terms, c_sessions = _extract_candidate_anchors(
        candidate_ids, id_to_memory, top_n
    )

    queries: List[str] = []
    seen: set = set()

    def _add(q_text: str) -> bool:
        q_lower = q_text.lower()
        if q_lower in seen:
            return False
        seen.add(q_lower)
        queries.append(q_text)
        return True

    primary_entity = q_entities[0] if q_entities else None
    q_qualifiers = [w for w in q_content if len(w) > 2][:3]

    if primary_entity:
        for term in c_key_terms:
            if len(queries) >= max_queries:
                break
            _add(f"{primary_entity} {term}")

    if primary_entity and q_qualifiers:
        for term in c_key_terms:
            if len(queries) >= max_queries:
                break
            for qual in q_qualifiers:
                if len(queries) >= max_queries:
                    break
                _add(f"{primary_entity} {term} {qual}")

    if not primary_entity and q_qualifiers:
        prefix = " ".join(q_qualifiers[:2])
        for term in c_key_terms:
            if len(queries) >= max_queries:
                break
            _add(f"{prefix} {term}")

    if c_speakers:
        for speaker in c_speakers[:2]:
            if speaker == primary_entity:
                continue
            for term in c_key_terms[:2]:
                if len(queries) >= max_queries:
                    break
                _add(f"{speaker} {term}")

    if c_sessions and primary_entity and len(queries) < max_queries:
        for session in c_sessions[:1]:
            _add(f"{primary_entity} {session}")

    q_lower_words = [w.rstrip("?,.!;:").lower() for w in question.split()]
    has_temporal = any(w in _TEMPORAL_WORDS for w in q_lower_words)
    if has_temporal and primary_entity:
        for term in c_key_terms[:2]:
            if len(queries) >= max_queries:
                break
            _add(f"{primary_entity} {term} when")

    return queries[:max_queries]


def _content_word_set(text: str) -> set:
    words = set()
    for w in text.split():
        stripped = w.rstrip("?,.!;:").lower()
        if len(stripped) > 2 and stripped not in _QUESTION_WORDS and not stripped.isdigit():
            words.add(stripped)
    return words


def _compute_completion_candidates(
    question: str,
    candidate_ids: List[int],
    id_to_memory: Dict[int, Dict[str, Any]],
    all_memory_ids: List[int],
    base_scores: Dict[int, float],
    seed_top_n: int = 20,
    window_radius: int = 2,
    same_speaker_limit: int = 5,
    max_candidates: int = 80,
) -> Dict[int, float]:
    seed_ids = candidate_ids[:seed_top_n]
    candidate_set = set(candidate_ids)

    q_words = _content_word_set(question)

    seed_metadata: Dict[int, Dict[str, Any]] = {}
    for rank_0, mid in enumerate(seed_ids):
        m = id_to_memory.get(mid)
        if m:
            seed_metadata[mid] = {
                "session": str(m.get("session", "")),
                "dia_num": _parse_int(m.get("dia_num", "")),
                "speaker": str(m.get("speaker", "")),
                "text_words": _content_word_set(str(m.get("text", ""))),
                "rank": rank_0,
            }

    completion_ids: Dict[int, float] = {}

    for mid in all_memory_ids:
        if mid in candidate_set:
            continue

        m = id_to_memory.get(mid)
        if not m:
            continue

        m_session = str(m.get("session", ""))
        m_dia_num = _parse_int(m.get("dia_num", ""))
        m_speaker = str(m.get("speaker", ""))
        m_text_words = _content_word_set(str(m.get("text", "")))

        score = 0.0
        hit = False

        for seed_mid, sm in seed_metadata.items():
            seed_rank_weight = 1.0 / (1 + sm["rank"])
            signal = 0.0

            if m_session and m_session == sm["session"] and m_dia_num is not None and sm["dia_num"] is not None:
                dia_dist = abs(m_dia_num - sm["dia_num"])
                if dia_dist <= window_radius:
                    signal += 0.4 * max(0, 1.0 - dia_dist / (window_radius + 1))
                    signal += 0.15
                    hit = True

            if m_session and m_session == sm["session"]:
                signal += 0.10
                hit = True

            if m_speaker and m_speaker == sm["speaker"]:
                signal += 0.20 * (1.0 / same_speaker_limit)
                hit = True

            q_seed_overlap = len(q_words & sm["text_words"])
            cand_overlap = len(m_text_words & sm["text_words"])
            if cand_overlap > 0:
                overlap_frac = cand_overlap / max(len(m_text_words), 1)
                signal += 0.15 * min(overlap_frac, 1.0)
                hit = True

            if q_seed_overlap > 0 and len(m_text_words & q_words) > 0:
                signal += 0.10
                hit = True

            score += signal * seed_rank_weight

        if hit and score > 0:
            completion_ids[mid] = score

    sorted_completion = sorted(completion_ids.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    return dict(sorted_completion[:max_candidates])


def _parse_int(val) -> Optional[int]:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def rrf_fuse_multi_query(
    ranked_lists: List[Tuple[str, List[int]]],
    final_k: int,
    rrf_k: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    ranked = rrf_fuse_multi_query_with_scores(ranked_lists, final_k, rrf_k, weights)
    return [mid for mid, _ in ranked]


def rrf_fuse_multi_query_with_scores(
    ranked_lists: List[Tuple[str, List[int]]],
    final_k: int,
    rrf_k: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[int, float]]:
    if weights is None:
        weight_per_list = 1.0 / len(ranked_lists) if ranked_lists else 1.0
        weights = {name: weight_per_list for name, _ in ranked_lists}

    scores: Dict[int, float] = {}
    for name, ranked in ranked_lists:
        w = weights.get(name, 1.0)
        for rank, mid in enumerate(ranked, start=1):
            scores[mid] = scores.get(mid, 0.0) + w / (rrf_k + rank)

    return sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:final_k]


def _memory_dia_num(memory: Dict[str, Any]) -> Optional[int]:
    dia_num = _parse_int(memory.get("dia_num"))
    if dia_num is not None:
        return dia_num
    dia_id = str(memory.get("dia_id", "") or "")
    matches = re.findall(r"\d+", dia_id)
    if not matches:
        return None
    return _parse_int(matches[-1])


def apply_local_neighbor_expansion(
    candidate_ids: List[int],
    source_scores: Dict[int, float],
    id_to_memory: Dict[int, Dict[str, Any]],
    radius: int = 2,
    anchor_k: int = 50,
    final_candidate_k: Optional[int] = None,
) -> Tuple[List[int], Dict[int, float], int, Set[int]]:
    if radius <= 0 or anchor_k <= 0:
        return candidate_ids, source_scores, 0, set()

    by_sample_session: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for mid, memory in id_to_memory.items():
        sample_id = str(memory.get("sample_id", "") or "")
        session = str(memory.get("session", "") or "")
        dia_num = _memory_dia_num(memory)
        if not sample_id or not session or dia_num is None:
            continue
        by_sample_session.setdefault((sample_id, session), []).append((dia_num, mid))

    for key in by_sample_session:
        by_sample_session[key].sort(key=lambda item: (item[0], item[1]))

    final_scores = dict(source_scores)
    base_rank = {mid: idx for idx, mid in enumerate(candidate_ids)}
    added: Set[int] = set()

    for anchor_mid in candidate_ids[:anchor_k]:
        anchor = id_to_memory.get(anchor_mid)
        if not anchor:
            continue
        sample_id = str(anchor.get("sample_id", "") or "")
        session = str(anchor.get("session", "") or "")
        anchor_dia = _memory_dia_num(anchor)
        if not sample_id or not session or anchor_dia is None:
            continue
        anchor_score = source_scores.get(anchor_mid, 0.0)
        for neighbor_dia, neighbor_mid in by_sample_session.get((sample_id, session), []):
            dist = abs(neighbor_dia - anchor_dia)
            if dist == 0 or dist > radius:
                continue
            decay = 1.0 - (dist / (radius + 1))
            neighbor_score = anchor_score * 0.35 * max(decay, 0.1)
            if neighbor_mid not in final_scores:
                added.add(neighbor_mid)
                final_scores[neighbor_mid] = neighbor_score
            else:
                final_scores[neighbor_mid] = max(final_scores[neighbor_mid], neighbor_score)

    ranked = sorted(
        final_scores.items(),
        key=lambda x: (-x[1], base_rank.get(x[0], len(base_rank) + 1)),
    )
    if final_candidate_k is not None:
        ranked = ranked[:final_candidate_k]
    ranked_ids = [mid for mid, _ in ranked]
    kept_scores = {mid: score for mid, score in ranked}
    kept_added = added & set(ranked_ids)
    return ranked_ids, kept_scores, len(kept_added), kept_added


def _memory_session(memory: Dict[str, Any]) -> str:
    return str(memory.get("session", "") or "")


def _main_entities(question: str) -> Set[str]:
    return {e.lower() for e in _extract_question_entities(question)}


def _memory_mentions_entity(memory: Dict[str, Any], entities: Set[str]) -> bool:
    if not entities:
        return False
    haystack = (
        f"{memory.get('speaker', '')} {memory.get('speaker_a', '')} "
        f"{memory.get('speaker_b', '')} {memory.get('text', '')}"
    ).lower()
    return any(entity in haystack for entity in entities)


def _memory_event_type(memory: Dict[str, Any]) -> str:
    return _classify_event_type(
        str(memory.get("text", "") or ""),
        str(memory.get("speaker", "") or ""),
    )


def coverage_select_source_ids(
    candidate_ids: List[int],
    source_scores: Dict[int, float],
    id_to_memory: Dict[int, Dict[str, Any]],
    question: str,
    k: int,
    preserve_top_n: int = 60,
    max_candidates: int = 300,
    neighbor_added_ids: Optional[Set[int]] = None,
) -> Tuple[List[int], Dict[str, int], Dict[str, int]]:
    if k <= 0:
        return [], {}, {}
    if not candidate_ids:
        return [], {}, {}

    candidate_slice = candidate_ids[:max(max_candidates, preserve_top_n, k)]
    if not any(id_to_memory.get(mid) for mid in candidate_slice):
        return candidate_ids[:k], {"rank": min(k, len(candidate_ids))}, {}

    preserve_n = min(preserve_top_n, k, len(candidate_ids))
    selected = list(candidate_ids[:preserve_n])
    selected_set = set(selected)
    counts = {"preserved": len(selected), "coverage": 0}
    reason_counts: Dict[str, int] = {}

    sessions = {
        _memory_session(id_to_memory.get(mid, {}))
        for mid in selected
        if _memory_session(id_to_memory.get(mid, {}))
    }
    event_types = {
        _memory_event_type(id_to_memory.get(mid, {}))
        for mid in selected
        if id_to_memory.get(mid)
    }
    entities = _main_entities(question)
    entity_sessions = {
        _memory_session(id_to_memory.get(mid, {}))
        for mid in selected
        if _memory_mentions_entity(id_to_memory.get(mid, {}), entities)
    }
    selected_texts = [_content_word_set(str(id_to_memory.get(mid, {}).get("text", ""))) for mid in selected]
    temporal_question = bool(set(re.findall(r"[a-z]+", question.lower())) & _TEMPORAL_WORDS)
    neighbor_added_ids = neighbor_added_ids or set()
    normalized = normalize_scores({mid: source_scores.get(mid, 0.0) for mid in candidate_slice})

    def _candidate_score(mid: int) -> Tuple[float, List[str]]:
        memory = id_to_memory.get(mid, {})
        session = _memory_session(memory)
        event_type = _memory_event_type(memory) if memory else ""
        text_words = _content_word_set(str(memory.get("text", "")))
        score = 0.70 * normalized.get(mid, 0.0)
        reasons = ["rank"]

        if session and session not in sessions:
            score += 0.18
            reasons.append("new_session")
        if session and entities and _memory_mentions_entity(memory, entities) and session not in entity_sessions:
            score += 0.16
            reasons.append("entity_new_session")
        if event_type and event_type not in event_types:
            score += 0.08
            reasons.append("event_type")
        if mid in neighbor_added_ids:
            score += 0.08
            reasons.append("neighbor_support")
        if temporal_question and _memory_dia_num(memory) is not None and session:
            score += 0.06
            reasons.append("temporal_spread")

        max_overlap = 0.0
        for selected_words in selected_texts[-20:]:
            if not text_words or not selected_words:
                continue
            max_overlap = max(max_overlap, len(text_words & selected_words) / max(len(text_words), 1))
        if max_overlap >= 0.80 and session in sessions:
            score -= 0.25
            reasons.append("redundancy_penalty")
        return score, reasons

    while len(selected) < k:
        best_mid = None
        best_tuple = None
        best_reasons: List[str] = []
        for mid in candidate_slice:
            if mid in selected_set:
                continue
            cov_score, reasons = _candidate_score(mid)
            tie = candidate_ids.index(mid) if mid in candidate_ids else len(candidate_ids) + mid
            current = (cov_score, -tie, -mid)
            if best_tuple is None or current > best_tuple:
                best_tuple = current
                best_mid = mid
                best_reasons = reasons
        if best_mid is None:
            break
        selected.append(best_mid)
        selected_set.add(best_mid)
        counts["coverage"] += 1
        memory = id_to_memory.get(best_mid, {})
        session = _memory_session(memory)
        if session:
            sessions.add(session)
        if entities and _memory_mentions_entity(memory, entities) and session:
            entity_sessions.add(session)
        if memory:
            event_types.add(_memory_event_type(memory))
            selected_texts.append(_content_word_set(str(memory.get("text", ""))))
        for reason in best_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return selected[:k], counts, reason_counts


def mmr_select_source_ids(
    candidate_ids: List[int],
    source_scores: Dict[int, float],
    id_to_memory: Dict[int, Dict[str, Any]],
    k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    if k <= 0 or not candidate_ids:
        return []

    candidate_set = list(dict.fromkeys(candidate_ids))
    if len(candidate_set) <= k:
        return candidate_set

    selected: List[int] = [candidate_set[0]]
    selected_set = {candidate_set[0]}

    selected_texts = [_content_word_set(str(id_to_memory.get(candidate_set[0], {}).get("text", "")))]

    normalized = normalize_scores({mid: source_scores.get(mid, 0.0) for mid in candidate_set})

    while len(selected) < k:
        best_mmr = -float("inf")
        best_mid = None

        for mid in candidate_set:
            if mid in selected_set:
                continue

            relevance = normalized.get(mid, 0.0)

            max_sim = 0.0
            mid_words = _content_word_set(str(id_to_memory.get(mid, {}).get("text", "")))
            for sel_words in selected_texts:
                if not mid_words or not sel_words:
                    continue
                intersection = len(mid_words & sel_words)
                union = len(mid_words | sel_words)
                if union > 0:
                    jaccard = intersection / union
                    max_sim = max(max_sim, jaccard)

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_mid = mid

        if best_mid is None:
            break

        selected.append(best_mid)
        selected_set.add(best_mid)
        selected_texts.append(_content_word_set(str(id_to_memory.get(best_mid, {}).get("text", ""))))

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "openai", "sentence_transformers", "nvidia"],
        default="hash",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument(
        "--embedding-cache",
        default="benchmarks/paper/locomo/data/embedding_cache.jsonl",
    )
    parser.add_argument(
        "--reranker-provider",
        choices=["none", "sentence_transformers"],
        default="none",
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument(
        "--reranker-cache",
        default="benchmarks/paper/locomo/data/rerank_cache_bge_base.jsonl",
    )
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--rerank-mode", choices=["pure", "blend"], default="blend")
    parser.add_argument("--rrf-final-weight", type=float, default=0.8)
    parser.add_argument("--reranker-final-weight", type=float, default=0.2)
    parser.add_argument("--debug-candidates", action="store_true")
    parser.add_argument("--candidate-debug-limit", type=int, default=100)

    parser.add_argument("--vector-backend", choices=["local", "sochdb", "local_hnsw"], default="local")
    parser.add_argument("--host", default=os.getenv("SOCHDB_HOST", "65.108.78.80"))
    parser.add_argument("--port", type=parse_port, default=parse_port(os.getenv("SOCHDB_PORT")))
    parser.add_argument("--collection-prefix", default="locomo_hybrid")
    parser.add_argument("--use-tls", action="store_true")
    parser.add_argument(
        "--sochdb-search-mode",
        choices=["single", "batch"],
        default="single",
    )
    parser.add_argument("--sochdb-ef", type=int, default=0)
    parser.add_argument(
        "--local-hnsw-m",
        type=int,
        default=32,
        help="HNSW M (max connections per node) for --vector-backend local_hnsw",
    )
    parser.add_argument(
        "--local-hnsw-ef-construction",
        type=int,
        default=256,
        help="HNSW ef_construction for --vector-backend local_hnsw",
    )
    parser.add_argument(
        "--local-hnsw-ef-search",
        type=int,
        default=500,
        help="HNSW ef_search for --vector-backend local_hnsw (set after construction)",
    )

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--use-grep", action="store_true", help="Enable trigram-indexed grep lexical search leg")
    parser.add_argument("--grep-weight", type=float, default=1.0, help="RRF weight for grep leg")
    parser.add_argument("--grep-trigram-threshold", type=float, default=0.3, help="Min trigram match ratio for grep candidates (0.0-1.0)")
    parser.add_argument(
        "--memory-render-mode",
        choices=["raw", "speaker", "speaker_time", "speaker_session", "metadata"],
        default="raw",
        help=(
            "raw: text only; speaker: speaker + text; speaker_time: speaker + timestamp "
            "+ resolved_time + text; speaker_session: speaker + session + dia_id + text; "
            "metadata: include compact memory metadata in indexed/embedded text"
        ),
    )

    parser.add_argument(
        "--query-mode",
        choices=["single", "multi", "entity_multi"],
        default="single",
        help="single: original baseline; multi: generate deterministic query variants and fuse via RRF; entity_multi: entity-constrained deterministic probes",
    )
    parser.add_argument("--max-query-probes", type=int, default=3)

    parser.add_argument(
        "--retrieval-plan",
        choices=["one_shot", "decomposed", "anchored_two_hop"],
        default="one_shot",
        help="one_shot: single-stage retrieval; decomposed: two-hop decomposed retrieval with anchor-based second-hop queries; anchored_two_hop: extract speakers from first-hop results and do second-hop search for cross-session evidence",
    )
    parser.add_argument("--decompose-top-n", type=int, default=10)
    parser.add_argument("--decompose-max-queries", type=int, default=6)
    parser.add_argument("--decompose-candidate-k", type=int, default=100)
    parser.add_argument("--anchor-top-n", type=int, default=20, help="Top-N first-hop results to extract speakers from for anchored_two_hop")
    parser.add_argument("--anchor-max-queries", type=int, default=4, help="Max second-hop queries per question for anchored_two_hop")
    parser.add_argument("--anchor-weight", type=float, default=0.5, help="RRF weight for second-hop results relative to first-hop (1.0)")

    parser.add_argument(
        "--evidence-completion",
        choices=["none", "conservative"],
        default="none",
        help="none: no completion; conservative: add nearby/same-speaker/overlap evidence conservatively",
    )
    parser.add_argument("--completion-seed-top-n", type=int, default=20)
    parser.add_argument("--completion-window-radius", type=int, default=2)
    parser.add_argument("--completion-same-speaker-limit", type=int, default=5)
    parser.add_argument("--completion-max-candidates", type=int, default=80)
    parser.add_argument("--completion-weight", type=float, default=0.20)

    parser.add_argument(
        "--memory-view-mode",
        choices=["turn", "multiview"],
        default="turn",
        help=(
            "turn: one searchable record per memory (default, backward compatible); "
            "multiview: create multiple searchable views per memory (turn, event, entity, neighbor_window) "
            "and map hits back to source_memory_id"
        ),
    )
    parser.add_argument(
        "--memory-view-types",
        default="turn,event,entity,neighbor_window",
        help="Comma-separated multiview types to build: turn,event,entity,neighbor_window. Ignored in turn mode.",
    )
    parser.add_argument(
        "--view-window-radius",
        type=int,
        default=2,
        help="Window radius for neighbor_window_view in multiview mode (default: 2)",
    )
    parser.add_argument("--local-neighbor-expansion", action="store_true")
    parser.add_argument("--neighbor-expansion-radius", type=int, default=2)
    parser.add_argument("--neighbor-expansion-anchor-k", type=int, default=50)

    parser.add_argument(
        "--selection-mode",
        choices=["rank", "coverage", "mmr"],
        default="rank",
    )
    parser.add_argument("--coverage-preserve-top-n", type=int, default=60)
    parser.add_argument("--coverage-max-candidates", type=int, default=300)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)

    parser.add_argument(
        "--retrieved-id-mode",
        choices=["memory", "parent"],
        default="memory",
        help="memory: output actual memory_ids; parent: output parent_memory_ids with deduplication",
    )

    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--run-id", default=str(int(time.time())))

    args = parser.parse_args()
    try:
        active_memory_view_types = parse_memory_view_types(args.memory_view_types)
    except ValueError as e:
        parser.error(str(e))
    active_view_count = len(active_memory_view_types) if args.memory_view_mode == "multiview" else 1

    memories = load_jsonl(args.memories)
    questions = load_jsonl(args.questions)

    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    memories_by_sample = group_by_sample(memories)
    questions_by_sample = group_by_sample(questions)

    sample_ids = sorted(questions_by_sample.keys())
    if args.limit_samples is not None:
        sample_ids = sample_ids[: args.limit_samples]

    embedder = Embedder(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dim=args.embedding_dim,
        cache_path=args.embedding_cache,
    )
    reranker = Reranker(
        provider=args.reranker_provider,
        model=args.reranker_model,
        cache_path=args.reranker_cache,
    )

    client = None
    local_hnsw_indices = {}
    LocalHnswIndex = None
    if args.vector_backend == "sochdb":
        SochDBClient = import_sochdb_client()
        client = SochDBClient(address=f"{args.host}:{args.port}", secure=args.use_tls)
    elif args.vector_backend == "local_hnsw":
        LocalHnswIndex = import_local_hnsw()

    output_rows = []

    system_name = (
        f"hybrid_bm25_{args.vector_backend}_{args.embedding_provider}_"
        f"{args.embedding_model.replace('-', '_').replace('.', '_')}"
    )
    if args.query_mode == "multi":
        system_name = f"multi_query_{system_name}"
    if args.query_mode == "entity_multi":
        system_name = f"entity_multi_query_{system_name}"
    if args.retrieval_plan == "decomposed":
        system_name = f"decomposed_{system_name}"
    if args.retrieval_plan == "anchored_two_hop":
        system_name = f"anchored_two_hop_{system_name}"
    if args.evidence_completion == "conservative":
        system_name = f"completion_{system_name}"
    if args.memory_view_mode == "multiview":
        system_name = f"multiview_{system_name}"
    if args.local_neighbor_expansion:
        system_name = f"neighbor_expanded_{system_name}"
    if args.selection_mode == "coverage":
        system_name = f"coverage_{system_name}"

    print(f"system={system_name}")
    print(
        f"samples={len(sample_ids)} k={args.k} candidate_k={args.candidate_k} "
        f"query_mode={args.query_mode} memory_render_mode={args.memory_render_mode} "
        f"retrieved_id_mode={args.retrieved_id_mode} retrieval_plan={args.retrieval_plan} "
        f"evidence_completion={args.evidence_completion} "
        f"memory_view_mode={args.memory_view_mode} selection_mode={args.selection_mode}"
    )
    final_candidate_k = max(args.k, args.rerank_top_n)

    for sample_id in sample_ids:
        sample_memories = memories_by_sample.get(sample_id, [])
        sample_questions = questions_by_sample.get(sample_id, [])

        if not sample_memories or not sample_questions:
            continue

        print(f"\n=== sample={sample_id} memories={len(sample_memories)} questions={len(sample_questions)} ===")

        memory_ids = [int(m["memory_id"]) for m in sample_memories]
        id_to_memory = {int(m["memory_id"]): m for m in sample_memories}

        search_records = build_memory_search_records(
            sample_memories,
            args.memory_render_mode,
            args.memory_view_mode,
            args.view_window_radius,
            active_memory_view_types,
        )
        record_ids = [r["record_id"] for r in search_records]
        record_texts = [r["rendered_text"] for r in search_records]
        record_to_source = {r["record_id"]: r["source_memory_id"] for r in search_records}
        record_to_view_type = {r["record_id"]: r["view_type"] for r in search_records}
        search_record_count = len(search_records)
        source_memory_count = len(set(record_to_source.values()))

        source_candidate_k, view_candidate_k = compute_view_overfetch(
            args.memory_view_mode, args.candidate_k, source_memory_count, active_view_count
        )
        source_final_candidate_k, view_final_candidate_k = compute_view_overfetch(
            args.memory_view_mode, final_candidate_k, source_memory_count, active_view_count
        )
        _, view_decompose_candidate_k = compute_view_overfetch(
            args.memory_view_mode, args.decompose_candidate_k, source_memory_count, active_view_count
        )
        _, view_anchor_candidate_k = compute_view_overfetch(
            args.memory_view_mode, args.anchor_top_n * 2, source_memory_count, active_view_count
        )

        print(f"  view_mode={args.memory_view_mode} search_records={search_record_count} source_memories={source_memory_count} source_candidate_k={source_candidate_k} view_candidate_k={view_candidate_k}")

        bm25 = BM25Okapi([tokenize(t) for t in record_texts])

        trigram_idx = None
        if args.use_grep:
            print(f"building trigram index for {len(record_texts)} records...")
            trigram_idx = TrigramIndex()
            for rid, text in zip(record_ids, record_texts):
                trigram_idx.add(str(rid), text.lower())
            trigram_idx.build()
            print(f"trigram index built")

        print("embedding search records...")
        if args.embedding_provider == "nvidia":
            record_vecs = embedder.embed_many_typed(record_texts, input_type="passage")
        else:
            record_vecs = embedder.embed_many(record_texts)

        index_name = f"{args.collection_prefix}_{safe_name(sample_id)}_{args.run_id}"

        if args.vector_backend == "sochdb":
            create_sochdb_index(client, index_name, args.embedding_dim)
            insert_sochdb_vectors(client, index_name, record_ids, record_vecs)
        elif args.vector_backend == "local_hnsw":
            hnsw_idx = LocalHnswIndex(
                args.embedding_dim,
                m=args.local_hnsw_m,
                ef_construction=args.local_hnsw_ef_construction,
                ef_search=args.local_hnsw_ef_search,
            )
            ids_arr = np.array(record_ids, dtype=np.uint64)
            vecs_arr = np.array(record_vecs, dtype=np.float32)
            hnsw_idx.insert_batch_with_ids(ids_arr, vecs_arr)
            local_hnsw_indices[sample_id] = hnsw_idx
            print(
                f"  local_hnsw: built index dim={args.embedding_dim} n={len(record_ids)} "
                f"m={args.local_hnsw_m} ef_construction={args.local_hnsw_ef_construction} "
                f"ef_search={args.local_hnsw_ef_search}"
            )

        question_texts = [q["question"] for q in sample_questions]
        print("embedding questions...")
        if args.embedding_provider == "nvidia":
            question_vecs = embedder.embed_many_typed(question_texts, input_type="query")
        else:
            question_vecs = embedder.embed_many(question_texts)

        batch_vector_ranked: Optional[List[List[int]]] = None
        batch_vector_wall_ms = 0.0
        batch_vector_amortized_ms = 0.0

        if args.vector_backend == "sochdb" and args.sochdb_search_mode == "batch":
            batch_start = time.perf_counter()
            batch_vector_ranked = search_sochdb_batch_ids(
                client,
                index_name,
                question_vecs,
                view_candidate_k,
                ef=args.sochdb_ef,
            )
            batch_vector_wall_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_vector_amortized_ms = batch_vector_wall_ms / max(len(question_vecs), 1)

        for q_idx, (q, qvec) in enumerate(zip(sample_questions, question_vecs)):
            start = time.perf_counter()

            if args.query_mode == "single":
                query_variants = [q["question"]]
                variant_vecs = [qvec]
            elif args.query_mode == "multi":
                if q.get("llm_query_variants"):
                    query_variants = q["llm_query_variants"]
                else:
                    query_variants = generate_query_variants(q["question"])
                if args.embedding_provider == "nvidia":
                    variant_vecs = embedder.embed_many_typed(query_variants, input_type="query")
                else:
                    variant_vecs = embedder.embed_many(query_variants)
            else:
                query_variants = generate_entity_constrained_probes(
                    q["question"], max_probes=args.max_query_probes
                )
                if args.embedding_provider == "nvidia":
                    variant_vecs = embedder.embed_many_typed(query_variants, input_type="query")
                else:
                    variant_vecs = embedder.embed_many(query_variants)

            variant_bm25_ranked: List[List[int]] = []
            variant_vector_ranked: List[List[int]] = []
            variant_grep_ranked: List[List[int]] = []

            for v_idx, (variant_text, variant_vec) in enumerate(zip(query_variants, variant_vecs)):
                v_bm25_scores = bm25.get_scores(tokenize(variant_text))
                v_bm25_ranked_idx = sorted(
                    range(len(v_bm25_scores)),
                    key=lambda i: v_bm25_scores[i],
                    reverse=True,
                )[:view_candidate_k]
                variant_bm25_ranked.append([record_ids[i] for i in v_bm25_ranked_idx])

                if trigram_idx is not None:
                    grep_results = trigram_idx.search(variant_text, k=view_candidate_k, min_trigram_ratio=args.grep_trigram_threshold)
                    variant_grep_ranked.append([int(doc_id) for doc_id, _ in grep_results])
                else:
                    variant_grep_ranked.append([])

                v_vector_search_wall_ms = 0.0
                v_vector_search_amortized_ms = 0.0

                if args.vector_backend == "local":
                    v_vector_ranked = topk_local_vector(
                        variant_vec,
                        record_ids,
                        record_vecs,
                        view_candidate_k,
                    )
                elif args.vector_backend == "local_hnsw":
                    vs = time.perf_counter()
                    q_arr = np.array(variant_vec, dtype=np.float32)
                    hnsw_ids, hnsw_dists = local_hnsw_indices[sample_id].search(q_arr, view_candidate_k)
                    v_vector_ranked = [int(rid) for rid in hnsw_ids]
                    v_vector_search_wall_ms = (time.perf_counter() - vs) * 1000.0
                    v_vector_search_amortized_ms = v_vector_search_wall_ms
                else:
                    if args.sochdb_search_mode == "batch":
                        if batch_vector_ranked is None:
                            raise RuntimeError("batch vector results were not initialized")
                        if v_idx == 0 and len(query_variants) > 1:
                            v_vector_ranked = batch_vector_ranked[q_idx] if q_idx < len(batch_vector_ranked) else []
                            v_vector_search_wall_ms = batch_vector_wall_ms
                            v_vector_search_amortized_ms = batch_vector_amortized_ms
                        else:
                            vs = time.perf_counter()
                            v_vector_ranked = search_sochdb(
                                client,
                                index_name,
                                variant_vec,
                                view_candidate_k,
                            )
                            v_vector_search_wall_ms = (time.perf_counter() - vs) * 1000.0
                            v_vector_search_amortized_ms = v_vector_search_wall_ms
                    else:
                        vs = time.perf_counter()
                        v_vector_ranked = search_sochdb(
                            client,
                            index_name,
                            variant_vec,
                            view_candidate_k,
                        )
                        v_vector_search_wall_ms = (time.perf_counter() - vs) * 1000.0
                        v_vector_search_amortized_ms = v_vector_search_wall_ms

                variant_vector_ranked.append(v_vector_ranked)

                if v_idx == 0:
                    vector_search_wall_ms = v_vector_search_wall_ms
                    vector_search_amortized_ms_per_query = v_vector_search_amortized_ms

            if args.query_mode == "single":
                if args.use_grep and variant_grep_ranked:
                    rrf_ranked = rrf_fuse_three_legs(
                        bm25_ranked=variant_bm25_ranked[0],
                        vector_ranked=variant_vector_ranked[0],
                        grep_ranked=variant_grep_ranked[0],
                        final_k=view_final_candidate_k,
                        rrf_k=args.rrf_k,
                        bm25_weight=args.bm25_weight,
                        vector_weight=args.vector_weight,
                        grep_weight=args.grep_weight,
                    )
                else:
                    rrf_ranked = rrf_fuse_with_scores(
                        bm25_ranked=variant_bm25_ranked[0],
                        vector_ranked=variant_vector_ranked[0],
                        final_k=view_final_candidate_k,
                        rrf_k=args.rrf_k,
                        bm25_weight=args.bm25_weight,
                        vector_weight=args.vector_weight,
                    )
                candidate_ids = [mid for mid, _ in rrf_ranked]
                rrf_scores = dict(rrf_ranked)
            else:
                per_variant_ranked: List[Tuple[str, List[int]]] = []
                for v_idx in range(len(query_variants)):
                    if args.use_grep and variant_grep_ranked:
                        v_fused = rrf_fuse_three_legs(
                            bm25_ranked=variant_bm25_ranked[v_idx],
                            vector_ranked=variant_vector_ranked[v_idx],
                            grep_ranked=variant_grep_ranked[v_idx],
                            final_k=view_candidate_k,
                            rrf_k=args.rrf_k,
                            bm25_weight=args.bm25_weight,
                            vector_weight=args.vector_weight,
                            grep_weight=args.grep_weight,
                        )
                    else:
                        v_fused = rrf_fuse_with_scores(
                            bm25_ranked=variant_bm25_ranked[v_idx],
                            vector_ranked=variant_vector_ranked[v_idx],
                            final_k=view_candidate_k,
                            rrf_k=args.rrf_k,
                            bm25_weight=args.bm25_weight,
                            vector_weight=args.vector_weight,
                        )
                    per_variant_ranked.append((f"variant_{v_idx}", [mid for mid, _ in v_fused]))

                multi_ranked = rrf_fuse_multi_query_with_scores(
                    ranked_lists=per_variant_ranked,
                    final_k=view_final_candidate_k,
                    rrf_k=args.rrf_k,
                )
                candidate_ids = [mid for mid, _ in multi_ranked]
                rrf_scores = dict(multi_ranked)

            decompose_queries: List[str] = []

            if args.retrieval_plan == "decomposed":
                if args.memory_view_mode == "multiview":
                    _seen_src = set()
                    decompose_top_source_ids = []
                    for rid in candidate_ids:
                        sid = record_to_source.get(rid, rid)
                        if sid not in _seen_src:
                            _seen_src.add(sid)
                            decompose_top_source_ids.append(sid)
                    decompose_top_source_ids = decompose_top_source_ids[:args.decompose_top_n]
                else:
                    decompose_top_source_ids = candidate_ids[:args.decompose_top_n]
                decompose_queries = generate_decomposed_queries(
                    question=q["question"],
                    candidate_ids=decompose_top_source_ids,
                    id_to_memory=id_to_memory,
                    top_n=args.decompose_top_n,
                    max_queries=args.decompose_max_queries,
                )

                if decompose_queries:
                    if args.embedding_provider == "nvidia":
                        decompose_vecs = embedder.embed_many_typed(decompose_queries, input_type="query")
                    else:
                        decompose_vecs = embedder.embed_many(decompose_queries)

                    decompose_ranked_lists: List[Tuple[str, List[int]]] = [
                        ("first_hop", candidate_ids)
                    ]

                    for dq_idx, (dq_text, dq_vec) in enumerate(zip(decompose_queries, decompose_vecs)):
                        dq_bm25_scores = bm25.get_scores(tokenize(dq_text))
                        dq_bm25_ranked_idx = sorted(
                            range(len(dq_bm25_scores)),
                            key=lambda i: dq_bm25_scores[i],
                            reverse=True,
                        )[:view_decompose_candidate_k]
                        dq_bm25_ranked = [record_ids[i] for i in dq_bm25_ranked_idx]

                        if args.vector_backend == "local":
                            dq_vector_ranked = topk_local_vector(
                                dq_vec, record_ids, record_vecs, view_decompose_candidate_k,
                            )
                        else:
                            dq_vector_ranked = search_sochdb(
                                client, index_name, dq_vec, view_decompose_candidate_k,
                            )

                        if args.use_grep and trigram_idx is not None:
                            dq_grep_results = trigram_idx.search(dq_text, k=view_decompose_candidate_k, min_trigram_ratio=args.grep_trigram_threshold)
                            dq_grep_ranked = [int(doc_id) for doc_id, _ in dq_grep_results]
                            dq_fused = rrf_fuse_three_legs(
                                bm25_ranked=dq_bm25_ranked,
                                vector_ranked=dq_vector_ranked,
                                grep_ranked=dq_grep_ranked,
                                final_k=view_decompose_candidate_k,
                                rrf_k=args.rrf_k,
                                bm25_weight=args.bm25_weight,
                                vector_weight=args.vector_weight,
                                grep_weight=args.grep_weight,
                            )
                        else:
                            dq_fused = rrf_fuse(
                                bm25_ranked=dq_bm25_ranked,
                                vector_ranked=dq_vector_ranked,
                                final_k=view_decompose_candidate_k,
                                rrf_k=args.rrf_k,
                                bm25_weight=args.bm25_weight,
                                vector_weight=args.vector_weight,
                            )
                        decompose_ranked_lists.append((f"decompose_{dq_idx}", dq_fused))

                    decompose_ranked = rrf_fuse_multi_query_with_scores(
                        ranked_lists=decompose_ranked_lists,
                        final_k=view_final_candidate_k,
                        rrf_k=args.rrf_k,
                    )
                    candidate_ids = [mid for mid, _ in decompose_ranked]

                    new_rrf_scores: Dict[int, float] = {}
                    for _, ranked in decompose_ranked_lists:
                        for rank, mid in enumerate(ranked, start=1):
                            new_rrf_scores[mid] = new_rrf_scores.get(mid, 0.0) + 1.0 / (args.rrf_k + rank)
                    rrf_scores = new_rrf_scores

            if args.retrieval_plan == "anchored_two_hop":
                anchor_source_ids: List[int] = []
                if args.memory_view_mode == "multiview":
                    _seen_src = set()
                    for rid in candidate_ids:
                        sid = record_to_source.get(rid, rid)
                        if sid not in _seen_src:
                            _seen_src.add(sid)
                            anchor_source_ids.append(sid)
                else:
                    anchor_source_ids = list(candidate_ids[:args.anchor_top_n])

                anchor_source_ids = anchor_source_ids[:args.anchor_top_n]

                c_speakers, c_key_terms, c_sessions = _extract_candidate_anchors(
                    anchor_source_ids, id_to_memory, top_n=args.anchor_top_n
                )

                q_entities, q_content = _extract_question_anchors(q["question"])

                anchor_queries: List[str] = []
                anchor_seen: set = set()

                primary_entity = q_entities[0] if q_entities else None

                if primary_entity:
                    for speaker in c_speakers[:2]:
                        aq = f"{speaker} {primary_entity}"
                        if aq.lower() not in anchor_seen:
                            anchor_seen.add(aq.lower())
                            anchor_queries.append(aq)

                if not primary_entity and q_content:
                    prefix = " ".join(q_content[:2])
                    for speaker in c_speakers[:2]:
                        aq = f"{speaker} {prefix}"
                        if aq.lower() not in anchor_seen:
                            anchor_seen.add(aq.lower())
                            anchor_queries.append(aq)

                for speaker in c_speakers[:2]:
                    for term in c_key_terms[:2]:
                        if len(anchor_queries) >= args.anchor_max_queries:
                            break
                        aq = f"{speaker} {term}"
                        if aq.lower() not in anchor_seen:
                            anchor_seen.add(aq.lower())
                            anchor_queries.append(aq)
                    if len(anchor_queries) >= args.anchor_max_queries:
                        break

                anchor_queries = anchor_queries[:args.anchor_max_queries]

                if anchor_queries:
                    if args.embedding_provider == "nvidia":
                        anchor_vecs = embedder.embed_many_typed(anchor_queries, input_type="query")
                    else:
                        anchor_vecs = embedder.embed_many(anchor_queries)

                    first_hop_set = set(candidate_ids)
                    anchor_novel: List[int] = []
                    anchor_novel_scores: Dict[int, float] = {}

                    for aq_idx, (aq_text, aq_vec) in enumerate(zip(anchor_queries, anchor_vecs)):
                        aq_bm25_scores = bm25.get_scores(tokenize(aq_text))
                        aq_bm25_ranked_idx = sorted(
                            range(len(aq_bm25_scores)),
                            key=lambda i: aq_bm25_scores[i],
                            reverse=True,
                        )[:view_anchor_candidate_k]
                        aq_bm25_ranked = [record_ids[i] for i in aq_bm25_ranked_idx]

                        if args.vector_backend == "local":
                            aq_vector_ranked = topk_local_vector(
                                aq_vec, record_ids, record_vecs, view_anchor_candidate_k,
                            )
                        else:
                            aq_vector_ranked = search_sochdb(
                                client, index_name, aq_vec, view_anchor_candidate_k,
                            )

                        if args.use_grep and trigram_idx is not None:
                            aq_grep_results = trigram_idx.search(aq_text, k=view_anchor_candidate_k, min_trigram_ratio=args.grep_trigram_threshold)
                            aq_grep_ranked = [int(doc_id) for doc_id, _ in aq_grep_results]
                            aq_fused = rrf_fuse_three_legs(
                                bm25_ranked=aq_bm25_ranked,
                                vector_ranked=aq_vector_ranked,
                                grep_ranked=aq_grep_ranked,
                                final_k=view_anchor_candidate_k,
                                rrf_k=args.rrf_k,
                                bm25_weight=args.bm25_weight,
                                vector_weight=args.vector_weight,
                                grep_weight=args.grep_weight,
                            )
                        else:
                            aq_fused = rrf_fuse_with_scores(
                                bm25_ranked=aq_bm25_ranked,
                                vector_ranked=aq_vector_ranked,
                                final_k=view_anchor_candidate_k,
                                rrf_k=args.rrf_k,
                                bm25_weight=args.bm25_weight,
                                vector_weight=args.vector_weight,
                            )
                        for rank, (mid, _) in enumerate(aq_fused, start=1):
                            if mid not in first_hop_set:
                                score = args.anchor_weight / (args.rrf_k + rank)
                                if mid not in anchor_novel_scores or score > anchor_novel_scores[mid]:
                                    anchor_novel_scores[mid] = score

                    anchor_novel = sorted(anchor_novel_scores, key=lambda mid: anchor_novel_scores[mid], reverse=True)
                    candidate_ids = candidate_ids + [mid for mid in anchor_novel if mid not in first_hop_set]

                    for mid in candidate_ids:
                        if mid not in rrf_scores:
                            rrf_scores[mid] = anchor_novel_scores.get(mid, 0.0)

            if args.memory_view_mode == "multiview":
                multiview_diagnostics = compute_multiview_diagnostics(
                    candidate_ids, rrf_scores, record_to_source, record_to_view_type, args.memory_view_mode
                )
                candidate_ids, rrf_scores = dedup_view_hits_to_source_ids(
                    candidate_ids, rrf_scores, record_to_source, source_final_candidate_k
                )
            else:
                multiview_diagnostics = {}

            neighbor_expansion_added_count = 0
            neighbor_added_ids: Set[int] = set()

            if args.local_neighbor_expansion:
                candidate_ids, rrf_scores, neighbor_expansion_added_count, neighbor_added_ids = apply_local_neighbor_expansion(
                    candidate_ids=candidate_ids,
                    source_scores=rrf_scores,
                    id_to_memory=id_to_memory,
                    radius=args.neighbor_expansion_radius,
                    anchor_k=args.neighbor_expansion_anchor_k,
                    final_candidate_k=final_candidate_k,
                )

            completion_candidate_count = 0
            completion_inserted_count = 0

            if args.evidence_completion == "conservative":
                completion_scores = _compute_completion_candidates(
                    question=q["question"],
                    candidate_ids=candidate_ids,
                    id_to_memory=id_to_memory,
                    all_memory_ids=memory_ids,
                    base_scores=rrf_scores,
                    seed_top_n=args.completion_seed_top_n,
                    window_radius=args.completion_window_radius,
                    same_speaker_limit=args.completion_same_speaker_limit,
                    max_candidates=args.completion_max_candidates,
                )
                completion_candidate_count = len(completion_scores)

                base_rank_map = {mid: rank for rank, mid in enumerate(candidate_ids, start=1)}
                max_base_score = max(rrf_scores.values()) if rrf_scores else 1.0

                final_scores: Dict[int, float] = {}
                for mid in candidate_ids:
                    final_scores[mid] = rrf_scores.get(mid, 0.0)

                for mid, cscore in completion_scores.items():
                    final_scores[mid] = (
                        final_scores.get(mid, 0.0)
                        + args.completion_weight * cscore * max_base_score
                    )

                fused_ranked = sorted(
                    final_scores.items(),
                    key=lambda x: (-x[1], base_rank_map.get(x[0], len(base_rank_map) + 1)),
                )
                seen: set = set()
                deduped_ids = []
                for mid, _ in fused_ranked:
                    if mid not in seen:
                        seen.add(mid)
                        deduped_ids.append(mid)

                pre_len = len(candidate_ids)
                candidate_ids = deduped_ids[:final_candidate_k]
                insertion_set = set(candidate_ids) - set(base_rank_map)
                completion_inserted_count = len(insertion_set)

                rrf_scores = final_scores

            coverage_selected_counts: Dict[str, int] = {}
            coverage_reason_counts: Dict[str, int] = {}
            if args.selection_mode == "coverage":
                candidate_ids, coverage_selected_counts, coverage_reason_counts = coverage_select_source_ids(
                    candidate_ids=candidate_ids,
                    source_scores=rrf_scores,
                    id_to_memory=id_to_memory,
                    question=q["question"],
                    k=args.k,
                    preserve_top_n=args.coverage_preserve_top_n,
                    max_candidates=args.coverage_max_candidates,
                    neighbor_added_ids=neighbor_added_ids,
                )
            elif args.selection_mode == "mmr":
                candidate_ids = mmr_select_source_ids(
                    candidate_ids=candidate_ids,
                    source_scores=rrf_scores,
                    id_to_memory=id_to_memory,
                    k=args.k,
                    lambda_param=args.mmr_lambda,
                )

            candidate_debug_scores = []

            if args.reranker_provider != "none":
                candidates = []
                for mid in candidate_ids:
                    m = id_to_memory.get(mid)
                    if m:
                        candidates.append((mid, m.get("text", "")))

                reranked = reranker.rerank(q["question"], candidates)
                reranker_scores = dict(reranked)

                if args.rerank_mode == "pure":
                    final_ranked = [
                        {
                            "memory_id": mid,
                            "rrf_score": rrf_scores.get(mid, 0.0),
                            "reranker_score": score,
                            "final_score": score,
                        }
                        for mid, score in reranked
                    ]
                else:
                    rrf_norm = normalize_scores(rrf_scores)
                    reranker_norm = normalize_scores(reranker_scores)
                    final_ranked = []

                    for mid in candidate_ids:
                        final_score = (
                            args.rrf_final_weight * rrf_norm.get(mid, 0.0)
                            + args.reranker_final_weight * reranker_norm.get(mid, 0.0)
                        )
                        final_ranked.append(
                            {
                                "memory_id": mid,
                                "rrf_score": rrf_scores.get(mid, 0.0),
                                "rrf_norm": rrf_norm.get(mid, 0.0),
                                "reranker_score": reranker_scores.get(mid, 0.0),
                                "reranker_norm": reranker_norm.get(mid, 0.0),
                                "final_score": final_score,
                            }
                        )

                    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)

                if args.retrieved_id_mode == "parent":
                    full_ranked_ids = [item["memory_id"] for item in final_ranked]
                    parent_ids, view_ids, memory_views = resolve_parent_ids(
                        full_ranked_ids, id_to_memory, args.k
                    )
                    final_ids = parent_ids
                    final_view_ids = view_ids
                    final_memory_views = memory_views
                else:
                    final_ids = [item["memory_id"] for item in final_ranked[:args.k]]
                    final_view_ids = final_ids
                    final_memory_views = [
                        id_to_memory.get(mid, {}).get("memory_view", "raw_turn")
                        for mid in final_ids
                    ]
                candidate_debug_scores = final_ranked[: args.candidate_debug_limit]
            else:
                if args.retrieved_id_mode == "parent":
                    parent_ids, view_ids, memory_views = resolve_parent_ids(
                        candidate_ids, id_to_memory, args.k
                    )
                    final_ids = parent_ids
                    final_view_ids = view_ids
                    final_memory_views = memory_views
                else:
                    final_ids = candidate_ids[:args.k]
                    final_view_ids = final_ids
                    final_memory_views = [
                        id_to_memory.get(mid, {}).get("memory_view", "raw_turn")
                        for mid in final_ids
                    ]
                candidate_debug_scores = [
                    {
                        "memory_id": mid,
                        "rrf_score": rrf_scores.get(mid, 0.0),
                        "final_score": rrf_scores.get(mid, 0.0),
                    }
                    for mid in candidate_ids[: args.candidate_debug_limit]
                ]

            processing_ms = (time.perf_counter() - start) * 1000.0
            if args.vector_backend == "sochdb" and args.sochdb_search_mode == "batch":
                latency_ms = processing_ms + vector_search_amortized_ms_per_query
            else:
                latency_ms = processing_ms
            context_ids = final_view_ids if args.retrieved_id_mode == "parent" else final_ids
            debug_context = build_context(context_ids, id_to_memory, args.retrieved_id_mode)

            if args.debug_candidates:
                print(f"DEBUG: final_ids type={type(final_ids)}, first_5={final_ids[:5]}")

            row = {
                "system": system_name,
                "question_id": q["question_id"],
                "sample_id": q["sample_id"],
                "question": q["question"],
                "gold_answer": q.get("gold_answer", ""),
                "category": q.get("category", "unknown"),
                "category_id": q.get("category_id"),
                "evidence_refs": q.get("evidence_refs", []),
                "evidence_memory_ids": q.get("evidence_memory_ids", []),
                "retrieved_memory_ids": final_ids,
                "retrieved_count": len(final_ids),
                "retrieved_id_mode": args.retrieved_id_mode,
                "retrieved_view_memory_ids": final_view_ids,
                "retrieved_parent_memory_ids": [
                    id_to_memory.get(mid, {}).get("parent_memory_id", mid) for mid in final_view_ids
                ],
                "retrieved_memory_views": final_memory_views,
                "approx_context_tokens": approx_tokens(debug_context),
                "latency_ms": latency_ms,
                "vector_search_wall_ms": vector_search_wall_ms,
                "vector_search_amortized_ms_per_query": vector_search_amortized_ms_per_query,
                "embedding_provider": args.embedding_provider,
                "embedding_model": args.embedding_model,
                "reranker_provider": args.reranker_provider,
                "reranker_model": args.reranker_model,
                "rerank_top_n": args.rerank_top_n,
                "final_candidate_k": final_candidate_k,
                "rerank_mode": args.rerank_mode,
                "rrf_final_weight": args.rrf_final_weight,
                "reranker_final_weight": args.reranker_final_weight,
                "vector_backend": args.vector_backend,
                "sochdb_search_mode": args.sochdb_search_mode,
                "memory_render_mode": args.memory_render_mode,
                "memory_view_mode": args.memory_view_mode,
                "view_window_radius": args.view_window_radius,
                "search_record_count": search_record_count,
                "source_memory_count": source_memory_count,
                "source_candidate_k": source_candidate_k,
                "view_candidate_k": view_candidate_k,
                "debug_context": debug_context,
            }

            if args.query_mode in ("multi", "entity_multi"):
                row["query_mode"] = args.query_mode
                row["query_variants"] = query_variants
                row["query_variant_count"] = len(query_variants)
                row["query_probes"] = query_variants
                row["query_probe_count"] = len(query_variants)
                row["max_query_probes"] = args.max_query_probes

            if args.retrieval_plan == "decomposed":
                row["retrieval_plan"] = "decomposed"
                row["decompose_top_n"] = args.decompose_top_n
                row["decompose_queries"] = decompose_queries
                row["decompose_query_count"] = len(decompose_queries)

            if args.retrieval_plan == "anchored_two_hop":
                row["retrieval_plan"] = "anchored_two_hop"
                row["anchor_top_n"] = args.anchor_top_n
                row["anchor_queries"] = anchor_queries
                row["anchor_query_count"] = len(anchor_queries)
                row["anchor_weight"] = args.anchor_weight

            if args.evidence_completion == "conservative":
                row["evidence_completion"] = "conservative"
                row["completion_seed_top_n"] = args.completion_seed_top_n
                row["completion_window_radius"] = args.completion_window_radius
                row["completion_candidate_count"] = completion_candidate_count
                row["completion_inserted_count"] = completion_inserted_count
                row["completion_weight"] = args.completion_weight

            if args.memory_view_mode == "multiview":
                row["memory_view_mode"] = "multiview"
                row["view_window_radius"] = args.view_window_radius
                row["memory_view_types"] = active_memory_view_types
                if multiview_diagnostics:
                    for dk, dv in multiview_diagnostics.items():
                        row[f"mv_{dk}"] = dv
                        row[dk] = dv

            if args.local_neighbor_expansion:
                row["local_neighbor_expansion"] = True
                row["neighbor_expansion_radius"] = args.neighbor_expansion_radius
                row["neighbor_expansion_anchor_k"] = args.neighbor_expansion_anchor_k
                row["neighbor_expansion_added_count"] = neighbor_expansion_added_count

            if args.selection_mode == "coverage":
                row["selection_mode"] = "coverage"
                row["coverage_preserve_top_n"] = args.coverage_preserve_top_n
                row["coverage_max_candidates"] = args.coverage_max_candidates
                row["coverage_selected_counts"] = coverage_selected_counts
                row["coverage_reason_counts"] = coverage_reason_counts
            elif args.selection_mode == "mmr":
                row["selection_mode"] = "mmr"
                row["mmr_lambda"] = args.mmr_lambda

            if args.debug_candidates:
                row["candidate_debug_ids"] = candidate_ids[: args.candidate_debug_limit]
                row["candidate_debug_scores"] = candidate_debug_scores
                if args.memory_view_mode == "multiview":
                    row["candidate_debug_view_types"] = [
                        {"memory_id": mid, "view_type": record_to_view_type.get(mid, "turn_view"),
                         "source_memory_id": record_to_source.get(mid, mid)}
                        for mid in candidate_ids[: args.candidate_debug_limit]
                    ]

            output_rows.append(row)

    write_jsonl(args.out, output_rows)
    print(f"\nWrote {len(output_rows)} rows to {args.out}")

    if client is not None and hasattr(client, "close"):
        client.close()


if __name__ == "__main__":
    main()
