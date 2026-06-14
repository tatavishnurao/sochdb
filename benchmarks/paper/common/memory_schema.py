#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Iterable


MISSING = (None, "")


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Missing input file: {src}")

    text = src.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if src.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {src}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append({"value": row})
        return rows

    data = json.loads(text)
    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"value": x} for x in data]
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "samples", "train", "validation", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [x if isinstance(x, dict) else {"value": x} for x in value]
        return [data]
    return [{"value": data}]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    require_safe_output_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def coerce_list(value: Any) -> list[Any]:
    if value in MISSING:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def first_present(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in MISSING:
            return value
    return default


def normalize_id(value: Any) -> str:
    return str(value).strip()


def normalize_evidence_ids(value: Any) -> list[str]:
    ids = []
    seen = set()
    for item in coerce_list(value):
        if item in MISSING:
            continue
        if isinstance(item, dict):
            item = first_present(
                item,
                ("memory_id", "message_id", "turn_id", "dia_id", "id", "evidence_id"),
            )
        if item in MISSING:
            continue
        sid = normalize_id(item)
        if sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def normalize_text_for_match(text: Any) -> str:
    chars = []
    for ch in str(text or "").lower():
        chars.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(chars).split())


def match_evidence_texts_to_memories(
    evidence_texts: Iterable[Any],
    memories: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Conservatively map evidence snippets to memory IDs by normalized containment."""
    memory_texts = [
        (normalize_id(m["memory_id"]), normalize_text_for_match(m.get("text", "")))
        for m in memories
        if m.get("memory_id") not in MISSING
    ]
    matched: list[str] = []
    failed: list[str] = []
    seen = set()

    for raw in evidence_texts:
        needle = normalize_text_for_match(raw)
        if not needle:
            continue

        candidates = []
        for mid, haystack in memory_texts:
            if not haystack:
                continue
            if needle in haystack or haystack in needle:
                candidates.append(mid)

        if len(candidates) == 1:
            mid = candidates[0]
            if mid not in seen:
                seen.add(mid)
                matched.append(mid)
        else:
            failed.append(str(raw))

    return matched, failed


def render_memory_text(memory: dict[str, Any], mode: str = "metadata") -> str:
    text = str(memory.get("text", ""))
    if mode == "raw":
        return text
    if mode != "metadata":
        raise ValueError(f"Unknown memory render mode: {mode}")

    lines = []
    for field in ("speaker", "timestamp", "session", "turn_id"):
        value = memory.get(field)
        if value not in MISSING:
            lines.append(f"{field}: {value}")
    lines.append(f"text: {text}")
    return "\n".join(lines)


def group_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id", "default")), []).append(row)
    return grouped


def require_file(path: str | Path, label: str) -> Path:
    src = Path(path)
    if not src.exists() or not src.is_file():
        raise SystemExit(f"{label} file does not exist: {src}")
    return src


def require_safe_output_path(path: str | Path) -> None:
    out = Path(path)
    if str(out).strip() in {"", ".", "/", "/tmp"}:
        raise SystemExit(f"Refusing unsafe output path: {out}")
    resolved = out.resolve()
    if resolved == Path("/tmp") or str(resolved).startswith("/tmp/"):
        raise SystemExit(f"Refusing to write benchmark output under /tmp: {out}")


def preflight_embedding(provider: str, model: str, dim: int) -> None:
    if provider == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is required for --embedding-provider nvidia")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --embedding-provider openai")
    if provider == "nvidia" and "llama-nemotron-embed-1b-v2" in model and dim != 2048:
        raise SystemExit(
            "nvidia/llama-nemotron-embed-1b-v2 requires --embedding-dim 2048"
        )


def preflight_sochdb(host: str | None, port: int | None, enabled: bool) -> None:
    if not enabled:
        return
    if not host:
        raise SystemExit("SochDB host is empty. Pass --host or set SOCHDB_HOST.")
    if not port:
        raise SystemExit("SochDB port is empty. Pass --port or set SOCHDB_PORT.")
    print(f"Testing TCP connection to {host}:{port} ...")
    with socket.create_connection((host, int(port)), timeout=8):
        print("OK: TCP connection works.")


def approximate_tokens(text: str) -> int:
    return len(str(text).split())
