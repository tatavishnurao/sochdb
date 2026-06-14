import json
import re
from pathlib import Path
from datetime import datetime, timedelta

SRC = Path("benchmarks/paper/locomo/data/locomo_memories.jsonl")
OUT = Path("benchmarks/paper/locomo/data/derived/locomo_memories_raw_fact_views.jsonl")

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

def parse_timestamp_date(timestamp):
    if not timestamp:
        return None

    text = str(timestamp)
    m = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r",?\s+(\d{4})\b",
        text,
    )
    if not m:
        return None

    day = int(m.group(1))
    month = MONTHS[m.group(2)]
    year = int(m.group(3))

    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None

def human_date(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"

def relative_time_notes(memory):
    text = str(memory.get("text") or "").lower()
    base_date = parse_timestamp_date(memory.get("timestamp"))
    if base_date is None:
        return []

    notes = []

    if "yesterday" in text:
        d = base_date - timedelta(days=1)
        notes.append(f"yesterday = {human_date(d)}")

    if "tomorrow" in text:
        d = base_date + timedelta(days=1)
        notes.append(f"tomorrow = {human_date(d)}")

    if "today" in text:
        notes.append(f"today = {human_date(base_date)}")

    return notes

def compact(value):
    if value is None:
        return ""
    return str(value).strip()

def build_fact_text(memory):
    speaker = compact(memory.get("speaker"))
    timestamp = compact(memory.get("timestamp"))
    session = compact(memory.get("session"))
    dia_id = compact(memory.get("dia_id"))
    text = compact(memory.get("text"))

    parts = []

    if speaker:
        parts.append(f"subject/speaker: {speaker}")

    if timestamp:
        parts.append(f"time: {timestamp}")

    notes = relative_time_notes(memory)
    for note in notes:
        parts.append(f"resolved_time: {note}")

    if session:
        parts.append(f"session: {session}")

    if dia_id:
        parts.append(f"dialogue_id: {dia_id}")

    if text:
        if speaker:
            parts.append(f"fact_view: {speaker} said or indicated: {text}")
        else:
            parts.append(f"fact_view: {text}")

    return "\n".join(parts).strip()

def make_view(memory, view_code, view_name, text):
    original_id = memory.get("memory_id")

    try:
        original_id_int = int(original_id)
    except Exception:
        raise ValueError(f"memory_id must be int-like, got {original_id!r}")

    out = dict(memory)
    out["original_memory_id"] = original_id_int
    out["parent_memory_id"] = original_id_int
    out["memory_view"] = view_name
    out["memory_id"] = original_id_int * 100 + view_code
    out["text"] = text
    return out

if not SRC.exists():
    print(f"ERROR: missing source file: {SRC}")
    raise SystemExit

rows = []
for line_no, line in enumerate(SRC.read_text().splitlines(), 1):
    line = line.strip()
    if not line:
        continue

    try:
        memory = json.loads(line)
    except Exception as e:
        print(f"ERROR: bad JSON line {line_no}: {e}")
        raise SystemExit

    raw_text = compact(memory.get("text"))
    fact_text = build_fact_text(memory)

    rows.append(make_view(memory, 1, "raw_turn", raw_text))
    rows.append(make_view(memory, 2, "fact_view", fact_text))

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"source={SRC}")
print(f"output={OUT}")
print(f"derived_rows={len(rows)}")
print(f"expected_views_per_memory=2")
print("\nPreview:")
for row in rows[:4]:
    print("-" * 80)
    print(f"memory_id={row.get('memory_id')} parent={row.get('parent_memory_id')} view={row.get('memory_view')}")
    print(row.get("text", "")[:700])
