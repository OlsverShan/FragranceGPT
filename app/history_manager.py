"""
History persistence layer — JSON file storage for past results.
Thread-safe, max 200 entries (FIFO), survives page refreshes.
"""
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent.parent / "history.json"
MAX_ENTRIES = 200
_lock = threading.Lock()


def save_entry(entry_type: str, input_summary: str, result_summary: str,
               data: dict, lang: str = "en") -> dict:
    """Append a history entry. Returns the new entry dict."""
    entry = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": entry_type,
        "input_summary": input_summary,
        "result_summary": result_summary,
        "lang": lang,
        "data": _trim_data(data),
    }
    with _lock:
        history = _read_file()
        history.insert(0, entry)  # newest first
        if len(history) > MAX_ENTRIES:
            history = history[:MAX_ENTRIES]
        _write_file(history)
    return entry


def load_history(limit: int = 100) -> list[dict]:
    """Load history entries, newest first."""
    with _lock:
        return _read_file()[:limit]


def clear_history() -> None:
    """Delete all entries."""
    with _lock:
        _write_file([])


def delete_entry(entry_id: str) -> bool:
    """Delete a single entry by id. Returns True if found and deleted."""
    with _lock:
        history = _read_file()
        new_history = [e for e in history if e.get("id") != entry_id]
        if len(new_history) < len(history):
            _write_file(new_history)
            return True
    return False


def _read_file() -> list[dict]:
    """Read history.json (call under lock)."""
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write_file(history: list[dict]) -> None:
    """Write history.json (call under lock)."""
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _trim_data(data: dict, max_str: int = 300) -> dict:
    """Truncate long strings in data dict to keep file size reasonable."""
    trimmed = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_str:
            trimmed[k] = v[:max_str] + "..."
        elif isinstance(v, list) and len(v) > 20:
            trimmed[k] = v[:20]
        else:
            trimmed[k] = v
    return trimmed
