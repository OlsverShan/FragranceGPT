"""
Async Task Runner for Streamlit — non-blocking background tasks.

Pattern:
    1. User clicks button
    2. start_task() spawns background thread, stores state in _task_store
    3. Page reruns immediately (st.rerun())
    4. On each rerun, sync() copies _task_store → st.session_state for display
    5. If still running → show spinner + auto-rerun after 0.8s
    6. When done → show results, stop polling

Thread-safety: background threads write to module-level _task_store dict
protected by threading.Lock. The main Streamlit thread reads from _task_store
via sync() which copies data into st.session_state for rendering.
"""
import json
import threading
import time
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

# ── Thread-safe task store ──────────────────────────────────────
# Background threads write here, main thread reads via sync()
_store_lock = threading.Lock()
_task_store: dict[str, dict] = {}


def _init_key(key: str):
    """Initialize task state in _task_store (call under lock)."""
    if key not in _task_store:
        _task_store[key] = {
            "running": False,
            "result": None,
            "progress": {},
            "partials": {},
        }


def _write(key: str, **kwargs):
    """Thread-safe write to _task_store."""
    with _store_lock:
        _init_key(key)
        for k, v in kwargs.items():
            _task_store[key][k] = v


def _read(key: str) -> dict:
    """Thread-safe read from _task_store."""
    with _store_lock:
        if key not in _task_store:
            return {"running": False, "result": None, "progress": {}, "partials": {}}
        return dict(_task_store[key])


def sync(key: str):
    """Copy _task_store state to st.session_state (call from main thread)."""
    data = _read(key)
    st.session_state[f"{key}_running"] = data["running"]
    st.session_state[f"{key}_result"] = data["result"]
    st.session_state[f"{key}_progress"] = data["progress"]
    st.session_state[f"{key}_partials"] = data["partials"]


def start_persona_ws(task_key: str, accords: list, top_notes: list,
                     mid_notes: list, base_notes: list, lang: str = "en"):
    """Start 8-Persona rating via WebSocket in background thread."""
    data = _read(task_key)
    if data["running"]:
        return

    _write(task_key, running=True, result=None, partials={}, progress={"completed": 0, "total": 8})
    sync(task_key)

    def _ws_thread():
        try:
            import websocket
            ws = websocket.create_connection(
                f"ws://localhost:8000/ws/persona-rate",
                timeout=90,  # recv timeout — matches backend as_completed(80) + buffer
            )
            payload = json.dumps({
                "accords": accords or [],
                "top_notes": top_notes or [],
                "middle_notes": mid_notes or [],
                "base_notes": base_notes or [],
                "lang": lang,
            })
            ws.send(payload)

            while True:
                msg = json.loads(ws.recv())
                msg_type = msg.get("type")

                if msg_type == "persona":
                    with _store_lock:
                        if task_key not in _task_store:
                            _init_key(task_key)
                        _task_store[task_key]["partials"][msg["key"]] = msg["data"]
                        _task_store[task_key]["progress"] = {
                            "completed": msg["completed"],
                            "total": msg["total"],
                        }
                elif msg_type == "done":
                    _write(task_key, result=msg.get("aggregated", {}), running=False)
                    break
                elif msg_type == "error":
                    _write(task_key, result={"error": msg.get("message", "Unknown")}, running=False)
                    break

            ws.close()
        except ImportError:
            _rest_fallback(task_key, accords, top_notes, mid_notes, base_notes, lang)
        except Exception as e:
            # If we already have partials, save them alongside the error
            partials = _read(task_key).get("partials", {})
            result = {"error": str(e)}
            if partials:
                result["partials"] = partials
            _write(task_key, result=result, running=False)

    thread = threading.Thread(target=_ws_thread, daemon=True)
    thread.start()


def _rest_fallback(task_key: str, accords: list, top_notes: list,
                   mid_notes: list, base_notes: list, lang: str):
    """Fallback: use REST /api/rate/persona when websocket-client unavailable."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/rate/persona",
            json={
                "accords": accords or [],
                "top_notes": top_notes or [],
                "middle_notes": mid_notes or [],
                "base_notes": base_notes or [],
                "lang": lang,
            },
            timeout=120,
        )
        data = resp.json()
        partials = {}
        for k, pdata in data.get("personas", {}).items():
            partials[k] = pdata
        _write(task_key, result=data, partials=partials,
               progress={"completed": 8, "total": 8}, running=False)
    except Exception as e:
        _write(task_key, result={"error": str(e)}, running=False)


def start_direct_task(task_key: str, endpoint: str, payload: dict):
    """Start a direct (non-polling) REST task. One POST, no polling overhead."""
    data = _read(task_key)
    if data["running"]:
        return

    _write(task_key, running=True, result=None)
    sync(task_key)

    def _thread():
        try:
            resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=60)
            resp_data = resp.json()
            if resp_data.get("status") == "done":
                _write(task_key, result=resp_data.get("result", {}), running=False)
            else:
                _write(task_key, result={"error": resp_data.get("error", "Request failed")}, running=False)
        except Exception as e:
            _write(task_key, result={"error": str(e)}, running=False)

    thread = threading.Thread(target=_thread, daemon=True)
    thread.start()


def start_rest_task(task_key: str, endpoint: str, payload: dict):
    """Start a REST-based async task (purchase, predict-notes)."""
    data = _read(task_key)
    if data["running"]:
        return

    _write(task_key, running=True, result=None)
    sync(task_key)

    def _poll_thread():
        try:
            # 1. Start task
            resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=10)
            task_info = resp.json()
            backend_task_id = task_info.get("task_id")

            if not backend_task_id:
                _write(task_key, result=task_info, running=False)
                return

            # 2. Poll until done
            for _ in range(120):  # max 60s
                time.sleep(0.5)
                poll_resp = requests.get(
                    f"{BACKEND_URL}/api/task/{backend_task_id}", timeout=5
                )
                poll_data = poll_resp.json()
                status = poll_data.get("status")

                if status == "done":
                    _write(task_key, result=poll_data.get("result", {}), running=False)
                    return
                elif status in ("error", "not_found"):
                    _write(task_key, result={"error": poll_data.get("error", "Task failed")}, running=False)
                    return

            _write(task_key, result={"error": "Request timed out"}, running=False)
        except Exception as e:
            _write(task_key, result={"error": str(e)}, running=False)

    thread = threading.Thread(target=_poll_thread, daemon=True)
    thread.start()


def start_persona_rest(task_key: str, accords: list, top_notes: list,
                       mid_notes: list, base_notes: list, lang: str = "en"):
    """Start persona rating via REST (quiet mode — no streaming, returns aggregate only)."""
    data = _read(task_key)
    if data["running"]:
        return
    _write(task_key, running=True, result=None, partials={}, progress={"completed": 0, "total": 8})
    sync(task_key)

    def _thread():
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/rate/persona",
                json={
                    "accords": accords or [],
                    "top_notes": top_notes or [],
                    "middle_notes": mid_notes or [],
                    "base_notes": base_notes or [],
                    "lang": lang,
                },
                timeout=120,
            )
            resp_data = resp.json()
            partials = {}
            for k, pdata in resp_data.get("personas", {}).items():
                partials[k] = pdata
            _write(task_key, result=resp_data, partials=partials,
                   progress={"completed": 8, "total": 8}, running=False)
        except Exception as e:
            _write(task_key, result={"error": str(e)}, running=False)

    thread = threading.Thread(target=_thread, daemon=True)
    thread.start()


def is_running(task_key: str) -> bool:
    """Check if a task is still running (reads from _task_store)."""
    return _read(task_key).get("running", False)


def get_result(task_key: str) -> dict | None:
    """Get final task result. Returns None if still running."""
    if is_running(task_key):
        return None
    return _read(task_key).get("result")


def get_partials(task_key: str) -> dict:
    """Get partial results for streaming persona display."""
    return _read(task_key).get("partials", {})


def get_progress(task_key: str) -> dict:
    """Get progress info {completed, total}."""
    return _read(task_key).get("progress", {})


def reset_task(task_key: str):
    """Clear task state for re-running."""
    _write(task_key, running=False, result=None, partials={}, progress={})
    sync(task_key)


def poll_loop(delay: float = 2.5):
    """
    Call at end of Streamlit script to auto-poll all running tasks.
    If any task is running, sync all keys, sleep, then rerun.
    """
    any_running = False
    with _store_lock:
        keys = list(_task_store.keys())

    for key in keys:
        data = _read(key)
        if data.get("running"):
            any_running = True
            sync(key)

    if any_running:
        time.sleep(delay)
        st.rerun()
