"""
Simple in-memory task manager for async task tracking.
Streamlit polls GET /api/task/{task_id} to check completion.
"""
import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    task_id: str
    status: str = "pending"  # pending | running | done | error
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    progress: dict = field(default_factory=dict)  # e.g. {"completed": 3, "total": 8}


# ── In-memory store ──
_tasks: dict[str, Task] = {}
_lock = threading.Lock()

# Auto-cleanup: remove tasks older than 10 minutes
_TASK_TTL = 600


def create_task() -> str:
    """Create a new task, return task_id."""
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        _tasks[task_id] = Task(task_id=task_id, status="pending")
    return task_id


def set_running(task_id: str):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].status = "running"


def set_progress(task_id: str, completed: int, total: int):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].progress = {"completed": completed, "total": total}


def set_done(task_id: str, result: dict):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].status = "done"
            _tasks[task_id].result = result


def set_error(task_id: str, error: str):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].status = "error"
            _tasks[task_id].error = error


def get_task(task_id: str) -> Optional[dict]:
    """Get task status. Returns None if not found or expired."""
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return None
        # Check TTL
        if time.time() - task.created_at > _TASK_TTL:
            del _tasks[task_id]
            return None
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "progress": task.progress,
        }


def cleanup_old():
    """Remove expired tasks."""
    now = time.time()
    with _lock:
        expired = [tid for tid, t in _tasks.items()
                   if now - t.created_at > _TASK_TTL]
        for tid in expired:
            del _tasks[tid]
