"""
FragranceGPT FastAPI Backend
Run: uvicorn backend.server:app --host 0.0.0.0 --port 8000

WebSocket endpoints for streaming, REST endpoints for simple calls.
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.llm_client import persona_rate_single, purchase_search, predict_notes, sales_expert_analyze

# ── CJK cleanup (guaranteed last-resort strip) ──────────────────
def _is_cjk(c: str) -> bool:
    """Check if a single character is in CJK Unicode ranges."""
    cp = ord(c)
    return (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or   # CJK Extension A
            0xF900 <= cp <= 0xFAFF or   # CJK Compatibility Ideographs
            0x3000 <= cp <= 0x303F or   # CJK Symbols and Punctuation
            0xFF00 <= cp <= 0xFFEF or   # Halfwidth and Fullwidth Forms
            0xFE30 <= cp <= 0xFE4F or   # CJK Compatibility Forms
            0x2E80 <= cp <= 0x2EFF)     # CJK Radicals Supplement

def _clean_cjk(val):
    """Strip CJK characters from a string, list, or nested persona result."""
    if isinstance(val, str):
        return ''.join(c for c in val if not _is_cjk(c)).strip()
    if isinstance(val, list):
        return [_clean_cjk(v) for v in val]
    if isinstance(val, dict):
        return {k: _clean_cjk(v) for k, v in val.items()}
    return val
from backend.task_manager import create_task, set_running, set_done, set_error, get_task
from src.multi_persona_rater import PERSONAS


# ── Global data systems (lazy-loaded on first use) ───────────
_df = None
_vector_store = None
_recommender = None
_data_lock = asyncio.Lock()


async def _ensure_data_loaded():
    """Lazy-load dataframe, vector store, and recommender on first call."""
    global _df, _vector_store, _recommender
    if _recommender is not None:
        return  # already loaded

    async with _data_lock:
        if _recommender is not None:
            return  # double-check
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_data_systems)


def _load_data_systems():
    """Load dataframe, vector store, and recommender. Called once at startup."""
    global _df, _vector_store, _recommender
    from utils import load_data, preprocess
    from src.rag import FragranceVectorStore
    from src.recommender import PerfumeRecommender

    print("[startup] Loading fragrance data...")
    _df = load_data()
    _df = preprocess(_df)
    print(f"[startup] Loaded {len(_df):,} perfumes")

    print("[startup] Initializing vector store...")
    _vector_store = FragranceVectorStore()
    if _vector_store.collection.count() == 0:
        _vector_store.build(_df)
    else:
        print(f"[startup] Vector store ready: {_vector_store.collection.count()} entries")

    print("[startup] Initializing recommender...")
    _recommender = PerfumeRecommender(_vector_store, _df)
    print("[startup] All data systems ready.")


def _run_recommendation_pipeline(accords: list[str]) -> dict:
    """Run full recommendation pipeline (sync, called via thread pool).
    Returns predicted notes + top-K recommendations from real database.
    """
    global _recommender, _df

    # Step 1: Predict top/mid/base notes via RAG + LLM
    predicted_notes = _recommender.predict_notes(accords)
    if predicted_notes is None:
        return {"error": "Notes prediction failed", "recommendations": []}

    # Step 2: Get top-5 real perfumes from database
    recommendations = _recommender.recommend(accords, predicted_notes, top_k=5)

    # Step 3: Build output with full details
    results = []
    for rec in recommendations:
        results.append({
            "name": rec.get("name", ""),
            "brand": rec.get("brand", ""),
            "accords": rec.get("accords", ""),
            "top_notes": rec.get("top_notes", ""),
            "middle_notes": rec.get("middle_notes", ""),
            "base_notes": rec.get("base_notes", ""),
            "rating_value": rec.get("rating_value", 0),
            "rating_count": rec.get("rating_count", 0),
            "bayesian_score": round(rec.get("bayesian_score", 0), 4),
            "content_overlap": rec.get("content_overlap", 0),
            "composite_score": rec.get("composite_score", 0),
            "matching_accords": rec.get("matching_accords", []),
            "similarity": rec.get("similarity", 0),
            "notes_raw": rec.get("notes", ""),
        })

    return {
        "predicted_top_notes": predicted_notes[0],
        "predicted_mid_notes": predicted_notes[1],
        "predicted_base_notes": predicted_notes[2],
        "recommendations": results,
    }


# ── FastAPI app ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure API key is set and preload data systems on startup."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")
    # Preload data systems so first user request is fast
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_data_systems)
    yield


app = FastAPI(title="FragranceGPT Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────

class PersonaRateRequest(BaseModel):
    accords: list[str] = []
    top_notes: list[str] = []
    middle_notes: list[str] = []
    base_notes: list[str] = []
    lang: str = "en"


class PurchaseRequest(BaseModel):
    name: str
    brand: str


class PredictNotesRequest(BaseModel):
    accords: list[str]
    rag_context: str = ""


class SalesExpertRequest(BaseModel):
    description: str
    scenario: str = ""
    target: str = ""
    lang: str = "en"


# ═══════════════════════════════════════════════════════════════
# WebSocket: 8-Persona Rating (streaming)
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/persona-rate")
async def ws_persona_rate(ws: WebSocket):
    await ws.accept()

    try:
        data = await ws.receive_json()
        req = PersonaRateRequest(**data)
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    accords = req.accords or []
    top_notes = req.top_notes or []
    mid_notes = req.middle_notes or []
    base_notes = req.base_notes or []
    lang = req.lang or "en"

    # Fire all 8 persona calls concurrently, wrap with key for completion-order streaming
    async def _rated(key):
        try:
            result = await asyncio.wait_for(
                persona_rate_single(key, accords, top_notes, mid_notes, base_notes, lang),
                timeout=45
            )
            return key, result, None
        except asyncio.TimeoutError:
            return key, None, "timeout after 45s"
        except Exception as e:
            return key, None, str(e)

    futures = [asyncio.create_task(_rated(key)) for key in PERSONAS]

    scores = []
    weights = []
    best_score = -1
    best_name = ""
    worst_score = 999
    worst_name = ""
    completed = 0

    try:
        for coro in asyncio.as_completed(futures, timeout=80):
            key, data, error = await coro
            if not error and data and lang == "en":
                data = _clean_cjk(data)
            persona = PERSONAS[key]

            if error:
                result = {
                    "name": persona["name"],
                    "name_zh": persona.get("name_zh", persona["name"]),
                    "school": persona["school"],
                    "error": error,
                    "weight": persona["weight"],
                }
            else:
                result = {
                    "name": persona["name"],
                    "name_zh": persona.get("name_zh", persona["name"]),
                    "school": persona["school"],
                    "overall": data.get("overall", 3),
                    "pros": data.get("pros", []),
                    "cons": data.get("cons", []),
                    "ratings": data.get("ratings", {}),
                    "comment": data.get("comment", ""),
                    "weight": persona["weight"],
                }
                scores.append(data.get("overall", 3))
                weights.append(persona["weight"])
                if data.get("overall", 3) > best_score:
                    best_score = data["overall"]
                    best_name = persona["name"]
                if data.get("overall", 999) < worst_score:
                    worst_score = data["overall"]
                    worst_name = persona["name"]

            completed += 1
            await ws.send_json({
                "type": "persona",
                "key": key,
                "completed": completed,
                "total": 8,
                "data": result,
            })
    except asyncio.TimeoutError:
        pass  # Some personas timed out — aggregate what we have

    # Aggregate
    import numpy as np
    valid = [(s, w) for s, w in zip(scores, weights) if s is not None]
    if valid:
        s_vals, w_vals = zip(*valid)
        overall = round(float(np.average(s_vals, weights=w_vals)), 2)
    else:
        overall = 0.0

    score_arr = [s for s in scores if s is not None]
    score_range = round(max(score_arr) - min(score_arr), 2) if score_arr else 0

    if score_range < 0.8:
        polarization = "consensus"
    elif score_range < 1.5:
        polarization = "moderate spread"
    elif score_range < 2.5:
        polarization = "divided opinions"
    else:
        polarization = "highly polarizing"

    await ws.send_json({
        "type": "done",
        "aggregated": {
            "overall_weighted": overall,
            "score_range": score_range,
            "polarization": polarization,
            "best": best_name,
            "worst": worst_name,
        }
    })


# ═══════════════════════════════════════════════════════════════
# REST: Purchase Links (task-based async)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/purchase")
async def api_purchase(req: PurchaseRequest):
    """Start purchase link search, return task_id immediately."""
    task_id = create_task()
    set_running(task_id)

    async def _run():
        try:
            result = await purchase_search(req.name, req.brand)
            set_done(task_id, result)
        except Exception as e:
            set_error(task_id, str(e))

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


@app.post("/api/purchase/direct")
async def api_purchase_direct(req: PurchaseRequest):
    """Purchase search — direct return, no polling. Faster for single LLM calls."""
    try:
        result = await purchase_search(req.name, req.brand)
        return {"status": "done", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# REST: Predict Notes (task-based async)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/predict-notes")
async def api_predict_notes(req: PredictNotesRequest):
    """Start notes prediction, return task_id immediately."""
    task_id = create_task()
    set_running(task_id)

    async def _run():
        try:
            result = await predict_notes(req.accords, req.rag_context)
            set_done(task_id, result)
        except Exception as e:
            set_error(task_id, str(e))

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


# ═══════════════════════════════════════════════════════════════
# REST: Sales Expert + Full Data Pipeline
# ═══════════════════════════════════════════════════════════════

@app.post("/api/sales-expert")
async def api_sales_expert(req: SalesExpertRequest):
    """Full pipeline: LLM needs analysis → RAG notes prediction → DB recommendations."""
    task_id = create_task()
    set_running(task_id)

    async def _run():
        try:
            # Lazy-load data systems on first call
            await _ensure_data_loaded()

            # Phase 1: LLM analysis (async)
            analysis = await sales_expert_analyze(
                req.description, req.scenario, req.target, req.lang
            )

            # Phase 2: Extract accords → run recommendation pipeline (sync, via thread pool)
            rec_accords = analysis.get("recommended_accords", [])
            accords_list = [item.get("accord", "") for item in rec_accords if item.get("accord")]

            pipeline_result = None
            if accords_list:
                loop = asyncio.get_event_loop()
                pipeline_result = await loop.run_in_executor(
                    None, _run_recommendation_pipeline, accords_list
                )

            # Merge everything
            result = {
                "needs_analysis": analysis.get("needs_analysis", ""),
                "recommended_accords": rec_accords,
                "ai_blend_direction": analysis.get("ai_blend_direction", ""),
                # Perfume recommendations now come exclusively from the data pipeline (RAG + DB)
                "data_pipeline": pipeline_result,
            }
            set_done(task_id, result)
        except Exception as e:
            set_error(task_id, str(e))

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


# ═══════════════════════════════════════════════════════════════
# REST: Persona Rating (non-streaming, for simple use)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/rate/persona")
async def api_rate_persona(req: PersonaRateRequest):
    """Rate with all 8 personas, return complete result (blocks ~3s)."""
    accords = req.accords or []
    top_notes = req.top_notes or []
    mid_notes = req.middle_notes or []
    base_notes = req.base_notes or []
    lang = req.lang or "en"

    # Fire all 8 persona calls concurrently
    async def _rated(key):
        try:
            result = await asyncio.wait_for(
                persona_rate_single(key, accords, top_notes, mid_notes, base_notes, lang),
                timeout=45
            )
            return key, result, None
        except asyncio.TimeoutError:
            return key, None, "timeout after 45s"
        except Exception as e:
            return key, None, str(e)

    futures = [asyncio.create_task(_rated(key)) for key in PERSONAS]

    results = {}
    scores, weights = [], []
    best_score, best_name = -1, ""
    worst_score, worst_name = 999, ""

    try:
        for coro in asyncio.as_completed(futures, timeout=80):
            key, data, error = await coro
            if not error and data and lang == "en":
                data = _clean_cjk(data)
            persona = PERSONAS[key]
            if error:
                results[key] = {
                    "name": persona["name"],
                    "name_zh": persona.get("name_zh", persona["name"]),
                    "school": persona["school"],
                    "error": error,
                    "weight": persona["weight"],
                }
                continue
            results[key] = {
                "name": persona["name"],
                "name_zh": persona.get("name_zh", persona["name"]),
                "school": persona["school"],
                "overall": data.get("overall", 3),
                "pros": data.get("pros", []),
                "cons": data.get("cons", []),
                "ratings": data.get("ratings", {}),
                "comment": data.get("comment", ""),
                "weight": persona["weight"],
            }
            scores.append(data.get("overall", 3))
            weights.append(persona["weight"])
            if data["overall"] > best_score:
                best_score, best_name = data["overall"], persona["name"]
            if data["overall"] < worst_score:
                worst_score, worst_name = data["overall"], persona["name"]
    except asyncio.TimeoutError:
        pass  # Some personas timed out — aggregate what we have

    import numpy as np
    valid = [(s, w) for s, w in zip(scores, weights) if s is not None]
    if valid:
        s_vals, w_vals = zip(*valid)
        overall = round(float(np.average(s_vals, weights=w_vals)), 2)
    else:
        overall = 0.0
    s_arr = [s for s in scores if s is not None]
    rng = round(max(s_arr) - min(s_arr), 2) if s_arr else 0

    if rng < 0.8: pol = "consensus"
    elif rng < 1.5: pol = "moderate spread"
    elif rng < 2.5: pol = "divided opinions"
    else: pol = "highly polarizing"

    return {
        "personas": results,
        "overall_weighted": overall,
        "score_range": rng,
        "scores": scores,
        "polarization": pol,
        "best": best_name,
        "worst": worst_name,
    }


# ═══════════════════════════════════════════════════════════════
# Task polling endpoint
# ═══════════════════════════════════════════════════════════════

@app.get("/api/task/{task_id}")
async def api_get_task(task_id: str):
    """Poll task status. Returns {status, result, error, progress}."""
    task = get_task(task_id)
    if task is None:
        return {"status": "not_found"}
    return task


# ── Health check ──
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "db_loaded": _df is not None and len(_df) > 0,
        "vector_store_size": _vector_store.collection.count() if _vector_store else 0,
    }
