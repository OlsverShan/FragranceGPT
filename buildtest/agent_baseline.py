"""
Baseline 5: Multi-Agent Orchestra (LangGraph)
===============================================
Decompose note prediction into specialized agents:
  Orchestrator → Top/Mid/Base Specialists (parallel) → Composer → Output

Each specialist focuses on ONE layer with layer-specific knowledge.
The composer validates cross-layer compatibility.

Run:
  export DEEPSEEK_API_KEY="sk-xxx"
  python agent_baseline.py
"""

import os
import json
import numpy as np
from typing import TypedDict, List, Set
from openai import OpenAI
from langgraph.graph import StateGraph, END

from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy, print_results
from rag_baseline import FragranceVectorStore, format_references, get_client


# ============================================================
# State Definition
# ============================================================

class PerfumeState(TypedDict):
    accords: List[str]
    rag_context: str          # formatted RAG references
    orchestration_notes: str  # orchestrator analysis
    top_notes: List[str]
    middle_notes: List[str]
    base_notes: List[str]
    composer_feedback: str
    final_top: List[str]
    final_middle: List[str]
    final_base: List[str]


# ============================================================
# Agent Prompts
# ============================================================

ORCHESTRATOR_PROMPT = """You are a master perfumer analyzing a fragrance brief. Given the main accords, write a concise analysis (2-3 sentences) covering:

1. What fragrance family does this belong to?
2. What are the characteristic ingredients for these accords?
3. Any special considerations for Top vs Middle vs Base layer balance?

Main Accords: {accords}

Reference perfumes with similar accords:
{rag_context}

Be brief and technical. Your analysis will guide three specialist perfumers who each design one layer."""

TOP_SPECIALIST_PROMPT = """You are a perfumer specialized in TOP NOTES — the first impression, lasting 5-15 minutes. Top notes are typically: citrus (bergamot, lemon, grapefruit), light fruits, green notes, fresh herbs, aldehydes, aquatic notes.

Fragrance Brief:
{orchestration_notes}

Reference perfumes:
{rag_context}

Predict exactly 5 Top notes for this fragrance. Reply ONLY with JSON:
{{"top_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Rules:
- Use ONLY real perfume ingredients (no abstract concepts)
- All 5 must be plausible TOP notes (volatile, light molecules)
- Standardize names: lowercase, singular form
- Be specific: prefer "sicilian lemon" over just "lemon" when appropriate"""

MID_SPECIALIST_PROMPT = """You are a perfumer specialized in MIDDLE (HEART) NOTES — the body of the fragrance, lasting 20-60 minutes. Middle notes are typically: florals (rose, jasmine, lavender), spices (cardamom, cinnamon), herbs (geranium, clary sage), medium-weight fruits.

Fragrance Brief:
{orchestration_notes}

Reference perfumes:
{rag_context}

Predict exactly 5 Middle notes for this fragrance. Reply ONLY with JSON:
{{"middle_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Rules:
- Use ONLY real perfume ingredients
- All 5 must be plausible HEART notes (medium volatility)
- Focus on the "character" notes that define the fragrance identity
- Be specific when appropriate"""

BASE_SPECIALIST_PROMPT = """You are a perfumer specialized in BASE NOTES — the foundation, lasting 2-24 hours. Base notes are typically: woods (sandalwood, cedar, vetiver), resins (amber, benzoin), musks, vanilla, patchouli, oakmoss, leather, tobacco.

Fragrance Brief:
{orchestration_notes}

Reference perfumes:
{rag_context}

Predict exactly 5 Base notes for this fragrance. Reply ONLY with JSON:
{{"base_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Rules:
- Use ONLY real perfume ingredients
- All 5 must be plausible BASE notes (heavy molecules, long-lasting)
- Include at least one fixative-type note (musk, amber, vanilla, or similar)
- Be specific when appropriate"""

COMPOSER_PROMPT = """You are a master perfumer reviewing a fragrance formula designed by three specialists. Your job is to validate and, if needed, correct the composition.

Top notes (by Top Specialist):    {top_notes}
Middle notes (by Heart Specialist): {middle_notes}
Base notes (by Base Specialist):   {base_notes}

Fragrance Brief:
{orchestration_notes}

Checklist:
1. Do the three layers form a coherent progression? (light → rich → deep)
2. Are any notes misplaced? (e.g., a base note listed as top note)
3. Are there any incompatible pairings? (e.g., conflicting styles)
4. Do the notes collectively match the intended accords?

If everything is fine, return the same notes. If adjustments needed, make them.
Reply ONLY with JSON:
{{"top_notes": [...], "middle_notes": [...], "base_notes": [...],
  "changes_made": false, "comments": "brief explanation"}}

Each layer must have exactly 5 notes."""


# ============================================================
# Agent Functions (LangGraph nodes)
# ============================================================

def make_llm_call(client, model, prompt):
    """Helper: single LLM call, return parsed JSON."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    return json.loads(raw)


def orchestrator_node(state: PerfumeState, client, model):
    """Analyze accords + RAG context → orchestration briefing."""
    prompt = ORCHESTRATOR_PROMPT.format(
        accords=", ".join(state["accords"]),
        rag_context=state["rag_context"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    state["orchestration_notes"] = resp.choices[0].message.content.strip()
    return state


def top_specialist_node(state: PerfumeState, client, model):
    """Predict Top notes only."""
    prompt = TOP_SPECIALIST_PROMPT.format(
        orchestration_notes=state["orchestration_notes"],
        rag_context=state["rag_context"],
    )
    result = make_llm_call(client, model, prompt)
    state["top_notes"] = [n.strip().lower() for n in result.get("top_notes", [])]
    return state


def mid_specialist_node(state: PerfumeState, client, model):
    """Predict Middle notes only."""
    prompt = MID_SPECIALIST_PROMPT.format(
        orchestration_notes=state["orchestration_notes"],
        rag_context=state["rag_context"],
    )
    result = make_llm_call(client, model, prompt)
    state["middle_notes"] = [n.strip().lower() for n in result.get("middle_notes", [])]
    return state


def base_specialist_node(state: PerfumeState, client, model):
    """Predict Base notes only."""
    prompt = BASE_SPECIALIST_PROMPT.format(
        orchestration_notes=state["orchestration_notes"],
        rag_context=state["rag_context"],
    )
    result = make_llm_call(client, model, prompt)
    state["base_notes"] = [n.strip().lower() for n in result.get("base_notes", [])]
    return state


def composer_node(state: PerfumeState, client, model):
    """Validate cross-layer compatibility, produce final output."""
    prompt = COMPOSER_PROMPT.format(
        top_notes=json.dumps(state["top_notes"]),
        middle_notes=json.dumps(state["middle_notes"]),
        base_notes=json.dumps(state["base_notes"]),
        orchestration_notes=state["orchestration_notes"],
    )
    result = make_llm_call(client, model, prompt)
    state["final_top"] = [n.strip().lower() for n in result.get("top_notes", state["top_notes"])]
    state["final_middle"] = [n.strip().lower() for n in result.get("middle_notes", state["middle_notes"])]
    state["final_base"] = [n.strip().lower() for n in result.get("base_notes", state["base_notes"])]
    state["composer_feedback"] = result.get("comments", "")
    return state


# ============================================================
# Build Graph
# ============================================================

def build_graph(client, model):
    """Construct the LangGraph StateGraph."""
    graph = StateGraph(PerfumeState)

    # Nodes
    graph.add_node("orchestrator", lambda s: orchestrator_node(s, client, model))
    graph.add_node("top_specialist", lambda s: top_specialist_node(s, client, model))
    graph.add_node("mid_specialist", lambda s: mid_specialist_node(s, client, model))
    graph.add_node("base_specialist", lambda s: base_specialist_node(s, client, model))
    graph.add_node("composer", lambda s: composer_node(s, client, model))

    # Edges: sequential pipeline
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "top_specialist")
    graph.add_edge("top_specialist", "mid_specialist")
    graph.add_edge("mid_specialist", "base_specialist")
    graph.add_edge("base_specialist", "composer")
    graph.add_edge("composer", END)

    return graph.compile()


# ============================================================
# Evaluation
# ============================================================

def evaluate_agent(df, vector_store, sample_size=50):
    client, model, provider = get_client()
    if client is None:
        print("  No API key found.")
        return None

    print(f"  Provider: {provider}  Model: {model}  Architecture: 5-Agent Orchestra")
    print(f"  Flow: Orchestrator → Top/Mid/Base Specialists → Composer")

    graph = build_graph(client, model)
    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue

        # RAG retrieval
        references = vector_store.retrieve(row['accords'], top_k=5)
        rag_text = format_references(references)

        # Initial state
        state: PerfumeState = {
            "accords": row['accords'],
            "rag_context": rag_text,
            "orchestration_notes": "",
            "top_notes": [],
            "middle_notes": [],
            "base_notes": [],
            "composer_feedback": "",
            "final_top": [],
            "final_middle": [],
            "final_base": [],
        }

        try:
            final_state = graph.invoke(state)
            pred_top = set(final_state.get("final_top", []))
            pred_mid = set(final_state.get("final_middle", []))
            pred_base = set(final_state.get("final_base", []))
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] Error: {e}")
            pred_top, pred_mid, pred_base = set(), set(), set()
            final_state = state

        results['top'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_top, row['Top_clean'])
        )))
        results['mid'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_mid, row['Middle_clean'])
        )))
        results['base'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_base, row['Base_clean'])
        )))

        pred_all = pred_top | pred_mid | pred_base
        true_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
        results['overall'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_all, true_all)
        )))

        results_fuzzy['top'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_top, row['Top_clean'])
        )))
        results_fuzzy['mid'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
        )))
        results_fuzzy['base'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_base, row['Base_clean'])
        )))
        results_fuzzy['overall'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_all, true_all)
        )))

        if (i + 1) % 10 == 0:
            running_f1 = np.mean([m['f1'] for m in results['overall']])
            running_f1f = np.mean([m['f1'] for m in results_fuzzy['overall']])
            print(f"  [{i+1}/{sample_size}] F1: {running_f1:.3f} (exact) / {running_f1f:.3f} (fuzzy)")

    return results, results_fuzzy


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Baseline 5: Multi-Agent Orchestra (LangGraph)")
    print("=" * 60)

    print("\n[1/3] Loading data + vector store...")
    df = load_data()
    df = preprocess(df)
    store = FragranceVectorStore()
    store.build(df)

    print("\n[2/3] Architecture:")
    print("  START → Orchestrator (analyzes accords)")
    print("        → Top Specialist (top notes only)")
    print("        → Middle Specialist (heart notes only)")
    print("        → Base Specialist (base notes only)")
    print("        → Composer (cross-layer validation)")
    print("        → END")

    print("\n[3/3] Running evaluation (50 samples)...")
    print("  Each sample = 5 LLM calls (Orch + Top + Mid + Base + Composer)")
    results, results_fuzzy = evaluate_agent(df, store, sample_size=50)

    if results:
        print_results(results, "Multi-Agent Orchestra (Exact Match)")
        print_results(results_fuzzy, "Multi-Agent Orchestra (Fuzzy Match)")

        overall_f1 = np.mean([m['f1'] for m in results['overall']])
        print(f"\n  {'='*50}")
        print(f"  Final Cross-Baseline Comparison")
        print(f"  {'='*50}")
        baselines = [
            ("Random guessing",          0.006),
            ("Frequency baseline",       0.299),
            ("LLM Zero-shot",            0.335),
            ("LLM + Few-shot",           0.360),
            ("LLM + RAG",                0.402),
            ("RAG + Few-shot (redundant)",0.403),
            ("Multi-Agent Orchestra",    overall_f1),
        ]
        prev_best = 0.402
        for name, f1 in baselines:
            marker = ""
            if f1 > prev_best:
                marker = " ← NEW BEST"
            print(f"  {name:<30} F1 = {f1:.3f}{marker}")


if __name__ == "__main__":
    main()
