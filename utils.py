"""
FragranceGPT Shared Utilities (compatibility re-exports)
=========================================================
Thin wrapper that re-exports from src/ for backward compatibility
with existing experiment scripts.
"""
from src.data import load_data, normalize_notes, preprocess
from src.evaluation import evaluate_single, evaluate_single_fuzzy, aggregate_results, print_results
from src.synonyms import canonicalize_note, canonicalize_notes, rule_normalize, load_synonym_map
