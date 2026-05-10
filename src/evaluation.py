"""
Evaluation metrics: exact match + fuzzy match, plus aggregation helpers.
"""
import numpy as np
from .synonyms import canonicalize_notes


def evaluate_single(pred_notes, true_notes):
    """Token-level Precision / Recall / F1 for a single perfume (exact match)."""
    pred = set(pred_notes)
    true = set(true_notes)

    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not true:
        return 0.0, 1.0, 0.0

    intersection = pred & true
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(true)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_single_fuzzy(pred_notes, true_notes):
    """
    Token-level F1 with fuzzy matching.
    Both predictions and ground truth are canonicalized before comparison.
    """
    pred = canonicalize_notes(pred_notes)
    true = canonicalize_notes(true_notes)

    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not true:
        return 0.0, 1.0, 0.0

    intersection = pred & true
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(true)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate_results(results):
    """Compute mean Precision/Recall/F1 from a list of per-sample dicts."""
    return {
        'precision': np.mean([m['precision'] for m in results]),
        'recall':    np.mean([m['recall']    for m in results]),
        'f1':        np.mean([m['f1']        for m in results]),
    }


def print_results(results, label):
    """Pretty-print evaluation results for all layers."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for layer in ['top', 'mid', 'base', 'overall']:
        agg = aggregate_results(results[layer])
        print(f"  {layer:<10}  Precision: {agg['precision']:.3f}  "
              f"Recall: {agg['recall']:.3f}  F1: {agg['f1']:.3f}")
