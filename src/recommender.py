"""
Perfume recommendation engine v2: content-based + quality + diversity hybrid.
Accords → notes prediction → multi-phase retrieval → diverse Top-5.
"""
import os
import json
import numpy as np
from src.rag import FragranceVectorStore, RAG_PROMPT, format_references
from src.synonyms import canonicalize_note


class PerfumeRecommender:
    """
    Hybrid recommendation: vector similarity + content match + Bayesian quality
    + brand diversity + discovery bonus.
    """

    def __init__(self, vector_store, df):
        self.store = vector_store
        self.df = df
        valid = df[df['Rating Value'] > 0]['Rating Value']
        self.global_mean_rating = float(valid.mean()) if len(valid) > 0 else 3.0
        self.m = 50

    # ── LLM client ──────────────────────────────────────────
    def _get_client(self):
        from openai import OpenAI
        if os.environ.get("DEEPSEEK_API_KEY"):
            return (
                OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"),
                "deepseek-chat",
            )
        elif os.environ.get("OPENAI_API_KEY"):
            return (OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o")
        return None, None

    # ── Notes prediction ────────────────────────────────────
    def predict_notes(self, accords):
        client, model = self._get_client()
        if client is None:
            return None
        references = self.store.retrieve(accords, top_k=5)
        refs_text = format_references(references)
        prompt = RAG_PROMPT.format(accords=", ".join(accords), references=refs_text)
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.3, max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"): raw = raw.split("\n", 1)[1]; raw = raw[:-3].strip() if raw.endswith("```") else raw.strip()
        result = json.loads(raw)
        top  = [n.strip().lower() for n in result.get("top_notes", [])]
        mid  = [n.strip().lower() for n in result.get("middle_notes", [])]
        base = [n.strip().lower() for n in result.get("base_notes", [])]
        return top, mid, base

    # ── Scoring functions ───────────────────────────────────
    def bayesian_weighted_rating(self, rating_value, rating_count):
        v = max(rating_count, 0)
        return (v / (v + self.m)) * rating_value + (self.m / (v + self.m)) * self.global_mean_rating

    def _content_score(self, predicted_notes, candidate):
        """Note overlap between predicted formula and candidate perfume."""
        if not predicted_notes:
            return 0.5
        pred_set = set(predicted_notes)
        cand_notes = set()
        for key in ['top_notes', 'middle_notes', 'base_notes']:
            cand_notes.update(n.strip().lower() for n in candidate.get(key, '').split(',') if n.strip())
        if not cand_notes:
            return 0.5
        overlap = len(pred_set & cand_notes) / max(len(pred_set | cand_notes), 1)
        return float(overlap)

    def _diversity_penalty(self, candidate, seen_brands, seen_accord_sets):
        """Penalize if brand already seen or accord combo too similar."""
        brand_key = candidate.get('brand', '').lower().strip()
        if brand_key in seen_brands:
            return 0.15
        # Check accord overlap with already-selected results
        cand_accords = set(a.strip().lower() for a in candidate.get('accords', '').split(','))
        max_overlap = 0
        for seen_accords in seen_accord_sets:
            if cand_accords and seen_accords:
                overlap = len(cand_accords & seen_accords) / max(len(cand_accords | seen_accords), 1)
                max_overlap = max(max_overlap, overlap)
        if max_overlap > 0.8:
            return 0.05
        return 0.0

    # ── Main recommendation ─────────────────────────────────
    def recommend(self, accords, predicted_notes=None, top_k=5):
        """
        Hybrid recommendation:
          Phase 1: Vector retrieval (top 30 by accord similarity)
          Phase 2: Score = 0.35×similarity + 0.20×content + 0.30×Bayesian + 0.15×diversity
          Phase 3: Diversity re-rank (select top-K greedily with penalty)
          Phase 4: Discovery bonus — inject 1 wildcard from outside cluster
        """
        candidates = self.store.retrieve_for_recommendation(accords, top_k=30)

        # Aggregate all predicted notes for content scoring
        all_pred_notes = set()
        if predicted_notes:
            for layer in predicted_notes:
                all_pred_notes.update(n.strip().lower() for n in layer)

        # Normalize scores
        max_sim = max(c['similarity'] for c in candidates) if candidates else 1
        ratings = [self.bayesian_weighted_rating(c['rating_value'], c['rating_count'])
                   for c in candidates]
        max_rating = max(ratings) if ratings else 1
        min_rating = min(ratings) if ratings else 0

        for i, c in enumerate(candidates):
            sim_norm = c['similarity'] / max_sim if max_sim > 0 else 0
            rating_norm = (ratings[i] - min_rating) / (max_rating - min_rating + 0.001)
            content = self._content_score(all_pred_notes, c)
            c['_sim'] = sim_norm
            c['_quality'] = rating_norm
            c['_content'] = content
            c['_bayesian_raw'] = ratings[i]

        # Greedy diversity re-ranking
        selected = []
        seen_brands = set()
        seen_accord_sets = []

        for _ in range(top_k + 1):  # +1 for possible discovery swap
            best_score = -1
            best_c = None
            best_idx = -1
            for i, c in enumerate(candidates):
                if c in selected:
                    continue
                penalty = self._diversity_penalty(c, seen_brands, seen_accord_sets)
                score = (0.35 * c['_sim'] + 0.20 * c['_content'] +
                         0.30 * c['_quality'] + 0.15 * (1.0 - penalty))
                c['_composite'] = score
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_idx = i
            if best_c is None:
                break
            selected.append(best_c)
            seen_brands.add(best_c.get('brand', '').lower().strip())
            seen_accord_sets.append(
                set(a.strip().lower() for a in best_c.get('accords', '').split(',')))

        # Discovery bonus: swap last result with a high-quality outlier
        if len(selected) >= top_k:
            # Find a candidate not yet selected with high Bayesian score but low accord similarity
            remaining = [c for c in candidates if c not in selected and c['rating_value'] > 4.0]
            if remaining:
                remaining.sort(key=lambda c: c['similarity'])  # diverse
                discovery = remaining[0]
                if discovery['_bayesian_raw'] > selected[-1]['_bayesian_raw'] + 0.1:
                    selected[-1] = discovery  # swap

        # Build final output
        results = []
        for c in selected[:top_k]:
            c['matching_accords'] = self._find_matching_accords(accords, c['accords'])
            c['bayesian_score'] = c['_bayesian_raw']
            c['content_overlap'] = round(c['_content'], 3)
            c['composite_score'] = round(c['_composite'], 3)
            results.append(c)
        return results

    # ── Fuzzy accord matching ──────────────────────────────
    def _find_matching_accords(self, query_accords, candidate_accords_str):
        query_canon = set()
        for a in query_accords:
            c = canonicalize_note(a.lower().strip())
            query_canon.add(c)
            query_canon.add(a.lower().strip())
        candidate_list = [a.strip().lower() for a in candidate_accords_str.split(',')]
        matched = []
        for ca in candidate_list:
            ca_canon = canonicalize_note(ca)
            if ca_canon in query_canon or ca in query_canon:
                matched.append(ca)
        return matched
