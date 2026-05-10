"""
Fused Rating Engine: XGBoost (calibration) + 8-Persona LLM (aesthetic signal).

XGBoost: MAE=0.17, r=0.54 — precise calibration, data-driven
8-Persona: MAE~1.3, r=0.50 — good ranking, aesthetic judgment

Fusion formula:
  base = XGBoost.predict(...)              # well-calibrated baseline
  persona_score = weighted average of 8     # aesthetic signal (biased low)
  calibrated = persona_score + bias_corr    # align to Fragrantica scale
  adjustment = calibrated - base            # persona disagrees with XGBoost?
  fused = base + alpha * adjustment         # nudge XGBoost toward persona

Where alpha is scaled by persona agreement (consensus = stronger nudge).
"""
import numpy as np

# Known systematic bias from validation (n=30): persona avg 2.64 vs Fragrantica 3.95
PERSONA_BIAS = 1.31
# How much persona signal to blend in (0 = pure XGBoost, 1 = pure persona)
BASE_ALPHA = 0.25


class FusedRater:
    """
    Combined rating: XGBoost baseline + 8-Persona aesthetic adjustment.

    Usage:
        rater = FusedRater(predictor, multi_rater)
        result = rater.rate(top_notes, mid_notes, base_notes,
                            accords=accords, brand=brand)
        print(f"Fused: {result['fused_score']:.2f} / 5")
    """

    def __init__(self, predictor, multi_rater):
        self.predictor = predictor
        self.multi_rater = multi_rater

    def rate(self, top_notes, mid_notes, base_notes,
             accords=None, brand=None, year=None, perfumer=None,
             gender=None, rating_count=None, persona_bias=None, lang="en"):
        """
        Produce a fused rating from XGBoost + 8-Persona.

        Returns dict with full breakdown: xgb_score, persona_score,
        calibrated_persona, adjustment, fused_score, confidence, reasoning.
        """
        if persona_bias is None:
            persona_bias = PERSONA_BIAS

        # ── 1. XGBoost baseline ──
        xgb_score = self.predictor.predict(
            top_notes, mid_notes, base_notes,
            accords=accords, brand=brand, year=year,
            perfumer=perfumer, gender=gender, rating_count=rating_count,
        )

        # ── 2. 8-Persona rating ──
        persona_result = self.multi_rater.rate(
            accords, top_notes, mid_notes, base_notes, lang,
        )

        if persona_result is None or persona_result.get("overall_weighted", 0) == 0:
            # Persona failed (no API key, network error, etc.) — fall back to XGBoost
            return {
                "fused_score": round(xgb_score, 2),
                "xgb_score": round(xgb_score, 2),
                "persona_score": None,
                "calibrated_persona": None,
                "adjustment": 0.0,
                "confidence": "medium",
                "confidence_label": "XGBoost only (persona unavailable)",
                "polarization": None,
                "persona_details": None,
                "reasoning": "XGBoost prediction. 8-Persona panel unavailable.",
            }

        persona_score = persona_result["overall_weighted"]
        calibrated = persona_score + persona_bias
        scores_arr = persona_result.get("scores", [])

        # ── 3. Calculate adjustment ──
        # Persona signal: do the 8 experts think this is better or worse than XGBoost says?
        adjustment_raw = calibrated - xgb_score

        # ── 4. Scale alpha by persona AGREEMENT ──
        # If all 8 personas agree (narrow spread) → stronger signal
        # If they're divided (wide spread) → weaker signal
        score_range = persona_result.get("score_range", 2.0)
        if score_range < 1.0:
            agreement_factor = 1.0      # consensus → full weight
            agreement_label = "Strong consensus — all 8 experts agree"
        elif score_range < 1.5:
            agreement_factor = 0.8      # moderate → reduced weight
            agreement_label = "Moderate agreement"
        elif score_range < 2.5:
            agreement_factor = 0.5      # divided → half weight
            agreement_label = "Divided opinions — some disagreement"
        else:
            agreement_factor = 0.25     # highly polarizing → minimal weight
            agreement_label = "Highly polarizing — experts strongly disagree"

        alpha = BASE_ALPHA * agreement_factor
        adjustment = alpha * adjustment_raw
        fused = xgb_score + adjustment
        fused = float(np.clip(fused, 1.0, 5.0))

        # ── 5. Confidence level ──
        if score_range < 1.0:
            confidence = "high"
        elif score_range < 2.0:
            confidence = "medium"
        else:
            confidence = "low"

        # ── 6. Build reasoning ──
        abs_adj = abs(adjustment)
        if abs_adj < 0.05:
            adj_desc = "XGBoost and the 8 experts agree — no adjustment needed."
        elif adjustment > 0:
            adj_desc = (f"The 8 experts see MORE potential than XGBoost suggests "
                        f"(+{adjustment:.2f}). {agreement_label}.")
        else:
            adj_desc = (f"The 8 experts are MORE critical than XGBoost suggests "
                        f"({adjustment:.2f}). {agreement_label}.")

        return {
            "fused_score": round(fused, 2),
            "xgb_score": round(xgb_score, 2),
            "persona_score": round(persona_score, 2),
            "calibrated_persona": round(calibrated, 2),
            "adjustment": round(adjustment, 2),
            "adjustment_raw": round(adjustment_raw, 2),
            "alpha_used": round(alpha, 2),
            "confidence": confidence,
            "confidence_label": agreement_label,
            "polarization": persona_result.get("polarization", "unknown"),
            "score_range": score_range,
            "persona_details": persona_result.get("personas", {}),
            "reasoning": adj_desc,
            "persona_best": persona_result.get("best", ""),
            "persona_worst": persona_result.get("worst", ""),
        }


def format_fused_result(result):
    """Format FusedRater result for terminal/print display."""
    if result.get("persona_score") is None:
        return (f"Fused: {result['fused_score']:.2f}/5  "
                f"(XGBoost only — persona unavailable)")

    adj = result["adjustment"]
    direction = "↑" if adj > 0.05 else ("↓" if adj < -0.05 else "—")

    lines = [
        f"{'='*50}",
        f"  FUSED RATING:  {result['fused_score']:.2f} / 5",
        f"  {'─'*46}",
        f"  XGBoost baseline:   {result['xgb_score']:.2f}",
        f"  8-Persona raw:      {result['persona_score']:.2f}",
        f"  8-Persona calibrated:{result['calibrated_persona']:.2f}",
        f"  Persona adjustment:  {result['adjustment']:+.2f} {direction}",
        f"  {'─'*46}",
        f"  Confidence: {result['confidence']}  |  "
        f"Alpha: {result['alpha_used']:.2f}  |  "
        f"Range: {result['score_range']:.2f}",
        f"  {result['reasoning']}",
        f"  Best: {result['persona_best']}  |  Worst: {result['persona_worst']}",
        f"{'='*50}",
    ]
    return "\n".join(lines)
