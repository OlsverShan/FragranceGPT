"""
Train XGBoost rating predictor v3 with LLM-enriched features.
3-Agent pipeline: Fame Collector + Data Miner + Synthesis Director.

Run once to build enrichments and train:
  python train_rating_predictor.py

To rebuild enrichments from scratch (re-run LLM):
  python train_rating_predictor.py --refresh
"""
import os
import sys
sys.path.insert(0, '.')
from utils import load_data, preprocess
from src.rating_predictor import RatingPredictor
from src.feature_enrichment import build_all_enrichments, load_enrichments


def main(force_refresh=False):
    print("=" * 60)
    print("  Training Rating Predictor v3 (XGBoost + LLM Enrichment)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data()
    df = preprocess(df)
    high_conf = df[df['Rating Count'] > 50]
    print(f"  Total perfumes:        {len(df):,}")
    print(f"  Rating Count > 50:     {len(high_conf):,}")
    print(f"  Rating Value range:    {high_conf['Rating Value'].min():.2f} "
          f"- {high_conf['Rating Value'].max():.2f}")
    print(f"  Rating Value mean:     {high_conf['Rating Value'].mean():.3f}")

    print("\n[2/5] 3-Agent Enrichment Pipeline...")
    if force_refresh:
        print("  --refresh: rebuilding LLM enrichments from scratch...")
    brands, perfumers, sophistication = build_all_enrichments(df, force_refresh)

    # Show enrichment stats
    from collections import Counter
    brand_sources = Counter(v.get("source", "unknown") for v in brands.values())
    perfumer_sources = Counter(v.get("source", "unknown") for v in perfumers.values())
    print(f"  Brand sources: {dict(brand_sources)}")
    print(f"  Perfumer sources: {dict(perfumer_sources)}")

    # Show some interesting enrichments
    print("\n  Sample brand enrichments:")
    for b in ['guerlain', 'dior', 'zara', 'avon', 'roja-dove', 'lattafa-perfumes']:
        if b in brands:
            bd = brands[b]
            print(f"    {b:25s}: tier={bd.get('prestige_tier','?')} "
                  f"pos={bd.get('market_position','?')} "
                  f"strength={bd.get('brand_strength','?')} "
                  f"[{bd.get('source','?')}]")

    print("\n[3/5] Training XGBoost v3 with enriched features...")
    predictor = RatingPredictor()
    predictor.fit(df, enrichments=(brands, perfumers, sophistication))

    print("\n[4/5] Quick evaluation...")
    metrics = predictor.evaluate(df)
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}  (p={metrics['pearson_p']:.4f})")

    baseline_mae = abs(high_conf['Rating Value'] - high_conf['Rating Value'].mean()).mean()
    print(f"\n  Baseline MAE (always predict mean): {baseline_mae:.4f}")
    print(f"  Improvement over baseline:          "
          f"{(1 - metrics['mae']/baseline_mae)*100:.1f}%")

    print("\n[5/5] Saving model...")
    save_dir = "models/rating_predictor"
    os.makedirs(save_dir, exist_ok=True)
    predictor.save(save_dir)
    print(f"  Model saved to {save_dir}/")

    # Quick test
    print("\n  Quick test:")
    test_pred = predictor.predict(
        ["bergamot", "lemon", "neroli", "orange blossom", "mandarin"],
        ["jasmine", "rose", "geranium", "ylang ylang", "iris"],
        ["sandalwood", "vanilla", "musk", "amber", "cedar"],
        brand="chanel",
    )
    print(f"  Predicted rating (Chanel floral): {test_pred:.2f} / 5")

    test_pred2 = predictor.predict(
        ["bergamot", "lemon", "neroli", "orange blossom", "mandarin"],
        ["jasmine", "rose", "geranium", "ylang ylang", "iris"],
        ["sandalwood", "vanilla", "musk", "amber", "cedar"],
        brand="avon",
    )
    print(f"  Predicted rating (Avon floral):   {test_pred2:.2f} / 5")
    print(f"  Brand effect (Chanel - Avon):     {test_pred - test_pred2:.2f}")


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    main(force_refresh=force)
