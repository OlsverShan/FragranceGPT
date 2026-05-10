"""
Market analytics: fragrance dataset insights for Streamlit display.
"""
import numpy as np
from collections import Counter
from src.recommender import PerfumeRecommender


class MarketAnalytics:
    """Simple analytics on the fragrance dataset."""

    def __init__(self, df):
        self.df = df
        self.df_high = df[df['Rating Count'] > 50].copy()
        # Pre-compute Bayesian scores for all perfumes
        recommender = PerfumeRecommender.__new__(PerfumeRecommender)
        valid = df[df['Rating Value'] > 0]['Rating Value']
        recommender.global_mean_rating = float(valid.mean()) if len(valid) > 0 else 3.0
        recommender.m = 50
        self.df_high['bayesian_score'] = self.df_high.apply(
            lambda r: recommender.bayesian_weighted_rating(r['Rating Value'], r['Rating Count']),
            axis=1,
        )

    def top_accord_combos(self, n=10):
        """Top-N accord combinations by average Bayesian score (min 10 perfumes)."""
        combo_stats = {}
        for _, row in self.df_high.iterrows():
            combo = tuple(sorted(row['accords']))
            if len(combo) < 3:
                continue
            if combo not in combo_stats:
                combo_stats[combo] = {'scores': [], 'count': 0}
            combo_stats[combo]['scores'].append(row['bayesian_score'])
            combo_stats[combo]['count'] += 1

        results = []
        for combo, stats in combo_stats.items():
            if stats['count'] >= 10:
                results.append({
                    'combo': ', '.join(combo),
                    'avg_score': np.mean(stats['scores']),
                    'count': stats['count'],
                })
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        return results[:n]

    def most_common_notes_high_rated(self, n=20):
        """Most common notes in perfumes rated >= 4.0 with >= 50 ratings."""
        high = self.df_high[self.df_high['Rating Value'] >= 4.0]
        counter = Counter()
        total = len(high)
        for _, row in high.iterrows():
            notes = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
            counter.update(notes)
        return [{'note': note, 'pct': round(cnt / total * 100, 1)}
                for note, cnt in counter.most_common(n)]

    def brand_rankings(self, n=10):
        """Top-N brands by average Bayesian score (min 5 perfumes)."""
        brand_stats = {}
        for _, row in self.df_high.iterrows():
            brand = str(row.get('Brand', 'unknown')).strip().lower()
            if not brand or brand == 'unknown':
                continue
            if brand not in brand_stats:
                brand_stats[brand] = {'scores': [], 'count': 0}
            brand_stats[brand]['scores'].append(row['bayesian_score'])
            brand_stats[brand]['count'] += 1

        results = []
        for brand, stats in brand_stats.items():
            if stats['count'] >= 5:
                results.append({
                    'brand': brand.title(),
                    'avg_score': np.mean(stats['scores']),
                    'count': stats['count'],
                })
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        return results[:n]

    def rating_distribution(self):
        """Rating histogram data."""
        vals = self.df_high['Rating Value'].dropna()
        bins = [2.0, 2.5, 3.0, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]
        labels = ['2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-3.75',
                  '3.75-4.0', '4.0-4.25', '4.25-4.5', '4.5-5.0']
        counts = []
        for i in range(len(bins) - 1):
            cnt = int(((vals >= bins[i]) & (vals < bins[i+1])).sum())
            counts.append(cnt)
        return labels, counts

    def accord_correlation(self, n=20):
        """Which accords are associated with higher ratings?"""
        all_accords = set()
        for accords_list in self.df_high['accords']:
            all_accords.update(accords_list)
        global_mean = self.df_high['bayesian_score'].mean()

        results = []
        for accord in all_accords:
            has = self.df_high[self.df_high['accords'].apply(lambda a: accord in a)]
            if len(has) < 20:
                continue
            results.append({
                'accord': accord,
                'avg_score': has['bayesian_score'].mean(),
                'vs_global': has['bayesian_score'].mean() - global_mean,
                'count': len(has),
            })
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        return results[:n]
