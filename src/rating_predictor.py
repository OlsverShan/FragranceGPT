"""
XGBoost rating predictor v3: LLM-enriched features via 3-Agent pipeline.
Features: TF-IDF of notes + note counts + brand target-encode + year + accords
          + perfumer target-encode + rating_count (log) + gender one-hot
          + brand prestige (LLM Agent1+2+3) + perfumer reputation (LLM)
          + note sophistication (statistical).
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr
import xgboost as xgb


class RatingPredictor:
    """
    XGBoost regressor v3: TF-IDF + structured + LLM-enriched features.
    3-Agent pipeline (Fame Collector + Data Miner + Synthesis Director)
    produces brand prestige and perfumer reputation scores.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                           sublinear_tf=True)
        self.model = xgb.XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.brand_encoder = {}
        self.perfumer_encoder = {}
        self.accord_encoder = {}
        self.gender_encoder = LabelEncoder()
        self.global_target_mean = 0.0
        self.is_fitted = False

        # Enrichment mappings (from 3-agent pipeline)
        self.brand_enrichments = {}
        self.perfumer_enrichments = {}
        self.market_position_vocab = {}
        self._sophistication_matrix = None
        # Feature layout tracking
        self._feat_layout = {}

    # ── Feature building ────────────────────────────────────

    def _build_notes_text(self, row):
        all_notes = sorted(row['Top_clean'] | row['Middle_clean'] | row['Base_clean'])
        return ' '.join(all_notes)

    def _target_encode(self, series, target, min_samples=5, smoothing=10):
        """Target encode with smoothing: (n*mean + smooth*global) / (n + smooth)."""
        global_mean = target.mean()
        stats = {}
        for val in series.unique():
            mask = series == val
            n = mask.sum()
            if n < min_samples:
                stats[val] = global_mean
            else:
                local_mean = target[mask].mean()
                stats[val] = (n * local_mean + smoothing * global_mean) / (n + smoothing)
        return stats

    def _fit_encoders(self, df, target, accord_cols):
        """Fit all target encoders on training data only."""
        self.global_target_mean = float(target.mean())

        self.brand_encoder = self._target_encode(
            df['Brand'].astype(str).str.lower().str.strip(), target)
        # Default for unseen brands
        self.brand_encoder['__unknown__'] = self.global_target_mean

        self.perfumer_encoder = self._target_encode(
            df['Perfumer1'].fillna('unknown').astype(str).str.lower().str.strip(), target)
        self.perfumer_encoder['__unknown__'] = self.global_target_mean

        # Accord vocabulary
        all_accords = set()
        for col in accord_cols:
            for val in df[col].dropna():
                all_accords.add(str(val).strip().lower())
        self.accord_encoder = {a: i for i, a in enumerate(sorted(all_accords))}

        self.gender_encoder.fit(df['Gender'].fillna('unisex').astype(str).str.lower())

    def _build_feature_matrix(self, df, accord_cols, fit=False):
        """Build complete feature matrix."""
        n = len(df)
        feature_blocks = []

        # 1. TF-IDF of notes (504 dims with counts)
        texts = df.apply(self._build_notes_text, axis=1)
        if fit:
            tfidf = self.vectorizer.fit_transform(texts)
        else:
            tfidf = self.vectorizer.transform(texts)
        feature_blocks.append(tfidf.toarray())

        # 2. Note counts per layer + total + unique (5 dims)
        counts = np.zeros((n, 5))
        for i, (_, row) in enumerate(df.iterrows()):
            t, m, b = len(row['Top_clean']), len(row['Middle_clean']), len(row['Base_clean'])
            counts[i, 0] = t
            counts[i, 1] = m
            counts[i, 2] = b
            counts[i, 3] = t + m + b
            counts[i, 4] = len(row['Top_clean'] | row['Middle_clean'] | row['Base_clean'])
        feature_blocks.append(counts)

        # 3. Brand target encoding (1 dim)
        brand_encoded = np.full(n, self.global_target_mean)
        for i, (_, row) in enumerate(df.iterrows()):
            brand_key = str(row.get('Brand', '')).lower().strip()
            brand_encoded[i] = self.brand_encoder.get(brand_key, self.brand_encoder['__unknown__'])
        feature_blocks.append(brand_encoded.reshape(-1, 1))

        # 4. Perfumer target encoding (1 dim)
        perfumer_encoded = np.full(n, self.global_target_mean)
        for i, (_, row) in enumerate(df.iterrows()):
            p = str(row.get('Perfumer1', '')).lower().strip()
            if p and p not in ('nan', 'none', 'unknown', ''):
                perfumer_encoded[i] = self.perfumer_encoder.get(p, self.perfumer_encoder['__unknown__'])
            else:
                perfumer_encoded[i] = self.perfumer_encoder['__unknown__']
        feature_blocks.append(perfumer_encoded.reshape(-1, 1))

        # 5. Year — fill missing with median, then normalize (1 dim)
        years = df['Year'].copy()
        median_year = years.median() if not years.isna().all() else 2010
        years = years.fillna(median_year).astype(float)
        years = (years - 1900) / 100.0  # scale to ~0-1.2
        feature_blocks.append(years.values.reshape(-1, 1))

        # 6. Rating Count — log transform (1 dim)
        rc = df['Rating Count'].clip(lower=1)
        log_rc = np.log1p(rc).values.reshape(-1, 1)
        feature_blocks.append(log_rc)

        # 7. Gender one-hot (3 dims)
        gender_raw = df['Gender'].fillna('unisex').astype(str).str.lower()
        gender_int = self.gender_encoder.transform(gender_raw)
        gender_oh = np.zeros((n, len(self.gender_encoder.classes_)))
        for i, g in enumerate(gender_int):
            gender_oh[i, g] = 1
        feature_blocks.append(gender_oh)

        # 8. Accords multi-hot (len(accord_encoder) dims)
        n_accords = len(self.accord_encoder)
        accord_mh = np.zeros((n, n_accords))
        for i, (_, row) in enumerate(df.iterrows()):
            for col in accord_cols:
                val = str(row[col]).strip().lower()
                if val and val in self.accord_encoder:
                    accord_mh[i, self.accord_encoder[val]] = 1
        feature_blocks.append(accord_mh)

        # 9. Brand prestige (from LLM 3-agent enrichment) — 2 dims
        brand_prestige = np.full((n, 2), 0.5)  # default: mid-tier
        for i, (_, row) in enumerate(df.iterrows()):
            brand_key = str(row.get('Brand', '')).lower().strip()
            enrich = self.brand_enrichments.get(brand_key, {})
            if enrich:
                brand_prestige[i, 0] = enrich.get("prestige_tier", 3) / 5.0
                brand_prestige[i, 1] = enrich.get("brand_strength", 3) / 5.0
            else:
                brand_prestige[i, 0] = 3.0 / 5.0
                brand_prestige[i, 1] = 3.0 / 5.0
        feature_blocks.append(brand_prestige)

        # 10. Brand market_position one-hot — N_position dims
        n_pos = len(self.market_position_vocab)
        if n_pos > 0:
            pos_oh = np.zeros((n, n_pos))
            for i, (_, row) in enumerate(df.iterrows()):
                brand_key = str(row.get('Brand', '')).lower().strip()
                enrich = self.brand_enrichments.get(brand_key, {})
                pos = enrich.get("market_position", "unknown")
                if pos in self.market_position_vocab:
                    pos_oh[i, self.market_position_vocab[pos]] = 1
            feature_blocks.append(pos_oh)

        # 11. Perfumer reputation (from LLM 3-agent enrichment) — 2 dims
        perf_reputation = np.full((n, 2), 0.4)  # default: unknown/working
        for i, (_, row) in enumerate(df.iterrows()):
            p = str(row.get('Perfumer1', '')).lower().strip()
            if p and p not in ('nan', 'none', 'unknown', ''):
                enrich = self.perfumer_enrichments.get(p, {})
                if enrich:
                    perf_reputation[i, 0] = enrich.get("reputation_tier", 2) / 5.0
                    perf_reputation[i, 1] = enrich.get("hit_rate", 3) / 5.0
                else:
                    perf_reputation[i, 0] = 2.0 / 5.0
                    perf_reputation[i, 1] = 3.0 / 5.0
            else:
                perf_reputation[i, 0] = 2.0 / 5.0
                perf_reputation[i, 1] = 3.0 / 5.0
        feature_blocks.append(perf_reputation)

        # 12. Note sophistication (statistical, no LLM needed) — 3 dims
        if self._sophistication_matrix is not None and len(self._sophistication_matrix) == n:
            feature_blocks.append(self._sophistication_matrix)
        else:
            feature_blocks.append(np.zeros((n, 3)))

        # Feature layout tracking (for importance reporting)
        self._feat_layout = {
            "tfidf": (0, 500),
            "note_counts": (500, 505),
            "brand_target": (505, 506),
            "perfumer_target": (506, 507),
            "year": (507, 508),
            "rating_count_log": (508, 509),
            "gender": (509, 509 + len(self.gender_encoder.classes_)),
            "accords": (509 + len(self.gender_encoder.classes_),
                        509 + len(self.gender_encoder.classes_) + n_accords),
            "brand_prestige": None,  # will be set after stack
            "market_position": None,
            "perfumer_reputation": None,
            "note_sophistication": None,
        }

        # Stack
        X = np.hstack(feature_blocks).astype(np.float32)

        # Update layout positions post-stack
        pos = 509 + len(self.gender_encoder.classes_) + n_accords
        self._feat_layout["brand_prestige"] = (pos, pos + 2); pos += 2
        self._feat_layout["market_position"] = (pos, pos + n_pos); pos += n_pos
        self._feat_layout["perfumer_reputation"] = (pos, pos + 2); pos += 2
        self._feat_layout["note_sophistication"] = (pos, pos + 3); pos += 3

        return X, counts  # return counts block separately for scaling

    # ── Train ───────────────────────────────────────────────

    def fit(self, df, enrichments=None):
        """Train on high-confidence samples with strict train/test split.
        Args:
            df: preprocessed DataFrame
            enrichments: optional (brand_map, perfumer_map, sophistication_matrix)
                         from src.feature_enrichment.build_all_enrichments()
        """
        mask = df['Rating Count'] > 50
        df_train = df[mask].copy()
        print(f"  High-confidence samples: {len(df_train):,}")

        # Load or use provided enrichments
        if enrichments is not None:
            brand_enrich, perfumer_enrich, sophistication = enrichments
            self.brand_enrichments = brand_enrich
            self.perfumer_enrichments = perfumer_enrich
            self._sophistication_matrix = sophistication
            print(f"  Loaded {len(brand_enrich)} brand + {len(perfumer_enrich)} "
                  f"perfumer enrichments")
        else:
            # Try loading from cache
            from src.feature_enrichment import load_enrichments
            cached = load_enrichments()
            self.brand_enrichments = cached.get("brands", {})
            self.perfumer_enrichments = cached.get("perfumers", {})
            if self.brand_enrichments:
                print(f"  Loaded {len(self.brand_enrichments)} brand enrichments "
                      f"from cache")
            if not self._sophistication_matrix:
                from src.feature_enrichment import compute_note_sophistication
                self._sophistication_matrix = compute_note_sophistication(df_train)

        # Build market_position vocabulary from enrichments
        positions = set()
        for enrich in self.brand_enrichments.values():
            positions.add(enrich.get("market_position", "unknown"))
        self.market_position_vocab = {p: i for i, p in enumerate(sorted(positions))}

        accord_cols = ['mainaccord1', 'mainaccord2', 'mainaccord3',
                       'mainaccord4', 'mainaccord5']
        target = df_train['Rating Value']

        # Strict 80/20 split BEFORE any fitting
        idx = np.arange(len(df_train))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        df_tr = df_train.iloc[train_idx].copy()
        df_te = df_train.iloc[test_idx].copy()
        y_tr = target.iloc[train_idx].values
        y_te = target.iloc[test_idx].values

        print(f"  Train: {len(df_tr):,}  Test: {len(df_te):,}")

        # Fit encoders on TRAIN ONLY
        self._fit_encoders(df_tr, target.iloc[train_idx], accord_cols)

        # Build features
        X_tr, count_tr = self._build_feature_matrix(df_tr, accord_cols, fit=True)
        X_te, count_te = self._build_feature_matrix(df_te, accord_cols, fit=False)

        # Scale count features on TRAIN only
        self.scaler.fit(X_tr[:, 500:505])
        X_tr[:, 500:505] = self.scaler.transform(X_tr[:, 500:505])
        X_te[:, 500:505] = self.scaler.transform(X_te[:, 500:505])

        print(f"  Features: {X_tr.shape[1]} dims "
              f"(TF-IDF:500 + counts:5 + structured + enriched)")

        # Train
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True

        # Evaluate on HELD-OUT test set
        y_pred = self.model.predict(X_te)
        mae = float(np.mean(np.abs(y_pred - y_te)))
        rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
        r, p_val = pearsonr(y_pred, y_te)
        r2 = 1 - np.sum((y_te - y_pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)

        baseline_mae = float(np.mean(np.abs(y_te - y_te.mean())))
        print(f"  Held-out MAE:        {mae:.4f}  (baseline: {baseline_mae:.4f}, "
              f"improvement: {(1 - mae/baseline_mae)*100:.1f}%)")
        print(f"  Held-out RMSE:       {rmse:.4f}")
        print(f"  Held-out Pearson r:  {r:.4f}  (p={p_val:.6f})")
        print(f"  Held-out R^2:        {r2:.4f}")

        # Feature importance by group
        imp = self.model.feature_importances_
        layout = self._feat_layout
        groups = {}
        for name, (s, e) in layout.items():
            if s is not None and e is not None and e <= len(imp):
                groups[name] = imp[s:e].sum()

        # Merge small groups for cleaner display
        display_groups = {
            'TF-IDF notes': groups.get('tfidf', 0),
            'Note counts': groups.get('note_counts', 0),
            'Brand (target enc)': groups.get('brand_target', 0),
            'Perfumer (target enc)': groups.get('perfumer_target', 0),
            'Year': groups.get('year', 0),
            'Rating Count (log)': groups.get('rating_count_log', 0),
            'Gender': groups.get('gender', 0),
            'Accords': groups.get('accords', 0),
            'Brand Prestige (LLM)': groups.get('brand_prestige', 0),
            'Brand Position (LLM)': groups.get('market_position', 0),
            'Perfumer Reputation (LLM)': groups.get('perfumer_reputation', 0),
            'Note Sophistication': groups.get('note_sophistication', 0),
        }

        total_llm = (display_groups['Brand Prestige (LLM)'] +
                     display_groups['Brand Position (LLM)'] +
                     display_groups['Perfumer Reputation (LLM)'])

        print(f"\n  Feature importance by group:")
        total = sum(display_groups.values())
        for name, val in sorted(display_groups.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(val / total * 40) if total > 0 else ''
            print(f"    {name:25s}: {val/total*100:5.1f}%  {bar}")
        print(f"\n  LLM-enriched features total: {total_llm/total*100:.1f}% "
              f"(vs TF-IDF: {display_groups['TF-IDF notes']/total*100:.1f}%)")

        return self

    # ── Predict single ──────────────────────────────────────

    def predict(self, top_notes, mid_notes, base_notes, accords=None, brand=None,
                year=None, perfumer=None, gender=None, rating_count=None):
        """Predict rating for a single fragrance."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        row = pd.Series({
            'Top_clean': set(n.strip().lower() for n in top_notes),
            'Middle_clean': set(n.strip().lower() for n in mid_notes),
            'Base_clean': set(n.strip().lower() for n in base_notes),
            'Brand': brand or '__unknown__',
            'Year': year if year and year > 0 else 2010,
            'Perfumer1': perfumer or 'unknown',
            'Gender': gender or 'unisex',
            'Rating Count': rating_count or 50,
        })
        # Fill accord columns
        accord_cols = ['mainaccord1', 'mainaccord2', 'mainaccord3',
                       'mainaccord4', 'mainaccord5']
        if accords:
            for i in range(5):
                row[accord_cols[i]] = accords[i] if i < len(accords) else None
        else:
            for col in accord_cols:
                row[col] = None

        df = pd.DataFrame([row])
        X, _ = self._build_feature_matrix(df, accord_cols, fit=False)
        X[:, 500:505] = self.scaler.transform(X[:, 500:505])
        pred = self.model.predict(X)[0]
        return float(np.clip(pred, 1.0, 5.0))

    # ── Evaluate ────────────────────────────────────────────

    def evaluate(self, df):
        """
        Evaluate on the full high-confidence dataset.
        NOTE: this uses data that includes training data — for honest
        evaluation, see the printed metrics during fit().
        """
        mask = df['Rating Count'] > 50
        df_ev = df[mask].copy()
        accord_cols = ['mainaccord1', 'mainaccord2', 'mainaccord3',
                       'mainaccord4', 'mainaccord5']
        X, _ = self._build_feature_matrix(df_ev, accord_cols, fit=False)
        # Scale counts using TRAINING scaler
        X[:, 500:505] = self.scaler.transform(X[:, 500:505])

        y_true = df_ev['Rating Value'].values
        y_pred = self.model.predict(X)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r, p = pearsonr(y_pred, y_true)
        return {'mae': mae, 'rmse': rmse, 'pearson_r': float(r), 'pearson_p': float(p)}

    # ── Persistence ─────────────────────────────────────────

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, 'model.json'))
        with open(os.path.join(path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(path, 'encoders.pkl'), 'wb') as f:
            pickle.dump({
                'brand': self.brand_encoder,
                'perfumer': self.perfumer_encoder,
                'accord': self.accord_encoder,
                'gender': self.gender_encoder,
                'global_mean': self.global_target_mean,
                'brand_enrichments': self.brand_enrichments,
                'perfumer_enrichments': self.perfumer_enrichments,
                'market_position_vocab': self.market_position_vocab,
                'feat_layout': self._feat_layout,
            }, f)
        print(f"  Saved to {path}/")

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = xgb.XGBRegressor(verbosity=0, n_jobs=-1)
        obj.model.load_model(os.path.join(path, 'model.json'))
        with open(os.path.join(path, 'vectorizer.pkl'), 'rb') as f:
            obj.vectorizer = pickle.load(f)
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
            obj.scaler = pickle.load(f)
        with open(os.path.join(path, 'encoders.pkl'), 'rb') as f:
            enc = pickle.load(f)
            obj.brand_encoder = enc['brand']
            obj.perfumer_encoder = enc['perfumer']
            obj.accord_encoder = enc['accord']
            obj.gender_encoder = enc['gender']
            obj.global_target_mean = enc['global_mean']
            obj.brand_enrichments = enc.get('brand_enrichments', {})
            obj.perfumer_enrichments = enc.get('perfumer_enrichments', {})
            obj.market_position_vocab = enc.get('market_position_vocab', {})
            obj._feat_layout = enc.get('feat_layout', {})
        obj.is_fitted = True
        return obj
        return obj


# ── LLM Direct Rating (alternative to XGBoost) ──────────────

LLM_RATING_PROMPT = """You are a fragrance expert. Rate this perfume formula on a 1-5 scale based on the notes composition.

Consider: balance between top/middle/base, quality of ingredient combinations, whether the formula is coherent vs random, similarity to successful known formulas.

{formula_text}

Reply ONLY with a JSON object:
{{"rating": <1.0-5.0 number>, "confidence": <0-1>, "reasoning": "one sentence"}}"""


def llm_direct_rate(top_notes, mid_notes, base_notes, accords=None,
                     api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
    """Single LLM call to predict perfume rating from notes."""
    import json as _json
    import os as _os
    from openai import OpenAI
    key = api_key or _os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    client = OpenAI(api_key=key, base_url=base_url)
    formula = f"Top notes: {', '.join(top_notes)}\nMiddle notes: {', '.join(mid_notes)}\nBase notes: {', '.join(base_notes)}"
    if accords:
        formula = f"Main Accords: {', '.join(accords)}\n{formula}"
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": LLM_RATING_PROMPT.format(formula_text=formula)}],
        response_format={"type": "json_object"}, temperature=0.3, max_tokens=200,
    )
    return _json.loads(response.choices[0].message.content.strip())