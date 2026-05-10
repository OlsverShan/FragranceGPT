"""
3-Agent LLM Feature Enrichment for XGBoost Rating Predictor.

Architecture:
  Agent 1 "Fame Collector" — LLM analyzes well-known brands/perfumers using
      world knowledge + data stats. Only rates entities it actually recognizes.
  Agent 2 "Data Miner"     — Pure statistical analysis of ALL entities.
      No LLM, no hallucination. Based on avg rating, volume, consistency.
  Agent 3 "Synthesis Director" — Receives Agent 1 + Agent 2 outputs.
      Cross-validates, resolves conflicts, flags disagreements as signals,
      produces final unified mapping.

All results cached to models/enrichments.json. LLM only runs once.
"""
import os
import json
import numpy as np
import pandas as pd

ENRICHMENT_PATH = "models/enrichments.json"

# ══════════════════════════════════════════════════════════════
# LLM client
# ══════════════════════════════════════════════════════════════

def _get_client():
    from openai import OpenAI
    if os.environ.get("DEEPSEEK_API_KEY"):
        return (
            OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com"),
            "deepseek-chat",
        )
    elif os.environ.get("OPENAI_API_KEY"):
        return (OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o")
    return None, None


# ══════════════════════════════════════════════════════════════
# Agent 1: Fame Collector — LLM world knowledge for known brands
# ══════════════════════════════════════════════════════════════

AGENT1_BRAND_PROMPT = """You are Agent 1 "Fame Collector" — a fragrance industry historian with encyclopedic knowledge of perfume brands worldwide.

You will receive a list of brands with their data statistics (perfume count, average rating, rating std).

CRITICAL RULES:
1. ONLY rate brands you are GENUINELY FAMILIAR WITH. If you've never heard of a brand, mark it as: {{"known": false}}
2. For brands you DO know, use your world knowledge as the PRIMARY signal. The data stats are SECONDARY (they help calibrate but shouldn't override clear industry knowledge).
3. Distinguish between: brand general fame (e.g. "Versace is a famous fashion house") vs perfume-specific reputation ("but their fragrances are mid-tier").

For each brand you KNOW, return:
  "brand_name": {
    "known": true,
    "prestige_tier": <1-5>,
    "market_position": "ultra_luxury"|"luxury"|"niche"|"designer"|"mass"|"drugstore"|"celebrity"|"clone_house",
    "brand_strength": <1-5>,
    "confidence": <0.0-1.0>,
    "reasoning": "one sentence on WHY you assigned this tier"
  }

Prestige tier definitions:
  5 = Ultra-luxury / Haute Parfumerie (Guerlain, Roja Dove, Clive Christian, Frederic Malle, Amouage)
  4 = Luxury / High-end (Dior, Chanel, Tom Ford, Hermes, Jo Malone, Byredo, Le Labo)
  3 = Premium / Mainstream Designer (YSL, Armani, Givenchy, Versace, Lancome)
  2 = Mass Market / Affordable (Zara, Avon, Bath & Body Works, celebrity fragrances, Adidas)
  1 = Budget / Generic / Drugstore (no-name drugstore brands)

Market position definitions:
  "ultra_luxury" — Roja Dove, Clive Christian, Frederic Malle, Amouage
  "luxury" — Dior, Chanel, Hermes, Tom Ford, Jo Malone
  "niche" — Byredo, Le Labo, Diptyque, Xerjoff, Creed, Mancera
  "designer" — YSL, Armani, Givenchy, Versace, Burberry
  "mass" — Zara, Avon, Bath & Body Works, Adidas
  "drugstore" — Old Spice, Axe, drugstore generics
  "celebrity" — Britney Spears, Ariana Grande, celebrity names
  "clone_house" — Lattafa, Armaf, Rasasi, houses that primarily clone popular fragrances

Brand strength definitions:
  5 = Legendary perfume house (century+ of perfume history, defined genres)
  4 = Major perfume player (many iconic fragrances, strong market presence)
  3 = Solid perfume reputation (known for fragrance, consistent quality)
  2 = Weak perfume reputation (known more for other products, or spotty fragrance track record)
  1 = No real perfume reputation (license-only, or purely functional scents)

Brands to analyze:
{brand_block}

Reply ONLY with a JSON object. For brands you DON'T know, explicitly set "known": false and omit other fields."""


AGENT1_PERFUMER_PROMPT = """You are Agent 1 "Fame Collector" — a fragrance industry historian who knows the careers of perfumers worldwide.

You will receive a list of perfumers with their data statistics (perfume count, average rating).

CRITICAL RULES:
1. ONLY rate perfumers you are GENUINELY FAMILIAR WITH. If you've never heard of a perfumer, mark them as: {{"known": false}}
2. For perfumers you DO know, use your world knowledge as PRIMARY signal. Data stats are SECONDARY.
3. A perfumer with few perfumes but all legendary (e.g. Edmond Roudnitska) should outrank a prolific but mediocre perfumer.

For each perfumer you KNOW, return:
  "perfumer_name": {
    "known": true,
    "reputation_tier": <1-5>,
    "specialization": "oriental"|"floral"|"woody"|"fresh"|"gourmand"|"chypre"|"versatile"|"unknown",
    "hit_rate": <1-5>,
    "confidence": <0.0-1.0>,
    "reasoning": "one sentence"
  }

Reputation tier definitions:
  5 = Legendary Master (defined modern perfumery: Alberto Morillas, Jacques Cavallier, Dominique Ropion, Jean-Claude Ellena, Olivier Cresp, Francis Kurkdjian, Thierry Wasser)
  4 = Highly Respected / Star Perfumer (many well-known perfumes, strong personal brand)
  3 = Established Professional (solid body of work, respected but not famous)
  2 = Working Perfumer (some perfumes, not particularly notable)
  1 = Unknown / Emerging (no significant track record)

Specialization: "oriental", "floral", "woody", "fresh", "gourmand", "chypre", "versatile", "unknown"

Hit rate: 5 = consistently creates 4.2+ rated perfumes, 1 = rarely exceeds 3.5.

Perfumers to analyze:
{perfumer_block}

Reply ONLY with a JSON object. For perfumers you DON'T know, explicitly set "known": false and omit other fields."""


def _repair_truncated_json(raw):
    """Attempt to repair truncated JSON by closing open braces/strings."""
    if not raw:
        return raw
    # Remove trailing incomplete key-value pairs (e.g., '"key":' or '"key": "val')
    # Find last complete object closing
    depth = 0
    in_string = False
    escape_next = False
    last_safe_pos = -1
    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_safe_pos = i
    # If we found a complete top-level object, trim to that
    if last_safe_pos > 0:
        raw = raw[:last_safe_pos + 1]
    else:
        # Close all open braces
        raw = raw.rstrip().rstrip(',')
        raw += '}' * depth
    return raw


def _parse_llm_json(raw):
    """Robust JSON parsing for LLM responses. Handles markdown, truncation."""
    if not raw:
        return None
    # Strip markdown code blocks
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    # Find JSON object boundaries
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try repairing truncated JSON
    repaired = _repair_truncated_json(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    return None


def run_agent1_brands(brand_stats, client, model):
    """Agent 1: LLM analyzes brands it knows. Returns dict + unknown list."""
    all_known = {}
    all_unknown = set()

    # Split: high-count brands first (more likely to be known)
    major = brand_stats[brand_stats["count"] >= 10].sort_values("count", ascending=False)
    minor = brand_stats[(brand_stats["count"] >= 5) & (brand_stats["count"] < 10)]
    tiny = brand_stats[brand_stats["count"] < 5]

    # Batch major brands to LLM
    batch_size = 50
    batches = [major.iloc[i:i + batch_size]
               for i in range(0, len(major), batch_size)]

    for bi, batch in enumerate(batches):
        lines = []
        for _, row in batch.iterrows():
            lines.append(
                f"{row['Brand']}: {int(row['count'])} perfumes, "
                f"avg {row['avg']:.2f}, std {row['std']:.2f}"
            )
        brand_block = "\n".join(lines)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": AGENT1_BRAND_PROMPT.replace(
                               "{brand_block}", brand_block)}],
                temperature=0.1,
                max_tokens=8192,
            )
            raw = response.choices[0].message.content.strip()
            results = _parse_llm_json(raw)
            if results is None:
                raise ValueError(f"Could not parse JSON from: {raw[:200]}")

            known_count = 0
            for name, data in results.items():
                if isinstance(data, dict) and data.get("known", False):
                    all_known[name] = {
                        "prestige_tier": data.get("prestige_tier", 3),
                        "market_position": data.get("market_position", "unknown"),
                        "brand_strength": data.get("brand_strength", 3),
                        "confidence": data.get("confidence", 0.7),
                        "source": "agent1_llm",
                        "reasoning": data.get("reasoning", ""),
                    }
                    known_count += 1
                else:
                    all_unknown.add(name)
            print(f"  Agent1 brands batch {bi+1}/{len(batches)}: "
                  f"{known_count} known, {len(results)-known_count} unknown")
        except Exception as e:
            print(f"  Agent1 brands batch {bi+1} error: {e}")
            # Print raw response for debugging
            try:
                print(f"    Raw response: {raw[:300]}")
            except Exception:
                pass
            for _, row in batch.iterrows():
                all_unknown.add(row["Brand"])

    # Mark minor+tiny brands as unknown (Agent 2 will handle)
    for _, row in pd.concat([minor, tiny]).iterrows():
        all_unknown.add(row["Brand"])

    print(f"  Agent1 total: {len(all_known)} known brands, "
          f"{len(all_unknown)} unknown brands")
    return all_known, all_unknown


def run_agent1_perfumers(perfumer_stats, client, model):
    """Agent 1: LLM analyzes perfumers it knows."""
    all_known = {}
    all_unknown = set()

    major = perfumer_stats[perfumer_stats["count"] >= 10].sort_values(
        "count", ascending=False)
    minor = perfumer_stats[(perfumer_stats["count"] >= 5) & (perfumer_stats["count"] < 10)]
    tiny = perfumer_stats[perfumer_stats["count"] < 5]

    batch_size = 50
    batches = [major.iloc[i:i + batch_size]
               for i in range(0, len(major), batch_size)]

    for bi, batch in enumerate(batches):
        lines = []
        for _, row in batch.iterrows():
            p = str(row["Perfumer1"]) if pd.notna(row["Perfumer1"]) else "unknown"
            lines.append(f"{p}: {int(row['count'])} perfumes, avg {row['avg']:.2f}")
        perfumer_block = "\n".join(lines)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": AGENT1_PERFUMER_PROMPT.replace(
                               "{perfumer_block}", perfumer_block)}],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()
            results = _parse_llm_json(raw)
            if results is None:
                raise ValueError(f"Could not parse JSON from: {raw[:200]}")

            known_count = 0
            for name, data in results.items():
                if isinstance(data, dict) and data.get("known", False):
                    all_known[name] = {
                        "reputation_tier": data.get("reputation_tier", 3),
                        "specialization": data.get("specialization", "unknown"),
                        "hit_rate": data.get("hit_rate", 3),
                        "confidence": data.get("confidence", 0.7),
                        "source": "agent1_llm",
                        "reasoning": data.get("reasoning", ""),
                    }
                    known_count += 1
                else:
                    all_unknown.add(name)
            print(f"  Agent1 perfumers batch {bi+1}/{len(batches)}: "
                  f"{known_count} known, {len(results)-known_count} unknown")
        except Exception as e:
            print(f"  Agent1 perfumers batch {bi+1} error: {e}")
            try:
                print(f"    Raw response: {raw[:300]}")
            except Exception:
                pass
            for _, row in batch.iterrows():
                all_unknown.add(str(row["Perfumer1"]) if pd.notna(row["Perfumer1"]) else "unknown")

    for _, row in pd.concat([minor, tiny]).iterrows():
        p = str(row["Perfumer1"]) if pd.notna(row["Perfumer1"]) else "unknown"
        all_unknown.add(p)

    print(f"  Agent1 total: {len(all_known)} known perfumers, "
          f"{len(all_unknown)} unknown")
    return all_known, all_unknown


# ══════════════════════════════════════════════════════════════
# Agent 2: Data Miner — Pure statistical analysis, no LLM
# ══════════════════════════════════════════════════════════════

def run_agent2_brands(brand_stats):
    """
    Agent 2: Statistical brand tiering. No hallucination possible.
    Computes prestige from: avg rating, volume (more perfumes = more established),
    rating consistency (lower std = more reliable quality).
    """
    global_avg = brand_stats["avg"].mean()
    global_std = brand_stats["avg"].std()

    results = {}
    for _, row in brand_stats.iterrows():
        count = row["count"]
        avg = row["avg"]
        std = row["std"] if pd.notna(row["std"]) else 0.2

        # Bayesian-adjusted avg: weight by count
        m = 20  # smoothing factor
        adj_avg = (count * avg + m * global_avg) / (count + m)

        # Prestige from adjusted rating
        if adj_avg >= 4.20:
            stat_tier = 5
        elif adj_avg >= 4.05:
            stat_tier = 4
        elif adj_avg >= 3.85:
            stat_tier = 3
        elif adj_avg >= 3.60:
            stat_tier = 2
        else:
            stat_tier = 1

        # Market position from volume + rating pattern
        if count >= 20 and adj_avg >= 4.20:
            position = "niche_or_luxury"
        elif count >= 50:
            position = "major_player"
        elif count >= 20:
            position = "established"
        elif count >= 5:
            position = "small_house"
        else:
            position = "micro_or_new"

        # Brand strength: combines volume, adjusted rating, and consistency
        vol_score = min(5, 1 + np.log1p(count) / 2)  # log scale for volume
        rating_score = adj_avg  # 3.0-4.5 range
        consistency_bonus = 1.0 if std < 0.25 else (0.5 if std < 0.35 else 0)
        strength = np.clip(round((vol_score + rating_score) / 2 + consistency_bonus), 1, 5)

        # Statistical confidence based on sample size
        if count >= 30:
            confidence = 0.9
        elif count >= 10:
            confidence = 0.7
        elif count >= 5:
            confidence = 0.5
        else:
            confidence = 0.3

        results[row["Brand"]] = {
            "prestige_tier": stat_tier,
            "market_position": position,
            "brand_strength": int(strength),
            "confidence": confidence,
            "source": "agent2_stats",
            "stats_detail": f"n={int(count)} adj_avg={adj_avg:.2f} std={std:.2f}",
        }

    print(f"  Agent2: analyzed {len(results)} brands statistically")
    return results


def run_agent2_perfumers(perfumer_stats):
    """Agent 2: Statistical perfumer tiering."""
    global_avg = perfumer_stats["avg"].mean()

    results = {}
    for _, row in perfumer_stats.iterrows():
        p = str(row["Perfumer1"]) if pd.notna(row["Perfumer1"]) else "unknown"
        count = row["count"]
        avg = row["avg"]

        m = 15
        adj_avg = (count * avg + m * global_avg) / (count + m)

        if adj_avg >= 4.20:
            stat_tier = 5
        elif adj_avg >= 4.05:
            stat_tier = 4
        elif adj_avg >= 3.85:
            stat_tier = 3
        elif adj_avg >= 3.60:
            stat_tier = 2
        else:
            stat_tier = 1

        # Hit rate: based on average rating
        hit_rate = max(1, min(5, round((adj_avg - 2.5) * 2)))

        confidence = 0.9 if count >= 20 else (0.7 if count >= 10 else
                     0.5 if count >= 5 else 0.3)

        results[p] = {
            "reputation_tier": stat_tier,
            "specialization": "unknown",
            "hit_rate": hit_rate,
            "confidence": confidence,
            "source": "agent2_stats",
            "stats_detail": f"n={int(count)} adj_avg={adj_avg:.2f}",
        }

    print(f"  Agent2: analyzed {len(results)} perfumers statistically")
    return results


# ══════════════════════════════════════════════════════════════
# Agent 3: Synthesis Director — Cross-validate & resolve conflicts
# ══════════════════════════════════════════════════════════════

AGENT3_BRAND_PROMPT = """You are Agent 3 "Synthesis Director" — resolve disagreements between two brand analyses.

AGENT 1 (Fame Collector — LLM world knowledge) classified well-known brands.
AGENT 2 (Data Miner — pure statistics) analyzed ALL brands from data.

DISAGREEMENTS to resolve (Agent1 tier vs Agent2 tier):
{disagreements}

Rules:
1. Trust Agent1 for well-known luxury/niche brands (Dior, Chanel, Guerlain, etc.)
2. Trust Agent2 for mass-market brands where stats are more reliable
3. Clone houses (Lattafa, Armaf, etc.) with high avg rating are NOT luxury
4. Legendary houses with slightly lower avg (many perfumes dilute mean) keep high tier

Reply ONLY with a JSON:
{{"brand_name": {{"prestige_tier": N, "market_position": "...", "brand_strength": N}}, ...}}"""


AGENT3_PERFUMER_PROMPT = """You are Agent 3 "Synthesis Director" — resolve disagreements between two perfumer analyses.

AGENT 1 (Fame Collector — LLM world knowledge) analyzed famous perfumers.
AGENT 2 (Data Miner — pure statistics) analyzed ALL perfumers from data.

DISAGREEMENTS (Agent1 tier vs Agent2 tier, +/- means Agent1 rated higher/lower):
{disagreements}

Rules:
1. Legendary masters (Morillas, Cavallier, Ropion, Ellena, Cresp, Kurkdjian, Wasser) = tier 5 ALWAYS
2. Great stats + unknown to Agent1 = solid pro (tier 3-4), NOT legend
3. Trust Agent1 for reputation_tier (FAME). Trust Agent2 for hit_rate (DATA).

Reply ONLY with a JSON:
{{"perfumer_name": {{"reputation_tier": N, "hit_rate": N}}, ...}}"""


def run_agent3_brands(agent1_known, agent2_all, agent1_unknown, client, model):
    """Agent 3: Synthesize brand analyses."""
    # Find disagreements where both agents rated the same brand
    disagreements = []
    for brand, a1 in agent1_known.items():
        if brand in agent2_all:
            a2 = agent2_all[brand]
            diff = a1["prestige_tier"] - a2["prestige_tier"]
            if abs(diff) >= 1:  # meaningful disagreement
                disagreements.append(
                    f"  {brand}: Agent1={a1['prestige_tier']} (conf={a1['confidence']:.1f}, "
                    f"{a1.get('reasoning','')}) vs "
                    f"Agent2={a2['prestige_tier']} (conf={a2['confidence']:.1f}, "
                    f"{a2.get('stats_detail','')}) [{diff:+d}]")

    # Sample of unknown brands for Agent3 to review
    unknown_list = list(agent1_unknown)
    unknown_sample = unknown_list[:30]  # show first 30

    # Agent2 summary stats
    a2_tiers = [v["prestige_tier"] for v in agent2_all.values()]
    a2_summary = (f"{len(agent2_all)} brands analyzed. "
                  f"Tier distribution: " +
                  ", ".join(f"tier{t}={a2_tiers.count(t)}" for t in range(1, 6)))

    if not disagreements:
        print("  Agent3: No significant disagreements — using Agent1 for known, "
              "Agent2 for unknown")
        # Fast path: merge directly
        final = {}
        for brand, data in agent2_all.items():
            if brand in agent1_known:
                final[brand] = {**agent1_known[brand], "source": "agent1"}
            else:
                final[brand] = {**data, "source": "agent2"}
        return final

    print(f"  Agent3: resolving {len(disagreements)} disagreements...")

    try:
        content = AGENT3_BRAND_PROMPT.replace(
            "{disagreements}", "\n".join(disagreements[:60]))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=8192,
        )
        raw = response.choices[0].message.content.strip()
        synthesized = _parse_llm_json(raw)
        if synthesized is None:
            raise ValueError(f"Agent3 brands: could not parse JSON")

        # Build final: start with Agent2 as base, override with Agent1 known,
        # then override with Agent3 synthesized (for disagreements)
        final = {}
        for brand, data in agent2_all.items():
            final[brand] = {**data, "source": "agent2"}

        for brand, data in agent1_known.items():
            final[brand] = {**data, "source": "agent1"}

        # Agent3 overrides for disagreements
        resolutions = synthesized.pop("disagreement_resolutions", {})
        for brand, data in synthesized.items():
            if brand in final:
                final[brand] = {**data, "source": "agent3_synthesized"}

        print(f"  Agent3: resolved {len(resolutions)} conflicts, "
              f"final map has {len(final)} brands")
        return final

    except Exception as e:
        print(f"  Agent3 error: {e} — falling back to merge")
        final = {}
        for brand, data in agent2_all.items():
            final[brand] = {**data, "source": "agent2"}
        for brand, data in agent1_known.items():
            final[brand] = {**data, "source": "agent1"}
        return final


def run_agent3_perfumers(agent1_known, agent2_all, agent1_unknown, client, model):
    """Agent 3: Synthesize perfumer analyses."""
    disagreements = []
    for perfumer, a1 in agent1_known.items():
        if perfumer in agent2_all:
            a2 = agent2_all[perfumer]
            diff = a1["reputation_tier"] - a2["reputation_tier"]
            if abs(diff) >= 1:
                disagreements.append(
                    f"  {perfumer}: Agent1={a1['reputation_tier']} "
                    f"(conf={a1['confidence']:.1f}) vs "
                    f"Agent2={a2['reputation_tier']} "
                    f"(conf={a2['confidence']:.1f}, {a2.get('stats_detail','')}) "
                    f"[{diff:+d}]")

    a2_tiers = [v["reputation_tier"] for v in agent2_all.values()]
    a2_summary = (f"{len(agent2_all)} perfumers. "
                  f"Reputation distribution: " +
                  ", ".join(f"tier{t}={a2_tiers.count(t)}" for t in range(1, 6)))

    if not disagreements:
        print("  Agent3 perfumers: No conflicts — merge directly")
        final = {}
        for p, data in agent2_all.items():
            final[p] = {**data, "source": "agent2"} if p not in agent1_known else {**agent1_known[p], "source": "agent1"}
        return final

    print(f"  Agent3: resolving {len(disagreements)} perfumer disagreements...")

    try:
        content = AGENT3_PERFUMER_PROMPT.replace(
            "{disagreements}", "\n".join(disagreements[:60]))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=8192,
        )
        raw = response.choices[0].message.content.strip()
        synthesized = _parse_llm_json(raw)
        if synthesized is None:
            raise ValueError(f"Agent3 perfumers: could not parse JSON")
        resolutions = synthesized.pop("disagreement_resolutions", {})

        final = {}
        for p, data in agent2_all.items():
            final[p] = {**data, "source": "agent2"}
        for p, data in agent1_known.items():
            final[p] = {**data, "source": "agent1"}
        for p, data in synthesized.items():
            if p in final:
                final[p] = {**data, "source": "agent3_synthesized"}

        print(f"  Agent3 perfumers: {len(final)} in final map")
        return final

    except Exception as e:
        print(f"  Agent3 perfumers error: {e} — fallback merge")
        final = {}
        for p, data in agent2_all.items():
            final[p] = {**data, "source": "agent2"}
        for p, data in agent1_known.items():
            final[p] = {**data, "source": "agent1"}
        return final


# ══════════════════════════════════════════════════════════════
# Orchestrator — Run the 3-agent pipeline
# ══════════════════════════════════════════════════════════════

def build_brand_analysis(df, force_refresh=False):
    """Run full 3-agent pipeline for brands."""
    if not force_refresh and os.path.exists(ENRICHMENT_PATH):
        with open(ENRICHMENT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "brands" in data:
            print(f"  Loaded {len(data['brands'])} brand enrichments from cache")
            return data["brands"]

    brand_stats = df.groupby("Brand").agg(
        count=("Rating Value", "count"),
        avg=("Rating Value", "mean"),
        std=("Rating Value", "std"),
    ).reset_index()

    client, model = _get_client()

    if client is None:
        print("  No API key — using Agent2 (statistical) only")
        brands = run_agent2_brands(brand_stats)
        brands["__unknown__"] = {
            "prestige_tier": 3, "market_position": "unknown",
            "brand_strength": 3, "source": "agent2_stats",
        }
    else:
        print("\n" + "="*60)
        print("  AGENT 1: Fame Collector — analyzing known brands...")
        print("="*60)
        a1_known, a1_unknown = run_agent1_brands(brand_stats, client, model)

        print("\n" + "="*60)
        print("  AGENT 2: Data Miner — statistical analysis of ALL brands...")
        print("="*60)
        a2_all = run_agent2_brands(brand_stats)

        print("\n" + "="*60)
        print("  AGENT 3: Synthesis Director — cross-validating...")
        print("="*60)
        brands = run_agent3_brands(a1_known, a2_all, a1_unknown, client, model)
        brands["__unknown__"] = {
            "prestige_tier": 3, "market_position": "unknown",
            "brand_strength": 3, "source": "agent2_stats",
        }

    _save_enrichments({"brands": brands})
    return brands


def build_perfumer_analysis(df, force_refresh=False):
    """Run full 3-agent pipeline for perfumers."""
    if not force_refresh and os.path.exists(ENRICHMENT_PATH):
        with open(ENRICHMENT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "perfumers" in data:
            print(f"  Loaded {len(data['perfumers'])} perfumer enrichments from cache")
            return data["perfumers"]

    perfumer_stats = df.groupby("Perfumer1").agg(
        count=("Rating Value", "count"),
        avg=("Rating Value", "mean"),
        std=("Rating Value", "std"),
    ).reset_index()

    client, model = _get_client()

    if client is None:
        print("  No API key — using Agent2 only")
        perfumers = run_agent2_perfumers(perfumer_stats)
        perfumers["__unknown__"] = {
            "reputation_tier": 2, "specialization": "unknown",
            "hit_rate": 3, "source": "agent2_stats",
        }
    else:
        print("\n" + "="*60)
        print("  AGENT 1: Fame Collector — analyzing known perfumers...")
        print("="*60)
        a1_known, a1_unknown = run_agent1_perfumers(perfumer_stats, client, model)

        print("\n" + "="*60)
        print("  AGENT 2: Data Miner — statistical analysis...")
        print("="*60)
        a2_all = run_agent2_perfumers(perfumer_stats)

        print("\n" + "="*60)
        print("  AGENT 3: Synthesis Director — cross-validating...")
        print("="*60)
        perfumers = run_agent3_perfumers(a1_known, a2_all, a1_unknown,
                                          client, model)
        perfumers["__unknown__"] = {
            "reputation_tier": 2, "specialization": "unknown",
            "hit_rate": 3, "source": "agent2_stats",
        }

    _save_enrichments({"perfumers": perfumers})
    return perfumers


# ══════════════════════════════════════════════════════════════
# Note Sophistication (no LLM needed)
# ══════════════════════════════════════════════════════════════

def compute_note_sophistication(df):
    """Score each perfume by note rarity, diversity, complexity."""
    note_freq = {}
    for _, row in df.iterrows():
        all_notes = row["Top_clean"] | row["Middle_clean"] | row["Base_clean"]
        for note in all_notes:
            note_freq[note] = note_freq.get(note, 0) + 1

    max_freq = max(note_freq.values()) if note_freq else 1
    note_rarity = {n: 1.0 - (c / max_freq) for n, c in note_freq.items()}

    n = len(df)
    sophistication = np.zeros((n, 3))
    for i, (_, row) in enumerate(df.iterrows()):
        all_notes = row["Top_clean"] | row["Middle_clean"] | row["Base_clean"]
        if all_notes:
            sophistication[i, 0] = np.mean([note_rarity.get(n, 0.5)
                                            for n in all_notes])
        total_notes = (len(row["Top_clean"]) + len(row["Middle_clean"]) +
                       len(row["Base_clean"]))
        unique_notes = len(all_notes)
        sophistication[i, 1] = unique_notes / max(total_notes, 1)
        sophistication[i, 2] = np.log1p(total_notes) / np.log1p(30)

    return sophistication


# ══════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════

def _save_enrichments(new_data):
    os.makedirs(os.path.dirname(ENRICHMENT_PATH), exist_ok=True)
    existing = {}
    if os.path.exists(ENRICHMENT_PATH):
        with open(ENRICHMENT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(new_data)
    with open(ENRICHMENT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_enrichments():
    if os.path.exists(ENRICHMENT_PATH):
        with open(ENRICHMENT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_all_enrichments(df, force_refresh=False):
    """Run full 3-agent pipeline. Returns (brand_map, perfumer_map, sophistication)."""
    brands = build_brand_analysis(df, force_refresh)
    perfumers = build_perfumer_analysis(df, force_refresh)
    sophistication = compute_note_sophistication(df)
    return brands, perfumers, sophistication
