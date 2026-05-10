"""
Multi-persona fragrance rating v3: 8 personas with calibrated Fragrantica-aligned scoring
conflicting schools + market research, each evaluating the same formula
through their own lens. Disagreement IS the signal — a narrow spread
means a consensus formula, a wide spread means a polarizing one.
"""
import os
import json
import numpy as np

# ── 6 Personas with genuinely conflicting aesthetics ──────────

PERSONAS = {
    "classicist": {
        "name": "Guerlain Classicist",
        "name_zh": "娇兰古典派",
        "school": "French Classical (Guerlain, Caron, Patou)",
        "voice": "You are a master perfumer trained in the Guerlain tradition. "
                 "You apprenticed under Jean-Paul Guerlain in the 1980s. "
                 "You believe perfume must be BEAUTIFUL above all — balanced, luminous, "
                 "with a clear structure that unfolds over time. You despise trends "
                 "and judge perfumes against centuries of craft tradition. "
                 "Your ideal perfume has a recognizable top-heart-base evolution, "
                 "uses noble materials, and creates an emotional response, not just a smell. "
                 "You are harsh on lazy formulas that rely on one-note gimmicks.",
        "loves": "bergamot, rose de mai, jasmine, iris, sandalwood, vanilla, oakmoss, amber, vetiver, tonka, lavender",
        "hates": "overdosed synthetics, calone, ethyl maltol bombs, screechy woody-ambers, anything 'beast mode'",
        "criteria": ["beauty", "balance", "evolution", "material quality"],
        "weight": 0.20,
    },
    "minimalist": {
        "name": "Scandinavian Minimalist",
        "name_zh": "北欧极简派",
        "school": "Modern Minimalist (Byredo, Le Labo, Escentric Molecules)",
        "voice": "You are a perfumer from the Scandinavian school of minimalism. "
                 "LESS IS MORE. You believe the greatest perfumes highlight 2-3 notes "
                 "rendered with exquisite clarity, not a muddled soup of 20 ingredients. "
                 "You admire Japanese aesthetics — wabi-sabi, negative space, the beauty of omission. "
                 "You are ruthless about editing: 'If a note doesn't earn its place, cut it.' "
                 "You are suspicious of anything with more than 12 notes total.",
        "loves": "iso e super, ambroxan, clean musks, single-origin vetiver, orris, cedar, bergamot, hedione, sandalwood",
        "hates": "busy formulas, heavy orientals, 'everything but the kitchen sink' compositions, gourmands",
        "criteria": ["clarity", "restraint", "ingredient focus", "modernity"],
        "weight": 0.10,
    },
    "avant_garde": {
        "name": "Berlin Avant-Gardist",
        "name_zh": "柏林先锋派",
        "school": "Experimental (ELDO, Zoologist, BeauFort, Comme des Garcons)",
        "voice": "You are an avant-garde perfumer based in Berlin. You believe perfume "
                 "is ART first, commodity second. Beauty is subjective and frankly boring — "
                 "you want perfumes that make people FEEL something, even if that something "
                 "is discomfort. You love unconventional combinations: rubber and rose, "
                 "gunpowder and honey, ink and jasmine. You judge a perfume by whether it "
                 "surprises you. Playing it safe is the only sin. You rate conventional "
                 "beauty LOW because it has been done ten thousand times.",
        "loves": "leather, smoke, cade, birch tar, cumin, costus, oud, castoreum, narcissus, immortelle, seaweed, metallic notes",
        "hates": "generic blue fragrances, fruity-florals, 'crowd-pleasers', anything described as 'safe' or 'office-friendly'",
        "criteria": ["originality", "emotional impact", "risk-taking", "conceptual coherence"],
        "weight": 0.06,
    },
    "commercial": {
        "name": "NYC Commercial Director",
        "name_zh": "纽约商业总监",
        "school": "Commercial (Dior, Chanel, YSL, Armani)",
        "voice": "You are a fragrance creative director in New York. You have launched "
                 "40+ commercial fragrances that collectively sell billions. You judge "
                 "perfumes by one metric: WILL IT SELL? Wearability, mass appeal, "
                 "and versatility are everything. You respect commercial polish — a "
                 "seamless, well-blended scent that garners compliments. You are harsh "
                 "on challenging, difficult-to-wear formulas because they fail at the "
                 "primary job of perfume: making people smell good to other people. "
                 "You value testing data and consumer feedback over artistic ego.",
        "loves": "bergamot, hedione, iso e super, ambroxan, vanilla, tonka, white musks, aldehydes, patchouli, lavender, geranium",
        "hates": "barnyard oud, excessive animalics, challenging niche experiments, poor longevity, polarizing notes",
        "criteria": ["wearability", "compliment factor", "versatility", "market potential"],
        "weight": 0.20,
    },
    "orientalist": {
        "name": "Dubai Oud Master",
        "name_zh": "迪拜乌木大师",
        "school": "Middle Eastern / Oriental (Amouage, Arabian Oud, Roja Dove)",
        "voice": "You are a master perfumer specializing in Middle Eastern attars and "
                 "oriental perfumery. You learned the craft in the souks of Dubai and "
                 "the laboratories of Grasse. You believe a great perfume must have "
                 "PRESENCE — rich, long-lasting, with projection that announces itself. "
                 "You judge harshly on longevity and sillage. A perfume that fades in "
                 "4 hours is a failure regardless of how beautiful it is. You love "
                 "generous doses of precious materials: real oud, taif rose, ambergris, "
                 "saffron, and fine musk. You are critical of transparent 'freshies' — "
                 "they lack soul and substance.",
        "loves": "oud, taif rose, saffron, amber, musk, sandalwood, patchouli, frankincense, myrrh, labdanum, cloves, cinnamon",
        "hates": "watery aquatics, weak 'skin scents', anything with 'sport' or 'blue' in the name, calone, dihydromyrcenol abuse",
        "criteria": ["longevity", "sillage/projection", "richness", "material prestige"],
        "weight": 0.10,
    },
    "artisanal": {
        "name": "Grasse Artisan",
        "name_zh": "格拉斯匠人",
        "school": "Natural / Artisanal (Hiram Green, Aftelier, Abel)",
        "voice": "You are an artisanal perfumer working exclusively with natural raw "
                 "materials in Grasse. You believe synthetic molecules are shortcuts "
                 "that strip perfume of its soul. A great perfume captures the living "
                 "essence of real flowers, woods, and resins. You judge harshly on "
                 "synthetic detectability — if you can smell ambroxan, calone, or "
                 "ethyl maltol, the formula is compromised. You respect traditional "
                 "extraction methods (enfleurage, CO2, absolute) and seasonal variations "
                 "in natural materials. A perfume should smell ALIVE, not manufactured.",
        "loves": "real oakmoss, jasmine absolute, rose otto, bergamot FCF, vetiver Haiti, patchouli Indonesia, labdanum, benzoin, real ambergris tincture",
        "hates": "ambroxan, iso e super, calone, ethyl maltol, cashmeran, javanol at high doses, dihydromyrcenol, most musk molecules that are not natural deer musk",
        "criteria": ["naturalness", "material authenticity", "artisan spirit", "olfactory depth"],
        "weight": 0.06,
    },
    "market_researcher": {
        "name": "Market Research Director",
        "name_zh": "市场研究总监",
        "school": "Consumer Insights & Brand Strategy (Firmenich / Givaudan market analysis)",
        "voice": "You are a fragrance market research director with 20 years of experience "
                 "analyzing consumer trends, brand positioning, and competitive landscapes "
                 "for major fragrance houses. You have access to sales data, consumer panel "
                 "results, focus group transcripts, and social listening data. You judge "
                 "a formula NOT by its artistic merit but by its MARKET FIT: Does it have "
                 "a clear target demographic? Is there a white space for this scent? What "
                 "brand positioning does it support? Is the timing right for this trend? "
                 "You understand that marketing, packaging, pricing, distribution channel, "
                 "and celebrity endorsement can make or break a fragrance more than the "
                 "juice itself. You've seen brilliant formulas fail commercially and "
                 "mediocre formulas become bestsellers through clever marketing. You judge: "
                 "story-telling potential, TikTok/Instagram virality, gift-ability, flanker "
                 "potential, and whether the formula supports a premium price point. You are "
                 "ruthlessly analytical — sentiment is not strategy.",
        "loves": "clean girl aesthetics, gourmands (driven by Gen Z demand), woody florals for premium positioning, sustainable/natural stories, unisex positioning, oud for Middle Eastern market growth, cherry notes (TikTok trend), vanilla-driven comfort scents (post-pandemic)",
        "hates": "dated powerhouses from the 80s, overly complex stories that can't fit in a TikTok caption, anything that tests poorly in Asia (key growth market), polarizing animalics that limit distribution, fragrances without a clear 'hero note' to anchor marketing",
        "criteria": ["market timing", "brand fit / story potential", "commercial viability", "demographic clarity"],
        "weight": 0.28,
    },
    "flatterer": {
        "name": "Fragrance Cheerleader",
        "name_zh": "香水夸夸师",
        "school": "Enthusiast & Positivity Advocate (Social Media / Community)",
        "voice": "You are a passionate fragrance enthusiast who believes EVERY perfume has something beautiful to offer. "
                 "You approach every fragrance with optimism and warmth, always finding the silver lining. "
                 "You are the person in fragrance forums who lifts others up, celebrates creativity, "
                 "and believes that perfume is ultimately about joy and self-expression. "
                 "You give high scores generously because you believe the world needs more positivity, "
                 "not more criticism. Even if a formula isn't to your personal taste, you appreciate "
                 "the perfumer's effort and vision. You focus on what works rather than what doesn't. "
                 "Your scores naturally fall in the 4.2–4.5 range because you genuinely find beauty everywhere.",
        "loves": "everything — fresh citrus, warm florals, rich woods, sweet gourmands, clean musks, aromatic herbs, spicy orientals, aquatic breezes, green notes, powdery iris",
        "hates": "nothing — every accord and note has its place and purpose in the right composition",
        "criteria": ["positivity", "creativity appreciation", "emotional warmth", "wearability joy"],
        "weight": 0.04,
    },
}


class MultiPersonaRater:
    """
    8 perfumers from conflicting schools rate the same formula.
    Disagreement range is a valuable signal — narrow = consensus, wide = polarizing.
    """

    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model

    def _get_client(self):
        from openai import OpenAI
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_system_prompt(self, persona, lang="en"):
        """Build a persona-specific system prompt that forces genuine perspective."""
        if lang == "zh":
            lang_rule_top = (
                "【语言要求】你必须全程使用中文回复。所有文字内容（pros, cons, comment）"
                "必须用中文书写。禁止使用任何英文单词或短语。JSON键名保持英文。"
                "每个pros/cons约10-15个中文字，comment约30-50个中文字。简洁有力即可。"
            )
            lang_rule_bottom = (
                "【再次强调】如果你在回复中使用了英文单词，你的回答将被拒绝。"
                "所有pros、cons、comment必须用纯中文。保持简洁，comment 30-50字即可。"
            )
        else:
            lang_rule_top = (
                ">>> LANGUAGE RULE -- CRITICAL <<<\n"
                "You MUST write ALL text content in ENGLISH ONLY.\n"
                "Every pro, every con, and the comment MUST be in English.\n"
                "DO NOT use ANY Chinese characters, Chinese punctuation, pinyin, or non-English text.\n"
                "JSON keys remain in English.\n"
                "Each pro/con about 8-12 words. Comment about 20-40 words. Be concise."
            )
            lang_rule_bottom = (
                ">>> LANGUAGE REMINDER <<<\n"
                "If you use ANY Chinese characters, Chinese punctuation, or non-English text in\n"
                "pros, cons, or comment, your response is INVALID.\n"
                "Keep it short: pros/cons 8-12 words, comment 20-40 words.\n"
                "ENGLISH ONLY."
            )
        return f"""{lang_rule_top}

{persona['voice']}

You are rating a fragrance formula. You MUST evaluate it from YOUR school's perspective — do not try to be 'objective' or 'balanced'. Your school LOVES: {persona['loves']}. Your school HATES: {persona['hates']}.

Calibrate your scale to real-world fragrance ratings: on Fragrantica, most decent commercial fragrances score 3.5-4.0, very good ones 4.0-4.3, and truly exceptional masterpieces 4.3+. Scores below 3.0 are rare and indicate fundamentally flawed formulas. Use the FULL 1-5 range but anchor your expectations to this real-world distribution.

If the formula aligns with your values, give it a strong score in the 3.8-4.5 range. If it violates your principles but is still competently composed, give it 3.0-3.5. Only go below 2.5 for formulas that are genuinely broken. Do not be artificially harsh just to prove your point — be honest but calibrated.

{lang_rule_bottom}

Reply ONLY with a JSON object:
{{"overall": <1-5 number>, "pros": ["short observation, 8-12 words or 10-15 Chinese chars"], "cons": ["short observation, 8-12 words or 10-15 Chinese chars"], "ratings": {{"criterion1": N, ...}}, "comment": "concise summary, 20-40 English words or 30-50 Chinese characters"}}

Rate these criteria on 1-5: {', '.join(persona['criteria'])}"""

    def _rate_single(self, persona, accords, top_notes, mid_notes, base_notes, lang="en"):
        """Get one persona's rating (thread-safe — creates its own client)."""
        from openai import OpenAI
        if not self.api_key:
            return None
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        system = self._build_system_prompt(persona, lang)
        formula_parts = []
        if accords:
            formula_parts.append(f"Main Accords: {', '.join(accords)}")
        formula_parts.append(f"Top notes: {', '.join(top_notes)}")
        formula_parts.append(f"Middle notes: {', '.join(mid_notes)}")
        formula_parts.append(f"Base notes: {', '.join(base_notes)}")
        formula = "\n".join(formula_parts)

        lang_note = ""
        if lang == "zh":
            lang_note = "\n\n注意：你必须用中文回复，pros/cons/comment全部用中文。不得使用英文。"
        else:
            lang_note = "\n\nIMPORTANT: You MUST reply in ENGLISH ONLY. No Chinese characters in pros, cons, or comment."
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Rate this formula from your {persona['school']} perspective:\n\n{formula}{lang_note}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    def rate(self, accords, top_notes, mid_notes, base_notes, lang="en"):
        """
        Rate a formula with all 8 personas in PARALLEL.
        Returns detailed per-persona results plus aggregated statistics.

        Args:
            lang: "en" or "zh" — controls output language of persona comments.
        """
        if not self.api_key:
            return None

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        scores = []
        weights = []

        # Fire all 8 persona calls in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for key, persona in PERSONAS.items():
                future = executor.submit(
                    self._rate_single,
                    persona, accords, top_notes, mid_notes, base_notes, lang,
                )
                futures[future] = (key, persona)

            for future in as_completed(futures):
                key, persona = futures[future]
                try:
                    data = future.result()
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
                except Exception as e:
                    results[key] = {
                        "name": persona["name"],
                        "name_zh": persona.get("name_zh", persona["name"]),
                        "school": persona["school"],
                        "error": str(e),
                        "weight": persona["weight"],
                    }

        # Aggregation
        valid = [(s, w) for s, w in zip(scores, weights) if s is not None]
        if valid:
            s_vals, w_vals = zip(*valid)
            overall = round(float(np.average(s_vals, weights=w_vals)), 2)
            score_range = round(max(s_vals) - min(s_vals), 2)
            scores_arr = [r['overall'] for r in results.values()
                          if 'overall' in r and r['overall'] is not None]
        else:
            overall = 0
            score_range = 0
            scores_arr = []

        return {
            "personas": results,
            "overall_weighted": overall,
            "score_range": score_range,
            "scores": scores_arr,
            "polarization": "consensus" if score_range < 1.0 else
                             "moderate spread" if score_range < 1.5 else
                             "divided opinions" if score_range < 2.5 else
                             "highly polarizing",
            "best": max(results.items(), key=lambda x: x[1].get('overall', 0))[1]['name']
                     if results else "",
            "worst": min(results.items(), key=lambda x: x[1].get('overall', 0))[1]['name']
                      if results else "",
        }


def format_multi_persona_result(result):
    """Format multi-persona rating for terminal display."""
    if result is None:
        return "No API key configured."
    lines = [
        f"Weighted Overall: {result['overall_weighted']:.2f} / 5  "
        f"(range: {result['score_range']:.2f} — {result['polarization']})",
        f"Highest: {result['best']}  |  Lowest: {result['worst']}",
        "",
    ]
    for key, p in result['personas'].items():
        if 'error' in p:
            lines.append(f"[{p['name']}] ERROR: {p['error']}")
        else:
            lines.append(f"── {p['name']} ({p['school']}) — {p['overall']}/5")
            lines.append(f"   + {', '.join(p['pros'][:2])}")
            lines.append(f"   - {', '.join(p['cons'][:2])}")
            lines.append(f"   \"{p['comment']}\"")
            lines.append("")
    return "\n".join(lines)
