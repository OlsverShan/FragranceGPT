"""
Shared AsyncOpenAI client — singleton used by all backend endpoints.
Wraps DeepSeek API calls for persona rating, purchase search, and RAG notes prediction.
"""
import os
import re
import sys
from pathlib import Path

# Allow importing from project root for persona prompts
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from src.multi_persona_rater import PERSONAS, MultiPersonaRater

# Reuse the system-prompt builder from existing code
_rater = MultiPersonaRater()

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Purchase link system prompt (reused from src/purchase_agent.py)
PURCHASE_SYSTEM_PROMPT = """You are a fragrance retail expert. For the given perfume, return a concise JSON with where to buy it.

Reply ONLY with:
{
  "retailers": [
    {"name": "Retailer Name", "url": "https://...", "type": "official|department|discounter|niche|marketplace"}
  ],
  "price_tier": "budget|mid-range|premium|luxury|ultra-luxury",
  "price_estimate": "$XX - $YY",
  "similar_perfumes": [
    {"name": "Perfume Name", "brand": "Brand Name", "why_similar": "one-line reason"}
  ]
}
Give 2-3 retailers and 3 similar perfumes. Be brief."""


_client: AsyncOpenAI | None = None

async def get_client() -> AsyncOpenAI:
    """Singleton AsyncOpenAI client — reused across all calls for connection pooling."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = "sk-5e871d95b53d4610a967488ec143fae9"
        os.environ["DEEPSEEK_API_KEY"] = api_key
    _client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE)
    return _client


def _strip_cjk(data: dict) -> dict:
    """Remove CJK characters from text fields in persona results."""
    def _is_cjk(c: str) -> bool:
        cp = ord(c)
        return (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or   # CJK Extension A
                0xF900 <= cp <= 0xFAFF or   # CJK Compatibility Ideographs
                0x3000 <= cp <= 0x303F or   # CJK Symbols and Punctuation (。、【】、)
                0xFF00 <= cp <= 0xFFEF or   # Halfwidth and Fullwidth Forms (，！)
                0xFE30 <= cp <= 0xFE4F or   # CJK Compatibility Forms
                0x2E80 <= cp <= 0x2EFF)     # CJK Radicals Supplement
    def _clean(s: str) -> str:
        result = ''.join(c for c in s if not _is_cjk(c)).strip()
        return re.sub(r'\s{2,}', ' ', result)
    for field in ("comment",):
        if field in data and isinstance(data[field], str):
            data[field] = _clean(data[field])
    for field in ("pros", "cons"):
        if field in data and isinstance(data[field], list):
            cleaned = []
            for item in data[field]:
                if isinstance(item, str):
                    s = _clean(item)
                    if s:
                        cleaned.append(s)
            data[field] = cleaned
    return data


def _safe_json_parse(raw: str) -> dict:
    """Parse JSON safely by stripping invalid control characters first."""
    import re
    import json
    # Strip control chars (0x00-0x1F) except \n \r \t which JSON allows
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
    return json.loads(cleaned)


async def persona_rate_single(persona_key: str, accords: list,
                               top_notes: list, mid_notes: list,
                               base_notes: list, lang: str = "en") -> dict:
    """Rate a formula from a single persona perspective (async)."""
    persona = PERSONAS[persona_key]
    system = _rater._build_system_prompt(persona, lang)

    formula_parts = []
    if accords:
        formula_parts.append(f"Main Accords: {', '.join(accords)}")
    formula_parts.append(f"Top notes: {', '.join(top_notes)}")
    formula_parts.append(f"Middle notes: {', '.join(mid_notes)}")
    formula_parts.append(f"Base notes: {', '.join(base_notes)}")
    formula = "\n".join(formula_parts)

    client = await get_client()
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Rate this formula from your {persona['school']} perspective:\n\n{formula}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
        max_tokens=600,
        timeout=45.0,
    )
    import json
    raw = response.choices[0].message.content.strip()
    data = _safe_json_parse(raw)
    if lang == "en":
        data = _strip_cjk(data)
    return data


async def purchase_search(name: str, brand: str) -> dict:
    """Search for purchase links (async)."""
    client = await get_client()
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": PURCHASE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Perfume: {name}\nBrand: {brand}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=500,
    )
    import json
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


SALES_EXPERT_SYSTEM_PROMPT = """You are a luxury fragrance consultant. A customer describes their ideal perfume scenario — translate it into accords and a creative formula direction.

Reply ONLY with JSON:
{
  "needs_analysis": "2-3 sentences summarizing the customer's true fragrance needs",
  "recommended_accords": [
    {"accord": "english_accord_name", "reason": "one-line reason"}
  ],
  "ai_blend_direction": "creative description of scent journey (top→mid→base)"
}

Rules:
- 3-5 accords only. Accords MUST be common English fragrance terms (citrus, woody, floral, musky, fresh spicy, amber, aquatic, etc.)
- Write analysis in the customer's language
- Be specific and personal"""


async def sales_expert_analyze(description: str, scenario: str = "",
                               target: str = "", lang: str = "en") -> dict:
    """Analyze customer needs and recommend perfumes + AI blend direction (async)."""
    user_parts = [f"Customer description: {description}"]
    if scenario:
        user_parts.append(f"Usage scenario: {scenario}")
    if target:
        user_parts.append(f"Target audience / occasion: {target}")
    user_msg = "\n".join(user_parts)

    lang_instruction = ""
    if lang == "zh":
        lang_instruction = ("\n请用中文回复。注意：recommended_accords 里的 accord 字段必须使用英文（如 citrus, woody, floral），"
                           "香水名和品牌名保持英文原名。其他分析文字用中文。")

    client = await get_client()
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SALES_EXPERT_SYSTEM_PROMPT + lang_instruction},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2000,
    )
    import json
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


async def predict_notes(accords: list, rag_context: str = "") -> dict:
    """Predict fragrance notes from accords via LLM (async)."""
    system = """You are a master perfumer. Given a set of fragrance accords, predict the most likely Top/Middle/Base notes for a well-balanced perfume formula.

Reply ONLY with a JSON object:
{"top_notes": ["note1", "note2", "note3", "note4", "note5"], "middle_notes": ["note1", "note2", "note3", "note4", "note5"], "base_notes": ["note1", "note2", "note3", "note4", "note5"]}

Use exactly 5 notes per layer. Choose notes that are realistic and complementary."""
    user_msg = f"Accords: {', '.join(accords)}"
    if rag_context:
        user_msg = f"Reference similar perfumes:\n{rag_context}\n\n{user_msg}"

    client = await get_client()
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=500,
    )
    import json
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)
