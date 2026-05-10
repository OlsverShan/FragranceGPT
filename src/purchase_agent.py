"""
Purchase Link Agent: LLM-powered shopping link finder for recommended perfumes.

Given a perfume name + brand, the agent uses DeepSeek's training knowledge
to suggest purchase channels, price tiers, and similar products.

Note: Links are generated from LLM knowledge, not real-time web search.
Users should verify availability and pricing.
"""
import os
import json
import hashlib
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "purchase_links"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PURCHASE_SYSTEM_PROMPT = """You are a fragrance retail expert. A user is looking for a specific perfume and wants to know where to buy it.

For the given perfume, provide:
1. **Purchase Links**: 3-5 real retail websites where this perfume is likely sold.
   Use your knowledge of major fragrance retailers. Construct plausible URLs.
2. **Price Tier**: Estimate the price range (budget / mid-range / premium / luxury / ultra-luxury).
3. **Similar Perfumes**: 3 perfumes with a similar scent profile the user might also enjoy.

IMPORTANT RULES:
- Only list retailers you are confident actually sell fragrances.
- For URLs, use the standard format for each retailer (e.g., amazon.com/s?k=perfume+name, sephora.com/search?keyword=...).
- Mark links as "likely available" since you cannot verify real-time stock.
- For similar perfumes, explain WHY each is similar (shared notes, same perfumer, same brand DNA).

Reply ONLY with a JSON object:
{
  "retailers": [
    {
      "name": "Retailer Name",
      "url": "https://...",
      "type": "official | department | discounter | niche | marketplace",
      "likely_available": true,
      "notes": "one-line tip"
    }
  ],
  "price_tier": "budget | mid-range | premium | luxury | ultra-luxury",
  "price_estimate": "$XX - $YY",
  "similar_perfumes": [
    {
      "name": "Perfume Name",
      "brand": "Brand Name",
      "why_similar": "Shares X notes / similar style / same perfumer",
      "price_tier": "..."
    }
  ]
}"""


def _cache_key(perfume_name, brand):
    key = f"{perfume_name.lower().strip()}|{brand.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()


def _load_cache(cache_key):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            # Cache valid for 30 days
            if time.time() - data.get("_cached_at", 0) < 30 * 86400:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_cache(cache_key, data):
    data["_cached_at"] = time.time()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class PurchaseAgent:
    """
    LLM agent that finds purchase links for perfumes.

    Usage:
        agent = PurchaseAgent()
        result = agent.search("Black Opium", "Yves Saint Laurent")
        for r in result["retailers"]:
            print(f"{r['name']}: {r['url']}")
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

    def search(self, perfume_name, brand):
        """
        Find purchase links for a given perfume.

        Returns dict with retailers, price_tier, price_estimate, similar_perfumes.
        Returns None if API key unavailable or call fails.
        """
        # Check cache first
        ck = _cache_key(perfume_name, brand)
        cached = _load_cache(ck)
        if cached:
            return cached

        client = self._get_client()
        if client is None:
            return None

        query = f"Perfume: {perfume_name}\nBrand: {brand}"
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PURCHASE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            _save_cache(ck, result)
            return result
        except Exception as e:
            return {"error": str(e), "retailers": [], "similar_perfumes": []}


def format_purchase_result(result, perfume_name, brand):
    """Format purchase result for terminal/print display."""
    if result is None:
        return "API key not configured — cannot search for purchase links."
    if "error" in result:
        return f"Purchase search failed: {result['error']}"

    lines = [
        f"{'='*60}",
        f"  🛒 Where to buy: {perfume_name} by {brand}",
        f"  💰 {result.get('price_tier', '?').upper()}  |  ~{result.get('price_estimate', '?')}",
        f"  {'─'*56}",
    ]
    for i, r in enumerate(result.get("retailers", []), 1):
        lines.append(f"  {i}. {r['name']} ({r.get('type', '')})")
        lines.append(f"     {r.get('url', 'N/A')}")
        if r.get("notes"):
            lines.append(f"     💡 {r['notes']}")
    if result.get("similar_perfumes"):
        lines.append(f"  {'─'*56}")
        lines.append(f"  Similar perfumes you might like:")
        for sp in result.get("similar_perfumes", []):
            lines.append(f"  • {sp['name']} by {sp['brand']} — {sp.get('why_similar', '')}")
    lines.append(f"{'='*60}")
    return "\n".join(lines)
