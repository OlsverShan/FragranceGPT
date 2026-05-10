"""
Build note synonym mapping table using LLM (DeepSeek).
Output: canonical_name → [all_variants] dictionary.

Run: export DEEPSEEK_API_KEY="sk-xxx" && python build_synonyms.py
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

# Load all unique notes
with open("results/all_unique_notes.json") as f:
    all_notes = json.load(f)

print(f"Loaded {len(all_notes)} unique notes")

# Split into chunks of ~300 notes (to fit in one response)
CHUNK_SIZE = 300
chunks = [all_notes[i:i+CHUNK_SIZE] for i in range(0, len(all_notes), CHUNK_SIZE)]
print(f"Split into {len(chunks)} chunks of ~{CHUNK_SIZE} notes each")

SYSTEM_PROMPT = """You are a professional perfumer and fragrance ingredient expert.

Your task: given a list of perfume note names, identify which names refer to the SAME ingredient and group them. Each group should have one canonical name (the most standard/common form).

Rules:
1. Group notes that are the same ingredient with different modifiers:
   "cedar" = "cedarwood" = "virginia cedar" = "atlas cedar" = "cedar wood"
   → canonical: "cedar"

2. Group notes that differ only by origin/quality adjectives:
   "italian bergamot" = "calabrian bergamot" = "bergamot"
   → canonical: "bergamot"

3. Group notes with "absolute" / "oil" / "essence" suffix with the base name:
   "rose absolute" = "rose oil" = "rose"
   → canonical: "rose"

4. Group spelling variants and abbreviations:
   "musc" = "musk" → canonical: "musk"

5. DO NOT group DIFFERENT ingredients:
   "lemon" ≠ "lemongrass" (different plants)
   "cedar" ≠ "cedar leaf" (different parts, different scent)
   "rose" ≠ "rose geranium" (different plants)

6. Notes that don't have synonyms → list them alone as their own group.

Output valid JSON ONLY:
{
  "groups": [
    {"canonical": "cedar", "variants": ["cedarwood", "virginia cedar", "atlas cedar", "cedar wood", "cedar oil"]},
    {"canonical": "bergamot", "variants": ["italian bergamot", "calabrian bergamot", "bergamotto"]},
    {"canonical": "sandalwood", "variants": []},
    ...
  ]
}

Include EVERY note from the input list. If a note has no variants, list it with empty variants array."""


all_groups = {}

for chunk_idx, chunk in enumerate(chunks):
    print(f"\nProcessing chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} notes)...")

    notes_text = "\n".join(f'- "{n}"' for n in chunk)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Group these perfume notes:\n\n{notes_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=8000,
    )

    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    result = json.loads(raw)
    groups = result.get("groups", [])

    for g in groups:
        canonical = g["canonical"].strip().lower()
        variants = [v.strip().lower() for v in g.get("variants", [])]

        all_groups[canonical] = variants

    print(f"  Got {len(groups)} groups")

# Build the final mapping: any variant → canonical name
synonym_map = {}
for canonical, variants in all_groups.items():
    synonym_map[canonical] = canonical  # canonical → itself
    for v in variants:
        synonym_map[v] = canonical

# Save
with open("results/note_synonyms.json", "w") as f:
    json.dump({
        "groups": all_groups,
        "map": synonym_map,
        "total_canonical": len(all_groups),
        "total_mappings": len(synonym_map),
    }, f, indent=2)

print(f"\n{'='*50}")
print(f"Total canonical groups: {len(all_groups)}")
print(f"Total mapped entries: {len(synonym_map)}")
print(f"Reduction: {len(all_notes)} → {len(all_groups)} ({(1-len(all_groups)/len(all_notes))*100:.1f}%)")
print(f"Saved to results/note_synonyms.json")
