"""Quick test: Few-shot ONLY (no RAG) to isolate contribution."""
import json, os, sys
import numpy as np
from openai import OpenAI
from utils import load_data, preprocess, evaluate_single

df = load_data()
df = preprocess(df)
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")

FEWSHOT = [
    {"accords":"citrus, woody, fresh, fruity, aromatic","top_notes":["sicilian lemon","apple","cedar","bellflower","green notes"],"middle_notes":["jasmine","white rose","bamboo","lily-of-the-valley","freesia"],"base_notes":["amber","musk","cedar","sandalwood","white musk"]},
    {"accords":"aromatic, warm spicy, lavender, woody, fresh spicy","top_notes":["cardamom","bergamot","lavender","pink pepper","lemon"],"middle_notes":["lavender","virginia cedar","bergamot","geranium","clary sage"],"base_notes":["vetiver","caraway","tonka bean","patchouli","cedar"]},
    {"accords":"musky, powdery, white floral, citrus, floral","top_notes":["bergamot","african orange flower","osmanthus","neroli","mandarin orange"],"middle_notes":["musk","amber","jasmine","rose","orange blossom"],"base_notes":["patchouli","vetiver","vanilla","sandalwood","white musk"]},
    {"accords":"floral, citrus, fresh, woody, fresh spicy","top_notes":["yuzu","pomegranate","bergamot","lemon","ice"],"middle_notes":["peony","magnolia","lotus","lily-of-the-valley","jasmine"],"base_notes":["musk","amber","mahogany","sandalwood","cedar"]},
    {"accords":"woody, floral, sweet, powdery, amber","top_notes":["pomegranate","persimmon","green notes","bergamot","pink pepper"],"middle_notes":["black orchid","lotus","champaca","jasmine","rose"],"base_notes":["amber","mahogany","black violet","musk","vanilla"]},
]

fs_text_lines = []
for i, ex in enumerate(FEWSHOT, 1):
    fs_text_lines.append(f"Example {i}:")
    fs_text_lines.append(f"  Accords: {ex['accords']}")
    fs_text_lines.append(f"  Top notes:    {json.dumps(ex['top_notes'])}")
    fs_text_lines.append(f"  Middle notes: {json.dumps(ex['middle_notes'])}")
    fs_text_lines.append(f"  Base notes:   {json.dumps(ex['base_notes'])}")
    fs_text_lines.append("")
fs_text = "\n".join(fs_text_lines)

sample = df.sample(n=50, random_state=42)
results = {"top":[],"mid":[],"base":[],"overall":[]}

for i, (_, row) in enumerate(sample.iterrows()):
    prompt = f"""You are a professional perfumer. Given the main accords, predict Top/Middle/Base notes.
Here are 5 examples of correct accord-to-notes mappings:

{fs_text}
Main Accords: {', '.join(row['accords'])}
Reply ONLY with JSON: {{"top_notes": [...], "middle_notes": [...], "base_notes": [...]}}
Include exactly 5 notes per layer. Use lowercase."""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
        temperature=0.3, max_tokens=500
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    pred = json.loads(raw)
    p_t,r_t,f_t = evaluate_single(
        set(n.strip().lower() for n in pred.get("top_notes",[])), row["Top_clean"])
    p_m,r_m,f_m = evaluate_single(
        set(n.strip().lower() for n in pred.get("middle_notes",[])), row["Middle_clean"])
    p_b,r_b,f_b = evaluate_single(
        set(n.strip().lower() for n in pred.get("base_notes",[])), row["Base_clean"])
    results["top"].append({"precision":p_t,"recall":r_t,"f1":f_t})
    results["mid"].append({"precision":p_m,"recall":r_m,"f1":f_m})
    results["base"].append({"precision":p_b,"recall":r_b,"f1":f_b})
    p_all = set(n.strip().lower() for n in
        pred.get("top_notes",[]) + pred.get("middle_notes",[]) + pred.get("base_notes",[]))
    t_all = row["Top_clean"] | row["Middle_clean"] | row["Base_clean"]
    p_a,r_a,f_a = evaluate_single(p_all, t_all)
    results["overall"].append({"precision":p_a,"recall":r_a,"f1":f_a})
    if (i+1)%10 == 0:
        print(f"  [{i+1}/50] running F1: {np.mean([m['f1'] for m in results['overall']]):.3f}")

f1 = np.mean([m["f1"] for m in results["overall"]])
print(f"\n  Few-shot ONLY (no RAG):    F1 = {f1:.3f}")
print(f"  Zero-shot (no RAG no FS):   F1 = 0.335")
print(f"  RAG only (no FS):           F1 = 0.402")
print(f"  RAG + Few-shot:             F1 = 0.403")
print(f"\n  Few-shot alone delta vs zero-shot: {(f1-0.335)/0.335*100:+.1f}%")
