"""
3-layer fuzzy note matching:
  Layer 1: Hardcoded synonym map (~120 entries)
  Layer 2: LLM-built synonym table (1,650 entries)
  Layer 3: Rule-based normalization (strip modifiers)
  Fallback: Levenshtein edit distance
"""
import json
import re
from pathlib import Path


# --- Layer 1: Rule-based normalization ---

_GEO_MODIFIERS = {
    'italian', 'french', 'bulgarian', 'turkish', 'egyptian', 'indian',
    'madagascar', 'calabrian', 'sicilian', 'californian', 'brazilian',
    'moroccan', 'indonesian', 'haitian', 'spanish', 'russian', 'chinese',
    'japanese', 'australian', 'african', 'persian', 'arabian', 'tahitian',
    'venezuelan', 'paraguayan', 'tunisian', 'himalayan', 'provence',
    'grasse', 'mysore', 'comoros', 'bourbon', 'java',
}

_COLOR_MODIFIERS = {
    'white', 'black', 'pink', 'green', 'red', 'blue', 'yellow', 'purple',
    'golden', 'silver', 'dark', 'light', 'blonde', 'clean', 'dirty',
    'sweet', 'bitter', 'fresh', 'soft', 'wild', 'sour',
}

_SUFFIXES_TO_STRIP = [
    'absolute', 'abs', 'oil', 'essence', 'extract', 'accord',
    'notes', 'co2', 'tincture', 'resin', 'resinoid', 'concrete',
    'butter', 'wax', 'water', 'juice',
]

_PROTECTED_PHRASES = {
    'black currant', 'black tea', 'black pepper', 'black cherry',
    'pink pepper', 'green tea', 'green apple', 'green pepper',
    'red apple', 'red currant', 'red pepper', 'red wine',
    'white tea', 'white wine', 'white chocolate',
    'blue lotus', 'blue cheese',
    'sweet pea', 'sweet orange', 'bitter orange',
}


def rule_normalize(note):
    """Strip geographic, color, and suffix modifiers from a note name."""
    note = note.strip().lower()
    original = note

    words = note.replace('-', ' ').split()
    if len(words) <= 1:
        return note

    if words[0] in _GEO_MODIFIERS:
        words = words[1:]
    if len(words) >= 2 and words[0] in _GEO_MODIFIERS:
        words = words[1:]

    note_joined = ' '.join(words)
    if note_joined not in _PROTECTED_PHRASES:
        if len(words) >= 2 and words[0] in _COLOR_MODIFIERS:
            if len(words[1:]) >= 1:
                words = words[1:]

    if len(words) >= 2 and words[-1] in _SUFFIXES_TO_STRIP:
        words = words[:-1]

    result = ' '.join(words)
    if result != original:
        return result
    return note


# --- Layer 2: Synonym tables ---

_synonym_map = None

_HARDCODED_MAP = {
    'cedarwood': 'cedar', 'cedar wood': 'cedar', 'virginia cedar': 'cedar',
    'atlas cedar': 'cedar', 'cedar oil': 'cedar', 'cedar leaf': 'cedar',
    'white cedar': 'cedar', 'red cedar': 'cedar', 'cedar tree': 'cedar',
    'sandal': 'sandalwood', 'indian sandalwood': 'sandalwood',
    'australian sandalwood': 'sandalwood', 'santal': 'sandalwood',
    'vetivert': 'vetiver', 'vetiver java': 'vetiver', 'vetiver bourbon': 'vetiver',
    'oak': 'oakmoss', 'oak moss': 'oakmoss', 'oak tree': 'oakmoss',
    'guaiac': 'guaiac wood', 'guaiacwood': 'guaiac wood',
    'white musk': 'musk', 'clean musk': 'musk', 'synthetic musk': 'musk',
    'musc': 'musk', 'musk notes': 'musk', 'transparent musk': 'musk',
    'skin musk': 'musk', 'velvet musk': 'musk', 'silky musk': 'musk',
    'soft musk': 'musk', 'powdery musk': 'musk', 'cotton musk': 'musk',
    'natural musk': 'musk', 'luminous musk': 'musk', 'cristal musk': 'musk',
    'italian bergamot': 'bergamot', 'calabrian bergamot': 'bergamot',
    'bergamotto': 'bergamot', 'bergamot oil': 'bergamot', 'bergamote': 'bergamot',
    'sicilian lemon': 'lemon', 'italian lemon': 'lemon', 'lemon oil': 'lemon',
    'lemon zest': 'lemon', 'lemon leaves': 'lemon',
    'mandarine': 'mandarin orange', 'mandarin': 'mandarin orange',
    'sicilian mandarin': 'mandarin orange', 'green mandarin': 'mandarin orange',
    'graperfruit': 'grapefruit', 'grapefruit peel': 'grapefruit',
    'sicilian orange': 'orange', 'bitter orange': 'orange', 'sweet orange': 'orange',
    'blood orange': 'orange', 'orange oil': 'orange',
    'rose absolute': 'rose', 'rose oil': 'rose', 'bulgarian rose': 'rose',
    'turkish rose': 'rose', 'rose essent': 'rose', 'damask rose': 'rose',
    'moroccan rose': 'rose', 'taif rose': 'rose', 'may rose': 'rose',
    'white rose': 'rose', 'rose centifolia': 'rose', 'tea rose': 'rose',
    'rose de mai': 'rose', 'wild rose': 'rose', 'red rose': 'rose',
    'jasmine absolute': 'jasmine', 'jasmine sambac': 'jasmine',
    'egyptian jasmine': 'jasmine', 'jasmine oil': 'jasmine',
    'indian jasmine': 'jasmine', 'night blooming jasmine': 'jasmine',
    'star jasmine': 'jasmine', 'white jasmine': 'jasmine',
    'orange blossom absolute': 'orange blossom', 'neroli oil': 'neroli',
    'lavender absolute': 'lavender', 'lavender oil': 'lavender',
    'french lavender': 'lavender', 'english lavender': 'lavender',
    'lavandin': 'lavender',
    'vanilla absolute': 'vanilla', 'vanilla bean': 'vanilla',
    'madagascar vanilla': 'vanilla', 'bourbon vanilla': 'vanilla',
    'vanilla orchid': 'vanilla', 'vanille': 'vanilla', 'vanilla extract': 'vanilla',
    'patchouli oil': 'patchouli', 'patchouli leaf': 'patchouli',
    'indonesian patchouli': 'patchouli',
    'frankincense oil': 'frankincense', 'olibanum': 'frankincense',
    'incense': 'frankincense', 'incense notes': 'frankincense',
    'saffron': 'saffron', 'safran': 'saffron',
    'blackcurrant': 'black currant', 'cassis': 'black currant',
    'passionfruit': 'passion fruit',
    'tonka': 'tonka bean', 'tonka absolute': 'tonka bean',
    'ambroxan': 'amber', 'ambre': 'amber', 'ambergris': 'amber',
    'iso e super': 'iso e super',
    'lily': 'lily-of-the-valley', 'muguet': 'lily-of-the-valley',
    'lily of the valley': 'lily-of-the-valley',
    'marine notes': 'sea notes', 'sea notes': 'sea notes',
    'ocean notes': 'sea notes', 'marine': 'sea notes', 'sea water': 'sea notes',
    'salt': 'sea salt', 'salt notes': 'sea salt',
}


def load_synonym_map():
    """Load the LLM-built note synonym mapping. Lazy-loaded once per session."""
    global _synonym_map
    if _synonym_map is None:
        map_path = Path(__file__).parent.parent / "results" / "note_synonyms.json"
        if map_path.exists():
            with open(map_path) as f:
                data = json.load(f)
            _synonym_map = data.get("map", {})
        else:
            _synonym_map = {}
    return _synonym_map


# --- Main canonicalize function ---

def canonicalize_note(note, edit_threshold=2):
    """
    Map a note name to its canonical form using 3-layer fallback:
      Layer 1: Hardcoded synonym table
      Layer 2: LLM-built synonym table
      Layer 3: Rule-based normalization (strip modifiers)
      Fallback: Edit distance against canonical dictionary
    """
    note = note.strip().lower()
    if not note:
        return note

    if note in _HARDCODED_MAP:
        return _HARDCODED_MAP[note]

    synonym_map = load_synonym_map()
    if note in synonym_map:
        return synonym_map[note]

    cleaned = rule_normalize(note)

    if cleaned in _HARDCODED_MAP:
        return _HARDCODED_MAP[cleaned]
    if cleaned in synonym_map:
        return synonym_map[cleaned]

    try:
        from Levenshtein import distance as lev_distance
    except ImportError:
        return cleaned

    candidates = list(_HARDCODED_MAP.keys()) + list(synonym_map.keys())

    best_match = cleaned
    best_dist = 999
    for cand in candidates:
        if abs(len(cleaned) - len(cand)) > edit_threshold + 4:
            continue
        d = lev_distance(cleaned, cand)
        if d < best_dist and d <= edit_threshold:
            best_dist = d
            best_match = _HARDCODED_MAP.get(cand) or synonym_map.get(cand, cand)

    return best_match


def canonicalize_notes(notes_set):
    """Convert a set of notes to their canonical forms."""
    return {canonicalize_note(n) for n in notes_set}
