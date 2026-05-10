"""
Data loading and preprocessing for Fragrantica dataset.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path


def parse_european_float(val):
    """Convert European decimal string like '1,42' -> float 1.42."""
    if pd.isna(val) or str(val).strip() == '':
        return 0.0
    s = str(val).strip()
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    elif ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_data():
    """Load the Fragrantica cleaned dataset, return DataFrame with parsed accords and ratings."""
    path = Path.home() / '.cache/kagglehub/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset/versions/3/fra_cleaned.csv'
    df = pd.read_csv(path, encoding='latin-1', sep=';')

    accord_cols = ['mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']
    df['accords'] = df[accord_cols].apply(
        lambda row: [a.strip().lower() for a in row.dropna().tolist() if a.strip()], axis=1
    )
    df['Rating Value'] = df['Rating Value'].apply(parse_european_float)
    df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0).astype(int)
    return df


def normalize_notes(note_str):
    """Parse a note string (comma-separated) -> set of cleaned note names."""
    if pd.isna(note_str) or not note_str.strip():
        return set()
    notes = re.split(r'[,;、/]', str(note_str))
    cleaned = set()
    for n in notes:
        n = n.strip().lower()
        n = re.sub(r'\b(notes?|essence|oil|absolute|extract|accord)\b', '', n).strip()
        if n and len(n) > 1:
            cleaned.add(n)
    return cleaned


def preprocess(df):
    """Parse Top/Middle/Base notes into cleaned sets. Returns a copy."""
    df = df.copy()
    for col in ['Top', 'Middle', 'Base']:
        df[f'{col}_clean'] = df[col].astype(str).apply(normalize_notes)
    df['all_notes'] = df.apply(
        lambda row: row['Top_clean'] | row['Middle_clean'] | row['Base_clean'], axis=1
    )
    return df
