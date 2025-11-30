#!/usr/bin/env python3
"""
Script to fix common issues in test predictions.
"""

import pandas as pd
import re

# Load predictions
df = pd.read_csv('test_predictions.csv')

# Italian stopwords and non-domain terms
STOPWORDS = {
    'del', 'di', 'a', 'e', 'essere', 'conferito', 'portare', 'buttare', 
    'esponi', 'esporre', 'delle', 'degli', 'dello', 'della', 'dei', 'delle',
    'umane', 'generato', 'accatastati', 'rubane', 'prefato', "all'", 'all',
    'degli', 'degli', 'degli', 'degli', 'degli', 'degli', 'degli', 'degli'
}

# English words to filter
ENGLISH_WORDS = {'waste', 'paper', 'plastic', 'iron', 'batterien', 'batteries', 'green'}

# Generic terms (too broad)
GENERIC_TERMS = {'sacchi', 'sacchetti', 'contenitori', 'sfuso', 'animali', 
                 'ambientale', 'elettronica', 'portare', 'buttare', 'esponi', 
                 'conferito', 'essere', 'a'}

# Valid acronyms in domain
VALID_ACRONYMS = {'raee', 'tari', 'cam', 'cer', 'ccr', 'rup', 'aro', 'tqrif', 
                  'arera', 'isola', 'ecologica'}

def normalize_term(term):
    """Normalize term formatting."""
    if pd.isna(term) or not term.strip():
        return term
    
    term = term.strip()
    
    # Remove spaces around punctuation
    term = re.sub(r'\s+/\s+', '/', term)  # carta / cartone -> carta/cartone
    term = re.sub(r'\s+-\s+', '-', term)  # pseudo - edili -> pseudo-edili
    term = re.sub(r'\s+,', ',', term)     # raccolta , trasporto -> raccolta, trasporto
    term = re.sub(r',\s*$', '', term)      # Remove trailing comma
    term = re.sub(r'\s+\.', '', term)      # Remove space before period
    term = re.sub(r'\.\s*$', '', term)     # Remove trailing period
    
    # Fix contractions
    term = re.sub(r"d'\s+", "d'", term)    # d' erba -> d'erba
    term = re.sub(r"dell'\s+", "dell'", term)  # dell' ambiente -> dell'ambiente
    term = re.sub(r"all'\s+", "all'", term)    # all' utenza -> all'utenza
    
    return term.strip()

def is_valid_term(term):
    """Check if term is valid domain-specific term."""
    if pd.isna(term) or not term.strip():
        return False
    
    term_lower = term.strip().lower()
    
    # Too short (unless it's a valid acronym)
    if len(term_lower) < 3 and term_lower not in VALID_ACRONYMS:
        return False
    
    # Single character
    if len(term_lower) == 1:
        return False
    
    # Stopword
    if term_lower in STOPWORDS:
        return False
    
    # English word
    if term_lower in ENGLISH_WORDS:
        return False
    
    # Generic term
    if term_lower in GENERIC_TERMS:
        return False
    
    # Incomplete term (starts with preposition)
    if re.match(r'^(del|di|a|da|in|con|su|per|tra|fra|delle|degli|dello|della|dei)\s+$', term_lower):
        return False
    
    # Incomplete term (ends with preposition)
    if re.search(r'\s+(del|di|a|da|in|con|su|per|tra|fra|dei|del|delle|degli|dello|della|su)$', term_lower):
        # Allow if it's a valid multi-word expression
        if len(term_lower.split()) < 3:
            return False
    
    # Very short incomplete fragments
    if len(term_lower.split()) == 1 and len(term_lower) < 4 and term_lower not in VALID_ACRONYMS:
        return False
    
    return True

def fix_predictions(df):
    """Fix predictions by normalizing and filtering."""
    fixed_df = df.copy()
    
    # Normalize terms
    fixed_df['term'] = fixed_df['term'].apply(normalize_term)
    
    # Filter invalid terms
    mask = fixed_df['term'].apply(is_valid_term)
    fixed_df = fixed_df[mask | fixed_df['term'].isna() | (fixed_df['term'].str.strip() == '')]
    
    return fixed_df

# Fix predictions
print("Fixing predictions...")
fixed_df = fix_predictions(df)

# Statistics
original_terms = len(df[df['term'].notna() & (df['term'].str.strip() != '')])
fixed_terms = len(fixed_df[fixed_df['term'].notna() & (fixed_df['term'].str.strip() != '')])

print(f"Original terms: {original_terms}")
print(f"Fixed terms: {fixed_terms}")
print(f"Removed: {original_terms - fixed_terms} terms ({((original_terms - fixed_terms)/original_terms*100):.1f}%)")

# Save fixed predictions
output_file = 'test_predictions_fixed.csv'
fixed_df.to_csv(output_file, index=False)
print(f"\nFixed predictions saved to: {output_file}")

# Show some examples of fixes
print("\nExample fixes:")
print("="*80)
examples = [
    ('carta / cartone', 'carta/cartone'),
    ('gestione dell\' ambiente', 'gestione dell\'ambiente'),
    ('raccolta , trasporto', 'raccolta, trasporto'),
    ('di raccolta rifiuti', '[REMOVED - incomplete]'),
    ('del', '[REMOVED - stopword]'),
    ('waste', '[REMOVED - English]'),
    ('sacchetti', '[REMOVED - generic]'),
]

for original, fixed in examples:
    print(f"  '{original}' → '{fixed}'")

