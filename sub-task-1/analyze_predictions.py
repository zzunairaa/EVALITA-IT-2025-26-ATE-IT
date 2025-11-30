#!/usr/bin/env python3
"""
Analysis script for test predictions to identify issues and suggest improvements.
"""

import pandas as pd
import re
from collections import Counter

# Load predictions
df = pd.read_csv('test_predictions.csv')

print("="*80)
print("TEST PREDICTIONS ANALYSIS")
print("="*80)

# Filter out empty terms
terms_df = df[df['term'].notna() & (df['term'].str.strip() != '')]
print(f"\nTotal rows: {len(df)}")
print(f"Rows with terms: {len(terms_df)}")
print(f"Unique sentences with terms: {terms_df.groupby(['document_id', 'paragraph_id', 'sentence_id']).ngroups}")
print(f"Total unique terms: {terms_df['term'].nunique()}")

# Issue categories
issues = {
    'format_issues': [],
    'incomplete_terms': [],
    'non_domain_terms': [],
    'stopwords': [],
    'english_words': [],
    'generic_terms': [],
    'fragmented_terms': []
}

# Format issues
format_patterns = [
    (r'\s+/\s+', 'spaces around /'),
    (r'\s+-\s+', 'spaces around -'),
    (r'\s+,', 'space before comma'),
    (r',\s*$', 'trailing comma'),
    (r'\s+\.', 'space before period'),
    (r"d'\s+", "space after d'"),
    (r"dell'\s+", "space after dell'"),
]

# Stopwords and non-domain terms
stopwords = {'del', 'di', 'a', 'e', 'essere', 'conferito', 'portare', 'buttare', 
             'esponi', 'esporre', 'delle', 'degli', 'dello', 'della', 'dei', 'delle',
             'umane', 'generato', 'accatastati', 'rubane', 'prefato', 'all\'', 'all'}

# English words
english_words = {'waste', 'paper', 'plastic', 'iron', 'batterien', 'batteries', 'green'}

# Generic terms (too broad, not domain-specific)
generic_terms = {'sacchi', 'sacchetti', 'contenitori', 'sfuso', 'animali', 'ambientale', 
                 'elettronica', 'portare', 'buttare', 'esponi', 'conferito', 'essere'}

# Incomplete term patterns
incomplete_patterns = [
    r'^di\s+',
    r'^a\s+',
    r'^del\s+',
    r'^delle\s+',
    r'^degli\s+',
    r'^dello\s+',
    r'^della\s+',
    r'^dei\s+',
    r'\s+di$',
    r'\s+a$',
    r'\s+su$',
    r'\s+dei$',
    r'\s+del$',
]

print("\n" + "="*80)
print("ISSUE ANALYSIS")
print("="*80)

# Analyze each term
for idx, row in terms_df.iterrows():
    term = row['term'].strip()
    sentence = row['sentence_text']
    
    # Format issues
    for pattern, desc in format_patterns:
        if re.search(pattern, term):
            issues['format_issues'].append((term, desc, sentence[:100]))
            break
    
    # Stopwords
    if term.lower() in stopwords:
        issues['stopwords'].append((term, sentence[:100]))
    
    # English words
    if term.lower() in english_words:
        issues['english_words'].append((term, sentence[:100]))
    
    # Generic terms
    if term.lower() in generic_terms:
        issues['generic_terms'].append((term, sentence[:100]))
    
    # Incomplete terms
    for pattern in incomplete_patterns:
        if re.search(pattern, term.lower()):
            issues['incomplete_terms'].append((term, sentence[:100]))
            break
    
    # Fragmented terms (very short or single words that are likely fragments)
    if len(term.split()) == 1 and len(term) < 4 and term.lower() not in {'raee', 'tari', 'cam', 'cer'}:
        issues['fragmented_terms'].append((term, sentence[:100]))

# Print findings
print(f"\n1. FORMAT ISSUES ({len(issues['format_issues'])}):")
for term, desc, sent in issues['format_issues'][:20]:
    print(f"   - '{term}' ({desc})")
    print(f"     Context: {sent}...")
if len(issues['format_issues']) > 20:
    print(f"   ... and {len(issues['format_issues']) - 20} more")

print(f"\n2. INCOMPLETE TERMS ({len(issues['incomplete_terms'])}):")
for term, sent in issues['incomplete_terms'][:20]:
    print(f"   - '{term}'")
    print(f"     Context: {sent}...")
if len(issues['incomplete_terms']) > 20:
    print(f"   ... and {len(issues['incomplete_terms']) - 20} more")

print(f"\n3. STOPWORDS/NON-DOMAIN TERMS ({len(issues['stopwords'])}):")
for term, sent in issues['stopwords'][:20]:
    print(f"   - '{term}'")
    print(f"     Context: {sent}...")
if len(issues['stopwords']) > 20:
    print(f"   ... and {len(issues['stopwords']) - 20} more")

print(f"\n4. ENGLISH WORDS ({len(issues['english_words'])}):")
for term, sent in issues['english_words']:
    print(f"   - '{term}'")
    print(f"     Context: {sent}...")

print(f"\n5. GENERIC TERMS ({len(issues['generic_terms'])}):")
for term, sent in issues['generic_terms'][:20]:
    print(f"   - '{term}'")
    print(f"     Context: {sent}...")
if len(issues['generic_terms']) > 20:
    print(f"   ... and {len(issues['generic_terms']) - 20} more")

print(f"\n6. FRAGMENTED TERMS ({len(issues['fragmented_terms'])}):")
for term, sent in issues['fragmented_terms'][:20]:
    print(f"   - '{term}'")
    print(f"     Context: {sent}...")
if len(issues['fragmented_terms']) > 20:
    print(f"   ... and {len(issues['fragmented_terms']) - 20} more")

# Summary statistics
total_issues = sum(len(v) for v in issues.values())
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total issues found: {total_issues}")
print(f"Total terms: {len(terms_df)}")
print(f"Issue rate: {total_issues/len(terms_df)*100:.1f}%")

# Most common problematic terms
all_problematic = []
for category, items in issues.items():
    for item in items:
        all_problematic.append(item[0].lower())

problematic_counts = Counter(all_problematic)
print(f"\nMost common problematic terms:")
for term, count in problematic_counts.most_common(10):
    print(f"  '{term}': {count} occurrences")

