#!/usr/bin/env python3
"""
Comprehensive compliance analysis for test predictions.
Checks against ATE-IT submission rules:
1. Terms must be lowercased only (no lemmatisation, stemming, or other transformations)
2. No duplicate terms are allowed within the same sentence
3. Nested terms are not permitted (unless they appear independently)
"""

import pandas as pd
import re
from collections import defaultdict

def analyze_compliance(csv_path):
    """Analyze predictions for compliance with ATE-IT rules."""
    
    print("="*80)
    print("ATE-IT SUBMISSION COMPLIANCE ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}\n")
    
    df = pd.read_csv(csv_path)
    
    # Filter out empty terms for analysis
    terms_df = df[df['term'].notna() & (df['term'].str.strip() != '')].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with terms: {len(terms_df)}")
    print(f"Unique sentences: {df.groupby(['document_id', 'paragraph_id', 'sentence_id']).ngroups}")
    print(f"Unique sentences with terms: {terms_df.groupby(['document_id', 'paragraph_id', 'sentence_id']).ngroups}")
    print(f"Total unique terms: {terms_df['term'].nunique()}")
    
    violations = {
        'not_lowercase': [],
        'duplicates': [],
        'nested_terms': [],
        'incomplete_terms': []
    }
    
    # Rule 1: Check for lowercase violations
    print("\n" + "="*80)
    print("RULE 1: Terms must be lowercased only")
    print("="*80)
    
    for idx, row in terms_df.iterrows():
        term = str(row['term']).strip()
        # Check if term has uppercase letters (excluding special cases like acronyms)
        if term != term.lower():
            violations['not_lowercase'].append({
                'term': term,
                'sentence_id': (row['document_id'], row['paragraph_id'], row['sentence_id']),
                'sentence': row['sentence_text'][:100]
            })
    
    print(f"Found {len(violations['not_lowercase'])} terms with uppercase letters")
    if violations['not_lowercase']:
        print("\nExamples:")
        for v in violations['not_lowercase'][:10]:
            print(f"  - '{v['term']}' in sentence {v['sentence_id']}")
    
    # Rule 2: Check for duplicates within sentences
    print("\n" + "="*80)
    print("RULE 2: No duplicate terms within the same sentence")
    print("="*80)
    
    sentence_groups = terms_df.groupby(['document_id', 'paragraph_id', 'sentence_id'])
    
    for (doc_id, para_id, sent_id), group in sentence_groups:
        terms = group['term'].str.strip().str.lower().tolist()
        term_counts = defaultdict(int)
        for term in terms:
            term_counts[term] += 1
        
        duplicates = {term: count for term, count in term_counts.items() if count > 1}
        if duplicates:
            sentence_text = group.iloc[0]['sentence_text']
            violations['duplicates'].append({
                'sentence_id': (doc_id, para_id, sent_id),
                'duplicates': duplicates,
                'sentence': sentence_text[:150]
            })
    
    print(f"Found {len(violations['duplicates'])} sentences with duplicate terms")
    if violations['duplicates']:
        print("\nExamples:")
        for v in violations['duplicates'][:10]:
            print(f"  Sentence {v['sentence_id']}:")
            for term, count in v['duplicates'].items():
                print(f"    - '{term}' appears {count} times")
            print(f"    Sentence: {v['sentence'][:100]}...")
    
    # Rule 3: Check for nested terms
    print("\n" + "="*80)
    print("RULE 3: No nested terms (unless they appear independently)")
    print("="*80)
    
    for (doc_id, para_id, sent_id), group in sentence_groups:
        terms = [t.strip().lower() for t in group['term'].str.strip().tolist() if t]
        sentence_text = group.iloc[0]['sentence_text'].lower()
        
        if len(terms) <= 1:
            continue
        
        # Sort by length (longest first)
        sorted_terms = sorted(set(terms), key=len, reverse=True)
        
        nested_pairs = []
        for i, longer_term in enumerate(sorted_terms):
            for shorter_term in sorted_terms[i+1:]:
                # Check if shorter term is nested in longer term
                pattern = r'\b' + re.escape(shorter_term) + r'\b'
                if re.search(pattern, longer_term, re.IGNORECASE):
                    # Check if shorter term appears independently
                    shorter_pattern = r'\b' + re.escape(shorter_term) + r'\b'
                    longer_pattern = r'\b' + re.escape(longer_term) + r'\b'
                    
                    shorter_matches = list(re.finditer(shorter_pattern, sentence_text))
                    longer_matches = list(re.finditer(longer_pattern, sentence_text))
                    
                    # Check if shorter term has independent occurrence
                    has_independent = False
                    for short_match in shorter_matches:
                        short_start, short_end = short_match.start(), short_match.end()
                        is_covered = False
                        for long_match in longer_matches:
                            long_start, long_end = long_match.start(), long_match.end()
                            if long_start <= short_start and short_end <= long_end:
                                is_covered = True
                                break
                        if not is_covered:
                            has_independent = True
                            break
                    
                    if not has_independent:
                        nested_pairs.append({
                            'shorter': shorter_term,
                            'longer': longer_term,
                            'sentence_id': (doc_id, para_id, sent_id),
                            'sentence': group.iloc[0]['sentence_text'][:150]
                        })
        
        if nested_pairs:
            violations['nested_terms'].extend(nested_pairs)
    
    print(f"Found {len(violations['nested_terms'])} nested term pairs")
    if violations['nested_terms']:
        print("\nExamples:")
        for v in violations['nested_terms'][:10]:
            print(f"  Sentence {v['sentence_id']}:")
            print(f"    - '{v['shorter']}' is nested in '{v['longer']}'")
            print(f"    Sentence: {v['sentence'][:100]}...")
    
    # Bonus: Check for incomplete terms
    print("\n" + "="*80)
    print("BONUS CHECK: Incomplete terms")
    print("="*80)
    
    incomplete_patterns = [
        (r'^(di|del|della|delle|degli|dello|dei)\s+\w+\s+(su|di|del|della|delle|degli|dello|dei)$', 'preposition sandwich'),
        (r'\s+(delle|degli|dello|della)$', 'ends with article'),
        (r'^(di|del|della|delle|degli|dello|dei)\s*$', 'only preposition'),
    ]
    
    for idx, row in terms_df.iterrows():
        term = str(row['term']).strip().lower()
        for pattern, desc in incomplete_patterns:
            if re.match(pattern, term):
                violations['incomplete_terms'].append({
                    'term': term,
                    'pattern': desc,
                    'sentence_id': (row['document_id'], row['paragraph_id'], row['sentence_id']),
                    'sentence': row['sentence_text'][:100]
                })
                break
    
    print(f"Found {len(violations['incomplete_terms'])} incomplete terms")
    if violations['incomplete_terms']:
        print("\nExamples:")
        for v in violations['incomplete_terms'][:10]:
            print(f"  - '{v['term']}' ({v['pattern']}) in sentence {v['sentence_id']}")
    
    # Summary
    print("\n" + "="*80)
    print("COMPLIANCE SUMMARY")
    print("="*80)
    
    total_violations = (
        len(violations['not_lowercase']) +
        len(violations['duplicates']) +
        len(violations['nested_terms']) +
        len(violations['incomplete_terms'])
    )
    
    print(f"\nTotal violations found: {total_violations}")
    print(f"  - Not lowercase: {len(violations['not_lowercase'])}")
    print(f"  - Duplicates: {len(violations['duplicates'])}")
    print(f"  - Nested terms: {len(violations['nested_terms'])}")
    print(f"  - Incomplete terms: {len(violations['incomplete_terms'])}")
    
    if total_violations == 0:
        print("\n✓ ALL CHECKS PASSED! Predictions are compliant with ATE-IT submission rules.")
    else:
        print("\n✗ VIOLATIONS FOUND. Please review and fix the issues above.")
    
    return violations

if __name__ == '__main__':
    violations = analyze_compliance('test_predictions_improved.csv')

