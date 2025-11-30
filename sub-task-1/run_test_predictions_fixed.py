#!/usr/bin/env python3
"""
Script to run test predictions with improved filtering.
This script extracts the necessary code from the notebook and runs predictions.
"""

import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy

# Try to load SpaCy model
try:
    nlp = spacy.load("it_core_news_sm")
except OSError:
    print("Warning: SpaCy Italian model not found. Using simple tokenization.")
    nlp = None

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"./ate_it_final_model"
TEST_CSV_PATH = r"test.csv"
OUTPUT_PATH = r"test_predictions_improved.csv"
OLD_PREDICTIONS_PATH = r"test_predictions.csv"

LABEL_LIST = ['O', 'B-TERM', 'I-TERM']
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ============================================================================
# HELPER FUNCTIONS (from notebook cell 31)
# ============================================================================

ITALIAN_STOPWORDS = {
    'del', 'di', 'a', 'e', 'essere', 'conferito', 'portare', 'buttare', 
    'esponi', 'esporre', 'delle', 'degli', 'dello', 'della', 'dei', 'delle',
    'umane', 'generato', 'accatastati', 'rubane', 'prefato', "all'", 'all',
    'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'il', 'lo', 'la', 'i', 'gli', 'le',
    'un', 'uno', 'una'
}

ENGLISH_WORDS = {'waste', 'paper', 'plastic', 'iron', 'batterien', 'batteries', 'green'}

GENERIC_TERMS = {'sacchi', 'sacchetti', 'contenitori', 'sfuso', 'animali', 
                 'ambientale', 'elettronica', 'portare', 'buttare', 'esponi', 
                 'conferito', 'essere', 'a'}

DAYS_OF_WEEK = {'lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì', 'sabato', 'domenica'}

ADMIN_HEADERS = {'data', 'argomenti', 'tipologia', 'descrizione', 'ultimo aggiornamento',
                 'a cura di', 'premesso', 'visto', 'considerato', 'ritenuto'}

VALID_ACRONYMS = {'raee', 'tari', 'cam', 'cer', 'ccr', 'rup', 'aro', 'tqrif', 
                  'arera', 'isola', 'ecologica'}

def normalize_term_format(term: str) -> str:
    """Normalize term formatting - remove spaces around punctuation, fix contractions."""
    if pd.isna(term) or not term.strip():
        return term
    
    term = term.strip()
    
    # Remove spaces around punctuation
    term = re.sub(r'\s+/\s+', '/', term)
    term = re.sub(r'\s+-\s+', '-', term)
    term = re.sub(r'\s+,', ',', term)
    term = re.sub(r',\s*$', '', term)
    term = re.sub(r'\s+\.', '', term)
    term = re.sub(r'\.\s*$', '', term)
    
    # Fix contractions
    term = re.sub(r"d'\s+", "d'", term)
    term = re.sub(r"dell'\s+", "dell'", term)
    term = re.sub(r"all'\s+", "all'", term)
    
    return term.strip().lower()

def is_valid_domain_term(term: str, sentence_context: str = "") -> bool:
    """
    Validate if term is a valid domain-specific term.
    Filters stopwords, generic terms, days of week, administrative headers, etc.
    """
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
    if term_lower in ITALIAN_STOPWORDS:
        return False
    
    # English word
    if term_lower in ENGLISH_WORDS:
        return False
    
    # Generic term
    if term_lower in GENERIC_TERMS:
        return False
    
    # Day of week
    if term_lower in DAYS_OF_WEEK:
        return False
    
    # Administrative header (check if sentence is just a header)
    if term_lower in ADMIN_HEADERS and len(sentence_context.split()) < 5:
        return False
    
    # Incomplete term (starts with preposition only)
    if re.match(r'^(del|di|a|da|in|con|su|per|tra|fra|delle|degli|dello|della|dei)\s*$', term_lower):
        return False
    
    # Incomplete term (ends with preposition - IMPROVED CHECK)
    if re.search(r'\s+(del|di|a|da|in|con|su|per|tra|fra|dei|del|delle|degli|dello|della)$', term_lower):
        # If it's a 2-word term ending with preposition, it's likely incomplete
        if len(term_lower.split()) <= 2:
            return False
        # For 3+ word terms, check if it looks incomplete (e.g., "di ritiro su")
        if re.search(r'^(di|del|della|delle|degli|dello|dei)\s+\w+\s+(su|di|del|della|delle|degli|dello|dei)$', term_lower):
            return False
    
    # Very short incomplete fragments
    if len(term_lower.split()) == 1 and len(term_lower) < 4 and term_lower not in VALID_ACRONYMS:
        return False
    
    # Additional check: Terms that are clearly incomplete fragments
    if re.match(r'^(di|del|della|delle|degli|dello|dei)\s+\w+\s+(su|di|del|della|delle|degli|dello|dei)$', term_lower):
        return False
    
    # Check for incomplete patterns like "spazzamento e lavaggio delle" (missing continuation)
    if re.search(r'\s+(delle|degli|dello|della)$', term_lower):
        # If it's a short phrase ending with these, it's likely incomplete
        if len(term_lower.split()) <= 3:
            return False
    
    return True

def clean_text(text: str) -> str:
    """Clean and lowercase text."""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).strip().lower()
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'\{([^\}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_with_spacy(text: str) -> list:
    """Tokenize using SpaCy."""
    if not text or text == '':
        return []
    if nlp is None:
        return text.split()
    doc = nlp(text)
    return [token.text for token in doc]

def extract_terms_from_bio(tokens: list, labels: list) -> list:
    """Extract terms from BIO labels."""
    terms = []
    current_term = []
    for token, label in zip(tokens, labels):
        if label == 'B-TERM':
            if current_term:
                terms.append(' '.join(current_term))
            current_term = [token.lower()]
        elif label == 'I-TERM':
            if current_term:
                current_term.append(token.lower())
            else:
                current_term = [token.lower()]
        else:  # 'O'
            if current_term:
                terms.append(' '.join(current_term))
                current_term = []
    if current_term:
        terms.append(' '.join(current_term))
    return terms

def reconstruct_terms_with_constraints(
    tokens: list, 
    labels: list,
    sentence_text: str = "",
    enforce_no_nested: bool = True,
    enforce_no_duplicates: bool = True,
    filter_invalid: bool = True
) -> list:
    """Reconstruct terms from BIO labels with ATE-IT constraints and enhanced filtering."""
    # First, extract all terms
    all_terms = extract_terms_from_bio(tokens, labels)
    
    if not all_terms:
        return []
    
    # Normalize to lowercase and format
    all_terms = [normalize_term_format(t) for t in all_terms if t and t.strip()]
    all_terms = [t for t in all_terms if t]
    
    # Filter invalid terms
    if filter_invalid:
        all_terms = [t for t in all_terms if is_valid_domain_term(t, sentence_text)]
    
    # Remove duplicates
    if enforce_no_duplicates:
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        all_terms = unique_terms
    
    # Enforce no nested terms (unless they appear independently)
    if enforce_no_nested and len(all_terms) > 1:
        sentence_text_lower = sentence_text.lower() if sentence_text else ' '.join(tokens).lower()
        
        # Sort by length (longest first)
        sorted_terms = sorted(all_terms, key=len, reverse=True)
        
        filtered_terms = []
        for term in sorted_terms:
            # Check if this term is nested in any already accepted term
            is_nested_in_accepted = False
            nested_in_terms = []
            
            for accepted_term in filtered_terms:
                if term in accepted_term and term != accepted_term:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, accepted_term):
                        is_nested_in_accepted = True
                        nested_in_terms.append(accepted_term)
            
            # If term is nested, check if it also appears independently
            if is_nested_in_accepted:
                term_pattern = r'\b' + re.escape(term) + r'\b'
                term_matches = list(re.finditer(term_pattern, sentence_text_lower))
                
                longer_term_positions = []
                for longer_term in nested_in_terms:
                    longer_pattern = r'\b' + re.escape(longer_term) + r'\b'
                    for match in re.finditer(longer_pattern, sentence_text_lower):
                        longer_term_positions.append((match.start(), match.end()))
                
                has_independent = False
                for term_match in term_matches:
                    t_start, t_end = term_match.start(), term_match.end()
                    is_covered = any(l_start <= t_start and t_end <= l_end 
                                   for l_start, l_end in longer_term_positions)
                    if not is_covered:
                        has_independent = True
                        break
                
                if has_independent:
                    filtered_terms.append(term)
            else:
                filtered_terms.append(term)
        
        all_terms = filtered_terms
    
    return all_terms

def remove_duplicates_and_nested(terms, sentence_text=""):
    """Remove duplicates and nested terms from a list of terms.
    
    IMPORTANT: Checks if nested terms appear independently in the sentence.
    Also filters invalid terms using is_valid_domain_term.
    """
    if not terms:
        return []
    
    # Remove empty terms and duplicates
    unique_terms = []
    seen = set()
    for term in terms:
        term_clean = term.strip().lower()
        if term_clean and term_clean not in seen:
            unique_terms.append(term_clean)
            seen.add(term_clean)
    
    if len(unique_terms) <= 1:
        return unique_terms
    
    # Filter invalid terms using is_valid_domain_term
    valid_terms = []
    for term in unique_terms:
        if is_valid_domain_term(term, sentence_text):
            valid_terms.append(term)
    unique_terms = valid_terms
    
    if len(unique_terms) <= 1:
        return unique_terms
    
    # Remove nested terms - but check for independent occurrences
    sentence_text_lower = sentence_text.lower() if sentence_text else ''
    
    # Sort by length (longest first) to check if shorter terms are nested in longer ones
    sorted_terms = sorted(unique_terms, key=len, reverse=True)
    final_terms = []
    
    for term in sorted_terms:
        is_nested_in_accepted = False
        nested_in_terms = []
        
        # Check if this term is nested in any already accepted term
        for accepted_term in final_terms:
            # Check if term appears as substring in accepted_term
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, accepted_term, re.IGNORECASE):
                is_nested_in_accepted = True
                nested_in_terms.append(accepted_term)
        
        # If term is nested, check if it also appears independently
        if is_nested_in_accepted and sentence_text_lower:
            # Find all occurrences of the shorter term in the sentence
            term_pattern = r'\b' + re.escape(term) + r'\b'
            term_matches = list(re.finditer(term_pattern, sentence_text_lower))
            
            # Find all occurrences of longer terms that contain it
            longer_term_positions = []
            for longer_term in nested_in_terms:
                longer_pattern = r'\b' + re.escape(longer_term) + r'\b'
                for match in re.finditer(longer_pattern, sentence_text_lower):
                    longer_term_positions.append((match.start(), match.end()))
            
            # Check if term has an independent occurrence (not covered by longer terms)
            has_independent_occurrence = False
            for term_match in term_matches:
                term_start = term_match.start()
                term_end = term_match.end()
                
                # Check if this occurrence is covered by any longer term
                is_covered = False
                for longer_start, longer_end in longer_term_positions:
                    if longer_start <= term_start and term_end <= longer_end:
                        is_covered = True
                        break
                
                # If this occurrence is not covered, it's independent
                if not is_covered:
                    has_independent_occurrence = True
                    break
            
            # Only add if it appears independently
            if has_independent_occurrence:
                final_terms.append(term)
            # Otherwise, skip it (it's nested and doesn't appear independently)
        else:
            # Not nested, add it
            final_terms.append(term)
    
    # Return in original order (but without duplicates and nested terms)
    result = []
    seen_result = set()
    for term in unique_terms:
        if term in final_terms and term not in seen_result:
            result.append(term)
            seen_result.add(term)
    
    return result

# ============================================================================
# MAIN PREDICTION CODE
# ============================================================================

print("="*60)
print("TEST SET PREDICTION AND EXPORT")
print("="*60)

# Load model
print("\nLoading model and tokenizer...")
try:
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load test data
print("\nLoading test set...")
test_df = pd.read_csv(TEST_CSV_PATH)
test_df.fillna('', inplace=True)
print(f" Loaded {len(test_df)} rows from test CSV")

sentence_groups = test_df.groupby(['document_id', 'paragraph_id', 'sentence_id'])
print(f" Found {len(sentence_groups)} unique sentences")

# Run predictions
print("\n Running predictions on test set...")
prediction_rows = []

with torch.no_grad():
    for (doc_id, para_id, sent_id), group in tqdm(sentence_groups, desc="Predicting"):
        sentence_text = group.iloc[0]['sentence_text']
        
        # Clean and tokenize
        cleaned_text = clean_text(sentence_text)
        tokens = tokenize_with_spacy(cleaned_text)
        
        doc_id_str = str(doc_id)
        para_id_str = str(para_id)
        sent_id_str = str(sent_id)
        sentence_text_str = str(sentence_text) if pd.notna(sentence_text) else ''
        
        if len(tokens) == 0:
            prediction_rows.append({
                'document_id': doc_id_str,
                'paragraph_id': para_id_str,
                'sentence_id': sent_id_str,
                'sentence_text': sentence_text_str,
                'term': ''
            })
            continue
        
        # Tokenize with transformer
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Predict
        outputs = model(**encoded)
        logits = outputs.logits
        pred_label_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get word_ids for alignment
        encoded_for_words = tokenizer(
            tokens,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=512
        )
        word_ids = encoded_for_words.word_ids()
        
        # Map predictions back to tokens
        pred_labels = []
        previous_word_idx = None
        for tokenizer_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx == previous_word_idx:
                continue
            else:
                if tokenizer_idx < len(pred_label_ids):
                    pred_labels.append(ID_TO_LABEL[pred_label_ids[tokenizer_idx]])
                else:
                    pred_labels.append('O')
                previous_word_idx = word_idx
        
        # Ensure alignment
        min_len = min(len(tokens), len(pred_labels))
        tokens_aligned = tokens[:min_len]
        pred_labels_aligned = pred_labels[:min_len]
        
        # Extract terms with constraints and filtering
        pred_terms = reconstruct_terms_with_constraints(
            tokens_aligned, 
            pred_labels_aligned,
            sentence_text=sentence_text,
            enforce_no_nested=True,
            enforce_no_duplicates=True,
            filter_invalid=True
        )
        
        # Create rows (one per term, or one empty row if no terms)
        if pred_terms:
            for term in pred_terms:
                normalized_term = term.strip().lower()
                if normalized_term:
                    prediction_rows.append({
                        'document_id': doc_id_str,
                        'paragraph_id': para_id_str,
                        'sentence_id': sent_id_str,
                        'sentence_text': sentence_text_str,
                        'term': normalized_term
                    })
        else:
            prediction_rows.append({
                'document_id': doc_id_str,
                'paragraph_id': para_id_str,
                'sentence_id': sent_id_str,
                'sentence_text': sentence_text_str,
                'term': ''
            })

print(f" Generated {len(prediction_rows)} prediction rows")

# Post-processing
print("\n Organizing predictions...")
test_predictions_df = pd.DataFrame(prediction_rows)

print("\nApplying ATE-IT output format requirements...")

# Step 1: Lowercase all terms
test_predictions_df['term'] = test_predictions_df['term'].apply(
    lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() else ''
)

# Step 2 & 3: Process each sentence to remove duplicates and nested terms
processed_rows = []
for (doc_id, para_id, sent_id), group in test_predictions_df.groupby(['document_id', 'paragraph_id', 'sentence_id']):
    sentence_text = group.iloc[0]['sentence_text']
    terms = group['term'].tolist()
    terms = [t for t in terms if t and str(t).strip()]
    
    # Process terms: remove duplicates, nested, and invalid terms
    processed_terms = remove_duplicates_and_nested(terms, sentence_text=sentence_text)
    
    # Add rows: one per term, or one empty row if no terms
    if processed_terms:
        for term in processed_terms:
            processed_rows.append({
                'document_id': doc_id,
                'paragraph_id': para_id,
                'sentence_id': sent_id,
                'sentence_text': sentence_text,
                'term': term
            })
    else:
        processed_rows.append({
            'document_id': doc_id,
            'paragraph_id': para_id,
            'sentence_id': sent_id,
            'sentence_text': sentence_text,
            'term': ''
        })

# Recreate DataFrame with processed terms
test_predictions_df = pd.DataFrame(processed_rows)

# Sort by document_id, paragraph_id, sentence_id to maintain order
test_predictions_df['_term_empty'] = test_predictions_df['term'].str.strip() == ''
test_predictions_df = test_predictions_df.sort_values(
    by=['document_id', 'paragraph_id', 'sentence_id', '_term_empty', 'term'],
    kind='stable',
    ascending=[True, True, True, True, True]
).reset_index(drop=True)

test_predictions_df = test_predictions_df.drop(columns=['_term_empty'])

print(f"Post-processing complete: {len(test_predictions_df)} rows")

# Save
column_order = ['document_id', 'paragraph_id', 'sentence_id', 'sentence_text', 'term']
test_predictions_df = test_predictions_df[column_order]

# Backup old predictions
import shutil
import datetime
if os.path.exists(OLD_PREDICTIONS_PATH):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = OLD_PREDICTIONS_PATH.replace('.csv', f'_backup_{timestamp}.csv')
    shutil.copy2(OLD_PREDICTIONS_PATH, backup_path)
    print(f"Backed up old predictions to: {backup_path}")

print(f"\nSaving predictions to: {OUTPUT_PATH}")
test_predictions_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"Successfully saved {len(test_predictions_df)} rows to {OUTPUT_PATH}")

# Also save to old path
test_predictions_df.to_csv(OLD_PREDICTIONS_PATH, index=False, encoding='utf-8')
print(f"Also saved to {OLD_PREDICTIONS_PATH} for compatibility")

# Summary
print("\n" + "="*60)
print("TEST SET PREDICTION SUMMARY")
print("="*60)
total_sentences = len(test_predictions_df.groupby(['document_id', 'paragraph_id', 'sentence_id']))
sentences_with_terms = len(test_predictions_df[
    (test_predictions_df['term'].notna()) & 
    (test_predictions_df['term'].str.strip() != '')
].groupby(['document_id', 'paragraph_id', 'sentence_id']))
total_terms = len(test_predictions_df[
    (test_predictions_df['term'].notna()) & 
    (test_predictions_df['term'].str.strip() != '')
])

print(f"Total sentences: {total_sentences}")
print(f"Sentences with terms: {sentences_with_terms}")
print(f"Sentences without terms: {total_sentences - sentences_with_terms}")
print(f"Total terms extracted: {total_terms}")
print(f"Average terms per sentence: {total_terms / total_sentences:.2f}" if total_sentences > 0 else "N/A")
print("="*60)

print("\nTEST SET PREDICTION EXPORT COMPLETE")
print(f"Improved predictions: {OUTPUT_PATH}")
print("Ready for ATE-IT submission")
print("="*60)

