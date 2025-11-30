#!/usr/bin/env python3
"""
Script to regenerate test predictions using improved notebook functions.
This applies all the fixes: normalization, filtering, and domain validation.
"""

import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"./ate_it_final_model"
TEST_CSV_PATH = r"test.csv"
OUTPUT_PATH = r"test_predictions_improved.csv"

LABEL_LIST = ['O', 'B-TERM', 'I-TERM']
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Domain filters (same as in improved notebook)
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

# Load SpaCy
import spacy
try:
    nlp = spacy.load("it_core_news_sm")
except OSError:
    print("SpaCy Italian model not found. Please install: python -m spacy download it_core_news_sm")
    nlp = None

def clean_text(text: str) -> str:
    """Improved text cleaning."""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).strip().lower()
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'\{([^\}]*)\}', r'\1', text)
    text = re.sub(r'\(([^)]*)\)', r'\1', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'[""]', '"', text)
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

def normalize_term_format(term: str) -> str:
    """Normalize term formatting."""
    if pd.isna(term) or not term.strip():
        return term
    term = term.strip()
    term = re.sub(r'\s+/\s+', '/', term)
    term = re.sub(r'\s+-\s+', '-', term)
    term = re.sub(r'\s+,', ',', term)
    term = re.sub(r',\s*$', '', term)
    term = re.sub(r'\s+\.', '', term)
    term = re.sub(r'\.\s*$', '', term)
    term = re.sub(r"d'\s+", "d'", term)
    term = re.sub(r"dell'\s+", "dell'", term)
    term = re.sub(r"all'\s+", "all'", term)
    return term.strip().lower()

def is_valid_domain_term(term: str, sentence_context: str = "") -> bool:
    """Validate if term is a valid domain-specific term."""
    if pd.isna(term) or not term.strip():
        return False
    term_lower = term.strip().lower()
    if len(term_lower) < 3 and term_lower not in VALID_ACRONYMS:
        return False
    if len(term_lower) == 1:
        return False
    if term_lower in ITALIAN_STOPWORDS:
        return False
    if term_lower in ENGLISH_WORDS:
        return False
    if term_lower in GENERIC_TERMS:
        return False
    if term_lower in DAYS_OF_WEEK:
        return False
    if term_lower in ADMIN_HEADERS and len(sentence_context.split()) < 5:
        return False
    if re.match(r'^(del|di|a|da|in|con|su|per|tra|fra|delle|degli|dello|della|dei)\s*$', term_lower):
        return False
    if re.search(r'\s+(del|di|a|da|in|con|su|per|tra|fra|dei|del|delle|degli|dello|della|su)$', term_lower):
        if len(term_lower.split()) < 3:
            return False
    if len(term_lower.split()) == 1 and len(term_lower) < 4 and term_lower not in VALID_ACRONYMS:
        return False
    return True

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

def reconstruct_terms_with_constraints(tokens: list, labels: list, sentence_text: str = "") -> list:
    """Reconstruct terms with ATE-IT constraints and filtering."""
    all_terms = extract_terms_from_bio(tokens, labels)
    if not all_terms:
        return []
    
    # Normalize and filter
    all_terms = [normalize_term_format(t) for t in all_terms if t and t.strip()]
    all_terms = [t for t in all_terms if t]
    all_terms = [t for t in all_terms if is_valid_domain_term(t, sentence_text)]
    
    # Remove duplicates
    seen = set()
    unique_terms = []
    for term in all_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    all_terms = unique_terms
    
    # Handle nested terms
    if len(all_terms) > 1:
        sentence_text_lower = sentence_text.lower() if sentence_text else ' '.join(tokens).lower()
        sorted_terms = sorted(all_terms, key=len, reverse=True)
        filtered_terms = []
        
        for term in sorted_terms:
            is_nested = False
            nested_in = []
            for accepted in filtered_terms:
                if term in accepted and term != accepted:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, accepted):
                        is_nested = True
                        nested_in.append(accepted)
            
            if is_nested:
                term_pattern = r'\b' + re.escape(term) + r'\b'
                term_matches = list(re.finditer(term_pattern, sentence_text_lower))
                longer_positions = []
                for longer_term in nested_in:
                    longer_pattern = r'\b' + re.escape(longer_term) + r'\b'
                    for match in re.finditer(longer_pattern, sentence_text_lower):
                        longer_positions.append((match.start(), match.end()))
                
                has_independent = False
                for term_match in term_matches:
                    t_start, t_end = term_match.start(), term_match.end()
                    is_covered = any(l_start <= t_start and t_end <= l_end 
                                   for l_start, l_end in longer_positions)
                    if not is_covered:
                        has_independent = True
                        break
                
                if has_independent:
                    filtered_terms.append(term)
            else:
                filtered_terms.append(term)
        
        all_terms = filtered_terms
    
    return all_terms

# ============================================================================
# MAIN PREDICTION LOOP
# ============================================================================
print("="*60)
print("REGENERATING IMPROVED TEST PREDICTIONS")
print("="*60)

# Load model
print("\n📦 Loading model...")
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
print(f"✓ Model loaded on {device}")

# Load test data
print("\n📂 Loading test data...")
test_df = pd.read_csv(TEST_CSV_PATH)
test_df.fillna('', inplace=True)
sentence_groups = test_df.groupby(['document_id', 'paragraph_id', 'sentence_id'])
print(f"✓ Found {len(sentence_groups)} unique sentences")

# Generate predictions
print("\n🔮 Generating improved predictions...")
prediction_rows = []

with torch.no_grad():
    for (doc_id, para_id, sent_id), group in tqdm(sentence_groups, desc="Predicting"):
        sentence_text = group.iloc[0]['sentence_text']
        
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
        
        # Extract terms with improved filtering
        pred_terms = reconstruct_terms_with_constraints(
            tokens_aligned, 
            pred_labels_aligned,
            sentence_text=sentence_text
        )
        
        # Create rows
        if pred_terms:
            for term in pred_terms:
                if term.strip():
                    prediction_rows.append({
                        'document_id': doc_id_str,
                        'paragraph_id': para_id_str,
                        'sentence_id': sent_id_str,
                        'sentence_text': sentence_text_str,
                        'term': term
                    })
        else:
            prediction_rows.append({
                'document_id': doc_id_str,
                'paragraph_id': para_id_str,
                'sentence_id': sent_id_str,
                'sentence_text': sentence_text_str,
                'term': ''
            })

# Create DataFrame and save
print("\n📊 Organizing predictions...")
test_predictions_df = pd.DataFrame(prediction_rows)
test_predictions_df['_term_empty'] = test_predictions_df['term'].str.strip() == ''
test_predictions_df = test_predictions_df.sort_values(
    by=['document_id', 'paragraph_id', 'sentence_id', '_term_empty', 'term'],
    kind='stable',
    ascending=[True, True, True, True, True]
).reset_index(drop=True)
test_predictions_df = test_predictions_df.drop(columns=['_term_empty'])

column_order = ['document_id', 'paragraph_id', 'sentence_id', 'sentence_text', 'term']
test_predictions_df = test_predictions_df[column_order]

print(f"\n💾 Saving improved predictions to: {OUTPUT_PATH}")
test_predictions_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

# Statistics
total_sentences = len(test_predictions_df.groupby(['document_id', 'paragraph_id', 'sentence_id']))
sentences_with_terms = len(test_predictions_df[
    (test_predictions_df['term'].notna()) & 
    (test_predictions_df['term'].str.strip() != '')
].groupby(['document_id', 'paragraph_id', 'sentence_id']))
total_terms = len(test_predictions_df[
    (test_predictions_df['term'].notna()) & 
    (test_predictions_df['term'].str.strip() != '')
])

print("\n" + "="*60)
print("IMPROVED PREDICTIONS SUMMARY")
print("="*60)
print(f"Total sentences: {total_sentences}")
print(f"Sentences with terms: {sentences_with_terms}")
print(f"Total terms extracted: {total_terms}")
print(f"Average terms per sentence: {total_terms / total_sentences:.2f}" if total_sentences > 0 else "N/A")
print("="*60)
print(f"\n✅ Improved predictions saved to: {OUTPUT_PATH}")

