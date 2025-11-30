# Improved Notebook - Code Fixes and Additions

This document contains the improved code blocks to replace/add to the notebook.

## CRITICAL FIXES TO APPLY

### Fix 1: Improved Preprocessing (Replace Cell 10)

```python
def clean_text(text: str) -> str:
    """
    Improved text cleaning: lowercase, remove brackets/parentheses, normalize punctuation.
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).strip()
    
    # Lowercase (no lemmatization/stemming as per requirements)
    text = text.lower()
    
    # Remove square brackets and curly braces but keep content
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'\{([^\}]*)\}', r'\1', text)
    
    # Handle parentheses - remove but keep content (for terms like "TARI (Tassa Rifiuti)")
    text = re.sub(r'\(([^)]*)\)', r'\1', text)
    
    # Normalize special quotation marks
    text = re.sub(r'["""]', '"', text)  # Convert various quote types to standard
    text = re.sub(r'[""]', '"', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_with_spacy(text: str) -> List[str]:
    """
    Tokenize text using SpaCy Italian model.
    Filters out punctuation-only tokens for better term matching.
    """
    if not text or text == '':
        return []
    
    if nlp is None:
        return text.split()
    
    doc = nlp(text)
    # Keep all tokens but note punctuation for later filtering
    tokens = [token.text for token in doc]
    
    return tokens
```

### Fix 2: Enhanced BIO Encoding with Better Term Matching (Replace Cell 12)

```python
def find_term_in_tokens(term: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Find all occurrences of a term in a tokenized sentence.
    Improved matching with normalization.
    """
    if not term or pd.isna(term):
        return []
    
    term = str(term).strip().lower()
    # Normalize the term the same way as sentences
    term = clean_text(term)
    term_tokens = tokenize_with_spacy(term)
    
    if len(term_tokens) == 0:
        return []
    
    matches = []
    for i in range(len(tokens) - len(term_tokens) + 1):
        # Normalize tokens for comparison
        token_slice = [clean_text(t) for t in tokens[i:i+len(term_tokens)]]
        if token_slice == term_tokens:
            matches.append((i, i + len(term_tokens)))
    
    return matches


def create_bio_labels(sentence_text: str, terms: List[str]) -> Tuple[List[str], List[str]]:
    """
    Create BIO labels for a sentence given the gold terms.
    Improved with better normalization matching.
    """
    # Clean and tokenize sentence
    cleaned_text = clean_text(sentence_text)
    tokens = tokenize_with_spacy(cleaned_text)
    
    if len(tokens) == 0:
        return [], []
    
    # Initialize all labels as 'O' (Outside)
    labels = ['O'] * len(tokens)
    
    # Process each term
    valid_terms = [t for t in terms if t and pd.notna(t) and str(t).strip() != '']
    
    # Process terms sorted by length (longest first) to handle nested terms correctly
    sorted_terms = sorted(valid_terms, key=lambda x: len(tokenize_with_spacy(clean_text(str(x)))), reverse=True)
    
    for term in sorted_terms:
        matches = find_term_in_tokens(term, tokens)
        for start, end in matches:
            # Only label if span is not already labeled
            if all(labels[i] == 'O' for i in range(start, end)):
                # Label first token as B-TERM
                labels[start] = 'B-TERM'
                # Label remaining tokens as I-TERM
                for i in range(start + 1, end):
                    labels[i] = 'I-TERM'
    
    return tokens, labels
```

### Fix 3: Enhanced Post-Processing with Filtering (Replace Cell 34)

```python
# Domain-specific stopwords and filters
ITALIAN_STOPWORDS = {
    'del', 'di', 'a', 'e', 'essere', 'conferito', 'portare', 'buttare', 
    'esponi', 'esporre', 'delle', 'degli', 'dello', 'della', 'dei', 'delle',
    'umane', 'generato', 'accatastati', 'rubane', 'prefato', "all'", 'all',
    'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'il', 'lo', 'la', 'i', 'gli', 'le',
    'un', 'uno', 'una', 'degli', 'degli'
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
    
    return term.strip().lower()


def is_valid_domain_term(term: str, sentence_context: str = "") -> bool:
    """
    Validate if term is a valid domain-specific term.
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
    
    # Incomplete term (ends with preposition - allow only if it's a valid MWE with 3+ words)
    if re.search(r'\s+(del|di|a|da|in|con|su|per|tra|fra|dei|del|delle|degli|dello|della|su)$', term_lower):
        if len(term_lower.split()) < 3:
            return False
    
    # Very short incomplete fragments
    if len(term_lower.split()) == 1 and len(term_lower) < 4 and term_lower not in VALID_ACRONYMS:
        return False
    
    return True


def reconstruct_terms_with_constraints(
    tokens: List[str], 
    labels: List[str],
    sentence_text: str = "",
    enforce_no_nested: bool = True,
    enforce_no_duplicates: bool = True,
    filter_invalid: bool = True
) -> List[str]:
    """
    Reconstruct terms from BIO labels with ATE-IT constraints and filtering.
    """
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
        # Reconstruct sentence text from tokens for independent occurrence checking
        sentence_text_lower = sentence_text.lower() if sentence_text else ' '.join(tokens).lower()
        
        # Sort by length (longest first)
        sorted_terms = sorted(all_terms, key=len, reverse=True)
        
        filtered_terms = []
        for term in sorted_terms:
            # Check if this term is nested in any already accepted term
            is_nested_in_accepted = False
            nested_in_terms = []
            
            for accepted_term in filtered_terms:
                # Check if term appears as substring in accepted_term
                if term in accepted_term and term != accepted_term:
                    # Check if it's a word boundary match (more strict)
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, accepted_term):
                        is_nested_in_accepted = True
                        nested_in_terms.append(accepted_term)
            
            # If term is nested, check if it also appears independently
            if is_nested_in_accepted:
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
                    filtered_terms.append(term)
                # Otherwise, skip it (it's nested and doesn't appear independently)
            else:
                # Not nested, add it
                filtered_terms.append(term)
        
        all_terms = filtered_terms
    
    return all_terms
```

### Fix 4: Add Class-Weighted Loss (Add after Cell 20)

```python
from torch.nn import CrossEntropyLoss
import torch.nn as nn

class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted loss to handle class imbalance."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            if torch.cuda.is_available():
                self.class_weights = self.class_weights.cuda()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Calculate class weights from training data
all_train_labels = [label for item in train_data for label in item['labels']]
label_counts = Counter(all_train_labels)
total_labels = sum(label_counts.values())

# Calculate inverse frequency weights
class_weights = [
    1.0 / (label_counts.get('O', 1) / total_labels),      # O class weight
    1.0 / (label_counts.get('B-TERM', 1) / total_labels), # B-TERM class weight
    1.0 / (label_counts.get('I-TERM', 1) / total_labels)   # I-TERM class weight
]

# Normalize weights
max_weight = max(class_weights)
class_weights = [w / max_weight for w in class_weights]

print(f"Class weights: {dict(zip(LABEL_LIST, class_weights))}")
```

### Fix 5: Improved Token Alignment (Update Cell 13 and prediction cells)

```python
def prepare_huggingface_dataset(data: List[Dict]) -> HFDataset:
    """
    Convert prepared data to HuggingFace Dataset format.
    Improved alignment with validation.
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=512
        )
        
        labels = []
        for i, label_seq in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD) get -100
                    aligned_labels.append(-100)
                elif word_idx == previous_word_idx:
                    # Same word as previous token (subword) - only first subword gets label
                    aligned_labels.append(-100)
                else:
                    # New word - get label from original label list
                    if word_idx < len(label_seq):
                        aligned_labels.append(LABEL_TO_ID[label_seq[word_idx]])
                    else:
                        # Word index out of range - should not happen, but handle gracefully
                        aligned_labels.append(LABEL_TO_ID['O'])
                previous_word_idx = word_idx
            
            labels.append(aligned_labels)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    # Convert to HuggingFace Dataset
    dataset_dict = {
        'tokens': [item['tokens'] for item in data],
        'labels': [item['labels'] for item in data]
    }
    
    hf_dataset = HFDataset.from_dict(dataset_dict)
    hf_dataset = hf_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=['tokens']
    )
    
    return hf_dataset
```

### Fix 6: Improved Prediction Alignment (Update Cell 37, 39, 43)

```python
# In prediction cells, replace the alignment logic with:

# Get word_ids for alignment
encoded_for_words = tokenizer(
    tokens,
    is_split_into_words=True,
    padding=False,
    truncation=True,
    max_length=512
)
word_ids = encoded_for_words.word_ids()

# Map predictions back to tokens with validation
pred_labels = []
previous_word_idx = None
word_idx_to_label = {}

# First pass: collect labels for each word
for tokenizer_idx, word_idx in enumerate(word_ids):
    if word_idx is None:
        continue
    elif word_idx == previous_word_idx:
        continue  # Skip subword tokens
    else:
        if tokenizer_idx < len(pred_label_ids):
            label_id = pred_label_ids[tokenizer_idx]
            word_idx_to_label[word_idx] = ID_TO_LABEL[label_id]
        previous_word_idx = word_idx

# Second pass: create label list aligned with tokens
for i, token in enumerate(tokens):
    if i in word_idx_to_label:
        pred_labels.append(word_idx_to_label[i])
    else:
        # Fallback for tokens without labels (shouldn't happen)
        pred_labels.append('O')

# Ensure alignment
if len(tokens) != len(pred_labels):
    min_len = min(len(tokens), len(pred_labels))
    tokens_aligned = tokens[:min_len]
    pred_labels_aligned = pred_labels[:min_len]
else:
    tokens_aligned = tokens
    pred_labels_aligned = pred_labels

# Extract terms with constraints and filtering
pred_terms = reconstruct_terms_with_constraints(
    tokens_aligned, 
    pred_labels_aligned,
    sentence_text=sentence_text,  # Pass original sentence for context
    enforce_no_nested=True,
    enforce_no_duplicates=True,
    filter_invalid=True  # Enable filtering
)
```

### Fix 7: Training Configuration Improvements (Update Cell 22)

```python
# Training arguments with better regularization
training_args = TrainingArguments(
    output_dir="./ate_it_model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Consider reducing to 3-4 if overfitting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,
    seed=42,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    report_to=None,
    # Add dropout regularization
    dropout=0.1,  # Explicit dropout
    # Add gradient clipping
    max_grad_norm=1.0,
)
```

### Fix 8: Update Trainer Initialization (Update Cell 23)

```python
# Use WeightedLossTrainer if class weights are defined
if 'class_weights' in globals():
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=dev_hf_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=dev_hf_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
```

