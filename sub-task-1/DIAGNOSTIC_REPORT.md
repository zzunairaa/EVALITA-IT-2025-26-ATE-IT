# ATE System Diagnostic Report

## PART 1: Code Review and Diagnostic

### 1.1 Preprocessing Issues

#### Issue 1.1.1: Incomplete Bracket/Parenthesis Handling
**Location**: `clean_text()` function (Cell 10)
**Problem**: 
- Only removes square brackets `[]` and curly braces `{}`
- Does NOT handle parentheses `()` which are common in Italian administrative text
- Example: "TARI (Tassa Rifiuti)" → should extract "TARI" and "Tassa Rifiuti" separately

**Impact**: May cause terms to be incorrectly tokenized or missed

#### Issue 1.1.2: No Punctuation Normalization
**Location**: `clean_text()` function
**Problem**:
- Does not normalize punctuation spacing (e.g., "carta / cartone" vs "carta/cartone")
- Does not handle special quotation marks (e.g., `""UMIDO""` → should be `umido`)
- Multiple spaces not fully normalized

**Impact**: Causes format inconsistencies in predictions

#### Issue 1.1.3: No Domain-Specific Filtering in Preprocessing
**Location**: Preprocessing pipeline
**Problem**:
- No filtering of non-domain tokens (days of week, administrative headers)
- All tokens passed to model, including metadata

**Impact**: Model learns to extract non-domain terms

### 1.2 BIO Encoding Issues

#### Issue 1.2.1: Token Matching May Fail Due to Normalization
**Location**: `find_term_in_tokens()` function (Cell 12)
**Problem**:
- Gold terms are tokenized separately from sentence
- If gold term has different spacing/normalization than sentence, matching fails
- Example: Gold term "servizio di raccolta" might not match if sentence has "servizio  di  raccolta" (extra spaces)

**Impact**: False negatives - gold terms not properly labeled

#### Issue 1.2.2: No Handling of Punctuation in Terms
**Location**: `create_bio_labels()` function
**Problem**:
- Terms with punctuation (e.g., "carta/cartone") are tokenized but punctuation tokens get 'O' labels
- This breaks multi-word terms that contain punctuation

**Impact**: Incomplete term extraction

#### Issue 1.2.3: Nested Term Handling is Correct but May Miss Independent Occurrences
**Location**: `create_bio_labels()` function
**Problem**:
- Uses longest-first strategy (correct)
- But if a shorter term appears independently elsewhere in sentence, it may still be blocked
- The check `if all(labels[i] == 'O' for i in range(start, end))` is too strict

**Impact**: Some valid independent occurrences of nested terms may be missed

### 1.3 Token Alignment Issues

#### Issue 1.3.1: Wordpiece Alignment Logic Has Edge Cases
**Location**: `prepare_huggingface_dataset()` and prediction code (Cell 13, 37, 39, 43)
**Problem**:
- Uses `word_ids()` to map subword tokens to words
- Logic: `elif word_idx == previous_word_idx: continue` skips subword tokens
- BUT: This means only FIRST subword of each word gets a label
- If a word is split into multiple subwords, only first subword prediction is used

**Impact**: 
- Loss of information from later subwords
- Potential misalignment if word boundaries don't match exactly

#### Issue 1.3.2: Alignment During Prediction May Mismatch Training
**Location**: Prediction code (Cell 37, 39, 43)
**Problem**:
- During training: labels aligned using `prepare_huggingface_dataset()`
- During prediction: labels aligned using different logic in evaluation cells
- The prediction alignment uses `word_ids()` but may have different indexing

**Impact**: Training/inference mismatch causing performance degradation

#### Issue 1.3.3: No Validation of Token Count Alignment
**Location**: Prediction code
**Problem**:
- Uses `min_len = min(len(tokens), len(pred_labels))` to handle mismatches
- But doesn't verify that alignment is correct
- If `pred_labels` is shorter than `tokens`, some tokens are silently ignored

**Impact**: Terms at end of sentences may be missed

### 1.4 Model Training Issues

#### Issue 1.4.1: No Class-Weighted Loss (Despite Mention in Documentation)
**Location**: Training configuration (Cell 22-23)
**Problem**:
- Documentation mentions "Class-Weighted Loss" but code doesn't implement it
- Training uses standard CrossEntropyLoss
- Class imbalance: O:35170, B-TERM:2176, I-TERM:2582 (O dominates)

**Impact**: Model biased toward predicting 'O', reducing recall

#### Issue 1.4.2: No Dropout Configuration
**Location**: Model initialization (Cell 20)
**Problem**:
- Model loaded with default dropout (0.1)
- No explicit dropout configuration
- Training loss: 0.1349 suggests possible overfitting

**Impact**: Overfitting to training data, poor generalization

#### Issue 1.4.3: Training Metrics Use Token-Level F1, Not Term-Level
**Location**: `compute_metrics()` function (Cell 23)
**Problem**:
- Uses `seqeval` which computes token-level F1
- But evaluation uses term-level F1 (Micro-F1, Type-F1)
- Training metric doesn't match evaluation metric

**Impact**: Model optimizes for wrong objective

### 1.5 Post-Processing Issues

#### Issue 1.5.1: No Stopword Filtering
**Location**: `reconstruct_terms_with_constraints()` function (Cell 34)
**Problem**:
- No filtering of stopwords (del, di, a, e, essere, etc.)
- No filtering of verbs (conferito, portare, buttare)
- No filtering of generic terms

**Impact**: False positives in predictions

#### Issue 1.5.2: No Domain-Specific Filtering
**Location**: Post-processing
**Problem**:
- No filtering of:
  - Days of week (Lunedì, Martedì, etc.)
  - Administrative headers (Data:, Argomenti:, etc.)
  - Generic nouns (sacchetti, contenitori)
  - English words

**Impact**: Many false positives

#### Issue 1.5.3: Format Normalization Missing
**Location**: Post-processing
**Problem**:
- No normalization of spacing around punctuation
- No handling of contractions (d' → d')
- No removal of trailing punctuation

**Impact**: Format inconsistencies

#### Issue 1.5.4: Incomplete Term Filtering
**Location**: Post-processing
**Problem**:
- No filtering of incomplete terms (starting/ending with prepositions)
- No minimum length validation
- No validation that terms are complete phrases

**Impact**: Fragmented terms in output

### 1.6 Evaluation Issues

#### Issue 1.6.1: Evaluation Functions Are Correct
**Location**: `compute_micro_f1()` and `compute_type_f1()` (Cell 31)
**Status**: ✅ CORRECT
- Micro-F1 implementation matches ATE-IT specification
- Type-F1 implementation is correct
- Normalization (lowercase) is applied correctly

### 1.7 Hard-Coded Logic Violations

#### Issue 1.7.1: No POS Tagging Constraints
**Problem**: ATE-IT allows "nouns, verbs, or adjectives" but code doesn't enforce this
**Impact**: May extract non-valid terms

#### Issue 1.7.2: No Validation of Domain-Specificity
**Problem**: Code doesn't verify terms are waste-management related
**Impact**: Extracts generic terms

---

## PART 2: Prediction Analysis

### 2.1 False Positives (from test_predictions.csv analysis)

#### Category 1: Stopwords and Prepositions
- `del`, `di`, `a`, `e`, `delle`, `degli` → Prepositions/articles
- `essere`, `conferito`, `portare`, `buttare` → Verbs
- **Reason**: No stopword filtering in post-processing
- **Count**: ~20 occurrences

#### Category 2: Incomplete/Fragmented Terms
- `di raccolta rifiuti` → Missing "servizio" prefix
- `avvio a` → Incomplete phrase
- `modalità di` → Incomplete phrase
- **Reason**: BIO alignment issues, incomplete term reconstruction
- **Count**: ~19 occurrences

#### Category 3: Generic/Non-Domain Terms
- `sacchetti`, `contenitori`, `sfuso` → Too generic
- `ambientale`, `elettronica`, `animali` → Not waste-management specific
- **Reason**: No domain-specific filtering
- **Count**: ~21 occurrences

#### Category 4: Administrative/Metadata Terms
- Days of week: `Lunedì`, `Martedì`, etc.
- Headers: `Data:`, `Argomenti:`, etc.
- **Reason**: No filtering of metadata
- **Count**: ~10+ occurrences

#### Category 5: English Words
- `waste`, `paper`, `plastic`, `iron`, `batteries`, `green`
- **Reason**: Multilingual text, no language filtering
- **Count**: 8 occurrences

#### Category 6: Format Issues
- `carta / cartone` → Spacing around `/`
- `gestione dell' ambiente` → Space in contraction
- `raccolta , trasporto` → Space before comma
- **Reason**: No format normalization
- **Count**: 12 occurrences

### 2.2 False Negatives (Inferred from Dev Performance Gap)

#### Category 1: Multi-word Terms Not Captured
- **Reason**: 
  - Token alignment issues
  - Model may predict B-TERM but miss I-TERM tokens
  - Long terms (>5 words) may be truncated

#### Category 2: Terms with Special Characters
- Terms like "carta/cartone" may be split incorrectly
- **Reason**: Punctuation handling in tokenization

#### Category 3: Domain-Specific Compounds
- Complex technical terms may be missed
- **Reason**: Model may not recognize domain patterns

### 2.3 Morphological/Semantic Analysis

#### False Positives by Type:
1. **Morphological**: Prepositions, articles (del, di, a) - function words
2. **Syntactic**: Verbs in isolation (essere, conferito) - should be part of phrases
3. **Semantic**: Generic nouns (sacchetti) - too broad, not domain-specific
4. **Pragmatic**: Administrative metadata (days, headers) - not content terms

#### False Negatives by Type:
1. **Boundary**: Multi-word terms where boundaries are unclear
2. **Domain**: Specialized terminology not in training data
3. **Context**: Terms that require sentence-level context to identify

---

## PART 3: Root Causes

### 3.1 Training-Dev Gap (0.8945 → 0.6889 Micro-F1)

**Primary Causes**:
1. **Overfitting**: Training loss 0.1349 is very low, suggests overfitting
2. **Class Imbalance**: No weighted loss, model biased toward 'O'
3. **Metric Mismatch**: Training optimizes token-F1, evaluation uses term-F1
4. **Alignment Issues**: Training/inference alignment may differ

### 3.2 False Positive Sources

1. **No Post-Processing Filters**: 86 issues (12.2% of terms)
2. **Model Overconfidence**: Predicts terms for non-domain tokens
3. **No Domain Validation**: Accepts any BIO sequence as valid term

### 3.3 False Negative Sources

1. **Alignment Bugs**: Token-to-word mapping may lose information
2. **Truncation**: Long sentences truncated at 512 tokens
3. **Boundary Detection**: Model struggles with term boundaries
4. **Class Imbalance**: Model biased toward 'O' predictions

---

## PART 4: Improvement Priorities

### High Priority (Immediate Impact)
1. ✅ Add stopword filtering
2. ✅ Add format normalization
3. ✅ Add incomplete term filtering
4. ✅ Fix token alignment consistency
5. ✅ Add domain-specific filtering

### Medium Priority (Significant Impact)
1. Add class-weighted loss
2. Improve punctuation handling
3. Add POS-based filtering
4. Better nested term handling

### Low Priority (Incremental Improvement)
1. Try different models (xlm-roberta, larger BERT)
2. Add CRF layer
3. Data augmentation
4. Adjust hyperparameters

