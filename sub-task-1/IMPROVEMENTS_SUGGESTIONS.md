# Test Predictions - Issues and Improvement Suggestions

## Summary
- **Total terms extracted**: 705
- **Total issues found**: 86 (12.2% issue rate)
- **Unique terms**: 381

## Critical Issues Found

### 1. Format Issues (12 occurrences)
**Problem**: Terms have incorrect spacing or punctuation formatting.

**Examples**:
- `carta / cartone` → should be `carta/cartone` or `carta e cartone`
- `gestione dell' ambiente` → should be `gestione dell'ambiente`
- `raccolta , trasporto` → should be `raccolta, trasporto` or `raccolta e trasporto`
- `sfalci d' erba` → should be `sfalci d'erba`
- `sacchi .` → should be `sacchi` (remove period)
- `pseudo - edili` → should be `pseudo-edili`

**Fix**: Improve post-processing to normalize spacing around punctuation and remove trailing punctuation.

### 2. Incomplete Terms (19 occurrences)
**Problem**: Terms are fragments that don't form complete domain concepts.

**Examples**:
- `di raccolta rifiuti` → should be `raccolta rifiuti` or `servizio di raccolta rifiuti`
- `di ritiro su` → incomplete fragment
- `avvio a` → should be `avvio a recupero` or `avvio a riciclo`
- `modalità di` → incomplete
- `servizio a` → incomplete
- `ritiro dei` → incomplete

**Fix**: 
- Filter out terms starting/ending with prepositions unless they form valid multi-word expressions
- Improve BIO label alignment to capture complete terms
- Add validation to ensure terms are complete phrases

### 3. Stopwords/Non-Domain Terms (20 occurrences)
**Problem**: Common words and verbs that are not domain-specific terms.

**Examples**:
- `del`, `di`, `a`, `e`, `delle`, `degli` → prepositions/articles
- `essere`, `conferito`, `portare`, `buttare`, `esponi` → verbs
- `umane`, `generato`, `rubane`, `prefato` → not waste management terms

**Fix**: 
- Add a stopword filter to remove common Italian prepositions, articles, and verbs
- Implement domain-specific term validation
- Only extract nouns, noun phrases, and domain-specific technical terms

### 4. English Words (8 occurrences)
**Problem**: English translations extracted instead of Italian terms.

**Examples**:
- `waste`, `paper`, `plastic`, `iron`, `batteries`, `green` → should be Italian equivalents

**Fix**: 
- Filter out English words (check against Italian vocabulary)
- When multilingual text appears, prefer Italian terms

### 5. Generic Terms (21 occurrences)
**Problem**: Terms too generic to be domain-specific.

**Examples**:
- `sacchi`, `sacchetti`, `contenitori`, `sfuso` → too generic
- `ambientale`, `elettronica`, `animali` → not waste-management specific
- `portare`, `buttare` → verbs, not terms

**Fix**: 
- Implement domain-specific filtering
- Only extract terms that appear in waste management context
- Use a domain dictionary/glossary to validate terms

### 6. Fragmented Terms (6 occurrences)
**Problem**: Very short terms that are likely fragments.

**Examples**:
- `del`, `a`, `e` → single characters/words
- `ccr`, `rup` → acronyms (may be valid but need context)

**Fix**: 
- Set minimum term length (e.g., at least 3 characters for single words)
- Validate acronyms against known domain acronyms

## Recommended Improvements

### Immediate Fixes (High Priority)

1. **Post-processing normalization**:
   - Remove spaces around punctuation (`/`, `-`, `,`)
   - Remove trailing punctuation
   - Normalize contractions (`d'` → no space after)

2. **Stopword filtering**:
   - Remove Italian prepositions: `del`, `di`, `a`, `da`, `in`, `con`, `su`, `per`, `tra`, `fra`
   - Remove articles: `il`, `lo`, `la`, `i`, `gli`, `le`, `un`, `uno`, `una`
   - Remove common verbs: `essere`, `avere`, `fare`, `conferire`, `portare`, `buttare`

3. **Incomplete term filtering**:
   - Remove terms starting with prepositions unless they form valid MWEs
   - Remove terms ending with prepositions
   - Ensure minimum term completeness (at least 2 words for phrases starting with prepositions)

4. **Language filtering**:
   - Detect and filter English words
   - Prefer Italian terms in multilingual contexts

### Model Improvements (Medium Priority)

1. **Better BIO alignment**:
   - Improve token-to-word alignment to capture complete terms
   - Handle subword tokenization better

2. **Domain-specific filtering**:
   - Train or fine-tune a classifier to identify waste management terms
   - Use a domain glossary/dictionary for validation

3. **Context-aware extraction**:
   - Consider sentence context when extracting terms
   - Filter out terms that don't appear in waste management context

### Advanced Improvements (Low Priority)

1. **Term validation**:
   - Check against training data vocabulary
   - Use linguistic patterns (noun phrases, technical terms)

2. **Nested term handling**:
   - Improve detection of independent occurrences
   - Better handling of acronyms and abbreviations

3. **Multi-word expression detection**:
   - Better handling of compound terms
   - Recognize domain-specific patterns

## Expected Impact

After implementing these fixes:
- **Format issues**: Should reduce to ~0% (easy fix)
- **Incomplete terms**: Should reduce by ~80% (moderate fix)
- **Stopwords**: Should reduce by ~90% (easy fix)
- **English words**: Should reduce to ~0% (easy fix)
- **Generic terms**: Should reduce by ~60% (moderate fix)

**Overall expected improvement**: Reduce issue rate from 12.2% to ~2-3%

