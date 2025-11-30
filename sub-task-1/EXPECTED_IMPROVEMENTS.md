# Expected Micro-F1 and Type-F1 Improvements

## Summary of Applied Fixes

### ✅ Fixes Applied to Notebook

1. **Enhanced Preprocessing** (Cell 10)
   - Added parentheses removal
   - Normalized special quotation marks
   - Better whitespace handling

2. **Improved Term Matching** (Cell 12)
   - Normalized term and sentence tokens before matching
   - Handles spacing variations better

3. **Comprehensive Post-Processing Filtering** (Cell 34)
   - Added stopword filtering (Italian prepositions, articles, verbs)
   - Added domain-specific term validation
   - Added format normalization (spacing, contractions)
   - Added filtering for:
     - Days of week
     - Administrative headers
     - English words
     - Generic terms
     - Incomplete/fragmented terms

4. **Updated Prediction Cells** (Cells 37, 39, 43)
   - All prediction cells now use enhanced filtering
   - Pass sentence context for better filtering

5. **Training Improvements** (Cells 22, 23)
   - Added gradient clipping
   - Added comments for class-weighted loss (optional)

---

## Expected Performance Improvements

### Current Performance
- **Training Set**: Micro-F1: 0.8945, Type-F1: 0.8319
- **Dev Set**: Micro-F1: 0.6889, Type-F1: 0.6123
- **Baseline**: Micro-F1: 0.513, Type-F1: 0.470

### Expected Improvements After Fixes

#### 1. False Positive Reduction

**Current Issue Rate**: 12.2% of terms (86 issues out of 705 terms)

**Expected Reduction**:
- **Format issues** (12): → **0%** (100% reduction) ✅
  - Normalization fixes spacing around punctuation
- **Stopwords** (20): → **~5%** (75% reduction) ✅
  - Filtering removes most stopwords
- **English words** (8): → **0%** (100% reduction) ✅
  - Language filter removes all English words
- **Generic terms** (21): → **~8%** (62% reduction) ✅
  - Domain filter removes most generic terms
- **Incomplete terms** (19): → **~4%** (79% reduction) ✅
  - Validation removes most incomplete fragments
- **Fragmented terms** (6): → **~1%** (83% reduction) ✅
  - Length and validation filters remove fragments

**Total Expected FP Reduction**: ~60-70 terms removed (8.5-10% of current terms)

#### 2. Precision Improvement

**Current Dev Precision**:
- Micro-Precision: 0.6695
- Type-Precision: 0.5900

**Expected Dev Precision** (after filtering):
- **Micro-Precision**: **0.75-0.80** (+12-19% improvement)
  - Removal of ~60-70 false positives improves precision significantly
- **Type-Precision**: **0.68-0.72** (+15-20% improvement)
  - Filtering removes invalid term types

#### 3. Recall Impact

**Current Dev Recall**:
- Micro-Recall: 0.7095
- Type-Recall: 0.6364

**Expected Dev Recall**:
- **Micro-Recall**: **0.70-0.72** (slight decrease or stable)
  - Filtering may remove some valid terms, but improved matching should compensate
- **Type-Recall**: **0.63-0.65** (stable or slight increase)
  - Better normalization may capture more valid terms

#### 4. F1 Score Improvements

**Expected Dev Performance**:
- **Micro-F1**: **0.72-0.76** (+4.5-10% improvement)
  - Precision improvement drives F1 increase
  - Formula: F1 = 2 * (P * R) / (P + R)
  - With P: 0.75-0.80, R: 0.70-0.72 → F1: 0.72-0.76

- **Type-F1**: **0.65-0.68** (+6-11% improvement)
  - Precision improvement drives F1 increase
  - With P: 0.68-0.72, R: 0.63-0.65 → F1: 0.65-0.68

#### 5. Training-Dev Gap Reduction

**Current Gap**: 0.8945 → 0.6889 (20.5 point drop)

**Expected Gap** (after fixes):
- Training: 0.89-0.90 (slight improvement from better filtering)
- Dev: 0.72-0.76 (significant improvement)
- **Gap**: **13-18 points** (reduced from 20.5 points)

**Reason**: Better generalization through:
- Improved normalization (handles variations)
- Better filtering (reduces overfitting to training noise)

---

## Detailed Breakdown by Fix Category

### A) Preprocessing Improvements
- **Impact**: Low-Medium
- **Benefit**: Better term matching, fewer false negatives
- **Expected Gain**: +1-2% F1

### B) Post-Processing Filtering
- **Impact**: **HIGH** (Primary improvement source)
- **Benefit**: Removes 60-70 false positives
- **Expected Gain**: +4-6% Micro-F1, +5-7% Type-F1

### C) Format Normalization
- **Impact**: Medium
- **Benefit**: Consistent output format, better matching
- **Expected Gain**: +1-2% F1

### D) Training Improvements (Gradient Clipping)
- **Impact**: Low
- **Benefit**: More stable training
- **Expected Gain**: +0.5-1% F1

---

## Conservative vs Optimistic Estimates

### Conservative Estimate (Lower Bound)
- **Micro-F1**: 0.72 (+4.5%)
- **Type-F1**: 0.65 (+6%)
- Assumes some valid terms are filtered out

### Realistic Estimate (Most Likely)
- **Micro-F1**: 0.74 (+7.4%)
- **Type-F1**: 0.67 (+9.4%)
- Balanced improvement from filtering and normalization

### Optimistic Estimate (Upper Bound)
- **Micro-F1**: 0.76 (+10.3%)
- **Type-F1**: 0.68 (+11.1%)
- Assumes minimal valid term loss from filtering

---

## Comparison to Baseline

### Current vs Baseline
- Micro-F1: 0.6889 vs 0.513 = **+34% improvement** ✅
- Type-F1: 0.6123 vs 0.470 = **+30% improvement** ✅

### Expected vs Baseline (After Fixes)
- Micro-F1: 0.72-0.76 vs 0.513 = **+40-48% improvement** ✅✅
- Type-F1: 0.65-0.68 vs 0.470 = **+38-45% improvement** ✅✅

---

## Key Improvements Summary

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| **Dev Micro-F1** | 0.6889 | 0.72-0.76 | +4.5-10% |
| **Dev Type-F1** | 0.6123 | 0.65-0.68 | +6-11% |
| **Dev Micro-Precision** | 0.6695 | 0.75-0.80 | +12-19% |
| **Dev Type-Precision** | 0.5900 | 0.68-0.72 | +15-20% |
| **False Positives** | 86 (12.2%) | ~15-25 (2-3%) | -70-85% reduction |

---

## Next Steps for Further Improvement

### If Performance Still Below Target:

1. **Enable Class-Weighted Loss** (uncomment in Cell 23)
   - Expected gain: +2-3% F1
   - Helps with class imbalance

2. **Reduce Training Epochs** (if overfitting)
   - Change from 5 to 3-4 epochs
   - Expected gain: +1-2% F1 on dev set

3. **Try Different Models**
   - `xlm-roberta-base` (multilingual)
   - `bert-base-italian-xxl-uncased` (larger Italian BERT)
   - Expected gain: +2-5% F1

4. **Add CRF Layer**
   - Better BIO sequence modeling
   - Expected gain: +1-3% F1

5. **Data Augmentation**
   - Synonym replacement
   - Expected gain: +1-2% F1

---

## Validation

To validate improvements:
1. Run evaluation on dev set with new code
2. Compare metrics before/after
3. Check false positive reduction
4. Verify format normalization works

Expected validation results:
- ✅ Format issues: 0 (down from 12)
- ✅ Stopwords: <5 (down from 20)
- ✅ English words: 0 (down from 8)
- ✅ Generic terms: <8 (down from 21)
- ✅ Incomplete terms: <4 (down from 19)

