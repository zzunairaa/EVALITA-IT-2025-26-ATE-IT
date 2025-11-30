# Subtask B - Term Variants Clustering: Production Pipeline

This repository contains a comprehensive production pipeline for clustering term variants in the Italian municipal waste management domain.

## Overview

The pipeline implements multiple clustering approaches:
1. **Lemma-based clustering**: Groups terms by their lemmas (handles inflectional variants)
2. **Embedding-based clustering**: Uses semantic embeddings to find similar terms
3. **LLM-based clustering**: Uses open-source Hugging Face models for intelligent clustering
4. **Hybrid approach**: Combines lemma grouping, embeddings, and optional LLM refinement

## Features

-  **100% Open Source**: No paid API keys required
-  **Multiple Methods**: Choose the best approach for your needs
-  **BCubed Evaluation**: Built-in evaluation metrics
-  **Error Analysis**: Identify and analyze clustering errors
-  **Visualization**: t-SNE cluster visualization
-  **Production Ready**: Handles edge cases and missing data

## Installation

### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

**Windows:**
```cmd
setup_venv.bat
venv\Scripts\activate
```

The setup script will:
- Create a virtual environment
- Install all dependencies from `requirements.txt`
- Download the Italian spaCy model

### Option 2: Manual Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download Italian spaCy model:
```bash
python -m spacy download it_core_news_sm
```

### LLM Setup (Optional)

The pipeline uses **open-source models** from Hugging Face. No paid API keys needed!

- **Default Model**: `microsoft/Phi-3-mini-4k-instruct` (~2GB, fast, good quality)
- **Alternative**: `mistralai/Mistral-7B-Instruct-v0.2` (~14GB, larger, more accurate)
- **First Run**: Downloads the model automatically (~2-14GB depending on model)
- **GPU Support**: Automatically uses GPU if available, otherwise uses CPU
- **Local Only**: All models run locally - no API calls or internet needed after download

To change the model, edit `config.LLM_MODEL` in the notebook.

## Usage

### Quick Start

1. **Activate virtual environment** (if not already active):
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. **Start Jupyter**:
```bash
jupyter notebook subtask_b_production_pipeline.ipynb
```

3. **Run all cells** (Cell → Run All) to load the pipeline

4. **Use examples** in Section 11 to test the pipeline

### Running on Your Data

1. **Prepare your input file** (CSV or JSON) with unique terms from Subtask A:
   - **CSV format**: Must have a `term` column
     ```csv
     term
     centro di raccolta
     isola ecologica
     ```
   - **JSON format**: 
     ```json
     {
       "data": [
         {"term": "centro di raccolta"},
         {"term": "isola ecologica"}
       ]
     }
     ```

2. **Configure the pipeline** in Section 12:
```python
PRODUCTION_CONFIG = {
    "input_file": "your_terms.csv",  # Your input file path
    "method": "hybrid",  # Options: "hybrid", "embedding", "llm", "lemma"
    "output_file": "output_b_1.json",  # Output file path
    "use_llm_refinement": False,  # Set to True if using LLM (slower)
    "evaluate": False  # Set to True if you have gold standard for evaluation
}
```

3. **Run the pipeline**:
```python
final_clustering = run_pipeline(
    input_file=PRODUCTION_CONFIG["input_file"],
    method=PRODUCTION_CONFIG["method"],
    output_file=PRODUCTION_CONFIG["output_file"],
    gold_standard_file=None,  # Path to gold standard if evaluating
    use_llm=PRODUCTION_CONFIG["use_llm_refinement"]
)
```

## Methods Comparison

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| **Hybrid** ⭐ | Fast | High | **Recommended** - Best balance |
| **Embedding** | Fast | Medium-High | Large datasets, quick results |
| **LLM** | Slow | Highest | Final refinement, best quality |
| **Lemma** | Fastest | Medium | Baseline, inflectional variants |

### Method Details

- **Hybrid** (Recommended): 
  - Combines lemma grouping + semantic embeddings
  - Fast and accurate
  - No LLM required (set `use_llm=False`)
  
- **Embedding**: 
  - Uses semantic similarity from multilingual embeddings
  - Fast, good for large datasets
  - Requires `sentence-transformers`
  
- **LLM**: 
  - Uses open-source Hugging Face models (no API key needed!)
  - Most accurate but slower
  - First run downloads model (~2-14GB)
  - Automatically uses GPU if available
  - Best for final refinement
  
- **Lemma**: 
  - Fast baseline using Italian lemmatization
  - Good for inflectional variants (e.g., "isola" → "isole")
  - Requires spaCy Italian model

## Configuration

Key parameters in `Config` class (Section 2):

### Clustering Parameters
- `SIMILARITY_THRESHOLD`: Threshold for embedding similarity (default: 0.75)
  - Higher = fewer clusters (more strict)
  - Lower = more clusters (more lenient)
- `EMBEDDING_MODEL`: Model for embeddings (default: "paraphrase-multilingual-MiniLM-L12-v2")
- `DBSCAN_EPS`: DBSCAN epsilon parameter (default: 0.3)
- `DBSCAN_MIN_SAMPLES`: DBSCAN min_samples parameter (default: 2)

### LLM Parameters
- `LLM_MODEL`: Hugging Face model name (default: "microsoft/Phi-3-mini-4k-instruct")
- `LLM_BATCH_SIZE`: Number of terms per LLM batch (default: 10)
  - Smaller = more accurate but slower
- `LLM_TEMPERATURE`: Generation temperature (default: 0.1)
- `LLM_USE_LOCAL`: Use local models (default: True)

### Output Parameters
- `OUTPUT_FORMAT`: "json" or "csv" (default: "json")
- `OUTPUT_PREFIX`: Prefix for output files (default: "production_run")

## Evaluation

The pipeline includes BCubed F1 score evaluation. To evaluate on dev set:

```python
clustering = run_pipeline(
    input_file="dev_terms.csv",
    method="hybrid",
    gold_standard_file="subtask_b_dev.csv",
    use_llm=False
)
```

The evaluation will print:
- BCubed Precision
- BCubed Recall
- BCubed F1 Score

You can also use the error analysis function (Section 10) to identify worst-performing terms.

## Output Format

The pipeline outputs in the required format:

**JSON:**
```json
{
  "data": [
    {"term": "centro di raccolta", "cluster": 1},
    {"term": "isola ecologica", "cluster": 1}
  ]
}
```

**CSV:**
```csv
term,cluster
centro di raccolta,1
isola ecologica,1
```

## Submission

Name your output files as: `[group_acronym]_[subtask]_[runID].json` or `.csv`

Example: `myteam_b_1.json`

## Project Structure

```
SUBTASK-B/
├── subtask_b_production_pipeline.ipynb  # Main pipeline notebook
├── subtask_b_train.csv                  # Training data
├── subtask_b_dev.csv                    # Development data
├── requirements.txt                     # Python dependencies
├── setup_venv.sh                        # Setup script (Linux/Mac)
├── setup_venv.bat                       # Setup script (Windows)
├── README.md                            # This file
└── QUICKSTART.md                        # Quick start guide
```

## Dependencies

All dependencies are listed in `requirements.txt`:

- **Core**: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
- **NLP**: spacy (with Italian model)
- **Embeddings**: sentence-transformers, torch
- **LLM**: transformers, accelerate, bitsandbytes
- **Jupyter**: jupyter, ipykernel, notebook

## Best Practices

1. **Data Preparation**: Extract unique terms from your Subtask A output before running
2. **Method Selection**: Start with `hybrid` method for best balance
3. **Parameter Tuning**: Use dev set to tune `SIMILARITY_THRESHOLD` before test
4. **LLM Usage**: Only use LLM if you need highest accuracy (it's slower)
5. **Memory Management**: Use smaller models or disable LLM if you have memory constraints

## Troubleshooting

### Installation Issues

1. **spaCy model not found**: 
   ```bash
   python -m spacy download it_core_news_sm
   ```

2. **sentence-transformers error**: 
   ```bash
   pip install torch sentence-transformers
   ```

3. **transformers error**: 
   ```bash
   pip install transformers torch accelerate
   ```

### Runtime Issues

1. **LLM model too large**: 
   - Use smaller model: `config.LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"`
   - Or disable LLM: Use `method="hybrid"` with `use_llm=False`

2. **Memory issues**: 
   - Reduce batch size: `config.LLM_BATCH_SIZE = 5`
   - Use CPU-only mode: Models automatically use CPU if no GPU
   - Use lemma-only or embedding methods instead
   - Close other applications to free memory

3. **Model download slow**: 
   - First run downloads the model (~2-14GB). Subsequent runs are faster
   - Models are cached locally after first download
   - Consider using smaller model if download is too slow

4. **CUDA/GPU errors**: 
   - Models automatically fall back to CPU if GPU unavailable
   - To force CPU: Set `device_map=None` in model loading (edit notebook)

5. **Import errors**: 
   - Make sure virtual environment is activated
   - Reinstall: `pip install -r requirements.txt --upgrade`

## Performance Tips

- **Fastest**: Use `method="lemma"` (no embeddings, no LLM)
- **Balanced**: Use `method="hybrid"` with `use_llm=False` (recommended)
- **Most Accurate**: Use `method="hybrid"` with `use_llm=True` (slower)
- **Large Datasets**: Use `method="embedding"` for better scalability

## License

This pipeline is provided for the ATE-IT Shared Task. Please refer to the task guidelines for usage terms.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the notebook comments and documentation
3. Check the ATE-IT Shared Task documentation

## Acknowledgments

- Uses open-source models from Hugging Face
- spaCy for Italian NLP
- sentence-transformers for multilingual embeddings
