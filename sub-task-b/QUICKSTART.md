# Quick Start Guide

## 1. Setup (One-time)

### Linux/Mac:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Windows:
```cmd
setup_venv.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Download the Italian spaCy model

## 2. Activate Virtual Environment

### Linux/Mac:
```bash
source venv/bin/activate
```

### Windows:
```cmd
venv\Scripts\activate
```

## 3. Run the Pipeline

1. Open Jupyter:
```bash
jupyter notebook subtask_b_production_pipeline.ipynb
```

2. Run all cells (Cell → Run All)

3. For production, update Section 12:
```python
PRODUCTION_CONFIG = {
    "input_file": "your_terms.csv",  # Your input file
    "method": "hybrid",  # Recommended
    "output_file": "output_b_1.json",
    "use_llm_refinement": False,  # Set True if you want LLM (slower)
    "evaluate": False
}
```

## Methods Quick Reference

- **`method="hybrid"`** (Recommended): Fast + accurate
- **`method="embedding"`**: Fast, good for large datasets
- **`method="lemma"`**: Fastest baseline
- **`method="llm"`**: Most accurate but slow (requires model download)

## Tips

1. **First time using LLM?** The model will download (~2-7GB). Be patient!
2. **No GPU?** Models will automatically use CPU (slower but works)
3. **Memory issues?** Use `method="lemma"` or `method="embedding"` instead
4. **Want faster LLM?** Use smaller model: `config.LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"`

## Troubleshooting

**Import errors?** Make sure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

**spaCy model missing?** Run:
```bash
python -m spacy download it_core_news_sm
```

**LLM too slow?** Disable it: `use_llm=False` in hybrid method

