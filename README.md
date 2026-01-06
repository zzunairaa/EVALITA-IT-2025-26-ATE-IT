# TermNinjas – ATE-IT @ EVALITA 2026

This repository contains the system developed for the **ATE-IT Shared Task at EVALITA 2026**, addressing both **Automatic Term Extraction** and **Term Variants Clustering** on Italian municipal waste management documents.

The approach follows a **hybrid neural–symbolic pipeline**, combining supervised transformer based sequence labeling with domain aware post processing, followed by a modular clustering strategy for grouping term variants.

## Overview

- Automatic Term Extraction formulated as a BIO sequence labeling task
- Constraint aware post processing to enforce task specific rules
- Hybrid term variants clustering using linguistic normalization and semantic embeddings
- Fully reproducible pipeline based exclusively on open source tools

## System Components

- Transformer based token classification for term extraction
- Symbolic filtering and constraint enforcement
- Lemma based and embedding based clustering for term variants
- Optional modular components for stricter cluster control

## Reproducibility

The entire system relies only on open source models and libraries and can be executed locally without external APIs or proprietary services.

## Tools and Libraries

- HuggingFace Transformers
- spaCy
- Sentence-Transformers
- PyTorch
- scikit-learn

## Repository Structure

- `data/` – dataset splits and preprocessing outputs  
- `ate/` – term extraction models and post processing  
- `clustering/` – term variants clustering pipeline  
- `scripts/` – training, inference, and evaluation utilities  
- `configs/` – configuration files for experiments  

## License

This project is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## Acknowledgments

This work was developed for the **ATE-IT Shared Task at EVALITA 2026**.  
We thank the task organizers for providing the dataset and evaluation framework.

We are also grateful to **Prof. Paolo Torroni (University of Bologna)** for his guidance and valuable discussions in the context of the Natural Language Processing course, which contributed to the development and refinement of this system.

Finally, we acknowledge the open source NLP community for the tools and libraries that made this work possible.
