# CBOW vs Spectral Embeddings

## Goal
Compare standard CBOW (dense vectors) with native spectral embeddings on the same corpus and evaluation benchmarks.

## Dataset
- **Text8** — quick experiments (`--dataset text8`)
- **FineWeb-Edu** — real experiments, same corpus as embeddings-js (`--dataset fineweb`)

## Files
- `dataset.py` — Data loading (Text8 + FineWeb-Edu), vocab, preprocessing
- `cbow.py` — CBOW model with negative sampling
- `evaluate.py` — WordSim-353, SimLex-999, Google Analogy, nearest neighbors
- `train_cbow.py` — Train and evaluate CBOW baseline

## Quick start
```bash
# Quick test with Text8
python train_cbow.py --dataset text8 --epochs 5

# Small test (1M words, 2 epochs)
python train_cbow.py --dataset text8 --max-words 1000000 --epochs 2

# Real experiment with FineWeb-Edu (needs Parquet files)
python train_cbow.py --dataset fineweb --epochs 5
```

## Status
- [x] CBOW baseline
- [ ] Spectral model (two-stage: amplitude then phase)
- [ ] Head-to-head comparison
