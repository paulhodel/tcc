# STEPS — Spectral Word Representations Research Plan

---

## 1. Word Structure

```
word = {
    amplitude: [A₁, A₂, ..., A₅₁₂],   # 512 grid, sparse via L1
    phase:     [φ₁, φ₂, ..., φ₅₁₂]     # meaningful only where A > 0
}
```

- Fixed grid of 512 candidate frequencies
- L1 sparsity → ~50-100 active per word
- Storage: only active (frequency_index, amplitude, phase) triples

---

## 2. Operations the Model Must Support

These are the benchmark targets — each must be compared against dense vector baselines (Word2Vec, GloVe).

### 2.1 Similarity
```
sim(w1, w2) = Σᵢ A₁ᵢ · A₂ᵢ · cos(φ₁ᵢ - φ₂ᵢ)
```
**Benchmark:** WordSim-353, SimLex-999, MEN-3000 (Spearman correlation)

### 2.2 Analogy (Arithmetic)
```
a:b :: c:? → find word closest to operation(a, b, c)
```
Test multiple operations:
- Op A: `E(c) · E(b) / E(a)` (complex division + multiplication)
- Op B: `A(c) - A(a) + A(b)` for amplitude, `φ(c) - φ(a) + φ(b)` for phase
- Op C: Superposition-based: `E(b) - E(a) + E(c)`

**Benchmark:** Google Analogy (19K), BATS (99K)

### 2.3 Clustering / Categorization
```
Words in the same category should share active frequencies
```
**Benchmark:** AP clustering, categorization purity (BLESS, BM datasets)

### 2.4 Sentiment Polarity
```
"good" and "bad" should differ in a systematic, identifiable way
(e.g., phase opposition on a VALENCE frequency)
```
**Benchmark:** SST-2 sentiment classification (use spectral embeddings as frozen features → simple classifier)

### 2.5 Composition (Phrases)
```
compose(w1, w2) = E(w1) + E(w2)   (superposition)
"ice cream" = E(ice) + E(cream) → should be similar to frozen dessert words
```
**Benchmark:** BiRD phrase similarity, qualitative analysis

### 2.6 Nearest Neighbors
```
Given a word, find the K most similar words by spectral similarity
```
**Benchmark:** Qualitative (do nearest neighbors make sense?) + sparse retrieval speed vs dense

---

## 3. Training

### Phase 1: Skip-Gram Spectral
- Objective: predict context from center word using spectral similarity
- Loss: negative sampling with spectral sim
- Regularization: L1 on amplitudes
- Corpus: Text8 (small) → Wikipedia (full)

### Phase 2: CBOW Spectral
- Objective: predict center from context superposition
- Forces composition (superposition) to work during training
- Compare spectral structure vs Phase 1

### Phase 3 (if needed): GloVe Spectral
- Factorize co-occurrence matrix into spectral parameters
- Uses global statistics

---

## 4. Evaluation Steps (In Order)

```
Step 1: SIMILARITY    → Does the representation encode meaning at all?
Step 2: ANALOGY       → Can we do math on it?
Step 3: CLUSTERING    → Does structure emerge (shared frequencies = categories)?
Step 4: SENTIMENT     → Can downstream tasks use it?
Step 5: COMPOSITION   → Does superposition produce meaningful combinations?
Step 6: EFFICIENCY    → Memory, FLOPs, speed vs dense baselines
Step 7: ANALYSIS      → What survived L1? Interpretable? How many active?
```

---

## 5. Baselines for Comparison

| Baseline | Purpose |
|----------|---------|
| Word2Vec skip-gram (d=300) | Direct comparison: same objective, different representation |
| GloVe (d=300) | Comparison with global statistics baseline |
| FFT of Word2Vec | Transform approach: what spectral structure exists in dense vectors? |
| Random spectral | Sanity check: untrained spectral representations should score ~0 |

---

## 6. Experiment Sequence

| Day | What | Success = | Fail = |
|-----|------|-----------|--------|
| 1 | Train Word2Vec baseline + FFT transform + test similarity & analogy in spectral domain | Similarity survives FFT. At least one analogy op works. | Spectral arithmetic doesn't inherit from vectors. Proceed to native training anyway. |
| 2 | Build native spectral skip-gram. Train on small corpus. Test similarity. | Training converges. Similarity > 0. | Debug training. Try different learning rates, initialization. |
| 3 | Full training on Wikipedia. Test all operations (2.1–2.6). | Competitive with Word2Vec on at least similarity + analogy. | Analyze gap. Try CBOW training (Phase 2). |
| 4 | CBOW spectral training. Compare composition quality with skip-gram. | CBOW spectra compose better. | Skip-gram is sufficient. Move to analysis. |
| 5 | Sparsity analysis. Vary L1. Plot N_effective vs quality. Find the elbow. | Clear elbow. N_effective discovered. | Model doesn't sparsify cleanly. Adjust regularization. |
| 6 | Emergent structure analysis. What do surviving frequencies mean? | Some frequencies are interpretable. Phase carries information. | Abstract patterns only. Still a finding. |
| 7 | Full benchmark table. Dense vs Transform vs Native. Write up results. | Clear comparison. At least one advantage for spectral. | Document negative results and insights. |

---

## 7. Deliverables

- [ ] Trained Word2Vec baseline with benchmark scores
- [ ] FFT transform analysis with benchmark scores
- [ ] Trained native spectral model with benchmark scores
- [ ] Comparison table: dense vs transform vs native (all operations)
- [ ] Sparsity convergence curve (N_effective vs quality)
- [ ] Frequency catalog (what each surviving frequency captures)
- [ ] Experiment report with daily entries
