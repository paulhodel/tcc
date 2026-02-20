# Experiment 0: Critical Path Validation

**Date started:** 2026-02-20
**Status:** Planning
**Goal:** Validate whether spectral representations can encode meaning AND support meaningful arithmetic operations.

---

## Motivation

We are investigating frequency-domain word representations as an efficient alternative to dense vector embeddings. Before building complex systems, we need to validate three hard requirements:

1. **ENCODE:** Can spectral representations capture semantic similarity?
2. **MATH:** Can we perform operations on spectra that produce meaningful semantic results (analogous to vector arithmetic)?
3. **EFFICIENT:** Are those operations cheaper than dense vector operations?

If any of these fail, the approach needs to pivot. This experiment series validates them in order.

---

## Day 1 — Baseline & Transform Test

### Objective
Train a dense baseline (Word2Vec), apply FFT transformation, and test whether semantic properties survive in the frequency domain.

### Plan
1. Train Word2Vec skip-gram on a corpus (English Wikipedia subset or Text8)
   - d=300 dimensions
   - Standard hyperparameters (window=5, min_count=5, negative=5)
2. Evaluate baseline on:
   - WordSim-353 (similarity correlation)
   - Google Analogy Dataset (analogy accuracy)
3. Apply FFT to the dense embedding matrix:
   - Each word's 300-dim vector → FFT → 300-dim complex spectrum
4. Evaluate the spectral (transformed) embeddings:
   - Similarity: spectral coherence vs cosine similarity → do rankings agree?
   - Analogies: try multiple spectral arithmetic operations (see below)

### Spectral Arithmetic Candidates
For analogy `a:b :: c:?` (e.g., king:queen :: man:?):

**Op A — Complex division + multiplication:**
```
result = E(c) * (E(b) / E(a))
```
Phase differences capture the relationship, multiplication applies it.

**Op B — Separate amplitude/phase arithmetic:**
```
result_amplitude = A(c) - A(a) + A(b)
result_phase = φ(c) - φ(a) + φ(b)
```
Direct analog of vector subtraction+addition, but on amplitude and phase separately.

**Op C — Inverse transform, vector arithmetic, forward transform:**
```
result = FFT( IFFT(E(c)) - IFFT(E(a)) + IFFT(E(b)) )
```
Sanity check: does round-tripping through FFT/IFFT preserve analogy quality?

**Op D — Magnitude-weighted phase manipulation:**
```
relationship = E(b) / E(a)  (captures the transformation)
result = E(c) * relationship  (applies only where amplitudes are significant)
```

### Success Criteria
- [ ] Similarity correlation within 10% of dense baseline
- [ ] At least ONE spectral arithmetic operation produces correct analogies (top-10 accuracy > 0%)
- [ ] Round-trip FFT/IFFT preserves analogy accuracy (sanity check for Op C)

### Results
*(To be filled after running)*

### Observations
*(To be filled after running)*

---

## Day 2 — Native Spectral Model (First Attempt)

### Objective
Build the learnable spectral model and train it from scratch — no dense vectors involved.

### Plan
1. Implement the spectral word representation:
   - K=256 candidate frequencies (global, learnable)
   - Per-word: amplitude (K floats) + phase (K floats)
   - Complex embedding: E(w) = A * e^(iφ)
2. Implement spectral skip-gram:
   - Similarity function: real part of complex dot product
   - Loss: negative sampling with spectral similarity
   - Regularization: L1 on amplitudes (sparsity)
3. Train on same corpus as Day 1
4. Evaluate:
   - Similarity benchmarks (WordSim-353, SimLex-999)
   - Analogy accuracy using the BEST operation from Day 1
   - Count N_effective: how many frequencies have meaningful amplitude?

### Success Criteria
- [ ] Training converges (loss decreases)
- [ ] Similarity correlation > 0 (captures SOME semantic structure)
- [ ] N_effective < K (sparsity is working — model self-selects)

### Results
*(To be filled after running)*

### Observations
*(To be filled after running)*

---

## Day 3 — Comparison & Gap Analysis

### Objective
Direct comparison: Dense (A) vs Transform (B) vs Native (C). Identify the gap.

### Plan
1. Run all three approaches on identical evaluation suite
2. Compare:
   - Quality: similarity correlation, analogy accuracy
   - Efficiency: parameters per word, FLOPs per operation
   - Sparsity: N_effective for native model vs full d for dense
3. Analyze the gap (C vs B):
   - Where does native training do BETTER? (what did it learn that vectors miss?)
   - Where does native training do WORSE? (what did it fail to capture?)
   - Qualitative: pick specific word pairs/analogies where the approaches differ

### Success Criteria
- [ ] Clear comparison table produced
- [ ] At least ONE dimension where native spectral outperforms transform
- [ ] Initial understanding of what the gap IS

### Results
*(To be filled after running)*

### Observations
*(To be filled after running)*

---

## Day 4 — Emergent Structure Analysis

### Objective
Analyze what the native spectral model learned. What are the surviving frequencies? Are they interpretable?

### Plan
1. For each surviving frequency (amplitude > threshold across vocabulary):
   - List top-20 words by amplitude
   - Check: do they form a coherent semantic category?
   - Label if interpretable, mark as ABSTRACT if not
2. Phase analysis:
   - For each surviving frequency: cluster words by phase value
   - Do phase clusters have semantic coherence? (e.g., male vs female at opposite phases)
3. Create a catalog of discovered dimensions
4. Post-hoc: compare to Wierzbicka's NSM primes (how many match?)

### Success Criteria
- [ ] At least SOME frequencies are interpretable
- [ ] Phase carries information (not random)
- [ ] Catalog produced

### Results
*(To be filled after running)*

### Observations
*(To be filled after running)*

---

## Day 5+ — Iteration & Deep Dives

Based on findings from Days 1-4, iterate on:
- Hyperparameter tuning (K, λ, training details)
- Alternative training objectives
- Dimensionality convergence curves (vary L1 strength)
- Additional benchmarks
- Specific deep dives into interesting findings

---

## Running Log

### 2026-02-20
- Created experiment plan and critical path
- Identified three hard requirements: ENCODE, MATH, EFFICIENT
- Key insight: test transform approach FIRST (2 days max) to know if spectral arithmetic exists at all
- Decided: Day 1 starts with Word2Vec baseline + FFT transform test

*(Continue log entries as work progresses)*
