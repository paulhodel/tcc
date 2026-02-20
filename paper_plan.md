# Paper Plan: Learnable Spectral Word Representations

**Target venues:** ACL, EMNLP, NeurIPS, ICML
**Working title:** "Beyond Dense Vectors: Emergent Semantic Structure in Learnable Spectral Word Representations"

---

## 1. The Core Argument

Dense word embeddings (Word2Vec, GloVe) compress semantic meaning into linear directions in Euclidean space. This geometric constraint is both their strength (enabling vector arithmetic) and their limitation — semantic structure that doesn't fit a linear geometry is lost. Furthermore, all dimensions in dense vectors are used equally and opaquely — we never know how many dimensions are actually needed or what each one captures.

We propose **learnable spectral word representations** where each word is a composition of fundamental frequencies with amplitudes and phases, trained directly via a spectral objective. Crucially, **we do not predefine what the frequencies should represent**. The model discovers its own fundamental dimensions of meaning through unsupervised training. We then analyze what emerges.

We hypothesize that:

1. Training natively in the frequency domain captures semantic structure that dense vectors lose.
2. The model will **spontaneously discover** a set of fundamental frequency components that organize into interpretable semantic dimensions — without being told to.
3. L1 sparsity pressure will cause the model to **self-select** how many frequencies it actually needs, revealing the natural dimensionality of semantic space.
4. Spectral operations (element-wise, O(d)) replace vector operations (matrix-based, O(d²)) without quality loss.
5. Post-hoc analysis may reveal correspondence with linguistic theories of semantic primitives (Wierzbicka's NSM) — but this is a finding to discover, not an assumption to enforce.

## 2. Three-Way Experimental Design

### Approach A — Baseline: Dense Embeddings
Standard Word2Vec (skip-gram) and GloVe trained on English Wikipedia.
- Establishes quality ceiling for dense vectors.
- Serves as reference for all benchmarks.

### Approach B — Transform: Dense → FFT Decomposition
Apply Fourier decomposition to trained dense embeddings from Approach A.
- Shows what spectral structure **already exists** in dense vectors.
- Tests whether semantic properties **survive** the transformation.
- Lower bound for frequency-domain quality.
- Analyze: how many frequency components needed to retain 95%, 99% of quality?
- Analyze: are the dominant frequencies interpretable?

### Approach C — Native: Learnable Spectral Model (Main Contribution)
Train word representations **directly in the frequency domain** — never passing through dense vectors.

Each word is parameterized as:
```
word_w = {(f₁, A₁, φ₁), (f₂, A₂, φ₂), ..., (fK, AK, φK)}
```

Where:
- fᵢ — frequency (shared across all words, learned globally)
- Aᵢ — amplitude (per-word, learned)
- φᵢ — phase (per-word, learned)

The **frequencies are shared and learned globally** — the model discovers what the fundamental semantic dimensions ARE. Each word then has its own amplitude (how much of each primitive) and phase (the orientation/polarity).

### The Key Comparison: C vs B

```
B finds: spectral structure that dense vectors ALREADY captured
C finds: spectral structure that EMERGES from native training

If C > B on quality: dense vectors are LOSING information
If C discovers primitives B doesn't: frequency domain has UNIQUE representational capacity
If C's primitives are more interpretable: native training produces CLEANER decomposition
```

**This gap is the paper's main finding.**

## 3. The Learnable Spectral Model — Architecture

### 3.1 Parameters

**Global parameters (shared across all words):**
- `F = [f₁, f₂, ..., fK]` — K candidate fundamental frequencies
- K is the **capacity** — the maximum number of primitives the model CAN use
- The model does NOT have to use all K. L1 sparsity will kill unused frequencies.

**Per-word parameters:**
- `A_w = [A₁, A₂, ..., AK]` — amplitudes for word w (most will be ~0 due to sparsity)
- `φ_w = [φ₁, φ₂, ..., φK]` — phases for word w

**Total parameters per word:** 2K floats (amplitude + phase)
**Total global parameters:** K floats (frequencies)

### 3.1.1 Self-Selecting Dimensionality

Unlike dense vectors where all d dimensions are used equally:

```
Dense (d=300):  All 300 dimensions active for every word. You chose 300. Why not 250? 400?
Spectral (K=512): Give 512 candidate frequencies. L1 pressure kills unused ones.
                   Model converges to N_effective << K active frequencies.
                   N_effective is DISCOVERED, not chosen.
```

**The convergence of N_effective is itself a finding:**
- If N_effective ≈ 65 → matches Wierzbicka's prime count (striking)
- If N_effective ≈ 200 → richer than linguistic theory predicted
- If N_effective varies by language → languages have different semantic complexity
- If N_effective is consistent across languages → evidence for universal structure

**Experiment:** Train with K=512 and vary L1 strength:
- Strong L1 → few survivors (aggressive compression, how far can we push?)
- Weak L1 → many survivors (maximum quality, what's the ceiling?)
- Plot: N_effective vs. quality (analogy accuracy, similarity correlation)
- Find the **elbow**: the point where adding more frequencies stops improving quality
- That elbow IS the natural dimensionality of semantic space

### 3.2 Wave Function

The "embedding" of word w is its wave function evaluated at the fundamental frequencies:

```
E(w) = [A₁·e^(iφ₁), A₂·e^(iφ₂), ..., AK·e^(iφK)]
```

This is a complex-valued vector of dimension K, where each component encodes how strongly (amplitude) and in what orientation (phase) the word expresses each semantic primitive.

### 3.3 Semantic Operations

**Similarity — Spectral Coherence:**
```
sim(w₁, w₂) = Re[ Σᵢ A_w₁[i] · A_w₂[i] · e^(i(φ_w₁[i] - φ_w₂[i])) ]
```
This is the real part of the complex dot product. It considers both amplitude agreement AND phase alignment. Two words are similar when they share the same primitives (amplitude match) AND express them in the same way (phase match).

**Composition — Superposition:**
```
compose(w₁, w₂) = E(w₁) + E(w₂)   (element-wise complex addition)
```
O(K) operation. Constructive interference for shared meaning, destructive for contradictions.

**Analogy — Phase Transformation:**
```
analogy(a, b, c) = E(c) · (E(b) / E(a))
                 = E(c) · [B_amp/A_amp · e^(i(φB - φA))]
```
The transformation from a→b is captured as amplitude ratio + phase shift. Apply the same transformation to c.

### 3.4 Training Objective

**Modified Skip-Gram with Spectral Similarity:**

Given a center word w and context word c:
```
P(c | w) = softmax( spectral_similarity(E(w), E(c)) / τ )
```

Where τ is a temperature parameter.

Loss: negative log-likelihood with negative sampling:
```
L = -log σ(sim(w, c_pos)) - Σⱼ log σ(-sim(w, c_negⱼ))
```

**Regularization:**
- L1 on amplitudes: encourages sparse decomposition — **this is what makes frequencies die or survive**. The model self-selects its dimensionality.
- Orthogonality loss on frequencies: encourages primitives to be independent (prevents two frequencies from encoding the same thing)
- Amplitude diversity loss: prevents frequency collapse (all words using the same few frequencies). Every surviving frequency should be used by a meaningful number of words.
- Phase utilization: soft penalty if phase is unused (all words have same phase on a frequency → phase carries no information → wasted parameter)

### 3.4.1 Training Philosophy: Unsupervised Discovery

**We do not tell the model what to learn.** There is:
- No predefined list of semantic categories
- No supervision signal saying "frequency 7 should encode GENDER"
- No linguistic knowledge injected into the architecture

The ONLY signal is: **predict context words from center words.** Everything else — which frequencies survive, what they represent, how many are needed — emerges from this single unsupervised objective plus sparsity pressure.

This is critical for the paper's scientific contribution. If we predefined primitives, the result would be "our model can reconstruct a linguistic theory." That's engineering. What we're doing is: "given NO linguistic knowledge, does the model INDEPENDENTLY discover structure that aligns with linguistic theory?" That's a scientific finding.

### 3.5 Why This Might Capture What Vectors Miss

Dense vectors have a fixed geometry:
- All dimensions are treated equally (no notion of "this dimension is more fundamental")
- Relationships must be linear (king - man + woman = queen works because it's a linear offset)
- Phase information doesn't exist (direction in vector space conflates amplitude and phase)
- **You choose the dimensionality (300), and the model uses ALL of it** — no way to know how many dimensions it actually needed

Spectral representations have richer structure:
- **Amplitude and phase are separate** — how much vs. in what direction. Dense vectors collapse these.
- **Frequencies have learned importance** — the global frequencies are ordered by utility, not arbitrary
- **Non-linear relationships via phase** — a π phase shift is a qualitative change (male→female), not just a different point in space
- **Natural sparsity** — most words only need a few dominant frequencies, automatically compressing
- **Self-selecting complexity** — the model tells YOU how many dimensions it needs, not the other way around

**Hypothesis:** The separation of amplitude (how much) from phase (what kind) captures semantic distinctions that dense vectors blur. For example, "hot" (temperature) and "hot" (attractive) might have the same amplitude on a INTENSITY primitive but different phases — a distinction that dense vectors struggle with.

### 3.6 What If Learned Dimensions Are NOT Interpretable?

This is a valid concern. The learned frequencies might be:

**Outcome A — Interpretable primitives:**
Frequencies map to recognizable categories (ANIMATE, POSITIVE, ABSTRACT...). We can label them, validate against linguistics. Best case for the paper.

**Outcome B — Partially interpretable:**
Some frequencies are clearly interpretable, others are abstract statistical patterns. This is actually the MOST LIKELY and MOST INTERESTING outcome — it suggests meaning has both "nameable" and "unnameable" dimensions.

**Outcome C — Entirely abstract:**
Frequencies are statistically optimal but semantically opaque (like PCA components). Even then:
- They still self-select dimensionality (a finding)
- They still separate amplitude from phase (a structural advantage)
- They still enable efficient spectral operations (a practical contribution)
- The question "why didn't interpretable primitives emerge?" is itself a publishable finding

**All three outcomes produce a paper.** This is crucial for a master's project — we can't afford a design where negative results kill the thesis.

## 4. Evaluation Plan

### 4.1 Semantic Quality (Does it work?)

| Benchmark | What it tests | Metric |
|-----------|--------------|--------|
| Google Analogy (19K questions) | Compositional semantics (king-man+woman=queen) | Accuracy |
| BATS (99K questions) | Broader analogy coverage | Accuracy |
| WordSim-353 | Semantic similarity judgments | Spearman correlation |
| SimLex-999 | Strict similarity (not relatedness) | Spearman correlation |
| MEN-3000 | Semantic relatedness | Spearman correlation |
| Downstream: sentiment (SST-2) | Practical utility as features | Accuracy |
| Downstream: NER (CoNLL-2003) | Practical utility as features | F1 |

### 4.2 Efficiency (Is it faster?)

| Metric | Dense baseline | Transform (B) | Native spectral (C) |
|--------|---------------|---------------|---------------------|
| Parameters per word | d floats | d complex | 2K floats |
| Similarity cost | O(d) | O(d) | O(K) |
| Composition cost | O(d²) matmul | O(d) element-wise | O(K) element-wise |
| Memory per word | d × 4 bytes | d × 8 bytes (complex) | 2K × 4 bytes |
| Training FLOPs | baseline | N/A (post-hoc) | measured |
| Training wall-clock | baseline | N/A | measured |

### 4.3 Dimensionality Convergence (How many primitives does meaning need?)

**The key experiment:**
- Train with K=512 (generous capacity)
- Vary L1 sparsity strength: λ ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
- For each λ, measure: N_effective (frequencies with mean amplitude > threshold) vs. quality (analogy accuracy)
- Plot the **convergence curve**: quality as a function of active frequencies

```
Expected plot:

Quality │          _______________
   ↑    │        /
        │      /
        │    /
        │  /
        │/
        └──────────────────────→ N_effective (active frequencies)
             Elbow here = natural dimensionality
```

**Report:**
- N_effective at 95% of max quality
- N_effective at 99% of max quality
- Compare N_effective to: dense vector dimensions (300), Wierzbicka primes (65), typical PCA analysis of embeddings

### 4.4 Emergent Structure Analysis (What did it learn?)

**Post-hoc interpretability (we do NOT prescribe, we discover):**
- For each surviving frequency fᵢ: which words have highest amplitude? Do they form a coherent semantic category?
- Example: if f₇ activates for [king, queen, prince, emperor, ruler] → label it GOVERNANCE
- Example: if f₁₂ activates for [red, blue, green, yellow, purple] → label it COLOR
- Example: if f₃₃ activates for a seemingly random set → label it ABSTRACT_33 (still a valid finding)
- Cluster the frequencies by the words they activate → create a catalog of discovered dimensions

**Phase analysis:**
- For each frequency: do words cluster into phase groups?
- Example: on a GENDER-like frequency, do male words cluster at φ≈0 and female words at φ≈π?
- Phase clusters reveal **sub-categories within a dimension** — richer than binary primitives

**Comparison with linguistic theory (post-hoc, not prescribed):**
- Map discovered dimensions to Wierzbicka's NSM primes where possible
- Score: how many NSM primes have a clear corresponding learned frequency?
- Score: how many learned frequencies DON'T match any NSM prime? (these are novel discoveries about meaning structure that linguistics hasn't named)
- This analysis is exploratory, not confirmatory — we report what we find

**Cross-lingual test (if time permits):**
- Train on English → catalog surviving frequencies
- Train on Chinese/Spanish/Arabic → catalog surviving frequencies independently
- Compare: are the same fundamental dimensions discovered across languages?
- If yes: evidence for universal structure (strong finding)
- If partially: which dimensions are universal, which are language-specific? (equally interesting)

### 4.4 The Gap Analysis (C vs B — the core contribution)

- Quality delta: does native training beat transform on analogy/similarity?
- Primitive quality: are natively-learned primitives more interpretable?
- Sparsity: does native training produce sparser decompositions?
- Phase utilization: does native training use phase more meaningfully?
- Novel primitives: does native training discover primitives that transform misses?

## 5. Timeline (5 Months)

| Month | Phase | Key Activities | Deliverables |
|-------|-------|---------------|-------------|
| **1** | Foundations | Implement baselines (Word2Vec, GloVe); build evaluation pipeline; implement FFT transform approach; literature review | Working baselines; evaluation framework; transform results |
| **2** | Native Model | Implement learnable spectral model; design training pipeline (spectral skip-gram); initial training on small corpus (PTB) | Working native spectral model; initial training curves |
| **3** | Full Training & Comparison | Train native model on Wikipedia; full benchmark suite; three-way comparison (A vs B vs C); iterate on regularization | Complete benchmark results; gap analysis |
| **4** | Primitive Analysis | Interpretability study; linguistic validation (NSM mapping); cross-lingual experiment (if time permits); ablation studies (K, regularization) | Primitive catalog; linguistic analysis; ablations |
| **5** | Paper Writing | Write paper; prepare figures; additional experiments to fill gaps; presentation | Submitted paper; thesis document |

## 6. Paper Structure (Draft Outline)

```
1. Introduction
   - Dense embeddings transformed NLP but have geometric limitations
   - We propose learnable spectral representations — unsupervised, no prescribed structure
   - Key findings preview: (1) the model self-selects its dimensionality,
     (2) native spectral training captures structure vectors miss,
     (3) interpretable semantic dimensions emerge without supervision

2. Related Work
   - Word embeddings (Word2Vec, GloVe, contextual)
   - Spectral methods in NLP (FNet, Fourier features)
   - Semantic primitives in linguistics (Wierzbicka NSM) — as context, not assumption
   - Efficient representations (HDC, binary embeddings, hash embeddings)
   - Dimensionality analysis of embeddings (PCA, intrinsic dimensionality studies)

3. Method
   - Learnable Spectral Word Representations
   - Global frequencies (learned, not prescribed) + per-word amplitude/phase
   - Training objective (spectral skip-gram + L1 sparsity)
   - Self-selecting dimensionality: how sparsity reveals natural complexity
   - Spectral semantic operations (similarity, composition, analogy)

4. Experimental Setup
   - Three approaches: Dense baseline (A), Transform (B), Native spectral (C)
   - Datasets, benchmarks, evaluation metrics
   - Hyperparameters: K (capacity), λ (sparsity), training details

5. Results
   5.1 Dimensionality convergence: how many frequencies survive? The elbow curve.
   5.2 Semantic quality comparison (A vs B vs C)
   5.3 Efficiency benchmarks
   5.4 The gap: what native training captures that vectors miss
   5.5 Emergent structure analysis: what the model discovered
       - Interpretable dimensions (if any)
       - Phase clustering patterns
       - Abstract dimensions that defy linguistic categories
   5.6 Post-hoc comparison with linguistic theory (NSM)
   5.7 Cross-lingual comparison (if included)

6. Analysis & Discussion
   - What does the natural dimensionality of meaning tell us?
   - Why does native training outperform transform? (if it does)
   - The role of phase: what amplitude can't capture alone
   - What emerges that linguistic theory hasn't named?
   - Limitations and failure cases

7. Conclusion
   - Spectral representations self-organize into interpretable dimensions
   - Frequency-domain training has unique representational advantages
   - The natural dimensionality of semantic space is N_effective (discovered)
   - Implications for efficient, interpretable NLP
   - Future work: resonance-based models, graph media, multimodal extensions
```

## 7. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Native model doesn't beat transform | Weakens main argument | Medium | Analyze WHY — characterize differences even if quality is similar. Focus on interpretability/efficiency. |
| Discovered primitives aren't interpretable | Weakens primitive narrative | Medium | Use multiple interpretability methods. Even uninterpretable primitives are a finding ("the model discovers abstract features beyond human categories"). |
| Training doesn't converge well | Blocks progress | Medium | Start with transform initialization (warm start). Extensive hyperparameter search. Curriculum: start with frequent words. |
| Quality significantly below dense baselines | Paper becomes negative result | Low-Medium | Frame as efficiency vs quality tradeoff. Even 90% quality at 10x speed is publishable. |
| Cross-lingual primitives don't match | Weakens universality claim | Medium | Make cross-lingual a bonus, not the core claim. Main contribution stands without it. |

## 8. What Makes This Publishable

1. **Novel model:** First learnable spectral word representation trained natively in frequency domain with self-selecting dimensionality.
2. **Novel finding #1:** The model discovers its own dimensionality — N_effective frequencies survive from K candidates. This is a measured property of semantic space.
3. **Novel finding #2:** The gap between transform and native — what vectors lose that frequencies preserve.
4. **Novel finding #3:** What emerges? Interpretable dimensions, abstract patterns, or both? Any outcome is a contribution.
5. **Scientifically clean:** We prescribe NOTHING. No predefined primitives, no linguistic assumptions. Everything is discovered. Post-hoc analysis connects to theory.
6. **Practical:** Concrete efficiency gains (O(d) composition, sparse self-compressed representations).
7. **Cross-disciplinary:** Post-hoc connection to linguistic theory (NSM) — rare in ML papers, reviewers love it.
8. **Opens new research direction:** Spectral representations → resonance models → graph media → multimodal (future work section writes itself).
9. **Robust to negative results:** Even if quality is lower, even if primitives aren't interpretable — the dimensionality convergence and the gap analysis are still contributions. Every outcome produces a paper.
