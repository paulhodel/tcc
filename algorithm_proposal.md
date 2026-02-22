if# Algorithm Proposal: Spectral Skip-Gram Training

**Date:** 2026-02-20
**Status:** Proposed — to be validated in experiments

---

## 1. Word Structure

```
Per word w:
    A[w] = [A₁, A₂, ..., A₅₁₂]    # amplitudes (512 floats, non-negative)
    φ[w] = [φ₁, φ₂, ..., φ₅₁₂]    # phases (512 floats, [0, 2π))
```

- Fixed grid of 512 candidate frequencies
- L1 sparsity → most amplitudes converge to 0
- ~50-100 active per word after training
- Storage at inference: only active (index, amplitude, phase) triples

---

## 2. Similarity Function

```
sim(w1, w2) = Σᵢ A[w1]ᵢ · A[w2]ᵢ · cos(φ[w1]ᵢ - φ[w2]ᵢ)
```

Three factors per frequency:
- `A[w1]ᵢ · A[w2]ᵢ` — both words must have amplitude for this frequency to contribute
- `cos(φ[w1]ᵢ - φ[w2]ᵢ)` — phases must be aligned for positive contribution

This gives the model TWO mechanisms to separate words:
1. **Don't share frequencies** — zero amplitude on different slots (categorical separation)
2. **Opposite phases on shared frequencies** — same category but opposite polarity

---

## 3. Gradient Dynamics

For a positive pair (center word w_c, context word w_pos):

**Amplitude gradient:**
```
∂sim/∂A[w_c]ᵢ = A[w_pos]ᵢ · cos(φ[w_c]ᵢ - φ[w_pos]ᵢ)
```
Pushes A[w_c]ᵢ UP when:
- w_pos has amplitude on this frequency
- AND phases are aligned

**Phase gradient:**
```
∂sim/∂φ[w_c]ᵢ = -A[w_c]ᵢ · A[w_pos]ᵢ · sin(φ[w_c]ᵢ - φ[w_pos]ᵢ)
```
Pushes φ[w_c]ᵢ TOWARD φ[w_pos]ᵢ (phase alignment).

For negative pairs: gradients push in the opposite direction (amplitudes decrease or phases misalign).

---

## 4. Training Dynamics: What Happens Over a Corpus

### Example Trace

```
Sentence 1: "The kind king is generous"
  → (king, kind):     Develop shared amplitude. Phases align.
  → (king, generous):  Develop shared amplitude. Phases align.

Sentence 2: "The cruel king is feared"
  → (king, cruel):     Develop shared amplitude. Phases align.
  → (king, feared):    Develop shared amplitude. Phases align.

Sentence 3: "The king rules the kingdom"
  → (king, rules):     Develop shared amplitude on GOVERNANCE frequencies.
  → (king, kingdom):   Develop shared amplitude on GOVERNANCE frequencies.

Sentence 4: "The queen rules wisely"
  → (queen, rules):    Queen develops amplitude on same GOVERNANCE frequencies as king.
```

### The Conflict Resolution: "kind" vs "cruel"

"King" appears in context with BOTH "kind" and "cruel". But "kind" and "cruel" don't share contexts with each other. How does the model resolve this?

On a VALENCE/sentiment frequency (say f₇):
- "kind" gradients pull king's phase toward 0
- "cruel" gradients pull king's phase toward π
- These cancel → king's amplitude on f₇ gets suppressed (not useful for king)

**Result after training:**

```
VALENCE frequency (f₇):
  kind:      A=0.8, φ=0.0     (high amplitude, positive polarity)
  cruel:     A=0.8, φ=π       (high amplitude, negative polarity)
  generous:  A=0.7, φ=0.1     (high amplitude, positive polarity)
  feared:    A=0.7, φ=3.0     (high amplitude, negative polarity)
  king:      A=0.1, φ=1.5     (LOW amplitude — valence not defining for king)

GOVERNANCE frequency (f₁₅):
  king:      A=0.9, φ=0.0     (high amplitude)
  queen:     A=0.9, φ=0.0     (high amplitude, same governance role)
  rules:     A=0.8, φ=0.1     (high amplitude)
  kind:      A=0.1, φ=?       (low amplitude — kindness isn't about governance)

GENDER frequency (f₂₃):
  king:      A=0.8, φ=0.0     (male)
  queen:     A=0.8, φ=π       (female — opposite phase, same amplitude)
  rules:     A=0.1, φ=?       (low amplitude — governance isn't gendered)
```

### What the Training Naturally Produces

- **Amplitude = category membership** — "this frequency is relevant to this word"
- **Phase = polarity within category** — "which side of this dimension"
- **Sparsity = category exclusion** — "this frequency is NOT relevant"

---

## 5. The Lifecycle of a Frequency

```
EARLY training (random):
    f₇: Every word has small random amplitude. No meaning yet.
    The frequency contributes noise to similarity.

MID training (emergence):
    f₇: Words about sentiment develop amplitude.
    Positive words cluster at φ≈0. Negative words at φ≈π.
    Neutral words lose amplitude on f₇ (conflicting gradients).

LATE training (convergence):
    f₇: Clear VALENCE primitive has emerged.
    Only sentiment-bearing words have significant amplitude.
    L1 has killed the amplitude for neutral words.

    A[good]=0.9 φ=0.1 | A[bad]=0.8 φ=3.1 | A[king]=0.1 φ=1.5
```

---

## 6. The Full Training Algorithm

```
INITIALIZE:
    For each word w in vocabulary:
        A[w] = small random positive values (512 floats)
        φ[w] = random uniform in [0, 2π) (512 floats)

TRAIN:
    For each epoch:
      For each sentence in corpus:
        For each center word w_c (sliding window):

          context_words = words within window of w_c
          negative_words = sample K random words (frequency-weighted)

          # --- FORWARD ---
          For each context word w_pos:
            s_pos = Σᵢ A[w_c]ᵢ · A[w_pos]ᵢ · cos(φ[w_c]ᵢ - φ[w_pos]ᵢ)

          For each negative word w_neg:
            s_neg = Σᵢ A[w_c]ᵢ · A[w_neg]ᵢ · cos(φ[w_c]ᵢ - φ[w_neg]ᵢ)

          # --- LOSS ---
          L = -Σ log σ(s_pos) - Σ log σ(-s_neg) + λ · Σᵢ |A[w_c]ᵢ|
                                                    └─── L1 sparsity ───┘

          # --- BACKWARD ---
          Compute gradients via autograd:
            ∂L/∂A[w_c], ∂L/∂φ[w_c]
            ∂L/∂A[w_pos], ∂L/∂φ[w_pos]
            ∂L/∂A[w_neg], ∂L/∂φ[w_neg]

          # --- UPDATE ---
          A[w_c]   -= lr_amp · ∂L/∂A[w_c]
          φ[w_c]   -= lr_phase · ∂L/∂φ[w_c]
          A[w_pos]  -= lr_amp · ∂L/∂A[w_pos]
          φ[w_pos]  -= lr_phase · ∂L/∂φ[w_pos]
          A[w_neg]  -= lr_amp · ∂L/∂A[w_neg]
          φ[w_neg]  -= lr_phase · ∂L/∂φ[w_neg]

          # --- CONSTRAINTS ---
          A[w_c] = max(A[w_c], 0)        # amplitudes stay non-negative
          φ[w_c] = φ[w_c] % (2π)         # phases stay in [0, 2π)
```

---

## 7. Update Rules Summary

| Event in text | Effect on amplitude | Effect on phase |
|---|---|---|
| Words co-occur (positive pair) | Both grow amplitude on shared frequencies | Phases align on shared frequencies |
| Words don't co-occur (negative pair) | Shared amplitudes decrease OR stay separate | Phases misalign |
| Word appears with contradictory contexts | Amplitude dies on conflicting frequencies | Phase gets pulled both ways → suppressed |
| L1 regularization (every step) | ALL amplitudes shrink toward 0 | No effect on phase |
| Frequency used by many words | Amplitude survives L1 (constantly reinforced) | Phase differentiates words within that frequency |
| Frequency used by few words | Amplitude dies (not reinforced enough to survive L1) | Irrelevant (amplitude ≈ 0) |

---

## 8. Why This Should Produce Meaningful Representations

The training signal is identical to Word2Vec: predict context words. This is proven to produce semantic structure (Mikolov et al., 2013).

The difference is the representation container:
- Word2Vec stores meaning in 300 opaque floats
- This model stores meaning in 512 × (amplitude, phase) with L1 sparsity

The distributional hypothesis (words in similar contexts have similar meanings) doesn't care about the container — it works through the similarity function. As long as the similarity function is differentiable and expressive enough, the training will push co-occurring words together and non-co-occurring words apart.

The spectral similarity function is MORE expressive than cosine similarity because it has two independent channels (amplitude agreement AND phase alignment). This means the model has more ways to organize words, which should produce richer structure.

---

## 9. Analogy: How Spectral Arithmetic Should Work

After training, the analogy king:queen :: man:woman should work because:

```
king  and queen  share GOVERNANCE frequencies (same A, same φ)
king  and queen  differ on GENDER frequency   (same A, opposite φ)
man   and woman  differ on GENDER frequency   (same A, opposite φ)
```

The transformation king→queen is: flip phase on GENDER frequency, keep everything else.

```
Operation: result = E(man) · conj(E(king)) · E(queen)

This captures:
1. conj(E(king)) · E(queen) = the transformation (phase flip on GENDER)
2. Apply to E(man) = man with GENDER phase flipped = woman
```

Alternative operation (simpler):
```
result_A = A[man]                             # keep man's amplitudes
result_φ = φ[man] + (φ[queen] - φ[king])     # apply king→queen phase shift to man

On GENDER frequency: φ[man] + (π - 0) = φ[man] + π = woman's phase ✓
On GOVERNANCE frequency: φ[man] + (0 - 0) = φ[man] = unchanged ✓
```

---

## 10. Open Implementation Questions

1. **Learning rate:** Should amplitude and phase have different learning rates? Phase is periodic — might need smaller steps to avoid oscillation.

2. **Initialization:** Random uniform? Initialize amplitudes from Word2Vec (warm start)? Start all amplitudes equal and let training differentiate?

3. **L1 strength (λ):** Critical hyperparameter. Too strong → everything dies. Too weak → no sparsity. Sweep: λ ∈ {0.001, 0.01, 0.05, 0.1, 0.5}.

4. **Amplitude constraint:** Clip to ≥ 0? Or use softplus `A = log(1 + exp(raw))` to ensure positivity smoothly? Clipping can zero out gradients.

5. **Phase representation:** Direct angle φ with modular arithmetic? Or Cartesian (r, i) where A = √(r²+i²), φ = atan2(i, r)? Cartesian avoids wrapping issues but makes L1 on amplitude indirect.

6. **Negative sampling distribution:** Same as Word2Vec (frequency^0.75)? Or uniform? Frequency-weighted avoids wasting gradients on very rare words.

7. **Batch size:** Standard Word2Vec uses SGD with small batches. Spectral model might benefit from larger batches for more stable gradient estimates on phases.

These will be resolved empirically during Day 1-2 experiments.

---

## 11. Evaluation After Training

| Operation | What it validates | Test |
|-----------|-------------------|------|
| Similarity | Amplitude sharing + phase alignment = semantic similarity | WordSim-353, SimLex-999 (Spearman correlation) |
| Analogy | Phase encodes relational structure | Google Analogy, BATS (accuracy) |
| Sentiment | VALENCE frequency emerged | SST-2 (classify using spectral features) |
| Clustering | Frequency SELECTION encodes categories | Words sharing active frequencies cluster by category |
| Composition | Superposition = meaning combination | E(ice) + E(cream) ≈ frozen dessert words |
| Sparsity | L1 produced meaningful compression | N_effective vs quality curve (elbow) |
| Interpretability | Surviving frequencies are nameable | Top-20 words per frequency, manual labeling |
