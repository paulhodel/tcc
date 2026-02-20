# Master's Research Roadmap: Toward a Modular, Matrix-Free Language Model

**Program:** Master's in Artificial Intelligence
**Author:** [Your Name]
**Date:** February 2026

---

## 1. Vision Statement

The goal of this research program is to systematically investigate whether each component of a modern Large Language Model can be replaced with an alternative computational block that does not rely on dense matrix multiplication — and ultimately, whether these alternative blocks can be composed into a complete, modular, heterogeneous language model that matches or exceeds transformer quality at significantly lower computational cost.

The key insight driving this research: **there is no reason every layer of a language model must use the same computational paradigm.** Current transformers stack identical blocks (attention + feed-forward), each performing the same types of matrix operations. A more efficient architecture might use different computational paradigms for different functions — a graph-based block where associative memory is needed, a frequency-domain block where composition is needed, a sparse activation block where selection is needed.

## 2. The Transformer Pipeline and Replacement Strategy

A standard transformer language model consists of:

```
Tokens → [Embedding Layer] → [Block 1] → [Block 2] → ... → [Block N] → [Output Layer] → Next Word
                                  |              |
                            [Attention]    [Attention]
                            [Feed-Fwd]    [Feed-Fwd]
```

Each component relies on dense matrix multiplications:

| Component | Operation | Complexity | Matrix Dependency |
|-----------|-----------|-----------|-------------------|
| **Embedding Layer** | Token → dense vector lookup | O(V × d) storage | Embedding matrix (V × d) |
| **Self-Attention** | Q, K, V projections + attention scores | O(n² × d) | Three weight matrices (d × d) + score matrix (n × n) |
| **Feed-Forward** | Two linear transformations with nonlinearity | O(d × d_ff) | Two weight matrices (d × d_ff, d_ff × d) |
| **Output Projection** | Hidden state → vocabulary logits | O(d × V) | Output weight matrix (d × V) |

**The replacement strategy** investigates each component independently, then composes them:

### Phase 1: Embedding Layer (Current Projects — 2026)

**Goal:** Replace the V × d embedding matrix with efficient alternatives.

Two parallel research tracks:

| Track | Approach | Core Operation | Complexity |
|-------|----------|---------------|-----------|
| **Track A** | Frequency-domain representations | FFT + element-wise multiply | O(d) composition, O(d log d) transform |
| **Track B** | Self-organizing neural graph | Spreading activation + pointer traversal | O(K × E_active) per prediction |

Additional explorations within these tracks: Hyperdimensional Computing, Sparse Distributed Representations, Spiking/Temporal encodings.

**Deliverable:** A semantic representation layer that provides word-level meaning without dense matrix storage, with verified emergent compositional properties.

### Phase 2: Attention Replacement (Future)

**Goal:** Replace the self-attention mechanism with a matrix-free context-mixing operation.

The attention mechanism does two things:
1. **Determines relevance:** Which other tokens are relevant to the current token?
2. **Mixes information:** Combines information from relevant tokens.

Potential replacement approaches informed by Phase 1 findings:

| Approach | Relevance Mechanism | Mixing Mechanism |
|----------|-------------------|-----------------|
| **Graph-based attention** | Spreading activation from current node determines which other nodes are relevant | Activated subgraph IS the mixed representation |
| **Spectral mixing** | FNet-style Fourier transform for token mixing (already demonstrated at 92% of BERT quality) | Element-wise operations in frequency domain |
| **Sparse activation** | Only top-K most relevant tokens activate (like MoE but for attention) | Weighted sum over sparse active set |
| **Graph + Spectral hybrid** | Graph structure determines connectivity, spectral operations perform mixing | Frequency-domain composition over graph neighborhoods |

**Key insight from FNet:** Replacing attention with Fourier transforms already achieves 92% of BERT's accuracy (Lee-Thorp et al., 2021). This suggests attention's matrix operations are partially redundant — a strong signal that matrix-free alternatives are viable.

### Phase 3: Feed-Forward Replacement (Future)

**Goal:** Replace the feed-forward layers (two dense matrices with nonlinearity) with alternative transformation blocks.

The feed-forward layer's role is to **transform representations** — applying nonlinear projections that extract features. Potential replacements:

| Approach | Description |
|----------|-------------|
| **Element-wise transformations** | Learned nonlinear functions applied independently per dimension (no matrix needed) |
| **Frequency-domain filtering** | Selective amplification/suppression of frequency components (band-pass filtering for meaning) |
| **Graph-based transformation** | Node compute functions (as in the neural graph proposal) applying local transformations |
| **Ternary/binary operations** | Following MatMul-free LM approach (Zhu et al., 2024) using {-1, 0, 1} weights |

### Phase 4: Output Layer Replacement (Future)

**Goal:** Replace the d × V output projection matrix.

The output layer converts hidden representations to vocabulary-sized probability distributions. If the embedding layer is replaced (Phase 1), the output layer naturally follows — in many architectures, the output projection shares weights with the embedding matrix.

Potential approaches:
- **Graph-based scoring:** If the model uses a neural graph, prediction is already a graph operation (spreading activation → softmax over activated candidates).
- **Spectral matching:** Compare the hidden state's spectrum to word spectra — nearest spectral neighbor is the prediction.
- **Hybrid:** Different scoring methods for different confidence levels — high-confidence predictions use fast approximate methods, uncertain predictions use more thorough computation.

## 3. The Modular, Heterogeneous Architecture

The ultimate vision: a language model where **each layer can use a different computational paradigm**, chosen for what it does best:

```
Tokens
  ↓
[Graph-Based Embedding Layer]     ← Organic vocabulary, associative memory
  ↓
[Spectral Context Mixing]         ← FFT-based token mixing, O(n log n)
  ↓
[Sparse Activation Transform]     ← Element-wise nonlinearities, no matrices
  ↓
[Graph-Based Context Layer]       ← Spreading activation for deep context
  ↓
[Frequency-Domain Composition]    ← Spectral composition of meaning
  ↓
[Activation-Based Scoring]        ← Graph traversal for next-word prediction
  ↓
Next Word
```

Each block is specialized. They communicate through a shared representation format (potentially frequency-domain vectors or activation patterns). The entire pipeline involves **zero dense matrix multiplications**.

### Why Heterogeneous Blocks?

Current transformers use identical blocks because:
1. It's simple to implement and scale.
2. The same hardware (GPU matmul units) handles every layer.
3. It works well enough.

But there's no theoretical reason every layer should use the same computation. In the brain:
- Different regions use different computational strategies.
- The hippocampus (memory) works differently from the visual cortex (feature extraction) works differently from the prefrontal cortex (planning).
- These heterogeneous systems compose into a coherent whole.

A heterogeneous language model could similarly assign different computational strategies to different functions, optimizing each independently.

## 4. Research Timeline Overview

```
Year 1 (2026):
├── Months 1-5:  Phase 1A — Frequency-domain embeddings
├── Months 1-5:  Phase 1B — Self-organizing neural graph (parallel)
└── Months 5-6:  Integration analysis, identify most promising directions

Year 1-2 (2026-2027):
├── Phase 2 — Attention replacement experiments
├── Phase 3 — Feed-forward replacement experiments
└── Cross-phase integration prototypes

Year 2 (2027):
├── Phase 4 — Output layer replacement
├── Full pipeline prototype (heterogeneous blocks)
└── Thesis writing and defense
```

## 5. Evaluation Strategy Across Phases

Each phase is evaluated on the same core criteria:

### Semantic Quality
- Does the replacement preserve the emergent properties of the component it replaces?
- For embeddings: analogy tests, similarity benchmarks.
- For attention: contextual understanding, long-range dependency capture.
- For feed-forward: representation quality, feature extraction.
- For the full pipeline: perplexity, downstream task performance.

### Efficiency
- **Training:** Convergence speed, data efficiency, hardware requirements.
- **Inference:** Latency per token, throughput, memory usage.
- **Hardware:** CPU vs GPU requirements, potential for edge/mobile deployment, neuromorphic hardware compatibility.

### Composability
- Can the replacement block interface with other blocks (both traditional and alternative)?
- What is the representation format between blocks?
- Does replacing one block degrade the performance of adjacent blocks?

## 6. Key Principles

1. **Exhaust each direction before combining.** Each alternative representation and mechanism must be thoroughly understood in isolation before attempting integration. Premature integration obscures which components contribute what.

2. **Emergent properties first, efficiency second.** An efficient representation that lacks compositional semantics is useless. Every alternative must first demonstrate emergent semantic properties, then demonstrate efficiency advantages.

3. **Modular and composable.** Each replacement block must define clear input/output interfaces so it can be swapped into existing architectures for fair comparison.

4. **Hardware-aware design.** The goal is not just algorithmic efficiency but practical speedup. Designs should consider CPU cache behavior, memory access patterns, and potential neuromorphic hardware targets.

5. **Biological plausibility as inspiration, not constraint.** The brain provides useful analogies (spreading activation, Hebbian learning, heterogeneous computation) but engineering decisions should prioritize performance over biological faithfulness.

## 7. Companion Proposals

This roadmap is supported by two detailed research proposals for Phase 1:

1. **`proposal_frequency_domain.md`** — Frequency-domain word representations as an efficient alternative to dense embeddings. Explores spectral encoding, phase-based semantic relationships, and FFT-based composition.

2. **`proposal_graph_neural_model.md`** — Self-organizing neural graph for language modeling. Explores dynamic graph architectures, spreading activation inference, and backpropagation on cyclic graphs.

Both proposals include detailed methodology, evaluation frameworks, timelines, and risk mitigation strategies for a 5-month research period.

## 8. References

1. Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
2. Lee-Thorp, J. et al. (2021). "FNet: Mixing Tokens with Fourier Transforms." arXiv:2105.03824.
3. Zhu, R. et al. (2024). "Scalable MatMul-free Language Modeling." arXiv:2406.02528.
4. Bai, S. et al. (2019). "Deep Equilibrium Models." NeurIPS.
5. Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation.
6. Collins, A. & Loftus, E. (1975). "A Spreading-Activation Theory of Semantic Processing." Psychological Review.
7. Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
8. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752.
9. Peng, B. et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048.
10. Maass, W. (1997). "Networks of Spiking Neurons." Neural Networks.

---

*This roadmap defines a multi-phase research program aimed at building a modular, matrix-free language model from the ground up. Starting with the embedding layer and progressively replacing each transformer component, the program investigates whether heterogeneous computational blocks — each using a different paradigm optimized for its function — can match or exceed the quality of homogeneous transformer architectures at dramatically lower computational cost.*
