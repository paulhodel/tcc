# Research Proposal: Frequency-Domain Word Representations as an Efficient Alternative to Dense Embeddings

**Program:** Master's in Artificial Intelligence
**Duration:** 5 months
**Author:** [Your Name]
**Date:** February 2026

---

## 1. Introduction

Large Language Models (LLMs) have achieved remarkable results across NLP tasks, but their computational cost remains a significant barrier to widespread deployment. At the foundation of these models lies the word embedding layer — dense vector representations that encode semantic meaning through high-dimensional floating-point matrices.

While effective, dense embeddings carry inherent inefficiencies: they require large matrix multiplications for composition, consume significant memory, and scale poorly as vocabulary size grows. This project investigates whether **frequency-domain representations** can serve as a viable, more efficient alternative to traditional dense vector embeddings while preserving — or enhancing — the emergent semantic properties that make embeddings useful.

### 1.1 Research Context and Motivation

This project is the first phase of a broader master's research program focused on **LLM efficiency**. The overarching goal is to remove matrices from the equation — to find representations and computational paradigms that eliminate the dependency on dense matrix multiplications that currently bottleneck both training and inference in language models.

The starting point is word embeddings, because they are the foundational representation layer. Traditional embeddings (Word2Vec, GloVe) demonstrated a magnificent emergent property: vector arithmetic captures semantic relationships (e.g., king - man + woman = queen). This emergent compositionality is, in the author's view, **the foundation of everything** that makes modern NLP work. Any alternative representation must preserve or replicate this property to be viable.

The research is scoped to be achievable start-to-end within a 5-month period, focusing specifically on the embedding layer rather than full model architectures. Subsequent phases will extend findings to full language model architectures, potentially integrating with a parallel research track on graph-based language modeling (see Section 10).

### 1.2 Two Fundamental Research Questions

This project is structured around two fundamental questions:

1. **Semantic emergent properties:** Does the alternative representation develop compositional structure? Can we perform operations analogous to vector arithmetic? Without this, the representation is merely an encoding — not a semantic space.

2. **Training and inference efficiency:** Can we design a training system for these representations, and do they offer concrete speedups? This must be evaluated separately for training (where gradient computation and convergence matter) and inference (where operation cost and memory access patterns matter).

## 2. Problem Statement

Traditional word embeddings (Word2Vec, GloVe, transformer embeddings) represent words as dense floating-point vectors in Euclidean space. All downstream operations — similarity computation, semantic composition, attention — rely on costly matrix multiplications (O(d^2) or worse). As models scale, this computational burden becomes the primary bottleneck for both training and inference.

**Core question:** Can we represent word-level semantics in a domain where the fundamental operations are cheaper than matrix multiplication, while retaining the emergent compositional properties (e.g., analogical reasoning) that make embeddings powerful?

## 3. Research Objectives

1. **Investigate frequency-domain word representations** — Encode word semantics as spectral signals rather than dense vectors, where composition becomes element-wise multiplication (O(d)) instead of matrix multiplication (O(d^2)).

2. **Evaluate emergent semantic properties** — Determine whether frequency-domain representations develop compositional structure analogous to vector arithmetic in traditional embeddings (e.g., king - man + woman = queen).

3. **Design a training methodology** — Develop and compare training approaches: native frequency-domain training vs. post-hoc spectral transformation of dense embeddings.

4. **Benchmark efficiency trade-offs** — Quantify improvements in memory footprint, training time, and inference speed against standard embedding baselines.

5. **Compare against alternative efficient representations** — Evaluate frequency-domain embeddings alongside Hyperdimensional Computing, Sparse Distributed Representations, and Spiking/Temporal encodings.

## 4. Background and Related Work

### 4.1 Dense Word Embeddings
- **Word2Vec** (Mikolov et al., 2013) — Skip-gram and CBOW models that learn distributed word representations. Demonstrated that vector arithmetic captures semantic relationships (king - man + woman = queen).
- **GloVe** (Pennington et al., 2014) — Global matrix factorization approach to word vectors using co-occurrence statistics.
- **Contextual Embeddings** (BERT, GPT) — Transformer-based dynamic representations where the same word gets different vectors based on context.

### 4.2 Frequency Domain in Neural Networks
- **FNet** (Lee-Thorp et al., 2021) — Replaced self-attention with Fourier transforms in a BERT-like architecture, achieving 92% of BERT's accuracy at 80% faster training speed on GPU and 70% faster on TPU. Demonstrates the viability of spectral methods for token mixing in NLP.
- **Fourier Features** (Tancik et al., 2020) — Random Fourier features for positional encoding, showing that mapping inputs to a higher-dimensional frequency space helps networks learn high-frequency functions.
- **Spectral Graph Theory** — Frequency analysis on graph structures; the Graph Fourier Transform defines convolution on irregular domains, potentially applicable to semantic networks.

### 4.3 Alternative Representation Paradigms
- **Hyperdimensional Computing (HDC)** (Kanerva, 2009) — Represents concepts as high-dimensional binary/ternary vectors (~10,000 dimensions). Operations are extremely cheap: bundling (element-wise majority), binding (XOR), permutation. Shows promise for lightweight semantic representation with mathematical guarantees on capacity.
- **Sparse Distributed Representations (SDR)** (Ahmad & Hawkins, 2016) — Neuroscience-inspired encoding where meaning emerges from sparse activation patterns. Only ~2% of bits are active, enabling cheap overlap-based similarity and union-based composition.
- **Spiking Neural Networks (SNNs)** (Maass, 1997) — Third generation of neural networks that process information as temporal spike trains, inherently operating in a time-frequency domain. Extremely energy-efficient (event-driven computation) and potentially aligned with frequency-based semantic encoding.

### 4.4 Embedding Compression Techniques
- **Binary/Ternary Embeddings** — Extreme quantization reducing embeddings to {-1, 0, 1}, achieving 32x memory reduction with moderate quality loss.
- **Hash Embeddings** (Svenstrup et al., 2017) — Using hash functions to map tokens to representations without a full learned embedding table.
- **Product Quantization** — Decomposing embedding vectors into sub-vectors quantized independently, widely used in approximate nearest neighbor search.

### 4.5 MatMul-Free Language Modeling
- **Scalable MatMul-free Language Modeling** (Zhu et al., 2024) — Demonstrated that matrix multiplications can be eliminated from language model inference entirely using ternary weights and element-wise operations, achieving competitive performance at scale.

### 4.6 Broader LLM Efficiency Landscape

The following areas of LLM efficiency research provide context for why alternative embeddings matter, and inform potential integration points for future phases of this research:

- **Model Compression:** Quantization (4-bit, 2-bit), pruning (removing redundant parameters/layers), and knowledge distillation (training smaller models to mimic larger ones).
- **Inference Optimization:** Speculative decoding (small draft model accelerates large model), KV-cache optimization (PagedAttention, multi-query attention), and early exit strategies (stopping computation early for "easy" inputs).
- **Training Efficiency:** Parameter-efficient fine-tuning (LoRA, QLoRA, adapters), data efficiency (achieving good performance with less data), and curriculum learning (ordering training data strategically for faster convergence).
- **Architecture-Level:** Mixture of Experts (MoE, activating only a subset of parameters per input), and alternative architectures challenging the transformer paradigm (state-space models like Mamba, linear attention, RWKV).
- **Edge Deployment:** Running LLMs on mobile/embedded devices — a direct motivation for lighter embedding representations.

## 5. Proposed Approach

### 5.1 Core Hypothesis

Word semantics can be encoded as spectral signals where:
- **Frequency bands** decompose meaning into interpretable semantic axes (e.g., syntax at low frequencies, fine-grained semantics at high frequencies).
- **Phase relationships** encode relational properties (e.g., gender, tense, plurality).
- **Semantic composition** reduces to element-wise operations in the frequency domain, offering significant computational savings over matrix multiplication.
- **Sparsity in the frequency domain** (few dominant frequency components per word) enables natural compression.

### 5.2 Methodology

#### Track A — Frequency-Domain Embeddings (Primary Focus)

**Representation:**
Each word w is represented as a complex-valued spectral vector:

```
E(w) = FFT(s_w) = [A_1 * e^(iφ_1), A_2 * e^(iφ_2), ..., A_d * e^(iφ_d)]
```

Where A_k is the amplitude (importance of frequency component k) and φ_k is the phase (relational encoding).

**Semantic Operations:**
- **Similarity:** Spectral coherence or cross-spectral density between two word spectra. Could also use cross-correlation in the frequency domain as a distance metric.
- **Composition:** Element-wise multiplication in frequency domain (equivalent to convolution in "meaning space"). Cost: O(d) vs O(d^2) for matrix multiply. **This is the key efficiency win:** if meaning composition becomes element-wise multiplication instead of matrix multiplication, it represents a fundamental reduction in computational complexity.
- **Analogy (Phase Manipulation):** If semantic relationships correspond to systematic phase shifts, then `E(king) * e^(iΔφ) ≈ E(queen)` where Δφ encodes the gender transformation. This would be the frequency-domain equivalent of vector arithmetic — the emergent property test.
- **Frequency Band Manipulation:** Different frequency bands could encode different semantic axes (gender, royalty, animacy). Targeted manipulation of specific bands = targeted meaning changes. For example, modifying only the "gender band" of "king" could yield "queen" while preserving the "royalty band."

**Critical Research Question — Emergent Properties:**
The make-or-break question for this entire approach is whether frequency-domain representations naturally develop a **structured, compositional semantic space** — or whether they become an opaque encoding. In traditional embeddings, the linear structure of vector space gives rise to analogical reasoning. For frequency-domain embeddings, the equivalent compositional structure must emerge from:
- Phase relationships (systematic phase offsets encoding semantic relationships)
- Spectral distance (semantic similarity measured via spectral coherence)
- Band-level semantics (different frequency ranges capturing different aspects of meaning)

This must be empirically tested: train the representations, then systematically check if analogical reasoning, semantic clustering, and compositional operations emerge.

**Training — Three Paradigms:**

| Approach | Description | Purpose |
|----------|-------------|---------|
| **Baseline** | Standard Word2Vec/GloVe dense training | Reference point for quality and speed |
| **Transform** | Train dense embeddings, then apply FFT | Lower-bound; tests if dense embeddings have spectral structure |
| **Native** | Train directly in frequency domain with spectral loss | Main contribution |

**Native Training Pipeline:**
1. Define word representations as learnable spectral parameters (amplitudes + phases).
2. Adapt skip-gram objective: predict context words using spectral similarity.
3. Loss function: contrastive spectral coherence — maximize coherence with true context words, minimize with negative samples.
4. Backpropagation through FFT/IFFT operations (both are differentiable).
5. Spectral regularization: encourage sparse frequency representations (L1 penalty on amplitudes).

**Key Training Consideration — Backprop Through FFT:**
Fourier transforms are differentiable operations, so backpropagation is theoretically sound. However, convergence properties in frequency domain may differ from standard vector spaces. The gradient landscape of spectral parameters (amplitudes and phases) may exhibit different smoothness characteristics than dense vector gradients. This is an open research question and a potential contribution of this work.

**Hybrid Training Pipeline (C):**
A third training option — **Spectral Loss Regularization** — trains in a space optimized for both domains simultaneously. The loss function penalizes representations that don't have clean spectral structure, encouraging the model to learn naturally compressible, spectrally-organized representations. This could bridge approaches A and B by starting with dense training but guiding it toward frequency-domain-friendly representations.

#### Track B — Hyperdimensional Computing Embeddings

1. Encode words as high-dimensional binary vectors (~10,000 dimensions, randomly initialized).
2. Learn associations through Hebbian-style updates: words in similar contexts develop overlapping representations.
3. Composition via binding (XOR) and bundling (majority vote).
4. Analogy test: `bind(unbind(king, man), woman) ≈ queen`.

#### Track C — Sparse Distributed Representations

1. Encode words as sparse binary vectors (e.g., 2,048 dimensions, 40 active bits).
2. Similarity measured by overlap (number of shared active bits).
3. Composition through union with competitive inhibition (maintain target sparsity).
4. Train using a modified skip-gram where the objective maximizes overlap between co-occurring words.

#### Track D — Spiking / Temporal Encoding

1. Represent words as spike timing patterns over a population of neurons.
2. Similarity measured by spike train correlation.
3. Explore whether spike-timing-dependent plasticity (STDP) can serve as an unsupervised training mechanism for word representations.
4. Connection to frequency domain: spike trains have a natural frequency decomposition.
5. Potential collaboration with spiking neural network researchers.

### 5.3 Evaluation Framework

#### Semantic Quality Benchmarks
- **Word Analogy** — Google Analogy Dataset (19,544 questions), BATS (99,200 questions). Tests emergent compositional properties.
- **Word Similarity** — WordSim-353, SimLex-999, MEN-3000. Tests whether semantic distances are meaningful.
- **Qualitative Analysis** — Visualization of frequency-band semantic decomposition. What do individual frequency components capture?
- **Downstream Tasks** — Sentiment classification, Named Entity Recognition using the embeddings as features (frozen).

#### Efficiency Benchmarks
- **Memory footprint** — Bytes per word representation (including any overhead for complex/sparse formats).
- **Training time** — Wall-clock time to convergence on equivalent corpus (English Wikipedia).
- **Training compute** — Total FLOPs during training.
- **Inference: similarity computation** — FLOPs and wall-clock time for batch similarity queries.
- **Inference: composition** — FLOPs for combining multiple word representations.
- **Sparsity analysis** — For frequency representations: how many components are needed to retain 95%, 99% of quality?

#### Detailed Efficiency Comparison Framework

**Training Efficiency:**

| Aspect | Dense Embeddings | Frequency Domain |
|--------|-----------------|-----------------|
| Parameter count | O(V x d) — full vocabulary x dimension | Potentially lower if sparse in frequency domain (few dominant frequencies per word) |
| Gradient computation | Standard backpropagation, well-understood | Backpropagation through FFT — feasible but convergence properties less studied |
| Convergence speed | Well-characterized, extensive literature | Unknown — this is a key research contribution |
| Hardware requirements | GPU-optimized (dense matmul) | Potentially CPU-friendly (FFT is well-optimized on CPUs) |

**Inference Efficiency:**

| Aspect | Dense Embeddings | Frequency Domain |
|--------|-----------------|-----------------|
| Similarity computation | Cosine similarity O(d) | Spectral coherence — potentially faster for batch operations |
| Semantic composition | Matrix multiply O(d^2) | Element-wise multiply in frequency domain O(d) — **key speedup** |
| Memory per word | Full dense vector (d x 4 bytes for float32) | Potentially sparse (few dominant frequencies → fewer stored values) |
| Batch processing | Requires GPU for efficient batching | FFT highly optimized on both CPU and GPU |

## 6. Timeline (5 Months)

| Month | Phase | Activities | Deliverables |
|-------|-------|-----------|-------------|
| **1** | Literature Review & Infrastructure | Deep survey of spectral methods in NLP, HDC, SDR, and SNN literature; set up evaluation pipeline; implement Word2Vec/GloVe baselines | Survey document; working evaluation framework; baseline results |
| **2** | Transform Approach & Analysis | Apply FFT to trained dense embeddings; analyze spectral structure of existing embeddings; identify whether semantic properties survive transformation | Spectral analysis report; transform-based embedding evaluation results |
| **3** | Native Frequency Training | Design spectral loss function; implement native frequency-domain training pipeline; train on Wikipedia corpus; iterate on training dynamics | Working native frequency embedding system; initial quality results |
| **4** | Alternative Tracks | Implement and evaluate HDC (Track B), SDR (Track C), and preliminary SNN exploration (Track D); cross-track comparison | Results for all four tracks; comparative analysis |
| **5** | Analysis, Benchmarking & Writing | Complete efficiency benchmarks across all approaches; spectral decomposition visualization; thesis writing and presentation preparation | Final benchmark report; thesis document; presentation |

## 7. Expected Contributions

1. **A novel frequency-domain word embedding method** — Complete with training pipeline, open-source implementation, and reproducibility package.
2. **Empirical analysis of spectral semantic properties** — First systematic study of whether analogical reasoning and semantic similarity emerge in frequency-domain representations.
3. **Spectral decomposition of meaning** — Insights into what frequency bands capture about language (syntax vs. semantics at different frequency scales).
4. **Comprehensive cross-paradigm efficiency benchmarks** — Fair comparison of frequency, HDC, sparse, spiking, and dense representations on both quality and computational cost.
5. **Practical guidelines** — Recommendations for when each representation paradigm is most appropriate based on resource constraints and quality requirements.

## 8. Required Resources

- **Compute:** GPU access for training experiments. A single mid-range GPU (e.g., RTX 3080/4070) should suffice, as the focus is on efficiency and most experiments involve embedding-level models, not full transformers.
- **Data:** English Wikipedia dump (~3GB text) for training; standard NLP benchmarks for evaluation. All freely available.
- **Software:** PyTorch (primary framework), NumPy/SciPy (FFT operations), Gensim (Word2Vec/GloVe baselines), standard NLP evaluation toolkits.
- **Storage:** Minimal — embedding models are small compared to full LLMs.

## 9. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Frequency representations don't develop semantic structure | Medium | Transform approach (Track A, paradigm 2) provides fallback; HDC/SDR tracks provide alternative contributions |
| Native training doesn't converge | Medium | Start with transform approach as initialization; gradually shift to native training |
| Insufficient compute for full experiments | Low | All experiments are designed for single-GPU; embedding models are lightweight |
| Lack of novelty (prior work already covers this) | Low | Preliminary literature search shows no systematic study of spectral word embeddings |

## 10. Potential for Future Work

This project is scoped as the first phase of a broader master's research trajectory:
- **Phase 2:** Replace embedding layers in full transformer architectures with frequency-domain alternatives.
- **Phase 3:** Extend to contextual (dynamic) frequency representations — position-dependent spectral modulation.
- **Phase 4:** Integration with spiking neural network architectures for ultra-efficient, neuromorphic NLP.
- **Long-term vision:** A fully matrix-free language model built on spectral and/or graph-based representations.

### 10.1 Integration with Graph-Based Language Modeling

A parallel research track (see companion proposal: "Self-Organizing Neural Graph for Language Modeling") explores representing language as a dynamic probabilistic graph where words are nodes and relationships are weighted edges. A natural integration point between the two tracks:

- Each node's **local neighborhood** (edge weights to all connected words) can be treated as a signal.
- Applying FFT to a node's edge-weight vector produces a **spectral fingerprint** of that word's usage pattern.
- Two words with similar spectral fingerprints would be semantically similar.
- This bridges frequency-domain representations with graph-based associative memory, potentially creating a unified framework where the graph provides structure and the spectral analysis provides efficient similarity and composition operations.

### 10.2 Spiking Neural Network Connection

Potential supervisors have expertise in spiking neural network models. The connection to this work is direct:
- Spike trains are inherently temporal signals with natural frequency decompositions.
- Spike-timing-dependent plasticity (STDP) could serve as an unsupervised learning mechanism for spectral word representations.
- Neuromorphic hardware (e.g., Intel Loihi, IBM TrueNorth) is optimized for spike-based computation, offering a hardware-native deployment target for frequency-domain representations.
- Collaboration with SNN researchers could yield a biologically plausible training pipeline for spectral embeddings.

## 11. References

1. Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
2. Pennington, J. et al. (2014). "GloVe: Global Vectors for Word Representation." EMNLP.
3. Lee-Thorp, J. et al. (2021). "FNet: Mixing Tokens with Fourier Transforms." arXiv:2105.03824.
4. Tancik, M. et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions." NeurIPS.
5. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation." Cognitive Computation, 1(2), 139-159.
6. Ahmad, S. & Hawkins, J. (2016). "How do neurons operate on sparse distributed representations?" arXiv:1601.00720.
7. Zhu, R. et al. (2024). "Scalable MatMul-free Language Modeling." arXiv:2406.02528.
8. Maass, W. (1997). "Networks of Spiking Neurons: The Third Generation of Neural Network Models." Neural Networks, 10(9), 1659-1671.
9. Svenstrup, D. et al. (2017). "Hash Embeddings for Efficient Word Representations." NeurIPS.
10. Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT.

---

*This proposal investigates efficient alternatives to dense word embeddings, with frequency-domain representations as the primary research direction. The work is designed to produce publishable results within the 5-month timeline while laying groundwork for a matrix-free approach to language modeling.*

---

## Appendix A: Design Discussion Notes

The following notes capture key design decisions and insights from the initial research discussions that shaped this proposal.

### A.1 On Removing Matrices From the Equation

The overarching research vision is to question whether matrices — specifically dense matrix multiplications — are necessary for language modeling. Current LLMs are built entirely on matrix operations: embedding lookup (matrix indexing), attention (matrix multiply), feed-forward layers (matrix multiply), output projection (matrix multiply). Each of these is O(d^2) or worse.

This project starts at the foundation: the embedding layer. If we can represent word semantics without a dense V x d matrix, and if semantic operations (similarity, composition) can be performed without matrix multiplication, we open the door to matrix-free alternatives for the entire pipeline.

The frequency domain is attractive because the Fourier transform converts convolution (expensive) to multiplication (cheap). If semantic composition can be formulated as convolution in a meaning space, then operating in the frequency domain reduces it to element-wise multiplication — O(d) instead of O(d^2).

### A.2 On The Fundamental Importance of Emergent Properties

The emergent property of traditional embeddings — where vector arithmetic captures semantic relationships — is, in the author's view, the single most important property that enables modern NLP. Without `king - man + woman = queen`, embeddings would be just another encoding scheme.

Any alternative representation must be evaluated first and foremost on whether it develops comparable emergent compositionality. This project treats emergent semantic properties as a **primary evaluation criterion**, not a secondary one. If the frequency-domain representation doesn't develop compositional structure, it is merely an efficient encoding, not a semantic representation — and its utility for downstream tasks would be fundamentally limited.

### A.3 On The Training vs. Inference Efficiency Distinction

Efficiency must be evaluated separately for training and inference because the computational profiles are different:

**Training concerns:**
- Can gradients flow through the representation effectively?
- Does the loss landscape have good convergence properties?
- How many data points are needed to learn good representations?
- What hardware is required for training?

**Inference concerns:**
- How fast is similarity computation?
- How fast is composition of multiple word representations?
- What is the memory footprint per word?
- Can it run on CPUs, edge devices, mobile phones?

A representation could be efficient at inference but expensive to train (acceptable for many applications) or vice versa. The experimental design must capture both dimensions.

### A.4 On Why Frequency Domain (Not Just Another Compression)

There are many ways to compress embeddings: quantization, pruning, distillation, hash tricks. These are engineering optimizations of the same underlying representation. The frequency-domain approach is fundamentally different because it proposes a **different mathematical domain** for representing meaning — not a compressed version of the same domain.

The hypothesis is that the frequency domain isn't just more efficient — it may capture different aspects of meaning that dense vectors don't:
- **Multi-scale semantics:** Low frequencies capture broad semantic categories, high frequencies capture fine distinctions.
- **Phase-encoded relationships:** Systematic phase shifts between words could encode semantic transformations more naturally than vector arithmetic.
- **Natural sparsity:** If word meaning has a "spectral signature" with few dominant frequencies, the representation is naturally sparse without any compression algorithm.

### A.5 On the FFT O(n log n) Advantage

The Fast Fourier Transform operates in O(n log n), which is important for several reasons:
- Converting between time/meaning domain and frequency domain is cheap.
- FFT is one of the most optimized algorithms in computing — hardware-level implementations exist on every platform.
- Operations that are O(n^2) in the original domain (convolution) become O(n) in the frequency domain (element-wise multiplication).
- For batch operations (comparing many word pairs simultaneously), frequency-domain operations can be further parallelized.

The practical implication: even with the overhead of FFT/IFFT conversions, frequency-domain operations could be faster for composition-heavy tasks (e.g., processing full sentences) than dense vector operations.
