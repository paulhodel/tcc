# Master's Research Project — Matrix-Free Language Modeling

## Owner
Paulo Hodel — Master's in AI, Phase 1 (5-month research period starting Feb 2026)

## Project Goal
Investigate alternatives to dense matrix multiplications in LLMs, starting with the embedding layer and progressing toward a fully matrix-free, multimodal language model.

## Research Tracks

### Track A — Frequency-Domain Embeddings (`proposal_frequency_domain.md`)
- Words as spectral signals (frequencies, amplitudes, phases)
- Composition via element-wise multiplication in frequency domain (O(d) vs O(d²))
- Emergent semantic properties through phase relationships
- Comparison with HDC, Sparse Distributed Representations, Spiking encodings
- Training: baseline (Word2Vec/GloVe), transform (dense→FFT), native (spectral loss)

### Track B — Self-Organizing Neural Graph (`proposal_graph_neural_model.md`)
- Words as nodes, relationships as weighted edges (edges = weights, nodes = neurons)
- Inference via spreading activation (no matrix multiply, pointer traversal)
- Typed edges: sequence, property, semantic, role
- Property edges enable context-dependent meaning (e.g., "kind king" vs "bad king")
- Organic growth: vocabulary and architecture emerge from data
- Backprop adapted for cyclic graphs (DEQ / Almeida-Pineda / truncated)
- Three scoring approaches: weighted activation sum, path scoring, subgraph matching

### Breakthrough Vision — Resonance Model (`breakthrough_resonance_model.md`)
- Unifies Track A + Track B: graph = medium, spectral representations = wave functions
- Words as collections of fundamental frequencies (semantic primitives, ~100-200)
- Sentences as standing waves (superposition)
- Prediction as resonance matching (O(d), no matrices)
- Medium has global frequency ("mood") that modulates all computation
- Mood bands (delta/theta/alpha/beta/gamma) control different processing aspects
- Native multimodal: sound, vision, text are ALL waves — same medium, natural cross-modal resonance
- Hardware paths: optical computing, neuromorphic chips, analog oscillator networks
- Connected to Wierzbicka's Natural Semantic Metalanguage (~65 universal primes)

## Key Documents
- `proposal_frequency_domain.md` — Detailed proposal for Track A
- `proposal_graph_neural_model.md` — Detailed proposal for Track B
- `research_roadmap.md` — Full master's plan (embedding → attention → FF → output replacement)
- `breakthrough_resonance_model.md` — Visionary resonance-based semantic computing framework
- `experiments/` — Experiment folders organized by date (YYYY-MM-DD_description/)

## Experiment Structure
Experiments are organized in `experiments/` with folders named by date:
```
experiments/
  2026-02-XX_baseline_word2vec/
  2026-02-XX_spectral_analysis/
  ...
```
Each experiment folder should contain:
- `README.md` — What was tested, hypothesis, results summary
- `code/` — Scripts and notebooks
- `results/` — Output data, plots, metrics
- `notes.md` — Observations, next steps

## Technical Context
- Primary language: Python
- Frameworks: PyTorch, NumPy/SciPy (FFT), NetworkX (graphs), Gensim (baselines)
- Evaluation benchmarks: WordSim-353, SimLex-999, MEN-3000, Google Analogy, BATS
- Training data: English Wikipedia, Penn Treebank, WikiText-2/103
- Hardware: Single GPU for baselines; CPU-native for graph experiments

## Supervisor Context
- Potential supervisors work on spiking neural network models
- SNN connection: spreading activation ≈ spike propagation, Hebbian ≈ STDP
- Neuromorphic hardware (Intel Loihi, IBM TrueNorth) as deployment target

## Key Principles
1. Exhaust each direction independently before combining
2. Emergent semantic properties first, efficiency second
3. Always evaluate: (a) semantic quality, (b) training efficiency, (c) inference efficiency
4. The "king - man + woman = queen" test is the minimum bar for any representation
5. Track experiments by date for progress tracking
