# Research Proposal: Self-Organizing Neural Graph for Language Modeling — A Dynamic, Matrix-Free Architecture

**Program:** Master's in Artificial Intelligence
**Duration:** 5 months
**Author:** [Your Name]
**Date:** February 2026

---

## 1. Introduction

Current language models are built on fixed architectures: a predetermined number of layers, attention heads, and embedding dimensions are chosen before training begins. The model's structure never changes — only its weights are updated. This rigidity means that vocabulary size, model capacity, and computational cost are all locked in at design time.

This project proposes a fundamentally different paradigm: a **self-organizing neural graph** where the architecture itself grows and adapts during training. Words are represented as nodes, semantic relationships as weighted edges, and inference is performed through spreading activation across the graph rather than matrix multiplication. The graph learns both **what to connect** (structure) and **how strongly** (weights) simultaneously, with a novel backpropagation mechanism adapted for cyclic graph structures.

This approach eliminates the need for fixed embedding matrices, attention layers, and the matrix multiplications that dominate modern LLM computation.

### 1.1 Core Motivation

The key motivation is to **not start with a fixed neural network for training**. In current systems, we pre-define the number of words (vocabulary), the embedding dimensions, and the network topology — then train within those constraints. This project asks: what if the model could grow organically?

The idea is to create **nodes that link to other nodes through memory-wise references (pointers)**, which are inherently fast at the hardware level. One word links to all other related words; those direct links carry pre-calculated softmax probabilities that update every time a new word or connection is strengthened or weakened. Stronger links encode higher transition frequencies (next-word likelihood), while weaker links represent rare or context-dependent transitions.

Beyond direct links, we can compute relationships at the **second, third, and higher degrees** — all depending on context. This creates a graph of relationships where, during inference, we construct a tree from the network: neurons that fire together activate a coherent branch, and from that branch we apply final scoring to predict the next word.

### 1.2 Research Context

This project is part of a broader master's research program on **LLM efficiency**. It represents the second research track, running in parallel with an investigation into frequency-domain word representations (see companion proposal). Together, these tracks aim to establish foundations for a fully matrix-free language model.

## 2. Problem Statement

Modern language models suffer from three fundamental constraints:

1. **Fixed vocabulary:** The embedding matrix has a predetermined number of rows. Adding new words requires retraining or workarounds (subword tokenization is a symptom of this limitation, not a solution).

2. **Fixed architecture:** The number of layers, dimensions, and connections is set before training. The model cannot grow its capacity where needed or shrink where redundant.

3. **Matrix multiplication dependency:** All core operations — embedding lookup, attention, feed-forward layers — rely on dense matrix multiplications, which scale as O(d^2) or O(n^2 * d) and require specialized hardware (GPUs/TPUs).

**Core question:** Can we design a language model where the architecture grows organically with the data, where inference is performed through graph traversal rather than matrix multiplication, and where the graph structure itself is trained through a form of backpropagation?

## 3. Research Objectives

1. **Design a self-organizing neural graph architecture** where words are nodes, relationships are weighted edges, and the graph grows as new words and contexts are encountered.

2. **Develop typed, property-aware edges** that encode not just transition probabilities but contextual semantic properties — enabling the same word to participate in fundamentally different prediction pathways depending on context.

3. **Formalize a spreading activation inference mechanism** where context words trigger a wave of activation through the graph, and the intersection of activated pathways predicts the next word.

4. **Adapt backpropagation for cyclic graphs** — develop a training algorithm that updates edge weights (the "neural network" aspect) through the graph structure, handling cycles via equilibrium-based or truncated propagation methods.

5. **Benchmark against traditional approaches** in terms of prediction quality, training efficiency, inference speed, and memory usage.

## 4. Background and Related Work

### 4.1 Graph-Based Language Models
- **Word Co-occurrence Graphs** — Representing word relationships as graphs is well-studied in NLP (TextRank, word co-occurrence networks), but these are typically used for analysis, not as the primary model for generation.
- **Knowledge Graphs** (Bordes et al., 2013) — Structured representations of entity relationships (TransE, RotatE). These use fixed typed relations and embedding-based scoring, unlike the dynamic probabilistic structure proposed here.

### 4.2 Graph Neural Networks (GNNs)
- **Message Passing Neural Networks** (Gilmer et al., 2017) — Nodes aggregate information from neighbors through learned edge functions. Related to our spreading activation, but GNNs operate on fixed graph structures, not growing ones.
- **Graph Attention Networks (GAT)** (Velickovic et al., 2018) — Attention-weighted message passing on graphs. Relevant but still requires matrix multiplications for attention computation.

### 4.3 Self-Organizing and Growing Networks
- **Growing Neural Gas** (Fritzke, 1995) — Self-organizing network that adds and removes nodes based on input distribution. Demonstrates that network structure can emerge from data.
- **Neural Architecture Search (NAS)** (Zoph & Le, 2017) — Automated discovery of network architecture. The graph growing in our proposal can be seen as a continuous, data-driven NAS.
- **Progressive Neural Networks** (Rusu et al., 2016) — Networks that grow by adding new columns for new tasks, preserving previous knowledge.

### 4.4 Spreading Activation and Cognitive Models
- **Spreading Activation Theory** (Collins & Loftus, 1975) — Cognitive model of human semantic memory where activating one concept spreads activation to related concepts. The theoretical foundation for our inference mechanism.
- **Sparse Distributed Memory** (Kanerva, 1988) — Associative memory model operating in high-dimensional binary space. Related to our graph structure in its distributed, content-addressable nature.

### 4.5 Training on Cyclic Graphs
- **Recurrent Backpropagation** (Almeida, 1987; Pineda, 1987) — Backpropagation algorithm for networks with cycles, based on finding equilibrium points. Directly applicable to training our graph.
- **Deep Equilibrium Models (DEQ)** (Bai et al., 2019) — Modern reformulation: find the fixed point of an implicit layer and differentiate through it using implicit differentiation. Handles cycles without unrolling.
- **Truncated Backpropagation Through Time (TBPTT)** — Practical approximation: limit gradient propagation to K steps. Applicable to our graph traversal depth.

### 4.6 Hebbian and Local Learning Rules
- **Hebbian Learning** (Hebb, 1949) — "Neurons that fire together wire together." Our edge strengthening mechanism is fundamentally Hebbian.
- **Spike-Timing-Dependent Plasticity (STDP)** — Temporal Hebbian rule in biological neural networks. Relevant to ordering-sensitive edge updates (word A before word B).
- **Spiking Neural Networks** — Third-generation neural networks that process temporal spike patterns. The proposed graph with activation dynamics shares structural similarities with SNN architectures.

## 5. Proposed Architecture

### 5.1 Core Components

#### 5.1.1 Nodes (Words as Computational Units)

Each node in the graph represents a word (or subword/token) and contains:

```
Node {
    id:             unique identifier
    token:          the word/subword string
    activation:     current activation level (float, 0.0 to 1.0)
    compute_fn:     local transformation function (learnable)
    edges_out:      list of outgoing edges with weights
    edges_in:       list of incoming edges
    properties:     contextual property connections
    created_at:     timestamp (for tracking growth)
}
```

**The node as a mini neural network:** Each node has a learnable `compute_fn` — a small transformation (e.g., a single-layer perceptron or a nonlinear activation with learnable parameters) that transforms incoming activation into outgoing activation. This means each node is not just a passive container but an active computational element.

#### 5.1.2 Edges (Relationships as Weights)

Edges are the core of the system — they ARE the weights:

```
Edge {
    source:         source node id
    target:         target node id
    weight:         connection strength (float, learnable)
    edge_type:      [sequence | property | semantic | role]
    context_vector: small learnable vector encoding edge semantics
    decay_rate:     how quickly this edge loses weight without reinforcement
}
```

**Edge types:**
- **Sequence edges:** word A is followed by word B (transition probability).
- **Property edges:** word A modifies the meaning of word B ("kind" modifies "king").
- **Semantic edges:** word A and word B share meaning (synonymy, hypernymy).
- **Role edges:** word A plays a grammatical role relative to word B (subject-of, object-of).

#### 5.1.3 Properties (Context-Dependent Meaning)

Properties are the mechanism by which context changes prediction:

```
"The kind king is ___"

Activated nodes: [the] → [kind] → [king] → [is]

[kind] --property--> [king] activates the "kind+king" context
    This context fires different edges than bare [king]:
    [king|kind] --sequence--> [generous: 0.15, loved: 0.12, merciful: 0.10, ...]
    [king|bare] --sequence--> [powerful: 0.11, crowned: 0.09, feared: 0.08, ...]
```

Properties are implemented as **activation modifiers**: when a property node is active simultaneously with a target node, it shifts which outgoing edges are favored. This is achieved through a gating mechanism:

```
effective_weight(edge) = edge.weight * gate(property_activations, edge.context_vector)
```

Where `gate` is a learnable function that modulates edge weights based on active properties.

### 5.2 Inference: Spreading Activation

Inference follows a **fire-and-converge** pattern:

#### Step 1: Seed Activation
Input words activate their corresponding nodes:
```
input = "The kind king is"
activate([the], [kind], [king], [is])  → activation = 1.0 for each
```

#### Step 2: Propagation (K rounds)
For each round k = 1 to K:
```
For each active node n:
    For each outgoing edge e from n:
        target_activation += n.activation * e.effective_weight
        target_activation = target.compute_fn(target_activation)
    Apply activation decay to n (prevents runaway activation)
```

Activation spreads outward from seed nodes, diminishing with distance.

#### Step 3: Convergence
After K rounds (or when activation changes fall below threshold), the graph reaches a stable state. Active nodes form a **context-specific subgraph** — the model's "understanding" of the input.

#### Step 4: Tree Extraction and Prediction
From the final activated state:
1. Collect all nodes reachable via sequence edges from the last input word.
2. Rank by accumulated activation.
3. Apply softmax over candidate activations:

```
P(next_word = w) = softmax(activation(w)) for all candidate w
```

**The tree metaphor:** The activated subgraph forms a tree-like structure (though the underlying graph has cycles). Each branch represents a possible continuation, weighted by accumulated activation. "Neurons that fire together" converge on the same branches, reinforcing the most contextually appropriate predictions.

#### Step 5: Disambiguation Through Activation Overlap

Context words create overlapping activation regions. Disambiguation emerges naturally:

```
Query: "The bank of the river"

[bank] fires → activates two neighborhoods:
    Financial: [money, account, loan, interest, ...]
    Geographic: [river, shore, slope, erosion, ...]

[river] fires → activates:
    Geographic: [water, bank, shore, flow, fish, ...]

Intersection: [shore, erosion, water, ...] — geographic meaning wins
Financial nodes receive weak/no reinforcement → decay
```

The key insight: **context as a subgraph activation pattern**. The recent words activate local neighborhoods in the graph, and prediction comes from the intersection/overlap of activated regions. No explicit disambiguation mechanism is needed — it emerges from the graph structure.

#### Inference Complexity
- **Per step:** O(E_active) where E_active is the number of edges from active nodes (sparse).
- **Total:** O(K * E_active) — linear in propagation depth and active edge count.
- **No matrix multiplication.** All operations are scalar multiplications and additions along edges.
- **Memory-native:** This maps naturally to how CPUs actually work — pointer chasing and cache-line access patterns. Graph traversal is CPU-friendly, potentially eliminating the need for GPU hardware entirely.

### 5.3 Training: Backpropagation on a Living Graph

#### 5.3.1 The Training Loop

```
For each sentence in corpus:
    1. GROW:   Add any new words as nodes; add new edges for unseen transitions
    2. FORWARD: Run spreading activation (Section 5.2) to predict next word
    3. LOSS:    Cross-entropy between predicted distribution and actual next word
    4. BACKWARD: Propagate gradients back through the activated subgraph
    5. UPDATE:  Adjust edge weights, node compute_fn parameters
    6. DECAY:   Apply weight decay to all edges (unused edges weaken)
    7. PRUNE:   Remove edges below minimum weight threshold
```

#### 5.3.2 Why Backpropagation, Not Just Statistical Updates

A simpler version of this system would only use frequency-based updates: strengthen edges that appear in the data, weaken those that don't. This would essentially be a sophisticated Markov chain. The decision to use **backpropagation through the graph** is deliberate and critical:

1. **Edge weights are literally neural network weights.** The edges between nodes serve the same role as weights in a neural network — they transform and route information. Training them via gradient descent (not just frequency counting) allows the system to optimize for prediction quality, not just co-occurrence statistics.

2. **Nodes are computational units.** Each node has a learnable `compute_fn` that transforms activation. These functions can only be trained via gradient-based optimization — Hebbian updates alone cannot optimize arbitrary nonlinear transformations.

3. **The graph is a circular/cyclic network.** Unlike feedforward networks, activation can flow in loops (word A → word B → word C → word A). This circular structure requires specialized backpropagation methods — but it also enables the model to capture reciprocal and recursive relationships that feedforward architectures cannot.

4. **Gradient-based training discovers indirect relationships.** Backprop can strengthen an edge between two words that never co-occur directly, if strengthening that edge reduces prediction loss through multi-hop paths. Frequency-based updates cannot discover these indirect relationships.

The theoretical foundation for this is solid: backpropagation on cyclic networks has been studied since Pineda (1987) and Almeida (1987), and modern Deep Equilibrium Models (Bai et al., 2019) provide practical, convergent methods for exactly this setting.

#### 5.3.3 Handling Cycles: Three Approaches

**Approach A — Deep Equilibrium (Primary)**
- During forward pass, iterate activation propagation until convergence (fixed point).
- During backward pass, use implicit differentiation at the fixed point.
- Mathematically exact; no need to unroll cycles.
- Based on DEQ (Bai et al., 2019) and recurrent backpropagation (Pineda, 1987).

**Approach B — Truncated Propagation**
- Limit activation to K hops (K = 3-5 in practice).
- Approximate gradients via truncated backpropagation through the K steps.
- Simpler to implement; faster but less exact.

**Approach C — Hebbian + Backprop Hybrid**
- Use Hebbian learning (strengthen co-activated edges) for structural learning.
- Use backpropagation for fine-tuning edge weights and node compute functions.
- Biologically plausible; may converge faster due to good Hebbian initialization.

#### 5.3.4 Organic Growth Protocol

The graph starts empty and grows as follows:

| Event | Action |
|-------|--------|
| New word encountered | Create new node; initialize with random compute_fn |
| New bigram (A, B) encountered | Create sequence edge A→B; initialize weight small (0.01) |
| Word A modifies word B in context | Create property edge A→B; weight proportional to frequency |
| Edge weight drops below threshold | Prune edge (synaptic pruning) |
| Node has no remaining edges | Remove node (dead neuron removal) |

**No predetermined vocabulary size.** The graph's capacity is determined by the data, not by architecture choices.

### 5.4 The Neural Network Equivalence

The key insight: **this graph IS a neural network**, just not a traditional one.

| Traditional NN | Neural Graph |
|---------------|-------------|
| Fixed layers | Dynamic depth (propagation hops) |
| Weight matrices | Edge weights |
| Neurons (fixed count) | Nodes (growing count) |
| Forward pass through layers | Spreading activation through graph |
| Backprop through layers | Backprop through activated subgraph |
| Architecture designed by human | Architecture emerges from data |

The edges between nodes are literally the weights. The nodes are literally the neurons (with learnable activation functions). The difference is that the topology is dynamic and data-dependent.

### 5.5 The Final Scoring Math: Three Candidate Approaches

The "final math to predict the next word" — the computation that converts the activated subgraph into a probability distribution — is a critical design choice. Three approaches are proposed for comparative evaluation:

#### Option A: Weighted Activation Sum
```
P(next_word = w) = softmax( Σ activation(node_i) × edge_weight(node_i → w) )
```
- **Pros:** Simple, fast O(K) where K is number of active nodes with edges to w.
- **Cons:** May lose higher-order interaction information.
- **Best for:** Baselines and fast inference settings.

#### Option B: Path Scoring
```
P(next_word = w) = Σ over all paths leading to w ( Π edge_weights along path )
```
- **Pros:** Captures multi-hop reasoning — a word reachable via multiple strong paths scores higher.
- **Cons:** Exponential number of paths in dense graphs; requires approximation (top-K paths or beam search).
- **Best for:** Capturing deep contextual dependencies.

#### Option C: Subgraph Matching (Most Novel)
Score candidate words by how well their **local neighborhood overlaps** with the activated tree:
```
P(next_word = w) ∝ |neighborhood(w) ∩ activated_subgraph| × strength_of_overlap
```
- **Pros:** Most novel approach; essentially pattern matching on graph topology. A word whose local context matches the current activation pattern scores highest.
- **Cons:** Computationally more expensive; requires efficient subgraph comparison.
- **Best for:** Capturing semantic coherence beyond simple co-occurrence.

All three approaches will be implemented and compared on quality and efficiency metrics.

### 5.6 Scaling Considerations

#### Edge Density
If every word links to every other word, the graph contains O(V^2) edges, which is prohibitive for large vocabularies. This is addressed through:

1. **Sparse initialization:** Only create edges for observed co-occurrences, not all possible pairs.
2. **Weight decay and pruning:** Edges that are never reinforced weaken and are removed (synaptic pruning). This is biologically plausible.
3. **Threshold-based storage:** Only edges above a minimum weight threshold are stored in memory.
4. **Dynamic equilibrium:** The graph naturally reaches a balance between edge creation (new co-occurrences) and edge pruning (decay), maintaining manageable density.

#### The Softmax Update Problem
Every time one edge is strengthened from a node, all outgoing edges from that node must be re-normalized (softmax update). For a node with K outgoing edges, this is O(K) per update. This is manageable because:
- The graph is sparse (K << V for most nodes after pruning).
- Updates can be batched and amortized.
- Lazy normalization: only re-normalize when queried, not on every update.

### 5.7 What This Architecture Is NOT

It is important to distinguish this approach from superficially similar systems:

| System | Similarity | Key Difference |
|--------|-----------|---------------|
| **Markov Chains** | Transition probabilities between states | Markov chains are memoryless (1st order); our graph captures multi-hop context via spreading activation |
| **N-gram Models** | Co-occurrence statistics | N-grams have fixed context windows; our graph captures variable-depth context through activation propagation |
| **Knowledge Graphs** | Typed relations between entities | Knowledge graphs have fixed relation types and use embedding-based scoring; our graph grows organically and uses spreading activation |
| **Standard GNNs** | Message passing on graphs | GNNs operate on fixed graph structures; our graph grows during training |
| **Probabilistic Automata** | State transitions with probabilities | Our nodes are computational units (mini NNs), not passive states |

**The fundamental difference:** In a Markov chain, `P(next | current)` is a lookup. In our system, `P(next | context)` is computed by **activating multiple nodes simultaneously** and letting the activation pattern converge — capturing interactions between context words that simple co-occurrence statistics cannot.

## 6. Evaluation Plan

### 6.1 Language Modeling Quality

- **Perplexity** on held-out test sets (Penn Treebank, WikiText-2, WikiText-103).
- **Next-word prediction accuracy** (top-1, top-5, top-10).
- **Comparison baselines:** N-gram models (2-gram, 3-gram, 5-gram with Kneser-Ney smoothing), Word2Vec + simple predictor, small LSTM, small Transformer.

### 6.2 Semantic Properties

- **Word similarity** — Extract node-to-node distances from graph; compare with human similarity judgments (WordSim-353, SimLex-999).
- **Analogical reasoning** — Test whether graph path patterns capture analogies. For example: the shortest path from "king" to "queen" should structurally resemble the path from "man" to "woman." Semantic similarity can be measured via **shared subgraph structure**: king → [crown, throne, rule] and queen → [crown, throne, rule] should share significant subgraph overlap, encoding their semantic relatedness through graph topology rather than vector proximity.
- **Context-dependent meaning** — Test whether property edges correctly disambiguate: "bank" in financial vs. river contexts should activate different subgraphs. This is a key test for whether the property-edge mechanism provides true contextual understanding.
- **Higher-degree semantic emergence** — The hypothesis is that meaning emerges from **higher-degree paths**, not just direct links. Direct links (1st degree) capture co-occurrence. The 2nd and 3rd degree connections — the "friends of friends" in the graph — are where genuine semantic relationships should appear. This is analogous to how graph-based recommendation systems discover latent preferences through indirect connections.

### 6.3 Efficiency Benchmarks

| Metric | Measurement |
|--------|-------------|
| **Inference latency** | Wall-clock time per next-word prediction |
| **Inference FLOPs** | Operations per prediction (no matmuls — count scalar ops) |
| **Training throughput** | Words processed per second |
| **Memory usage** | Total bytes for graph (nodes + edges) vs. equivalent embedding matrix |
| **Growth dynamics** | Nodes and edges over time; pruning rate; convergence of graph structure |
| **Hardware** | CPU-only performance (graph traversal doesn't need GPU) |

### 6.4 Structural Analysis

- **Graph topology** — Degree distribution, clustering coefficient, small-world properties. Does the semantic graph exhibit known properties of human semantic networks?
- **Property edge analysis** — Visualization of how property edges reshape prediction. Qualitative examples of context-dependent activation patterns.
- **Interpretability** — For any prediction, trace the exact activation path from input to output. Full explainability by design.

### 6.5 Comparative Analysis with Existing Paradigms

| Aspect | Neural Embeddings | This Graph Model |
|--------|------------------|-----------------|
| Vocabulary | Fixed at training time | Grows organically, no limit |
| Training | Batch processing, epochs, requires GPU | Incremental, real-time updates, CPU-native |
| Inference | Matrix multiplication | Pointer traversal + scalar operations |
| Memory | Dense V x d matrix (all words same size) | Sparse adjacency + weights (varies per word) |
| Interpretability | Opaque (black box) | Fully inspectable activation paths |
| Semantic compositionality | Vector arithmetic | Subgraph overlap / path analysis |
| Adding new words | Requires retraining or heuristics | Simply add a node, immediate integration |
| Context handling | Fixed context window (attention) | Variable depth via propagation hops |

## 7. Timeline (5 Months)

| Month | Phase | Activities | Deliverables |
|-------|-------|-----------|-------------|
| **1** | Design & Foundations | Formalize graph data structure; implement node/edge system; implement basic spreading activation; literature deep-dive on DEQ and recurrent backprop | Working graph infrastructure; basic activation demo |
| **2** | Growth & Basic Training | Implement organic growth protocol; implement Hebbian edge updates; train on small corpus (PTB); evaluate basic next-word prediction | Growing graph trained on PTB; initial perplexity numbers |
| **3** | Backpropagation & Properties | Implement DEQ-based backprop for cyclic graph; add property edges and context gating; train on larger corpus (WikiText) | Full training pipeline; property-aware predictions |
| **4** | Scaling & Optimization | Optimize graph traversal (sparse data structures, cache-friendly memory layout); train on Wikipedia; full benchmark suite | Optimized system; complete benchmarks |
| **5** | Analysis & Writing | Graph topology analysis; interpretability case studies; comparison with baselines; thesis writing; presentation preparation | Final results; thesis document; presentation |

## 8. Expected Contributions

1. **A novel self-organizing neural graph architecture** for language modeling that grows its own structure during training — the first architecture where both topology and weights are learned simultaneously from raw text.

2. **Property-aware contextual edges** — A mechanism for context-dependent prediction without attention mechanisms or matrix multiplications.

3. **Backpropagation on dynamic cyclic graphs** — Adaptation of equilibrium-based gradient methods to growing graph structures with typed edges.

4. **Spreading activation as inference** — Formalization and empirical validation of spreading activation as a practical inference mechanism for language modeling.

5. **Complete interpretability** — Every prediction can be traced through exact activation paths, offering full transparency that black-box neural networks cannot provide.

6. **CPU-native language modeling** — A model that performs inference through pointer traversal and scalar operations, potentially running efficiently on CPUs without GPU requirements.

## 9. Required Resources

- **Compute:** Primarily CPU-based (graph traversal is CPU-friendly). GPU useful for baseline comparisons only. A standard multi-core workstation should suffice.
- **Data:** Penn Treebank, WikiText-2/103, English Wikipedia. All freely available.
- **Software:** Python, NetworkX or custom graph library, PyTorch (for backprop infrastructure and baselines), memory profiling tools.
- **Storage:** Graph storage requirements are a research question — expected to be manageable for vocabulary-scale graphs.

## 10. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Spreading activation doesn't converge | Medium | Implement activation decay and max-iteration limits; fallback to truncated propagation (Approach B) |
| Quality significantly below neural baselines | High (initially) | Focus narrative on efficiency and interpretability advantages; position as first step, not final system |
| Memory explosion from dense connectivity | Medium | Aggressive pruning; sparse data structures; edge weight threshold for storage |
| Backprop through cycles is unstable | Medium | Start with Hebbian-only training; gradually introduce backprop; use DEQ's proven convergence guarantees |
| Graph traversal too slow in practice | Low-Medium | Cache-friendly memory layout; pre-computed activation indices; batch processing of independent queries |

## 11. Connection to Broader Research Vision

This project represents the second phase of a research program aimed at **matrix-free language modeling**:

- **Phase 1 (parallel project):** Frequency-domain word representations — efficient alternatives to dense embedding matrices.
- **Phase 2 (this project):** Self-organizing neural graph — efficient alternative to fixed neural network architectures.
- **Future Phase 3:** Integration — frequency-domain representations as node features within the neural graph, creating a fully matrix-free, dynamically growing language model.

The intersection with **spiking neural networks** is particularly promising: the spreading activation mechanism proposed here shares structural similarities with spike propagation in biological neural networks. Collaboration with SNN researchers could yield a biologically plausible, hardware-efficient architecture suitable for neuromorphic chips.

### 11.1 Connection to Spiking Neural Networks

Potential supervisors have expertise in spiking neural network (SNN) models. The connection to this work is direct and deep:

- **Spreading activation ≈ Spike propagation:** The proposed activation spreading through the graph is structurally analogous to spike propagation in biological neural networks. Spiking models process information as temporal spike trains — our nodes firing and propagating activation through weighted edges mirrors this mechanism.
- **Hebbian learning ≈ STDP:** Our edge strengthening (co-occurring words strengthen their link) is a direct analog of spike-timing-dependent plasticity in SNNs.
- **Event-driven computation:** Both SNNs and our graph model are inherently event-driven — computation only happens where activation is present, not across the entire network. This is fundamentally more efficient than dense matrix operations.
- **Neuromorphic hardware target:** Neuromorphic chips (Intel Loihi, IBM TrueNorth) are designed for exactly this kind of sparse, event-driven, graph-structured computation. Our model could potentially run natively on neuromorphic hardware.
- **Potential collaboration:** Working with SNN researchers could accelerate this project by leveraging their expertise in temporal dynamics, stability analysis, and efficient implementation of spike-like computation.

### 11.2 Integration with Frequency-Domain Representations

A natural integration point exists between this graph model and the companion frequency-domain research:

- Each node's **local neighborhood** (the pattern of edge weights to all connected words) can be treated as a discrete signal.
- Applying FFT to a node's edge-weight vector produces a **spectral fingerprint** — a compact frequency-domain representation of how that word relates to the rest of the vocabulary.
- This spectral fingerprint could serve as the node's compute_fn input, bridging graph-structural semantics with efficient spectral operations.
- The long-term vision: a self-organizing graph where structural relationships provide the knowledge representation, and frequency-domain operations provide efficient similarity and composition — a fully matrix-free, dynamically growing language model.

## 12. References

1. Collins, A. & Loftus, E. (1975). "A Spreading-Activation Theory of Semantic Processing." Psychological Review, 82(6), 407-428.
2. Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
3. Hebb, D. (1949). "The Organization of Behavior." Wiley.
4. Pineda, F. (1987). "Generalization of Back-Propagation to Recurrent Neural Networks." Physical Review Letters, 59(19), 2229.
5. Almeida, L. (1987). "A Learning Rule for Asynchronous Perceptrons with Feedback in a Combinatorial Environment." IEEE ICNN.
6. Fritzke, B. (1995). "A Growing Neural Gas Network Learns Topologies." NeurIPS.
7. Bai, S. et al. (2019). "Deep Equilibrium Models." NeurIPS.
8. Gilmer, J. et al. (2017). "Neural Message Passing for Quantum Chemistry." ICML.
9. Velickovic, P. et al. (2018). "Graph Attention Networks." ICLR.
10. Bordes, A. et al. (2013). "Translating Embeddings for Modeling Multi-relational Data." NeurIPS.
11. Zoph, B. & Le, Q. (2017). "Neural Architecture Search with Reinforcement Learning." ICLR.
12. Rusu, A. et al. (2016). "Progressive Neural Networks." arXiv:1606.04671.
13. Maass, W. (1997). "Networks of Spiking Neurons: The Third Generation of Neural Network Models." Neural Networks, 10(9), 1659-1671.
14. Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.

---

*This proposal introduces a self-organizing neural graph architecture that challenges the fixed-architecture paradigm of current language models. By treating the graph structure as both the data representation and the computational substrate — with edges as weights and nodes as neurons — it offers a path toward dynamic, interpretable, CPU-native language modeling. The 5-month timeline focuses on proving the core mechanism and establishing baselines for future scaling.*

---

## Appendix A: Design Discussion Notes

The following notes capture key design decisions and insights from the initial research discussions that shaped this proposal.

### A.1 On the Relationship Between Graph Density and Pruning

Initial concern: if every word links to every other word, the graph becomes O(V^2) — unmanageable. Two solutions were considered:

1. **Pruning (passive):** Weak links are removed based on weight threshold. This is biologically analogous to synaptic pruning.
2. **Backpropagation (active):** Instead of only pruning, use gradient-based learning to actively adjust all edge weights. This is fundamentally different because backprop can *strengthen* previously weak edges if doing so improves predictions — something that frequency-based pruning alone cannot do.

The decision to use backpropagation rather than relying solely on pruning transforms this from a statistical model into a true neural architecture. The links between nodes are not just statistics — they are learned parameters optimized for prediction quality.

### A.2 On the "Circular Network" Challenge

The graph contains cycles (e.g., "king" → "crown" → "royal" → "king"). This is not a bug — it's a feature. Circular paths encode reciprocal semantic relationships. The challenge is training through these cycles.

Three established methods address this:
- **Almeida-Pineda algorithm (1987):** Original solution for recurrent network training.
- **Deep Equilibrium Models (2019):** Modern, scalable approach using implicit differentiation.
- **Truncated BPTT:** Practical approximation limiting gradient flow depth.

All three have proven convergence properties, making the circular network trainable, not just theoretically but practically.

### A.3 On Emergent Semantic Properties

The most critical question for this research: does the graph develop emergent semantic properties comparable to vector arithmetic in word embeddings?

The hypothesis is that **semantics emerge from graph topology**, not from vector geometry:
- **Similarity:** Two words with similar neighborhoods (shared connections) are semantically similar — measured by subgraph overlap rather than cosine distance.
- **Analogy:** The path structure from "king" to "queen" should resemble the path from "man" to "woman" — same topological transformation, different starting points.
- **Composition:** "kind" + "king" activates a specific subgraph that is different from "cruel" + "king" — compositional meaning from intersection of activation patterns.

This is fundamentally different from vector-space semantics, and whether it works is the primary research question.

### A.4 On Pre-Computed Softmax and Incremental Updates

Each node maintains a pre-computed softmax distribution over its outgoing edges. When new data arrives:
1. The relevant edge weight is incremented.
2. The softmax is re-normalized for that node (O(K) where K = outgoing edges).
3. During backprop, gradient updates adjust weights and trigger re-normalization.

This means the model always has up-to-date probability distributions without requiring a full training epoch. Combined with the organic growth protocol, this enables **truly online, incremental learning** — each sentence immediately updates the model, no batching required.

### A.5 On Hardware Efficiency

Traditional neural networks are optimized for GPU execution (dense parallel matrix operations). This graph model is optimized for **CPU execution**:
- **Pointer traversal:** Following edges is pointer chasing — native CPU operation.
- **Scalar operations:** Edge weight multiplication and activation updates are scalar ops.
- **Cache locality:** Nodes and their edges can be stored contiguously in memory, exploiting CPU cache lines.
- **No GPU dependency:** This model could run on any CPU, including embedded processors, without the memory bandwidth requirements of matrix-heavy models.

This is a significant practical advantage: deploying language models without GPU requirements dramatically reduces infrastructure costs.
