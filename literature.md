What already exists in this direction
You're not alone in this intuition — some relevant prior work:

Complex-valued word embeddings (Trouillon et al., "Complex Embeddings for Simple Link Prediction") — used complex vectors for knowledge graph embeddings, where the imaginary part captures antisymmetric relations (e.g., "parent of" vs "child of")
Rotational embeddings (RotatE) — relations are modeled as rotations in complex space
Quantum-inspired NLP — represents words as density matrices in complex Hilbert spaces


Suggested learning path
To build this up step by step:

Foundations → Refresh on Euler's formula: eiθ=cos⁡θ+isin⁡θe^{i\theta} = \cos\theta + i\sin\theta
eiθ=cosθ+isinθ — this is the bridge between rotation, complex numbers, and frequencies

DFT basics → Understand how a discrete signal becomes a set of complex coefficients
Complex-valued neural networks — these exist and backprop works (you just need Wirtinger derivatives)
Experiment → Take a small vocabulary, train complex embeddings, apply FFT, and visualize the spectra
