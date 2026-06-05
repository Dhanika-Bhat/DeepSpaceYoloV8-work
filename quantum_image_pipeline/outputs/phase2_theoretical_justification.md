# Phase 2: Theoretical Justification & Quantum Encoding Analysis

## 1. Information-Theoretic Justification of Qubit Count (6-8 Qubits)

Choosing **6 to 8 qubits** strikes an optimal balance between classical information retention and quantum simulation tractability on NISQ hardware:
* **Hilbert Space Capacity**: An $n$-qubit system resides in a $2^n$-dimensional Hilbert space. For $n=6$, the space is 64-dimensional; for $n=8$, it is 256-dimensional. This exponential growth allows complex feature interactions to be resolved in higher dimensions than classical linear models can handle.
* **Information Density & Von Neumann Entropy**: The maximum Von Neumann entropy of an $n$-qubit density matrix is given by $S_{max} = n \ln(2)$. An 8-qubit system allows a maximum entropy of $pprox 5.54$ nats, which represents a rich quantum state space capable of representing highly non-linear boundaries.
* **PCA Dimensionality Compression**: If we use **Amplitude Encoding**, 8 qubits can represent a $2^8 = 256$-dimensional feature vector. Looking at the PCA variance curve, 256 components capture a highly significant fraction of the image information (typically $>90\%$), which meets our 85% target threshold. If we use **Angle Embedding**, 8 qubits represents 8 features, which only captures a smaller fraction of the variance. However, the VQC can apply deep entangling layers to combine these 8 features non-linearly.

## 2. State-Preparation Cost ($O(2^n)$) Limitation

Quantum State Preparation (QSP) is the process of mapping a classical vector $x \in \mathbb{R}^N$ to a quantum state $|\psi(x)angle = \sum_i x_i |iangle$:
* **Amplitude Encoding Gates**: For an arbitrary normalized vector of size $N = 2^n$, state preparation requires $O(2^n)$ elementary gates (CNOTs and single-qubit rotations). Specifically, the Möttönen method requires $2^{n+1} - 2$ CNOT gates.
* **Coherence Time Bottleneck**: For $n=8$ qubits, preparing a general state requires $2^9 - 2 = 510$ CNOT gates. In the current NISQ era, where CNOT error rates are around $10^{-3}$ to $10^{-2}$, a circuit of this depth will completely decohere before execution completes, resulting in output noise.
* **Angle Embedding Alternative**: Angle Embedding encodes features as rotations (e.g., $|\psiangle = igotimes_{i=1}^n R_Y(x_i)|0angle$). This requires only $n$ single-qubit rotation gates and zero CNOTs, resulting in a constant depth $O(1)$ state preparation. However, it requires $n$ qubits for $n$ features, sacrificing the exponential compression of amplitude encoding.

This demonstrates that while Amplitude Encoding is highly storage-efficient, its $O(2^n)$ gate complexity makes it impractical for larger qubit sizes on current physical hardware, justifying our focus on Angle Embedding with PCA-reduced feature sets.
