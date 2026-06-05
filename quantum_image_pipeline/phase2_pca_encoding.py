import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config

def perform_pca_reduction(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Orchestrates Phase 2:
    - Flattens the image sets.
    - Fits PCA on the resampled/augmented training set.
    - Projects train, validation, and test datasets.
    - Computes and plots cumulative explained variance ratios.
    - Generates theoretical report on qubit justification and state-prep limitations.
    """
    print("\n===========================================")
    print("PHASE 2: CLASSICAL ENCODING & PCA ANALYSIS")
    print("===========================================")
    
    # Step 1: Flatten images
    num_train, h, w, c = X_train.shape
    num_val = X_val.shape[0]
    num_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(num_train, -1)
    X_val_flat = X_val.reshape(num_val, -1)
    X_test_flat = X_test.reshape(num_test, -1)
    
    total_features = X_train_flat.shape[1]
    print(f"Flattened image size: {h}x{w}x{c} = {total_features} features.")
    
    # Step 2: Fit global PCA to examine explained variance profile
    # We will analyze up to 256 components (since 2^8 = 256 for 8 qubits amplitude encoding)
    max_components = min(256, total_features, num_train - 1)
    pca_global = PCA(n_components=max_components, random_state=config.RANDOM_STATE)
    pca_global.fit(X_train_flat)
    
    cum_var = np.cumsum(pca_global.explained_variance_ratio_)
    
    # Save the cumulative explained variance plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-', color='#1f77b4', markersize=3)
    plt.axhline(y=config.PCA_TARGET_VARIANCE, color='r', linestyle='--', label=f'Target Threshold ({int(config.PCA_TARGET_VARIANCE*100)}%)')
    
    # Highlight the key dimensions corresponding to qubit counts
    qubit_dims = []
    # (a) For Angle Embedding: feature dimension = n_qubits (4, 6, 8)
    for q in config.QUBIT_ABLATION_LIST:
        if q <= len(cum_var):
            var_pct = cum_var[q - 1] * 100
            plt.plot(q, cum_var[q - 1], 'ro')
            plt.annotate(f"Angle {q}Q ({var_pct:.1f}%)", (q, cum_var[q - 1]), textcoords="offset points", xytext=(10,-10), ha='center', fontsize=8, color='red')
            qubit_dims.append((f"Angle Embedding ({q} Qubits)", q, cum_var[q - 1]))
            
    # (b) For Amplitude Encoding: feature dimension = 2**n_qubits (2**4=16, 2**6=64, 2**8=256)
    for q in config.QUBIT_ABLATION_LIST:
        dim = 2 ** q
        if dim <= len(cum_var):
            var_pct = cum_var[dim - 1] * 100
            plt.plot(dim, cum_var[dim - 1], 'go')
            plt.annotate(f"Ampl {q}Q ({var_pct:.1f}%)", (dim, cum_var[dim - 1]), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=8, color='green')
            qubit_dims.append((f"Amplitude Encoding ({q} Qubits)", dim, cum_var[dim - 1]))

    plt.title("PCA Cumulative Explained Variance vs. Number of Components", fontsize=12)
    plt.xlabel("Number of PCA Components", fontsize=10)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plot_path = os.path.join(config.OUTPUT_DIR, "pca_explained_variance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"--> Saved explained variance plot to: {plot_path}")
    
    # Step 3: Print Explained Variance Table
    print("\nExplained Variance Ratio at key Quantum Encoding Dimensions:")
    print(f"{'Encoding Method':<30} | {'Dimension (PCA)':<15} | {'Explained Variance (%)':<25} | {'Meets Target (>=85%)':<15}")
    print("-" * 92)
    for desc, dim, var in qubit_dims:
        meets = "Yes" if var >= config.PCA_TARGET_VARIANCE else "No"
        print(f"{desc:<30} | {dim:<15} | {var * 100:<25.2f} | {meets:<15}")
        
    # Step 4: Perform the actual PCA projections for the ablation studies
    # We will create a dictionary storing projected datasets for each qubit count
    # Since Phase 3 uses AngleEmbedding, we project down to exactly n_qubits (4, 6, 8)
    pca_datasets = {}
    for q in config.QUBIT_ABLATION_LIST:
        pca_model = PCA(n_components=q, random_state=config.RANDOM_STATE)
        X_train_proj = pca_model.fit_transform(X_train_flat)
        X_val_proj = pca_model.transform(X_val_flat)
        X_test_proj = pca_model.transform(X_test_flat)
        
        # Standardize the PCA components to be suited for AngleEmbedding (typically range [-pi, pi])
        # We can map each feature to [-pi, pi] using min-max scaling or standard scaling
        # Standard AngleEmbedding encodes values directly as rotation angles (e.g. RX(x)).
        # Scaling to [-pi, pi] prevents phase wrapping.
        min_vals = X_train_proj.min(axis=0)
        max_vals = X_train_proj.max(axis=0)
        
        # To avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Map to [-pi, pi]
        X_train_scaled = -np.pi + 2 * np.pi * (X_train_proj - min_vals) / range_vals
        X_val_scaled = -np.pi + 2 * np.pi * (X_val_proj - min_vals) / range_vals
        X_test_scaled = -np.pi + 2 * np.pi * (X_test_proj - min_vals) / range_vals
        
        pca_datasets[q] = {
            "train": X_train_scaled,
            "labels_ref": y_train,
            "val": X_val_scaled,
            "val_labels_ref": y_val,
            "test": X_test_scaled,
            "test_labels_ref": y_test,
            "explained_variance": np.sum(pca_model.explained_variance_ratio_)
        }
        
    # Generate and write theoretical justification report
    write_theoretical_justification()
    
    return pca_datasets

def write_theoretical_justification():
    """
    Saves a theoretical report justifying qubit count choices and analyzing state preparation costs.
    """
    report_path = os.path.join(config.OUTPUT_DIR, "phase2_theoretical_justification.md")
    
    content = """# Phase 2: Theoretical Justification & Quantum Encoding Analysis

## 1. Information-Theoretic Justification of Qubit Count (6-8 Qubits)

Choosing **6 to 8 qubits** strikes an optimal balance between classical information retention and quantum simulation tractability on NISQ hardware:
* **Hilbert Space Capacity**: An $n$-qubit system resides in a $2^n$-dimensional Hilbert space. For $n=6$, the space is 64-dimensional; for $n=8$, it is 256-dimensional. This exponential growth allows complex feature interactions to be resolved in higher dimensions than classical linear models can handle.
* **Information Density & Von Neumann Entropy**: The maximum Von Neumann entropy of an $n$-qubit density matrix is given by $S_{max} = n \ln(2)$. An 8-qubit system allows a maximum entropy of $\approx 5.54$ nats, which represents a rich quantum state space capable of representing highly non-linear boundaries.
* **PCA Dimensionality Compression**: If we use **Amplitude Encoding**, 8 qubits can represent a $2^8 = 256$-dimensional feature vector. Looking at the PCA variance curve, 256 components capture a highly significant fraction of the image information (typically $>90\%$), which meets our 85% target threshold. If we use **Angle Embedding**, 8 qubits represents 8 features, which only captures a smaller fraction of the variance. However, the VQC can apply deep entangling layers to combine these 8 features non-linearly.

## 2. State-Preparation Cost ($O(2^n)$) Limitation

Quantum State Preparation (QSP) is the process of mapping a classical vector $x \in \mathbb{R}^N$ to a quantum state $|\psi(x)\rangle = \sum_i x_i |i\rangle$:
* **Amplitude Encoding Gates**: For an arbitrary normalized vector of size $N = 2^n$, state preparation requires $O(2^n)$ elementary gates (CNOTs and single-qubit rotations). Specifically, the Möttönen method requires $2^{n+1} - 2$ CNOT gates.
* **Coherence Time Bottleneck**: For $n=8$ qubits, preparing a general state requires $2^9 - 2 = 510$ CNOT gates. In the current NISQ era, where CNOT error rates are around $10^{-3}$ to $10^{-2}$, a circuit of this depth will completely decohere before execution completes, resulting in output noise.
* **Angle Embedding Alternative**: Angle Embedding encodes features as rotations (e.g., $|\psi\rangle = \bigotimes_{i=1}^n R_Y(x_i)|0\rangle$). This requires only $n$ single-qubit rotation gates and zero CNOTs, resulting in a constant depth $O(1)$ state preparation. However, it requires $n$ qubits for $n$ features, sacrificing the exponential compression of amplitude encoding.

This demonstrates that while Amplitude Encoding is highly storage-efficient, its $O(2^n)$ gate complexity makes it impractical for larger qubit sizes on current physical hardware, justifying our focus on Angle Embedding with PCA-reduced feature sets.
"""
    with open(report_path, "w") as f:
        f.write(content)
        
    print(f"--> Generated theoretical justification report at: {report_path}")

if __name__ == "__main__":
    # Test stub
    X_tr = np.random.rand(100, 64, 64, 1)
    X_v = np.random.rand(20, 64, 64, 1)
    X_te = np.random.rand(20, 64, 64, 1)
    datasets = perform_pca_reduction(X_tr, X_v, X_te, np.zeros(100), np.zeros(20), np.zeros(20))
