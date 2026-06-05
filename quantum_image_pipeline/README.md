# Quantum Image Classification Pipeline

This repository contains a modular, end-to-end **Quantum Machine Learning (QML) Pipeline** designed for classifying deep space objects (DSOs) or general image datasets into 4 classes. It implements classical preprocessing, class balancing via SMOTE, PCA dimensionality reduction, variational quantum circuit (VQC) design and evaluation using Sim et al. (2019) metrics, and QNN training.

---

## 📁 Repository Structure

```
quantum_image_pipeline/
├── config.py                 # Central configurations (image size, splits, qubits, path)
├── phase1_preprocessing.py   # Preprocessing, augmentation, splitting, and SMOTE balancing
├── phase2_pca_encoding.py     # PCA analysis, explained variance ratio reporting, and theoretical justification
├── phase3_circuit_design.py  # VQC architectures, expressibility & entangling metrics, ablation study
├── phase4_training.py        # Parameter optimization, VQC training loop, evaluation metrics
├── main.py                   # End-to-end pipeline orchestrator
├── requirements.txt          # Python dependencies
└── README.md                 # Detailed walkthrough, theoretical writeups, and running instructions
```

---

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install all classical and quantum library dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Modify `config.py` to customize the dataset paths, image dimensions, qubit counts, and optimization hyperparameters:
- `IMAGE_SIZE`: Choose resolution, e.g., `(64, 64)` or `(128, 128)`.
- `QUBIT_ABLATION_LIST`: Array of qubit sizes to test, e.g., `[4, 6, 8]`.
- `TRAINING_EPOCHS`: Adjust number of training runs.

### 3. Run Pipeline End-to-End
Execute the orchestrator script to run all 4 phases:
```bash
python main.py
```
> [!NOTE]
> If a local YOLO dataset is not found in the parent directory, the code automatically falls back to a synthetic galaxy generator to demonstrate pipeline functionality out-of-the-box.

---

## 📖 Theoretical Documentation

### Phase 1: Preprocessing, Imbalance Correction & Augmentation
* **Imbalance Correction**: Since dataset classes are often highly imbalanced, we flatten the images to 1D vectors and apply **SMOTE (Synthetic Minority Over-sampling Technique)** on the training split. SMOTE draws synthetic samples along the line segments joining k-nearest neighbors in feature space, restoring class balance.
* **Image Augmentation**: To prevent overfitting, resampled training samples are reshaped back to 2D and augmented using rotation jitter ($\pm15^{\circ}$), horizontal/vertical flips, and brightness scaling.
* **Data Split**: Standardized $70/15/15$ split for train, validation, and test datasets.

### Phase 2: Classical PCA & Information-Theoretic Justification
* **PCA explained variance ratio**: Principle Component Analysis (PCA) maps the high-dimensional flattened images ($D = 64 \times 64 = 4096$) to a low-dimensional space suited for quantum embedding. The pipeline generates cumulative explained variance plots to track if the target $\ge85\%$ explained variance is met.
* **Qubit Count Justification**: Selecting 6-8 qubits is justified by:
  - **Hilbert Space Size**: $n$ qubits provide $2^n$ dimensions ($2^6 = 64$, $2^8 = 256$), which can fit high-dimensional features.
  - **Von Neumann Entropy**: Represents the maximum information content of the mixed state $\rho$. An $n$-qubit state has a maximum entropy of $S = n \ln 2$, which is highly dense.
* **State Preparation Cost Bottleneck**:
  - **Amplitude Encoding** packs $2^n$ features into $n$ qubits, but preparing the state requires $O(2^n)$ depth, making it prone to decoherence on noisy (NISQ) hardware.
  - **Angle Embedding** requires $n$ qubits for $n$ features but has $O(1)$ preparation depth (single-qubit rotations only), rendering it much more robust for current hardware.

### Phase 3: Variational Quantum Circuit (VQC) Design
We implement the VQC using `AngleEmbedding` on $n$ qubits, followed by layers containing parameterized $RY$ and $RZ$ rotations and CNOT gates.
* **Entanglement Variants**:
  1. **Variant (a): Full Entanglement**: Applies CNOT gates between all pairs of qubits in each layer.
  2. **Variant (b): Nearest-Neighbor Entanglement**: Applies CNOT gates between adjacent qubits $i$ and $i+1$ in a periodic 1D chain.
* **Sim et al. (2019) Metrics**:
  - **Expressibility (KL)**: Measures how uniform the circuit's output states are over the Haar measure. We calculate the KL-divergence between the fidelity distribution of random pairs of states and the Haar distribution $P_{Haar}(F) = (2^n - 1)(1 - F)^{2^n - 2}$. Lower KL values indicate greater expressibility.
  - **Entangling Capability (MW)**: Computes the average Meyer-Wallach entanglement measure $Q(\psi) = 2 \left( 1 - \frac{1}{n} \sum_{k=0}^{n-1} \text{Tr}(\rho_k^2) \right)$ over random weights. Higher values indicate stronger multi-qubit correlations.

### Phase 4: Training & Evaluation
* **Hybrid Architecture**: Outputs Pauli-Z expectation values from the VQC and feeds them into a classical linear layer with softmax activation to calculate 4-class probabilities.
* **Optimization**: Multi-class Cross-Entropy loss is minimized. The parameters (VQC weights + classical weights/biases) are optimized jointly using the `qml.AdamOptimizer`.
* **Visual Reports**: Training curves (loss/accuracy) and confusion matrices are generated and saved automatically to the `outputs/` folder.
