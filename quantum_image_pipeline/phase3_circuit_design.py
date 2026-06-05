import os
import numpy as np
import pennylane as qml
import config

def get_vqc_state_circuit(n_qubits, n_layers, entanglement):
    """
    Creates and returns a QNode that executes the VQC ansatz
    and returns the quantum state vector. Used for metric evaluation.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def state_circuit(inputs, weights):
        # State Prep: Angle Embedding using RX
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        
        # Variational Layers
        for l in range(n_layers):
            # Rotations
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
                
            # Entangling layers
            if entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "nearest":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Periodic boundary
                    
        return qml.state()
        
    return state_circuit

def get_vqc_expval_circuit(n_qubits, n_layers, entanglement):
    """
    Creates and returns a QNode that executes the VQC ansatz
    and returns Pauli-Z expectation values for all qubits. Used for classification.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def expval_circuit(inputs, weights):
        # State Prep: Angle Embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
        
        # Variational Layers
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
                
            if entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "nearest":
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
                    
        # Outputs: Pauli-Z expectation values for all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return expval_circuit

def get_reduced_density_matrix_numpy(state_vector, qubit_idx, n_qubits):
    """
    Computes the reduced density matrix of a single qubit by taking the
    partial trace over all other qubits using NumPy's einsum.
    """
    # Reshape the 2^n state vector to a tensor of shape (2, 2, ..., 2)
    psi = state_vector.reshape([2] * n_qubits)
    psi_conj = np.conj(psi)
    
    # Generate indices for einsum
    # We contract all indices except the target qubit_idx
    indices_psi = list(range(n_qubits))
    indices_psi_conj = list(range(n_qubits))
    
    # Rename target qubit's index in conjugate to create a 2x2 matrix output
    new_idx = n_qubits
    indices_psi_conj[qubit_idx] = new_idx
    
    # Character maps for einsum labels
    char_map = "abcdefghijklmnopqrstuvwxyz"
    sub_psi = "".join(char_map[i] for i in indices_psi)
    sub_psi_conj = "".join(char_map[i] for i in indices_psi_conj)
    sub_out = char_map[qubit_idx] + char_map[new_idx]
    
    einsum_str = f"{sub_psi},{sub_psi_conj}->{sub_out}"
    rho_k = np.einsum(einsum_str, psi, psi_conj)
    return rho_k

def compute_meyer_wallach_entanglement(state_vector, n_qubits):
    """
    Computes the Meyer-Wallach entanglement measure Q for a given state vector.
    Q ranges from 0 (product states) to 1 (maximally entangled states).
    """
    sum_tr_rho_sq = 0.0
    for k in range(n_qubits):
        rho_k = get_reduced_density_matrix_numpy(state_vector, k, n_qubits)
        # Trace of rho_k squared
        tr_rho_sq = np.real(np.trace(np.dot(rho_k, rho_k)))
        sum_tr_rho_sq += tr_rho_sq
        
    # Meyer-Wallach formula: Q = 2 * (1 - (1/n) * sum(Tr(rho_k^2)))
    Q = 2.0 * (1.0 - (1.0 / n_qubits) * sum_tr_rho_sq)
    return Q

def compute_expressibility_and_entanglement(n_qubits, n_layers, entanglement, num_samples=200):
    """
    Computes Expressibility (KL divergence from Haar fidelity) and
    Entangling Capability (mean Meyer-Wallach entanglement) according to Sim et al. (2019).
    """
    state_circuit = get_vqc_state_circuit(n_qubits, n_layers, entanglement)
    
    # We fix the input vector to zero to assess the expressibility of the ansatz weights
    inputs_fixed = np.zeros(n_qubits)
    
    # Sample random weights
    # weights shape: (n_layers, n_qubits, 2)
    weights_samples = np.random.uniform(0, 2 * np.pi, size=(num_samples, n_layers, n_qubits, 2))
    
    # 1. Compute Entangling Capability (Meyer-Wallach)
    mw_values = []
    state_vectors = []
    
    for i in range(num_samples):
        state_vec = state_circuit(inputs_fixed, weights_samples[i])
        # Convert Pennylane tensor to numpy array
        state_vec_np = np.array(state_vec)
        state_vectors.append(state_vec_np)
        
        q_val = compute_meyer_wallach_entanglement(state_vec_np, n_qubits)
        mw_values.append(q_val)
        
    entangling_capability = np.mean(mw_values)
    
    # 2. Compute Expressibility (Fidelity overlaps compared to Haar)
    fidelities = []
    # Sample pairs of state vectors to compute overlaps
    for i in range(num_samples // 2):
        v1 = state_vectors[2 * i]
        v2 = state_vectors[2 * i + 1]
        
        # Fidelity = |<v1|v2>|^2
        fidelity = np.abs(np.vdot(v1, v2)) ** 2
        fidelities.append(fidelity)
        
    fidelities = np.array(fidelities)
    
    # Bin the fidelities to get distribution P_PQC
    num_bins = 50
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_width = bins[1] - bins[0]
    
    pqc_hist, _ = np.histogram(fidelities, bins=bins, density=True)
    # Convert density to probability mass
    P_pqc = pqc_hist * bin_width
    
    # Compute Haar distribution
    # P_Haar(F) = (d-1) * (1-F)^(d-2) where d = 2^n
    d = 2 ** n_qubits
    P_haar_density = (d - 1) * ((1.0 - bin_centers) ** (d - 2))
    P_haar = P_haar_density * bin_width
    
    # Normalize distributions to sum to 1
    P_pqc = P_pqc / (np.sum(P_pqc) + 1e-12)
    P_haar = P_haar / (np.sum(P_haar) + 1e-12)
    
    # Compute KL divergence: sum( P(x) * log(P(x) / Q(x)) )
    eps = 1e-10
    kl_div = np.sum(P_pqc * np.log((P_pqc + eps) / (P_haar + eps)))
    
    return kl_div, entangling_capability

def run_vqc_ablation_study():
    """
    Executes the ablation study for circuit configurations:
    - Qubit count: 4, 6, 8
    - Variational layers: 3, 4, 5
    - Entanglement: 'full', 'nearest'
    Saves circuit layouts as text diagrams and compiles a performance report.
    """
    print("\n===========================================")
    print("PHASE 3: QUANTUM CIRCUIT DESIGN & ABLATION")
    print("===========================================")
    
    results = []
    
    print("Running Ablation on Circuit Topologies (Sim et al. 2019 Metrics):")
    print(f"{'Qubits':<8} | {'Layers':<8} | {'Entanglement':<13} | {'Expressibility (KL)':<22} | {'Entangling Cap (MW)':<22}")
    print("-" * 83)
    
    for q in config.QUBIT_ABLATION_LIST:
        for l in config.LAYER_ABLATION_LIST:
            for ent in config.ENTANGLEMENT_VARIANTS:
                # Run the evaluation
                kl_div, ent_cap = compute_expressibility_and_entanglement(q, l, ent, num_samples=100)
                results.append({
                    "qubits": q,
                    "layers": l,
                    "entanglement": ent,
                    "expressibility": kl_div,
                    "entangling_capability": ent_cap
                })
                print(f"{q:<8} | {l:<8} | {ent:<13} | {kl_div:<22.5f} | {ent_cap:<22.5f}")
                
    # Save the ablation results to a markdown file
    report_path = os.path.join(config.OUTPUT_DIR, "phase3_circuit_ablation_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase 3: Quantum Circuit Ablation Study Report\n\n")
        f.write("Evaluation based on metrics proposed by Sim et al. (2019):\n")
        f.write("- **Expressibility (KL)**: Measures how close the state distribution is to Haar random. Lower values mean better coverage of Hilbert space.\n")
        f.write("- **Entangling Capability (MW)**: Measures the average Meyer-Wallach entanglement. Higher values mean stronger multi-qubit correlations.\n\n")
        f.write("| Qubits | Layers | Entanglement Topology | Expressibility (KL) | Entangling Capability (MW) |\n")
        f.write("|--------|--------|-----------------------|---------------------|----------------------------|\n")
        for res in results:
            f.write(f"| {res['qubits']} | {res['layers']} | {res['entanglement']} | {res['expressibility']:.5f} | {res['entangling_capability']:.5f} |\n")
            
    print(f"\n--> Saved ablation reports to: {report_path}")
    
    # Save text drawings for representative circuits (e.g. 4 qubits, 3 layers)
    for ent in config.ENTANGLEMENT_VARIANTS:
        exp_circuit = get_vqc_expval_circuit(n_qubits=4, n_layers=3, entanglement=ent)
        # Dummy inputs and weights for drawing
        dummy_inputs = np.zeros(4)
        dummy_weights = np.zeros((3, 4, 2))
        
        drawer = qml.draw(exp_circuit)(dummy_inputs, dummy_weights)
        draw_path = os.path.join(config.OUTPUT_DIR, f"circuit_draw_4q_3l_{ent}.txt")
        with open(draw_path, "w") as fd:
            fd.write(drawer)
            
    print("--> Generated text drawings of circuits in the outputs folder.")
    return results

if __name__ == "__main__":
    # Test stub
    run_vqc_ablation_study()
