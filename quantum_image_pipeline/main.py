import os
import sys
import numpy as np
import config
from phase1_preprocessing import preprocess_and_balance_data
from phase2_pca_encoding import perform_pca_reduction
from phase3_circuit_design import run_vqc_ablation_study
from phase4_training import train_and_evaluate_vqc

def main():
    """
    Executes the entire QML Pipeline end-to-end:
    - Phase 1: Preprocessing, splitting, SMOTE balancing, image augmentation.
    - Phase 2: PCA feature extraction, explained variance reporting, and theoretical documentation.
    - Phase 3: VQC architecture expressibility and entangling capability ablation.
    - Phase 4: Hybrid QNN training and classification metric reporting.
    """
    print("=" * 60)
    print("QUANTUM IMAGE CLASSIFICATION PIPELINE - END-TO-END RUN")
    print("=" * 60)
    
    # ----------------------------------------------------
    # Phase 1: Data Load, Split, SMOTE, and Augmentation
    # ----------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_and_balance_data()
    
    # ----------------------------------------------------
    # Phase 2: Dimensionality Reduction via PCA
    # ----------------------------------------------------
    pca_datasets = perform_pca_reduction(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # ----------------------------------------------------
    # Phase 3: Quantum Circuit Design & Ablation Studies
    # ----------------------------------------------------
    ablation_results = run_vqc_ablation_study()
    
    # ----------------------------------------------------
    # Phase 4: Model Training and Performance Evaluation
    # ----------------------------------------------------
    # We train a representative QNN using:
    # - 6 qubits (provides good feature representation with 6 PCA components)
    # - 3 layers (reasonable parameters to train quickly)
    # - Nearest-neighbor entanglement (highly practical for NISQ devices)
    selected_qubits = 6
    selected_layers = 3
    selected_entanglement = "nearest"
    
    metrics = train_and_evaluate_vqc(
        pca_datasets,
        n_qubits=selected_qubits,
        n_layers=selected_layers,
        entanglement=selected_entanglement
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)
    print(f"All reports, diagrams, and figures have been saved to: {config.OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
