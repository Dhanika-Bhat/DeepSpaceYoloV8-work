# Phase 3: Quantum Circuit Ablation Study Report

Evaluation based on metrics proposed by Sim et al. (2019):
- **Expressibility (KL)**: Measures how close the state distribution is to Haar random. Lower values mean better coverage of Hilbert space.
- **Entangling Capability (MW)**: Measures the average Meyer-Wallach entanglement. Higher values mean stronger multi-qubit correlations.

| Qubits | Layers | Entanglement Topology | Expressibility (KL) | Entangling Capability (MW) |
|--------|--------|-----------------------|---------------------|----------------------------|
| 4 | 3 | full | 0.13325 | 0.74748 |
| 4 | 3 | nearest | 0.11574 | 0.84322 |
| 4 | 4 | full | 0.21930 | 0.79450 |
| 4 | 4 | nearest | 0.13013 | 0.81802 |
| 4 | 5 | full | 0.10793 | 0.81432 |
| 4 | 5 | nearest | 0.12535 | 0.82473 |
| 6 | 3 | full | 0.08173 | 0.84502 |
| 6 | 3 | nearest | 0.01581 | 0.94820 |
| 6 | 4 | full | 0.10510 | 0.89723 |
| 6 | 4 | nearest | 0.02375 | 0.95159 |
| 6 | 5 | full | 0.04686 | 0.92477 |
| 6 | 5 | nearest | 0.08217 | 0.95292 |
| 8 | 3 | full | 0.00562 | 0.87507 |
| 8 | 3 | nearest | 0.01126 | 0.97115 |
| 8 | 4 | full | 0.15733 | 0.91767 |
| 8 | 4 | nearest | 0.00562 | 0.98154 |
| 8 | 5 | full | 0.01126 | 0.95096 |
| 8 | 5 | nearest | 0.00562 | 0.98699 |
