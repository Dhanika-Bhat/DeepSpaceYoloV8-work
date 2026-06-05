import os
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import config
from phase3_circuit_design import get_vqc_expval_circuit

def softmax(x):
    """Numerically stable softmax."""
    # If 2D (batch_size, classes)
    if len(x.shape) == 2:
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        shift_x = x - np.max(x)
        exps = np.exp(shift_x)
        return exps / np.sum(exps)

def hybrid_qnn_predict(inputs, vqc_weights, class_w, class_b, expval_qnode):
    """
    Computes logits and class probabilities using VQC output and a classical linear layer.
    """
    batch_size = inputs.shape[0]
    n_qubits = inputs.shape[1]
    
    # Get expectation values from VQC for all inputs in the batch
    expvals = []
    for i in range(batch_size):
        # expval_qnode returns list of expectations
        ev = expval_qnode(inputs[i], vqc_weights)
        expvals.append(ev)
        
    expvals = pnp.stack(expvals) # shape: (batch_size, n_qubits)
    
    # Classical linear layer projection: (batch_size, n_qubits) @ (n_qubits, 4) + (4,)
    logits = pnp.dot(expvals, class_w) + class_b
    
    # Apply softmax to get probabilities
    probs = softmax(logits)
    return probs

def cross_entropy_loss(labels, probs):
    """Computes categorical cross-entropy loss."""
    batch_size = len(labels)
    # One-hot encode labels
    one_hot = pnp.zeros((batch_size, config.NUM_CLASSES))
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
        
    # Categorical cross entropy
    # Add a small epsilon to prevent log(0)
    eps = 1e-15
    loss = -pnp.sum(one_hot * pnp.log(probs + eps)) / batch_size
    return loss

def train_and_evaluate_vqc(pca_datasets, n_qubits=6, n_layers=3, entanglement="nearest"):
    """
    Orchestrates Phase 4:
    - Sets up the VQC classifier using AngleEmbedding and specified layer/entanglement settings.
    - Initialises hybrid network parameters (VQC weights + classical linear layer).
    - Runs the training loop using Pennylane's AdamOptimizer.
    - Computes and plots validation losses and accuracies.
    - Runs evaluation on test data, reporting Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    print(f"\n===========================================")
    print(f"PHASE 4: VQC TRAINING & EVALUATION ({n_qubits} Qubits)")
    print(f"===========================================")
    
    # Retrieve projected data for this qubit count
    train_data = pca_datasets[n_qubits]["train"]
    train_labels = pca_datasets[n_qubits]["labels_ref"]  # Reference labels
    
    val_data = pca_datasets[n_qubits]["val"]
    val_labels = pca_datasets[n_qubits]["val_labels_ref"]
    
    test_data = pca_datasets[n_qubits]["test"]
    test_labels = pca_datasets[n_qubits]["test_labels_ref"]
    
    print(f"Training shapes: X={train_data.shape}, y={train_labels.shape}")
    print(f"Validation shapes: X={val_data.shape}, y={val_labels.shape}")
    
    # Instantiate the expectation value circuit
    expval_qnode = get_vqc_expval_circuit(n_qubits, n_layers, entanglement)
    
    # Initialize parameters as PennyLane numpy arrays (with requires_grad=True)
    # VQC weights: (layers, qubits, 2)
    vqc_weights = pnp.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
    # Classical weight projection: (qubits, classes)
    class_w = pnp.random.normal(0, 0.1, size=(n_qubits, config.NUM_CLASSES), requires_grad=True)
    # Classical bias: (classes,)
    class_b = pnp.zeros(config.NUM_CLASSES, requires_grad=True)
    
    # Pack parameters into a list for optimization
    params = [vqc_weights, class_w, class_b]
    
    # Set up optimizer
    opt = qml.AdamOptimizer(stepsize=config.LEARNING_RATE)
    
    # Metrics logging
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    batch_size = config.BATCH_SIZE
    epochs = config.TRAINING_EPOCHS
    
    def cost_func(p, x_batch, y_batch):
        probs = hybrid_qnn_predict(x_batch, p[0], p[1], p[2], expval_qnode)
        return cross_entropy_loss(y_batch, probs)
        
    print("\nStarting Hybrid Quantum-Classical Training Loop...")
    for epoch in range(epochs):
        # Shuffle training set each epoch
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        
        # Batch training
        epoch_loss = 0.0
        num_batches = int(np.ceil(len(train_data) / batch_size))
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            x_batch = train_data[batch_idx]
            y_batch = train_labels[batch_idx]
            
            # Perform optimization step
            # Note: We must pass x_batch and y_batch to opt.step as extra args
            params = opt.step(lambda p: cost_func(p, x_batch, y_batch), params)
            
        # Compute metrics at the end of epoch
        # (Using batches to avoid memory overhead for validation/training prediction)
        train_probs = hybrid_qnn_predict(train_data, params[0], params[1], params[2], expval_qnode)
        t_loss = cross_entropy_loss(train_labels, train_probs)
        t_preds = np.argmax(train_probs, axis=1)
        t_acc = accuracy_score(train_labels, t_preds)
        
        val_probs = hybrid_qnn_predict(val_data, params[0], params[1], params[2], expval_qnode)
        v_loss = cross_entropy_loss(val_labels, val_probs)
        v_preds = np.argmax(val_probs, axis=1)
        v_acc = accuracy_score(val_labels, v_preds)
        
        train_losses.append(float(t_loss))
        val_losses.append(float(v_loss))
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.1f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.1f}%")
        
    # --- Evaluation on Test Set ---
    test_probs = hybrid_qnn_predict(test_data, params[0], params[1], params[2], expval_qnode)
    test_preds = np.argmax(test_probs, axis=1)
    
    # Calculate performance metrics
    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')
    conf_mat = confusion_matrix(test_labels, test_preds)
    
    print("\n" + "="*30)
    print("FINAL TEST METRICS")
    print("="*30)
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"Confusion Matrix:\n{conf_mat}")
    
    # Save training history curves plot
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='#ff7f0e', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', color='#2ca02c', marker='o')
    plt.plot(range(1, epochs + 1), val_accs, label='Val Acc', color='#d62728', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    curves_path = os.path.join(config.OUTPUT_DIR, f"vqc_training_curves_{n_qubits}q.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"\n--> Saved training curve plots to: {curves_path}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({n_qubits} Qubits VQC)')
    plt.colorbar()
    tick_marks = np.arange(config.NUM_CLASSES)
    plt.xticks(tick_marks, [f"Class {i}" for i in range(config.NUM_CLASSES)])
    plt.yticks(tick_marks, [f"Class {i}" for i in range(config.NUM_CLASSES)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add values text to cells
    thresh = conf_mat.max() / 2.
    for i, j in np.ndindex(conf_mat.shape):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
                 
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUT_DIR, f"vqc_confusion_matrix_{n_qubits}q.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"--> Saved confusion matrix plot to: {cm_path}")
    
    # Save report summary
    report_path = os.path.join(config.OUTPUT_DIR, f"phase4_evaluation_report_{n_qubits}q.txt")
    with open(report_path, "w") as f:
        f.write("VQC Classifier Performance Report\n")
        f.write("=================================\n")
        f.write(f"Configuration: {n_qubits} Qubits, {n_layers} Layers, {entanglement} Entanglement\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test Precision: {precision*100:.2f}%\n")
        f.write(f"Test Recall: {recall*100:.2f}%\n")
        f.write(f"Test F1-Score: {f1*100:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n{np.array2string(conf_mat)}\n")
        
    print(f"--> Saved metrics report to: {report_path}")
    
    return {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    # Test script in isolated mode
    # Create fake PCA data
    fake_datasets = {
        6: {
            "train": np.random.uniform(-np.pi, np.pi, (40, 6)),
            "labels_ref": np.random.randint(0, 4, 40),
            "val": np.random.uniform(-np.pi, np.pi, (10, 6)),
            "val_labels_ref": np.random.randint(0, 4, 10),
            "test": np.random.uniform(-np.pi, np.pi, (10, 6)),
            "test_labels_ref": np.random.randint(0, 4, 10)
        }
    }
    train_and_evaluate_vqc(fake_datasets, n_qubits=6)
