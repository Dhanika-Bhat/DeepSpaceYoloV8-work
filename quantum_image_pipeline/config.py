import os

# Central configuration parameters for the Quantum Image Classification Pipeline

# --- Dataset and Directory Settings ---
DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DeepSpaceYoloDatasetV2", "DeepSpaceYoloDatasetV2")
)
# Output directories for reports and plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Preprocessing & Split Parameters (Phase 1) ---
IMAGE_SIZE = (64, 64)       # Default resolution. Can toggle between (64, 64) or (128, 128)
NORMALIZE_RANGE = (0.0, 1.0) # Normalization interval
NUM_CLASSES = 4             # Targeted label classes
SPLIT_RATIO = (0.70, 0.15, 0.15) # Train / Val / Test splitting ratio
RANDOM_STATE = 42

# Data Augmentation configurations
AUGMENTATION_CONFIG = {
    "rotation_range_deg": 15,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_jitter_range": (0.8, 1.2) # Jitter factor multiplier
}

# --- SMOTE Configurations (Phase 1) ---
SMOTE_STRATEGY = "auto"     # Resample all classes except majority class

# --- PCA Dimensions (Phase 2) ---
PCA_TARGET_VARIANCE = 0.85  # Target explained cumulative variance (>= 85%)

# --- Quantum Circuit and VQC Parameters (Phases 3 & 4) ---
QUBIT_ABLATION_LIST = [4, 6, 8] # Qubits to test/ablate
LAYER_ABLATION_LIST = [3, 4, 5] # Variational layers to test
ENTANGLEMENT_VARIANTS = ["full", "nearest"] # Variant (a) and (b) topologies

# --- Optimization and Training (Phase 4) ---
LEARNING_RATE = 0.05
TRAINING_EPOCHS = 15
BATCH_SIZE = 16
OPTIMIZER = "adam"          # Can select 'adam' or 'sgd'
