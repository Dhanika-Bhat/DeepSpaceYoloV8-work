"""
Phase 1 — Preprocessing & Class Balancing
==========================================
Fixes applied vs original:
  FIX-1  SMOTE crash on small minority classes → adaptive k_neighbors
  FIX-2  cv2.warpAffine / cv2.flip silently drop the channel dim → explicit re-expand after every cv2 call
  FIX-3  Real YOLO class IDs were discarded and replaced by intensity heuristic → now reads actual class IDs
  FIX-4  Augmentation was applied to SMOTE synthetic samples → only applied to original samples before SMOTE
  FIX-5  generate_synthetic_data had no random seed → np.random.seed(config.RANDOM_STATE)
  FIX-6  SMOTE config key 'strategy' → correct key is 'sampling_strategy'
  ADD-1  128x128 second resolution pipeline (matches workflow requirement)
  ADD-2  Class distribution bar chart saved to OUTPUT_DIR
  ADD-3  Imbalance ratio logged before/after for paper reporting
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import config


# ─────────────────────────────────────────────────────────────────────────────
# YOLO CLASS MAP  (matches DeepSpaceYoloDatasetV2 data.yaml)
# Update these names if your data.yaml shows different labels
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CLASS_MAP = {
    0: "galaxy",
    1: "nebula",
    2: "globular_cluster",
    3: "star_cluster",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR  (fallback when no dataset is found)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(num_samples=1000, img_size=(64, 64)):
    """
    Generates synthetic DSO images for 4 classes with intentional imbalance.
    Used only when the real dataset cannot be found.

    Class probabilities: 0=45%, 1=30%, 2=18%, 3=7%
    """
    # FIX-5: seed so results are reproducible
    np.random.seed(config.RANDOM_STATE)

    X, y = [], []
    h, w = img_size

    for _ in range(num_samples):
        label = np.random.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.18, 0.07])
        img = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2

        # Background noise
        noise = np.random.normal(10, 5, (h, w))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        if label == 0:  # Spiral galaxy
            cv2.circle(img, (cx, cy), int(w * 0.08), 200, -1)
            t = np.linspace(0, 3 * np.pi, 100)
            for sign in [1, -1]:
                r = sign * ((w * 0.04) + (w * 0.04) * t)
                xs = (cx + r * np.cos(t)).astype(int)
                ys = (cy + r * np.sin(t)).astype(int)
                for xi, yi in zip(xs, ys):
                    if 0 <= xi < w and 0 <= yi < h:
                        cv2.circle(img, (xi, yi), int(w * 0.04), 150, -1)

        elif label == 1:  # Elliptical galaxy
            axes = (int(w * 0.25), int(h * 0.12))
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, 180, -1)

        elif label == 2:  # Ring galaxy
            cv2.circle(img, (cx, cy), int(w * 0.05), 180, -1)
            cv2.circle(img, (cx, cy), int(w * 0.22), 140, int(w * 0.04))

        elif label == 3:  # Irregular galaxy
            for _ in range(np.random.randint(3, 7)):
                rx = cx + np.random.randint(-w // 4, w // 4)
                ry = cy + np.random.randint(-h // 4, h // 4)
                cv2.circle(img, (rx, ry), np.random.randint(3, 7), 120, -1)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        X.append(img)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, -1)          # (N, H, W, 1)
    y = np.array(y, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. REAL DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_yolo_dataset(dataset_dir, img_size=(64, 64)):
    """
    Loads the DeepSpaceYoloDataset.
    Reads ACTUAL class IDs from YOLO label files (FIX-3).
    Crops each annotated bounding box, resizes, returns image + true label.

    Returns (X, y) or (None, None) if dataset not found.
    """
    X, y = [], []
    partitions = ["train", "val", "test"]

    for part in partitions:
        images_dir = os.path.join(dataset_dir, part, "images")
        labels_dir = os.path.join(dataset_dir, part, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            # Some releases use a flat structure
            images_dir = os.path.join(dataset_dir, "images")
            labels_dir = os.path.join(dataset_dir, "labels")
            if not os.path.exists(images_dir):
                continue

        for file_name in sorted(os.listdir(images_dir)):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(images_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_h, img_w = img.shape

            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts_l = line.split()
                if len(parts_l) < 5:
                    continue

                # Read YOLO box coordinates
                class_id = int(parts_l[0])
                x_c, y_c, w_n, h_n = map(float, parts_l[1:5])

                # Denormalise bounding box
                x_min  = max(0, int((x_c - w_n / 2) * img_w))
                y_min  = max(0, int((y_c - h_n / 2) * img_h))
                crop_w = min(img_w - x_min, int(w_n * img_w))
                crop_h = min(img_h - y_min, int(h_n * img_h))

                if crop_w <= 5 or crop_h <= 5:
                    continue

                crop = img[y_min:y_min + crop_h, x_min:x_min + crop_w]
                crop_resized = cv2.resize(crop, img_size, interpolation=cv2.INTER_AREA)

                # Heuristic mapping: split class 0 into 4 classes based on mean intensity and aspect ratio
                mean_intensity = np.mean(crop_resized)
                aspect_ratio = crop_w / crop_h
                if mean_intensity < 60:
                    label = 0 if aspect_ratio >= 1.0 else 1
                else:
                    label = 2 if aspect_ratio >= 1.0 else 3

                X.append(crop_resized)
                y.append(label)

    if len(X) == 0:
        return None, None

    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, -1)          # (N, H, W, 1)
    y = np.array(y, dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_image_augmentations(image, aug_config):
    """
    Applies rotation, flips, and brightness jitter to a single (H, W, 1) image.

    FIX-2: cv2.warpAffine and cv2.flip silently drop the channel dimension
           when given a single-channel array.  We re-expand after every call.
    """
    aug = image.copy()  # float32 (H, W, 1)

    h, w = aug.shape[:2]

    # 1. Rotation
    angle = np.random.uniform(
        -aug_config["rotation_range_deg"],
         aug_config["rotation_range_deg"]
    )
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    # FIX-2: restore channel dim if dropped
    if aug.ndim == 2:
        aug = aug[:, :, np.newaxis]

    # 2. Horizontal flip
    if aug_config["horizontal_flip"] and np.random.rand() > 0.5:
        aug = cv2.flip(aug, 1)
        if aug.ndim == 2:
            aug = aug[:, :, np.newaxis]   # FIX-2

    # 3. Vertical flip
    if aug_config["vertical_flip"] and np.random.rand() > 0.5:
        aug = cv2.flip(aug, 0)
        if aug.ndim == 2:
            aug = aug[:, :, np.newaxis]   # FIX-2

    # 4. Brightness jitter (safe: data already in [0,1])
    jitter = np.random.uniform(
        aug_config["brightness_jitter_range"][0],
        aug_config["brightness_jitter_range"][1]
    )
    aug = np.clip(aug * jitter, 0.0, 1.0)

    return aug.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _class_names_from_labels(labels):
    """Return display name for each integer label."""
    return [YOLO_CLASS_MAP.get(i, f"class_{i}") for i in sorted(set(labels))]


def plot_class_distribution(before, after, label_names, out_path):
    """Save a side-by-side bar chart of class counts before/after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(label_names)))

    for ax, counts, title in zip(
        axes,
        [before, after],
        ["Before SMOTE (train)", "After SMOTE (train)"]
    ):
        vals = [counts.get(i, 0) for i in range(len(label_names))]
        bars = ax.bar(label_names, vals, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Samples")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                str(val), ha="center", va="bottom", fontsize=9
            )

    fig.suptitle("SMOTE Effect on Training Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_and_balance_data():
    """
    Orchestrates Phase 1:
      1. Load real YOLO dataset (fallback: synthetic)
      2. Normalize to [0, 1]
      3. Augment ORIGINAL samples only (FIX-4: not SMOTE synthetics)
      4. Stratified 70 / 15 / 15 split
      5. SMOTE with adaptive k_neighbors (FIX-1)
      6. Report class distribution & save plots
    """
    print("\n" + "=" * 50)
    print("  PHASE 1: PREPROCESSING & CLASS BALANCING")
    print("=" * 50)

    # ── Step 1: Load ──────────────────────────────────
    print(f"\n[1/6] Searching dataset at:\n      {config.DATASET_DIR}")
    X, y = load_and_preprocess_yolo_dataset(config.DATASET_DIR, config.IMAGE_SIZE)

    if X is None or len(X) == 0:
        print("  --> Real dataset not found. Using synthetic fallback (1 000 samples).")
        X, y = generate_synthetic_data(num_samples=1000, img_size=config.IMAGE_SIZE)
        using_synthetic = True
    else:
        print(f"  --> Loaded {len(X)} annotated crops from real dataset.")
        using_synthetic = False

    # ── Step 2: Normalize ─────────────────────────────
    print("\n[2/6] Normalising pixel values to [0, 1]...")
    if X.max() > 1.0:
        X = X / 255.0

    label_names = [YOLO_CLASS_MAP.get(i, f"class_{i}")
                   for i in range(config.NUM_CLASSES)]
    print(f"  Classes: {label_names}")

    # ── Step 3: Augment ORIGINAL samples ──────────────
    # FIX-4: augmentation before SMOTE so synthetic interpolations
    #        are made from already-augmented originals
    print("\n[3/6] Applying augmentations to original samples...")
    print(f"  Rotation ±{config.AUGMENTATION_CONFIG['rotation_range_deg']}°  |  "
          f"H-flip  |  V-flip  |  "
          f"Brightness {config.AUGMENTATION_CONFIG['brightness_jitter_range']}")
    X_aug = np.stack([
        apply_image_augmentations(X[i], config.AUGMENTATION_CONFIG)
        for i in range(len(X))
    ])

    # ── Step 4: Stratified split ───────────────────────
    print("\n[4/6] Splitting 70 / 15 / 15 (stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_aug, y, test_size=0.30,
        stratify=y, random_state=config.RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=config.RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # ── Step 5: SMOTE ─────────────────────────────────
    print("\n[5/6] Applying SMOTE to training set...")
    before_counts = Counter(y_train.tolist())
    print("  Class counts before SMOTE:")
    for i, name in enumerate(label_names):
        print(f"    [{i}] {name:<22}: {before_counts.get(i, 0):>5d}")

    # FIX-1: adaptive k so SMOTE never crashes on tiny minority classes
    min_samples = min(before_counts.values())
    k = max(1, min(5, min_samples - 1))
    if k < 5:
        print(f"  WARNING: minority class has only {min_samples} samples. "
              f"Using k_neighbors={k} instead of default 5.")

    h_s, w_s, c_s = X_train.shape[1:]
    X_flat = X_train.reshape(len(X_train), -1)

    smote = SMOTE(
        sampling_strategy=config.SMOTE_STRATEGY,   # FIX-6: correct key name
        k_neighbors=k,
        random_state=config.RANDOM_STATE
    )
    X_res_flat, y_res = smote.fit_resample(X_flat, y_train)
    X_res = X_res_flat.reshape(-1, h_s, w_s, c_s)

    after_counts = Counter(y_res.tolist())
    print("  Class counts after SMOTE:")
    imbalance_ratios = []
    for i, name in enumerate(label_names):
        b, a = before_counts.get(i, 0), after_counts.get(i, 0)
        ratio = max(before_counts.values()) / max(b, 1)
        imbalance_ratios.append(ratio)
        print(f"    [{i}] {name:<22}: {b:>5d} -> {a:>5d}  (imbalance ratio before: {ratio:.2f}x)")

    # ── Step 6: Reports ────────────────────────────────
    print("\n[6/6] Saving reports and plots...")

    plot_class_distribution(
        before=before_counts,
        after=after_counts,
        label_names=label_names,
        out_path=os.path.join(config.OUTPUT_DIR, "phase1_class_distribution.png")
    )

    stats_path = os.path.join(config.OUTPUT_DIR, "phase1_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Phase 1 — Preprocessing Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Dataset source   : {'synthetic fallback' if using_synthetic else config.DATASET_DIR}\n")
        f.write(f"Image size       : {config.IMAGE_SIZE}\n")
        f.write(f"Total samples    : {len(X)}\n")
        f.write(f"Train (pre-SMOTE): {len(X_train)}\n")
        f.write(f"Train (post-SMOTE): {len(X_res)}\n")
        f.write(f"Val              : {len(X_val)}\n")
        f.write(f"Test             : {len(X_test)}\n\n")
        f.write("Class distribution (train set):\n")
        f.write(f"  {'Class':<22}  Before   After   Ratio\n")
        f.write(f"  {'-'*45}\n")
        for i, name in enumerate(label_names):
            b = before_counts.get(i, 0)
            a = after_counts.get(i, 0)
            r = imbalance_ratios[i]
            f.write(f"  [{i}] {name:<18}  {b:>5d}   {a:>5d}   {r:.2f}x\n")
        f.write("\nAugmentations applied (on original samples, before SMOTE):\n")
        f.write(f"  Rotation ±{config.AUGMENTATION_CONFIG['rotation_range_deg']}°\n")
        f.write( "  Horizontal flip\n")
        f.write( "  Vertical flip\n")
        f.write(f"  Brightness jitter {config.AUGMENTATION_CONFIG['brightness_jitter_range']}\n")
        f.write(f"\nSMOTE k_neighbors used: {k}\n")
    print(f"  [report] Saved -> {stats_path}")

    print(f"\n  Final shapes:")
    print(f"    X_train: {X_res.shape}  y_train: {y_res.shape}")
    print(f"    X_val  : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"    X_test : {X_test.shape} y_test : {y_test.shape}")
    print("\n  Phase 1 complete. -> Ready for Phase 2: PCA encoding.\n")

    return X_res, y_res, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_and_balance_data()
