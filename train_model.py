"""
=============================================================
  Face Mask Detection — Model Training Script
  Author : Wisdom (Santandave961)
  Stack  : OpenCV, scikit-learn, numpy, pickle
=============================================================

This script:
  1. Generates synthetic training data (simulating mask/no-mask
     face crops with realistic colour & texture differences)
  2. Extracts HOG + colour histogram features from each image
  3. Trains a MLP classifier
  4. Saves the model as mask_model.pkl

In production, replace synthetic data with a real dataset such as:
  - Kaggle Face Mask Detection Dataset
  - MaskedFace-Net
  - RMFD (Real-world Masked Face Dataset)
=============================================================
"""

import numpy as np
import cv2
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

np.random.seed(42)

IMG_SIZE   = 64
N_SAMPLES  = 2000    # per class
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "mask_model.pkl")


# ── Feature extraction ────────────────────────────────────────

def extract_features(img_bgr):
    """
    Extract HOG + colour histogram features from a BGR image.
    Returns a 1-D feature vector.
    """
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # HOG descriptor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    win_size   = (IMG_SIZE, IMG_SIZE)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size  = (8, 8)
    nbins      = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                             cell_size, nbins)
    hog_feat = hog.compute(gray).flatten()

    # Colour histograms (HSV)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        hist_feats.append(hist.flatten())
    colour_feat = np.concatenate(hist_feats)

    return np.concatenate([hog_feat, colour_feat])


# ── Synthetic data generation ─────────────────────────────────

def make_face_no_mask(n):
    """Simulate face crops WITHOUT mask — skin tones visible."""
    imgs = []
    for _ in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Skin-tone background
        skin = (
            np.random.randint(120, 200),
            np.random.randint(80,  150),
            np.random.randint(60,  130),
        )
        img[:] = skin
        # Add noise (texture)
        noise = np.random.randint(-20, 20,
                                   img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Eyes region (darker)
        ey = np.random.randint(15, 25)
        img[ey:ey+8, 10:24] = (40, 30, 20)
        img[ey:ey+8, 40:54] = (40, 30, 20)
        # Mouth region — visible lips (no mask)
        my = np.random.randint(38, 48)
        lip_col = (
            np.random.randint(80, 140),
            np.random.randint(40, 90),
            np.random.randint(100, 180),
        )
        img[my:my+6, 20:44] = lip_col
        imgs.append(img)
    return imgs

def make_face_with_mask(n):
    """Simulate face crops WITH mask — lower half covered."""
    imgs = []
    for _ in range(n):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Skin-tone upper face
        skin = (
            np.random.randint(120, 200),
            np.random.randint(80,  150),
            np.random.randint(60,  130),
        )
        img[:] = skin
        noise = np.random.randint(-20, 20,
                                   img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Eyes
        ey = np.random.randint(15, 25)
        img[ey:ey+8, 10:24] = (40, 30, 20)
        img[ey:ey+8, 40:54] = (40, 30, 20)
        # Mask covering lower half — random mask colour
        mask_col = (
            np.random.randint(180, 255),
            np.random.randint(180, 255),
            np.random.randint(180, 255),
        )
        mask_top = np.random.randint(28, 36)
        img[mask_top:, :] = mask_col
        # Mask strap lines
        img[mask_top:mask_top+3, :] = (
            max(0, mask_col[0]-40),
            max(0, mask_col[1]-40),
            max(0, mask_col[2]-40),
        )
        imgs.append(img)
    return imgs


# ═══════════════════════════════════════════════════════════════
# 1. GENERATE DATA
# ═══════════════════════════════════════════════════════════════

print("Generating synthetic training data...")
no_mask_imgs   = make_face_no_mask(N_SAMPLES)
with_mask_imgs = make_face_with_mask(N_SAMPLES)

labels = ([0] * N_SAMPLES) + ([1] * N_SAMPLES)   # 0=no mask, 1=mask

print("Extracting features...")
features = []
all_imgs = no_mask_imgs + with_mask_imgs
for i, img in enumerate(all_imgs):
    features.append(extract_features(img))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(all_imgs)} done")

X = np.array(features)
y = np.array(labels)
print(f"Feature matrix: {X.shape}")


# ═══════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")


# ═══════════════════════════════════════════════════════════════
# 3. TRAIN MODEL
# ═══════════════════════════════════════════════════════════════

print("\nTraining MLP classifier...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )),
])

pipeline.fit(X_train, y_train)


# ═══════════════════════════════════════════════════════════════
# 4. EVALUATE
# ═══════════════════════════════════════════════════════════════

y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*50}")
print(classification_report(y_test, y_pred,
      target_names=["No Mask", "With Mask"]))


# ═══════════════════════════════════════════════════════════════
# 5. SAVE MODEL
# ═══════════════════════════════════════════════════════════════

model_data = {
    "pipeline"  : pipeline,
    "img_size"  : IMG_SIZE,
    "classes"   : ["No Mask", "With Mask"],
    "accuracy"  : acc,
}

with open(SAVE_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved → {SAVE_PATH}")
print("   Run: streamlit run app.py")