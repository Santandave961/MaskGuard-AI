"""
=============================================================
  Face Mask Detection — Streamlit App
  Author : Wisdom (Santandave961)
  Stack  : Streamlit, OpenCV, scikit-learn, numpy, PIL
=============================================================
  Run:  streamlit run app.py
=============================================================
"""

import os
import pickle
import tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MaskGuard AI",
    page_icon=":mask:",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

[data-testid="stAppViewContainer"] {
    background: #080B12;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #1E2330;
}
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #E8F4FD !important; }
p, li, span { color: #A8B8CC !important; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF, #0066FF, #7B2FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 8px;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #5A7A99 !important;
    letter-spacing: 0.04em;
}
.stat-card {
    background: #0D1117;
    border: 1px solid #1E2330;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF);
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #E8F4FD !important;
}
.stat-label {
    font-size: 0.75rem;
    color: #3A5A77 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.result-mask {
    background: linear-gradient(135deg, #002A1A, #004D2A);
    border: 1px solid #00C853;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
.result-nomask {
    background: linear-gradient(135deg, #2A0000, #4D0000);
    border: 1px solid #FF1744;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}
.result-conf {
    font-size: 0.85rem;
    margin-top: 6px;
    color: #A8B8CC !important;
}
.upload-zone {
    background: #0D1117;
    border: 2px dashed #1E2330;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    transition: border-color 0.3s;
}
.info-pill {
    display: inline-block;
    background: #0D1117;
    border: 1px solid #1E2330;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #3A6A99 !important;
    margin: 3px;
}
.divider { border-top: 1px solid #1E2330; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mask_model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def extract_features(img_bgr, img_size=64):
    img  = cv2.resize(img_bgr, (img_size, img_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog  = cv2.HOGDescriptor(
        (img_size, img_size), (16,16), (8,8), (8,8), 9
    )
    hog_feat = hog.compute(gray).flatten()
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        hist_feats.append(h.flatten())
    return np.concatenate([hog_feat] + hist_feats)

def predict_face(face_bgr, model_data):
    feat  = extract_features(face_bgr, model_data["img_size"])
    proba = model_data["pipeline"].predict_proba([feat])[0]
    label = model_data["classes"][np.argmax(proba)]
    conf  = np.max(proba)
    return label, conf, proba

def annotate_image(img_bgr, face_cascade, model_data, scale=1.1, neighbors=5, min_size=60):
    """Detect faces, classify each, draw annotated bounding boxes."""
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=neighbors,
        minSize=(min_size, min_size)
    )

    results     = []
    annotated   = img_bgr.copy()

    if len(faces) == 0:
        return annotated, results

    for (x, y, w, h) in faces:
        pad    = int(0.1 * w)
        x1     = max(0, x - pad)
        y1     = max(0, y - pad)
        x2     = min(img_bgr.shape[1], x + w + pad)
        y2     = min(img_bgr.shape[0], y + h + pad)
        face   = img_bgr[y1:y2, x1:x2]

        if face.size == 0:
            continue

        label, conf, proba = predict_face(face, model_data)

        # Box colour
        colour = (0, 220, 80) if label == "With Mask" else (30, 30, 255)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label background
        txt     = f"{label}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated,
                      (x1, y1 - th - 14),
                      (x1 + tw + 10, y1),
                      colour, -1)
        cv2.putText(annotated, txt,
                    (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

        results.append({
            "label"    : label,
            "confidence": conf,
            "proba_no" : proba[0],
            "proba_yes": proba[1],
            "bbox"     : (x1, y1, x2, y2),
        })

    return annotated, results


def process_video(video_path, face_cascade, model_data,
                  scale, neighbors, min_size, max_frames=120):
    """Process video, annotate frames, return output path + stats."""
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = video_path.replace(".mp4", "_annotated.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx    = 0
    all_results  = []
    progress_bar = st.progress(0, text="Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        annotated, res = annotate_image(
            frame, face_cascade, model_data, scale, neighbors, min_size
        )
        writer.write(annotated)
        all_results.extend(res)
        frame_idx += 1
        pct = min(frame_idx / min(total, max_frames), 1.0)
        progress_bar.progress(pct, text=f"Frame {frame_idx}/{min(total, max_frames)}")

    cap.release()
    writer.release()
    progress_bar.empty()
    return out_path, all_results, frame_idx


# ═══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════

model_data   = load_model()
face_cascade = load_face_cascade()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px'>
        <span style='font-size:2.5rem'>😷</span>
        <div style='font-family:Space Mono,monospace; font-size:1rem;
                    color:#00D4FF; margin-top:6px'>MaskGuard AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Detection Settings")

    scale     = st.slider("Scale Factor",     1.05, 1.5,  1.1,  0.05,
                          help="Lower = more detections, slower")
    neighbors = st.slider("Min Neighbours",   2, 10,      5,    1,
                          help="Higher = fewer false positives")
    min_size  = st.slider("Min Face Size (px)", 20, 120,  60,   10,
                          help="Minimum face size to detect")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🧠 Model Info")
    st.markdown(f"""
    <span class="info-pill">HOG Features</span>
    <span class="info-pill">HSV Histogram</span>
    <span class="info-pill">MLP Classifier</span>
    <span class="info-pill">Accuracy: {model_data['accuracy']*100:.1f}%</span>
    <span class="info-pill">OpenCV Haar Cascade</span>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📌 About")
    st.caption(
        "Built by **Wisdom** · Data Science Portfolio\n\n"
        "[@Santandave961](https://github.com/Santandave961) 🇳🇬"
    )


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 8px 0 24px'>
    <div class='hero-title'>MaskGuard AI 😷</div>
    <div class='hero-sub'>REAL-TIME FACE MASK DETECTION · POWERED BY HOG + MLP</div>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Model Accuracy", f"{model_data['accuracy']*100:.1f}%"),
    ("Detection Engine", "Haar Cascade"),
    ("Feature Type", "HOG + HSV"),
    ("Classes", "2 (Mask / No Mask)"),
]
for col, (label, value) in zip([c1,c2,c3,c4], stats):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab_img, tab_vid, tab_info = st.tabs(["🖼️  Image Detection", "🎬  Video Detection", "📖  How It Works"])

# ── IMAGE TAB ─────────────────────────────────────────────────
with tab_img:
    st.markdown("#### Upload an image to detect face masks")
    st.caption("Supports JPG, JPEG, PNG · Multiple faces supported")

    uploaded = st.file_uploader(
        "Drop image here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read image. Please try another file.")
        else:
            col_orig, col_result = st.columns(2)

            with col_orig:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            with st.spinner("Detecting faces and classifying masks..."):
                annotated, results = annotate_image(
                    img_bgr, face_cascade, model_data,
                    scale, neighbors, min_size
                )

            with col_result:
                st.markdown("**Detection Result**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            if len(results) == 0:
                st.warning("⚠️ No faces detected. Try adjusting the detection settings in the sidebar (lower Scale Factor or Min Face Size).")
            else:
                n_mask   = sum(1 for r in results if r["label"] == "With Mask")
                n_nomask = len(results) - n_mask

                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"""<div class="stat-card">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Faces Detected</div></div>""",
                    unsafe_allow_html=True)
                m2.markdown(f"""<div class="stat-card" style="border-top-color:#00C853">
                    <div class="stat-value" style="color:#00C853 !important">{n_mask}</div>
                    <div class="stat-label">Wearing Mask ✅</div></div>""",
                    unsafe_allow_html=True)
                m3.markdown(f"""<div class="stat-card" style="border-top-color:#FF1744">
                    <div class="stat-value" style="color:#FF1744 !important">{n_nomask}</div>
                    <div class="stat-label">No Mask ❌</div></div>""",
                    unsafe_allow_html=True)
                compliance = n_mask / len(results) * 100
                m4.markdown(f"""<div class="stat-card">
                    <div class="stat-value">{compliance:.0f}%</div>
                    <div class="stat-label">Compliance Rate</div></div>""",
                    unsafe_allow_html=True)

                st.markdown("<br>**Per-Face Results**", unsafe_allow_html=True)
                face_cols = st.columns(min(len(results), 4))
                for i, (res, col) in enumerate(zip(results, face_cols)):
                    css_class = "result-mask" if res["label"] == "With Mask" else "result-nomask"
                    icon      = "✅" if res["label"] == "With Mask" else "❌"
                    colour    = "#00C853" if res["label"] == "With Mask" else "#FF1744"
                    col.markdown(f"""
                    <div class="{css_class}">
                        <div class="result-label" style="color:{colour} !important">
                            {icon} Face {i+1}
                        </div>
                        <div style="font-size:1.1rem; color:#E8F4FD !important;
                                    font-weight:600; margin-top:6px">
                            {res['label']}
                        </div>
                        <div class="result-conf">
                            Confidence: {res['confidence']*100:.1f}%
                        </div>
                        <div style="margin-top:8px; font-size:0.78rem; color:#5A7A99 !important">
                            Mask prob: {res['proba_yes']*100:.1f}%<br>
                            No-mask prob: {res['proba_no']*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem">📸</div>
            <div style="color:#3A5A77 !important; margin-top:8px">
                Upload a JPG or PNG image above
            </div>
            <div style="font-size:0.8rem; color:#2A3A4A !important; margin-top:4px">
                Works best with clear, front-facing faces
            </div>
        </div>""", unsafe_allow_html=True)


# ── VIDEO TAB ─────────────────────────────────────────────────
with tab_vid:
    st.markdown("#### Upload a video to detect face masks across frames")
    st.caption("Supports MP4 · First 120 frames processed")

    vid_file = st.file_uploader(
        "Drop video here",
        type=["mp4"],
        label_visibility="collapsed",
        key="video_uploader",
    )

    if vid_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("🚀 Run Detection", type="primary"):
            with st.spinner("Annotating video..."):
                out_path, all_results, n_frames = process_video(
                    tmp_path, face_cascade, model_data,
                    scale, neighbors, min_size
                )

            st.success(f"✅ Done — {n_frames} frames processed, {len(all_results)} face detections")

            if os.path.exists(out_path):
                st.markdown("**Annotated Video**")
                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=f,
                        file_name="maskguard_output.mp4",
                        mime="video/mp4",
                    )

            if all_results:
                n_mask   = sum(1 for r in all_results if r["label"] == "With Mask")
                n_nomask = len(all_results) - n_mask

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("**Aggregate Video Stats**")
                v1, v2, v3 = st.columns(3)
                v1.metric("Total Detections", len(all_results))
                v2.metric("With Mask",  n_mask,  delta=f"{n_mask/len(all_results)*100:.0f}%")
                v3.metric("No Mask",    n_nomask, delta=f"-{n_nomask/len(all_results)*100:.0f}%",
                          delta_color="inverse")
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem">🎬</div>
            <div style="color:#3A5A77 !important; margin-top:8px">
                Upload an MP4 video above
            </div>
            <div style="font-size:0.8rem; color:#2A3A4A !important; margin-top:4px">
                First 120 frames will be annotated and available for download
            </div>
        </div>""", unsafe_allow_html=True)


# ── HOW IT WORKS TAB ──────────────────────────────────────────
with tab_info:
    st.markdown("### How MaskGuard AI Works")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**🔍 Step 1 — Face Detection**

Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to locate all faces in the image. Each detected face is cropped and padded before classification.

---

**📐 Step 2 — Feature Extraction**

Two feature types are extracted from each 64×64 face crop:

- **HOG (Histogram of Oriented Gradients)** — captures edge and shape information across 8×8 cells. Excellent at distinguishing the structural difference a mask makes on the lower half of a face.
- **HSV Colour Histograms** — captures colour distribution across Hue, Saturation, and Value channels (32 bins each). Masks typically introduce a distinct colour signature vs bare skin.

Final feature vector: **1,860 dimensions**.
        """)

    with col_b:
        st.markdown("""
**🧠 Step 3 — MLP Classification**

A Multi-Layer Perceptron (neural network) with architecture:

```
Input (1860) → Dense(256) → Dense(128) → Dense(64) → Output(2)
```

Trained on 4,000 synthetic face images (2,000 masked, 2,000 unmasked) with:
- StandardScaler normalisation
- ReLU activations
- Early stopping (patience on validation loss)
- **Test accuracy: 100%** on synthetic data

---

**📊 Output**

For each detected face, the model returns:
- **Label**: `With Mask` or `No Mask`
- **Confidence**: probability score (0–100%)
- **Compliance rate**: % of detected faces wearing masks
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
**⚠️ Production Note**

This model is trained on synthetic data for demonstration purposes. For production deployment, retrain `train_model.py` using a real-world dataset such as:
- [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
- [RMFD — Real-world Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)

Simply swap the data generation section in `train_model.py` with real image loading and re-run.
    """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(
        "Built by **Wisdom** · Data Science Portfolio · "
        "[@Santandave961](https://github.com/Santandave961) · "
        "Targeting Nigerian Fintech & Tech Roles 🇳🇬"
    )