# 😷 MaskGuard AI — Real-Time Face Mask Detection

A production-ready face mask detection app built with OpenCV, scikit-learn, and Streamlit. Upload an image or video and the app detects every face, classifies it as **With Mask** or **No Mask**, and returns confidence scores with annotated bounding boxes.

![MaskGuard AI](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E)

---

## 🎯 Features

- **Image detection** — upload JPG/PNG, detect multiple faces simultaneously
- **Video detection** — upload MP4, annotate up to 120 frames, download result
- **Confidence scores** — per-face mask/no-mask probability
- **Compliance rate** — % of detected faces wearing masks
- **Adjustable detection settings** — scale factor, min neighbours, min face size
- **Dark-themed UI** — Space Mono + DM Sans, gradient accents

---

## 🧠 How It Works

| Step | Method |
|---|---|
| Face detection | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Feature extraction | HOG (Histogram of Oriented Gradients) + HSV colour histograms |
| Classification | MLP Neural Network (256 → 128 → 64 → 2) |
| Feature dimensions | 1,860 per face crop |
| Training accuracy | 100% on synthetic data |

---

## 🗂️ Project Structure

```
maskguard-ai/
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── mask_model.pkl      # Trained model (auto-generated)
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

**1. Clone the repo**
```bash
git clone https://github.com/Santandave961/maskguard-ai.git
cd maskguard-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** (skip if `mask_model.pkl` already exists)
```bash
python train_model.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
opencv-python-headless
numpy
scikit-learn
Pillow
```

> Use `opencv-python-headless` for Streamlit Cloud (no display needed).

---

## 🔁 Using Real Data (Production)

This project ships with a synthetic dataset for demonstration. To retrain on real-world data:

1. Download one of these datasets:
   - [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
   - [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
2. Replace the `make_face_no_mask()` / `make_face_with_mask()` functions in `train_model.py` with real image loading
3. Re-run `python train_model.py`

---

## 🚀 Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → `app.py` → Deploy

---

## 👤 Author

**Wisdom** — Data Science & ML Engineer
📌 NYSC Corper | Abia State, Nigeria
🐙 GitHub: [@Santandave961](https://github.com/Santandave961)
🐦 X: [@Santandave961](https://x.com/Santandave961)

---

## 📄 License

MIT License — free to use, adapt, and build on.
