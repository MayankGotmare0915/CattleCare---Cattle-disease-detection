<div align="center">

<img src="https://img.shields.io/badge/Status-Model%20Deployed-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/TFLite-v1.0-blue?style=for-the-badge&logo=tensorflow" />
<img src="https://img.shields.io/badge/Accuracy-87%25-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Platform-React%20Native%20Expo-black?style=for-the-badge&logo=expo" />
<img src="https://img.shields.io/badge/Offline-First-purple?style=for-the-badge" />

# 🐄 CattleCare AI

**Edge AI-powered livestock disease detection — fully offline, on-device inference.**

*Point your phone at a cow. Get a diagnosis in under 200ms. No internet required.*

[📱 Mobile Integration Guide](./MOBILE_INTEGRATION_GUIDE.md) · [🚀 Quick Start](#-quick-start) · [📊 Model Performance](#-model-performance) · [🗺️ Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 📌 What Is This?

CattleCare AI is the machine learning backbone of a React Native mobile application that helps farmers and veterinarians diagnose **4 cattle diseases** from a single photo — entirely on-device, without any internet connection or cloud API calls.

The model runs as a `.tflite` file embedded inside the mobile app and performs inference in ~50–150ms on a modern Android/iOS device.

| Disease | Description |
|---|---|
| **FMD** | Foot-and-Mouth Disease — highly contagious viral infection |
| **LSD** | Lumpy Skin Disease — nodular skin lesions, fever |
| **Mastitis** | Udder inflammation — leading cause of dairy loss |
| **Healthy Cow** | No disease detected |

---

## 📁 Repository Structure

```
CattleCare/
├── 📂 deploy/                        # ← Mobile app gets these files
│   ├── cattlecare_v1.tflite          #   Float16 model (2.1 MB)
│   └── labels.txt                    #   4 class names
│
├── 📂 cattlecare_dataset/            # ML data pipeline scripts
│   ├── train.py                      #   Main training script
│   ├── augmentation_pipeline.py      #   Data augmentation
│   ├── finalize_dataset.py           #   Dataset preparation
│   ├── validate_final.py             #   Integrity checks
│   └── *.py                          #   Other utility scripts
│
├── 📄 MOBILE_INTEGRATION_GUIDE.md    # React Native integration docs
└── 📄 README.md                      # You are here
```

> **Note:** The `cattlecare_dataset/ready_dataset/` (~3 GB) and `models/*.keras`
> files are excluded from Git. See [Large Files](#-large-files) below.

---

## 🚀 Quick Start

### For ML / Training

```bash
# 1. Clone the repo
git clone https://github.com/mayankgotmare15/CattleCare.git
cd CattleCare

# 2. Set up Python environment
python -m venv ml_env
ml_env\Scripts\activate          # Windows
# source ml_env/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install tensorflow scikit-learn numpy matplotlib seaborn

# 4. Download dataset (see Large Files section below)
#    Place in: cattlecare_dataset/ready_dataset/

# 5. Train the model
cd cattlecare_dataset
python train.py
```

### For Mobile Development (React Native)

```bash
# 1. Clone the repo
git clone https://github.com/mayankgotmare15/CattleCare.git

# 2. Copy deploy files into your Expo project
cp deploy/cattlecare_v1.tflite  ../YourExpoApp/assets/ml/
cp deploy/labels.txt            ../YourExpoApp/assets/ml/

# 3. Install TFLite runtime
npx expo install react-native-fast-tflite expo-image-manipulator expo-camera

# 4. See MOBILE_INTEGRATION_GUIDE.md for full integration code
```

---

## 📊 Model Performance

### Architecture
| Component | Detail |
|---|---|
| **Backbone** | MobileNetV3-Small (ImageNet pretrained) |
| **Head** | GAP → BatchNorm → Dropout(0.4) → Dense(256) → Dropout(0.2) → Dense(4) |
| **Training** | 2-phase fine-tuning with mixed precision (float16) |
| **Total params** | 1,090,164 |

### Training Setup
| Config | Value |
|---|---|
| Platform | Google Colab (T4 GPU) |
| Image size | 224 × 224 |
| Optimizer | Adam (Phase 1: lr=1e-3, Phase 2: lr=1e-4) |
| Loss | Categorical Crossentropy + Class Weights |
| Epochs | Phase 1: 8 · Phase 2: up to 60 (early stopping) |

### Results on Test Set (3,631 images)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **FMD** | 0.598 | 1.000 | 0.749 | 76 |
| **Healthy Cow** | 0.723 | 0.662 | 0.691 | 515 |
| **LSD** | 0.891 | 0.796 | 0.840 | 900 |
| **Mastitis** | 0.907 | 0.944 | 0.925 | 2,140 |
| **Overall** | — | — | — | **86.86%** |

### Exported Model Sizes

| Format | Size | Use Case |
|---|---|---|
| `cattlecare_float32.tflite` | 4.1 MB | Debug / accuracy reference |
| `cattlecare_float16.tflite` | 2.1 MB | ✅ **Mobile deployment (shipped)** |
| `cattlecare_v1_int8.tflite` | 1.3 MB | Not used — too much accuracy loss with BatchNorm |

> **Why float16 instead of int8?**  
> MobileNetV3 + BatchNormalization layers are highly sensitive to int8 rounding.  
> Accuracy dropped from 87% → 59% with int8. Float16 maintains ~87% at 2.1 MB.

---

## 📂 Dataset

### Overview

| Split | Images | Classes |
|---|---|---|
| Train | ~14,500 | FMD, Healthy_Cow, LSD, Mastitis |
| Val | ~1,810 | same |
| Test | 3,631 | same |

### Sources
- **Kaggle** — publicly available cattle disease datasets
- **Roboflow** — curated livestock image collections
- Cleaned, deduplicated, and validated with zero cross-split leakage

### Pipeline Scripts

```
download_kaggle.py         Download from Kaggle
download_roboflow.py       Download from Roboflow
clean.py                   Remove corrupt/duplicate images
augmentation_pipeline.py   Data augmentation (flip, rotate, brightness)
finalize_dataset.py        Final train/val/test split
validate_final.py          Integrity check (no leakage, no corruption)
```

---

## 🗂️ Large Files

These are excluded from Git due to size. Access via Google Drive:

| File / Folder | Size | Location |
|---|---|---|
| `ready_dataset/` | ~3 GB | Google Drive (contact maintainer) |
| `ready_dataset.zip` | ~3 GB | Google Drive |
| `cattlecare_final_*.keras` | ~9 MB | Google Drive |
| `best_phase2.keras` | ~9 MB | Google Drive |

To reproduce training from scratch, download the dataset and place it at:
```
cattlecare_dataset/ready_dataset/
  ├── train/
  ├── val/
  └── test/
```

---

## 📱 Mobile Integration

The deploy bundle is self-contained:

```
deploy/
├── cattlecare_v1.tflite   # Float16 TFLite model
└── labels.txt             # ['FMD', 'Healthy_Cow', 'LSD', 'Mastitis']
```

**Key integration facts:**
- Input: `float32` tensor `[1, 224, 224, 3]`, pixel values `[0, 255]`
- Output: `float32` tensor `[1, 4]` — softmax probabilities per class
- Confidence threshold: `0.60` (below this → show "Uncertain")
- Inference time: ~50–150ms on modern devices

👉 See **[MOBILE_INTEGRATION_GUIDE.md](./MOBILE_INTEGRATION_GUIDE.md)** for the complete React Native / Expo integration with TypeScript code, image preprocessing, error handling, and testing checklist.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model training** | TensorFlow 2.x · Keras · Google Colab (T4 GPU) |
| **Architecture** | MobileNetV3-Small (transfer learning) |
| **Data pipeline** | Python · NumPy · scikit-learn · PIL |
| **Evaluation** | Matplotlib · Seaborn · sklearn metrics |
| **Mobile runtime** | react-native-fast-tflite · Expo |
| **Mobile platform** | React Native · Expo SDK |

---

## 🗺️ Roadmap

### ✅ Completed
- [x] Dataset collection, cleaning, deduplication
- [x] Train/val/test split with zero leakage
- [x] MobileNetV3-Small 2-phase training (87% accuracy)
- [x] Evaluation — confusion matrix + saliency maps
- [x] TFLite export (float32 · float16 · int8)
- [x] Deploy bundle created (`cattlecare_v1.tflite` + `labels.txt`)
- [x] Mobile integration guide documented

### 🔄 In Progress
- [ ] React Native Expo app integration
- [ ] `diseases.json` knowledge database (treatments + precautions)
- [ ] On-device inference validation

### 📋 Planned
- [ ] "Not a Cow" rejection class (out-of-distribution detection)
- [ ] Veterinary location finder (nearby vet stores + GPS)
- [ ] OTA model updates (without app store release)
- [ ] Multi-modal input (symptoms + image)
- [ ] Model v2 — expand to 6 classes (BRD, Orf)

---

## 🤝 Contributing

This project is currently in active development.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is for educational and research purposes.  
Dataset images are sourced from publicly available Kaggle and Roboflow collections under their respective licenses.

---

<div align="center">

**Built with ❤️ for farmers and veterinarians.**

*If this helped you, give it a ⭐ on GitHub!*

</div>
