# CattleCare AI Model Status

## Overview
CattleCare (AgriVet) is an offline-first, Edge AI-powered livestock disease detection application. The system is designed to use a MobileNetV2-based model to classify images of cattle into various disease classes locally on a user's mobile device.

## Current ML Project Status: **Dataset Ready**

### 1. Dataset Preparation - `COMPLETED`
The dataset pipeline is fully implemented inside the `cattlecare_dataset` directory. It includes comprehensive scripts to download, compile, organize, and augment images to prevent data leakage and handle class imbalance. 

* **Source Location:** `cattlecare_dataset/ready_dataset/`
* **Splits:** `train`, `val`, `test` splits are successfully created without any cross-split overlap.
* **Classes (6 Total):** 
  * BRD (Bovine Respiratory Disease)
  * FMD (Foot-and-Mouth Disease)
  * Healthy_Cow
  * LSD (Lumpy Skin Disease)
  * Mastitis
  * Orf
* **Validation:** The dataset has officially passed integrity checks via `validate_final.py` (checked for corrupt images, data leakage, and labels correctness). 

### 2. Model Training - `PENDING`
Currently, there is no `train.py` or compiled model (`.h5`, `.tflite`) in the repository. The model has not been trained yet. 

**Next Steps / Training Roadmap:**
1. **Write Training Script:** Create the `train.py` script.
2. **Architecture Implementation:** Use **MobileNetV2** (pre-trained on ImageNet) designed for efficient mobile deployment. 
3. **Handle Imbalance:** The validation steps indicate a minor imbalance. The training loop should leverage class weights (`WeightedRandomSampler` or `Focal Loss`) to prioritize minority classes rather than over-predicting the majority class (`Healthy_Cow`).
4. **Compile for Edge Inference:** Export the completed model to `.tflite` format so it can be packaged tightly with the React Native Expo mobile frontend.

### 3. Application Integration (Mobile App) - `PENDING`
The exported offline machine learning model will eventually be bundled into the React Native frontend for fast on-device inference, allowing offline diagnostics and symptom fusion.
