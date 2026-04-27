# CattleCare AI — Mobile Integration Reference Guide

> **Purpose:** Everything you need to know about integrating the trained TFLite model
> into the React Native mobile app. Read this before starting mobile development.
> Last updated: 2026-04-28

---

## Table of Contents
1. [The Big Picture](#1-the-big-picture)
2. [What's Inside the TFLite Model](#2-whats-inside-the-tflite-model)
3. [The Prediction Pipeline](#3-the-prediction-pipeline)
4. [React Native Integration](#4-react-native-integration)
5. [The diseases.json Knowledge Database](#5-the-diseasesjson-knowledge-database)
6. [Critical Things to Get Right](#6-critical-things-to-get-right)
7. [Performance & Battery Optimization](#7-performance--battery-optimization)
8. [Error Handling & Edge Cases](#8-error-handling--edge-cases)
9. [Model Versioning & Updates](#9-model-versioning--updates)
10. [Security & Privacy](#10-security--privacy)
11. [Testing the Model in the App](#11-testing-the-model-in-the-app)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. The Big Picture

```
App Launch
    ├─ Load TFLite model into memory (once, ~200ms)
    ├─ Load labels.txt (4 class names)
    └─ Load diseases.json (treatment database)

User Takes Photo of Cow
    ├─ Resize image to 224×224 pixels
    ├─ Convert pixel values to uint8 [0-255]
    └─ Shape: [1, 224, 224, 3]

TFLite Inference (ON-DEVICE, NO INTERNET)
    ├─ Feed tensor into interpreter (~50-150ms on modern phones)
    └─ Output: [0.02, 0.05, 0.91, 0.02]  one prob per class

Result Processing
    ├─ Find highest probability → index 2 = "LSD" at 91%
    ├─ Apply confidence threshold (reject if < 60%)
    └─ Look up "LSD" in diseases.json

Show Diagnosis Report
    ├─ Disease name + confidence %
    ├─ Severity level (HIGH / MEDIUM / LOW / NONE)
    ├─ Symptoms list
    ├─ Treatment steps
    ├─ Medicine recommendations
    └─ Emergency vet contact (if severity = HIGH)
```

---

## 2. What's Inside the TFLite Model

### Model Architecture
- **Base:** MobileNetV3-Small (pre-trained on ImageNet)
- **Head:** GlobalAveragePooling → BatchNorm → Dropout → Dense(256) → Dense(4)
- **Output activation:** Softmax (outputs sum to 1.0 = probabilities)

### The 4 Classes (in exact order — ORDER MATTERS)
```
Index 0 → FMD         (Foot-and-Mouth Disease)
Index 1 → Healthy_Cow
Index 2 → LSD         (Lumpy Skin Disease)
Index 3 → Mastitis
```
> ⚠️ The index order is fixed at training time. Always read it from `labels.txt`,
> never hardcode it. If the model is retrained the order may change.

### File Details
| File | Size | Input | Output |
|---|---|---|---|
| `cattlecare_v1.tflite` | ~3-4 MB | `[1, 224, 224, 3]` uint8 | `[1, 4]` uint8 |
| `labels.txt` | < 1 KB | — | — |

### What "int8 quantized" means
The model weights were converted from 32-bit floats to 8-bit integers.
- Size is ~4× smaller than the original
- Speed is ~2-3× faster on phone hardware
- Accuracy loss: < 1% (acceptable)
- Both input AND output are `uint8` (0-255 range, not 0.0-1.0 floats)

---

## 3. The Prediction Pipeline

### Step-by-Step with Exact Data Transformations

```
RAW PHOTO (e.g. 4032×3024 pixels, JPEG)
    ↓ resize to 224×224
[224, 224, 3]  float32  values 0.0 to 255.0
    ↓ convert dtype (for int8 model)
[224, 224, 3]  uint8    values 0 to 255
    ↓ add batch dimension
[1, 224, 224, 3]  uint8  ← model input
    ↓ TFLite interpreter.invoke()
[1, 4]  uint8  ← raw output (dequantize before using!)
    ↓ dequantize:  real_value = (uint8_value - zero_point) * scale
[1, 4]  float32   e.g. [0.02, 0.05, 0.91, 0.02]
    ↓ argmax
index = 2  →  labels[2] = "LSD"   confidence = 0.91 = 91%
```

### Dequantization (Critical Step)
The int8 model output is raw uint8, NOT probabilities. You MUST dequantize it:
```javascript
const outputDetails = interpreter.getOutputDetails()[0];
const scale     = outputDetails.quantization.scale;
const zeroPoint = outputDetails.quantization.zeroPoint;

const realProbabilities = rawOutput.map(
  v => (v - zeroPoint) * scale
);
```

### Confidence Threshold — VERY IMPORTANT
```javascript
const CONFIDENCE_THRESHOLD = 0.60; // 60% minimum

if (maxConfidence < CONFIDENCE_THRESHOLD) {
  return {
    result: "UNCERTAIN",
    message: "Image quality too low or animal not clearly visible.",
    action: "Please retake the photo in good lighting."
  };
}
```

---

## 4. React Native Integration

### Recommended Library
**`react-native-fast-tflite`** — best choice for `.tflite` files in React Native.
- Directly runs `.tflite` without format conversion
- Native C++ performance bridge
- Supports both Android and iOS

```bash
npm install react-native-fast-tflite
```

### Project Asset Structure
```
mobile/
├── assets/
│   └── model/
│       ├── cattlecare_v1.tflite    ← bundle with app
│       ├── labels.txt              ← bundle with app
│       └── diseases.json           ← bundle with app
└── src/
    ├── utils/
    │   └── inference.ts            ← prediction logic
    ├── data/
    │   └── diseases.ts             ← loads diseases.json
    └── screens/
        └── DiagnosisScreen.tsx     ← shows results
```

### Core Inference Module (`inference.ts`)
```typescript
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';

const MODEL_ASSET  = require('../assets/model/cattlecare_v1.tflite');
const IMG_SIZE     = 224;
const MIN_CONFIDENCE = 0.60;

let model: TensorflowModel | null = null;
let CLASS_NAMES: string[] = [];

// Call ONCE when app starts
export async function initModel() {
  model = await loadTensorflowModel(MODEL_ASSET);
  const labelsText = await fetch(require('../assets/model/labels.txt'))
    .then(r => r.text());
  CLASS_NAMES = labelsText.trim().split('\n').map(s => s.trim());
  console.log('Model ready. Classes:', CLASS_NAMES);
}

// Call when user captures a photo
export async function predict(imageUri: string) {
  if (!model) throw new Error('Model not initialized');

  const pixels = await imageToUint8Array(imageUri, IMG_SIZE);
  const output = await model.run([pixels]);
  const rawOutput = output[0] as Uint8Array;

  // Dequantize
  const { scale, zeroPoint } = model.getOutputDetails()[0].quantization;
  const probs = Array.from(rawOutput).map(v => (v - zeroPoint) * scale);

  const maxIdx  = probs.indexOf(Math.max(...probs));
  const maxConf = probs[maxIdx];

  if (maxConf < MIN_CONFIDENCE) {
    return { status: 'UNCERTAIN', disease: null, confidence: maxConf };
  }

  return {
    status: 'OK',
    disease:    CLASS_NAMES[maxIdx],
    confidence: maxConf,
    allProbs:   Object.fromEntries(CLASS_NAMES.map((n, i) => [n, probs[i]]))
  };
}
```

### Image Pre-processing
```typescript
import * as ImageManipulator from 'expo-image-manipulator';

async function imageToUint8Array(uri: string, size: number): Promise<Uint8Array> {
  // Resize to 224×224
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: size, height: size } }],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );
  // Decode base64 → raw pixel bytes
  const binaryString = atob(resized.base64!);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}
```

---

## 5. The diseases.json Knowledge Database

This file powers all treatment info. The model says WHAT disease.
This file says WHAT TO DO about it.

```json
{
  "FMD": {
    "full_name": "Foot-and-Mouth Disease",
    "severity": "HIGH",
    "contagious": true,
    "symptoms": [
      "Blisters on hooves, mouth and tongue",
      "Excessive salivation and drooling",
      "Lameness and reluctance to walk",
      "Fever above 40 degrees C",
      "Drop in milk production"
    ],
    "immediate_actions": [
      "ISOLATE the animal from the herd immediately",
      "Do NOT move the animal off the farm",
      "Report to local veterinary authority (legally required)"
    ],
    "treatment": "No specific cure. Supportive therapy: clean and disinfect lesions, provide soft feed and clean water. Antibiotics for secondary infections.",
    "medicines": [
      { "name": "Oxytetracycline", "use": "Prevent secondary bacterial infection", "prescription": true },
      { "name": "Ibuprofen / Meloxicam", "use": "Reduce fever and pain", "prescription": false },
      { "name": "Potassium permanganate solution", "use": "Disinfect hoof lesions", "prescription": false }
    ],
    "prevention": "Vaccination every 6 months. Strict biosecurity. No sharing of equipment.",
    "recovery_days": "14-21 days",
    "emergency": true,
    "vet_required": true
  },
  "LSD": {
    "full_name": "Lumpy Skin Disease",
    "severity": "HIGH",
    "contagious": true,
    "symptoms": [
      "Multiple raised nodular lumps (2-5cm) on skin",
      "Fever above 41 degrees C",
      "Swollen lymph nodes",
      "Reduced milk production",
      "Nasal and eye discharge"
    ],
    "immediate_actions": [
      "Isolate the affected animal",
      "Vaccinate the remaining herd within 48 hours",
      "Control insects (flies, mosquitoes) — they spread the virus"
    ],
    "treatment": "Supportive care. Treat secondary skin infections. Anti-inflammatory for fever.",
    "medicines": [
      { "name": "Neomycin ointment", "use": "Topical treatment of skin nodules", "prescription": false },
      { "name": "Oxytetracycline", "use": "Secondary bacterial infections", "prescription": true },
      { "name": "Meloxicam", "use": "Anti-inflammatory / fever", "prescription": false }
    ],
    "prevention": "Annual LSD vaccination. Insect vector control.",
    "recovery_days": "28-42 days",
    "emergency": true,
    "vet_required": true
  },
  "Mastitis": {
    "full_name": "Mastitis (Udder Infection)",
    "severity": "MEDIUM",
    "contagious": false,
    "symptoms": [
      "Swollen, red, hard or hot udder",
      "Pain when touching udder",
      "Abnormal milk: watery, clots, blood-tinged",
      "Reduced milk yield",
      "Mild fever"
    ],
    "immediate_actions": [
      "Milk the affected quarter more frequently (3-4x per day)",
      "Do NOT consume or sell affected milk",
      "Consult vet for antibiotic selection"
    ],
    "treatment": "Intramammary antibiotic infusion. Anti-inflammatory treatment.",
    "medicines": [
      { "name": "Intramammary Amoxicillin tube", "use": "Direct udder antibiotic", "prescription": true },
      { "name": "Ceftiofur", "use": "Systemic antibiotic for severe cases", "prescription": true },
      { "name": "Meloxicam", "use": "Reduce udder inflammation", "prescription": false }
    ],
    "prevention": "Post-milking teat dipping. Dry cow therapy. Regular milking hygiene.",
    "recovery_days": "7-14 days",
    "emergency": false,
    "vet_required": true
  },
  "Healthy_Cow": {
    "full_name": "Healthy Animal",
    "severity": "NONE",
    "contagious": false,
    "symptoms": [],
    "immediate_actions": ["No action required"],
    "treatment": "Continue regular feeding, vaccination schedule, and routine health checks.",
    "medicines": [],
    "prevention": "Maintain vaccination schedule. Regular deworming. Clean water and nutrition.",
    "recovery_days": "N/A",
    "emergency": false,
    "vet_required": false
  }
}
```

---

## 6. Critical Things to Get Right

### #1 — Image Must Be of the COW, Not Landscape
The model was trained on cow images. If a user points the camera at grass or a wall,
it will still output a prediction — it has no "not a cow" class.

**Mitigation:**
- Add a UI overlay guide frame: "Center the cow in the frame"
- Show warning if confidence < 80%
- Future: add a binary "is this a cow?" classifier before the disease classifier

### #2 — Single Disease Model (Not Multi-Label)
The model predicts ONE disease per image. A cow can have BOTH Mastitis AND LSD.

**Mitigation:** Show ALL class probabilities:
```
Primary diagnosis:  LSD         (87%)
Also detected:      Mastitis    (11%)   ← show if > 10%
```

### #3 — Lighting Matters
Low-light photos produce unreliable results.

**Mitigation:**
- Warn if photo is dark before running inference
- Suggest using flash

### #4 — Retrain When Adding New Diseases
The model currently recognizes: FMD, Healthy_Cow, LSD, Mastitis (4 classes).
You CANNOT add a class without full retraining.

### #5 — Model is NOT a Vet
Always display this disclaimer:
> "This AI diagnosis is for guidance only. Always consult a licensed veterinarian
> before administering any treatment."

---

## 7. Performance & Battery Optimization

### Inference Time Benchmarks (MobileNetV3-Small int8)
| Device Tier | Approximate Inference Time |
|---|---|
| High-end (Pixel 8, iPhone 15) | 30–60 ms |
| Mid-range (Redmi Note 12) | 60–120 ms |
| Low-end (entry-level Android) | 150–300 ms |

### Load Model Once, Not Every Time
```typescript
// CORRECT: Load on app start, reuse
useEffect(() => { initModel(); }, []);

// WRONG: Loading on every prediction wastes 200ms + memory
async function predict(uri) {
  const model = await loadTensorflowModel(...); // don't do this
}
```

### Release Model When App Backgrounds
```typescript
AppState.addEventListener('change', state => {
  if (state === 'background') model?.close(); // free GPU memory
  if (state === 'active')     initModel();
});
```

---

## 8. Error Handling & Edge Cases

### All Possible Outcomes
| Status | Meaning | What to Show |
|---|---|---|
| `OK` (≥ 60%) | Clear diagnosis | Full report with treatment |
| `UNCERTAIN` (< 60%) | Low confidence | "Retake photo in better lighting" |
| `ERROR` | Model/image failure | "Something went wrong — Try Again" |
| `Healthy_Cow` | No disease detected | "Animal appears healthy" |
| HIGH severity | Dangerous disease | Red warning + "Call Vet Now" button |

---

## 9. Model Versioning & Updates

### Over-The-Air Updates via Firebase ML
Instead of forcing app store update for new model:
1. Upload new `.tflite` to Firebase ML Console
2. App checks for updates on launch
3. Downloads in background, applies next session

```typescript
import ml from '@react-native-firebase/ml';
const model = ml().model('cattlecare_disease_classifier');
await model.downloadIfNeeded();
const localPath = model.downloadedFilePath;
```

### File Naming Convention
```
cattlecare_v1.tflite   ← current (4 classes, MobileNetV3-Small, ~84% accuracy)
cattlecare_v2.tflite   ← future  (6+ classes, retrained with more data)
```

---

## 10. Security & Privacy

### What NEVER leaves the device
- Camera photos (never uploaded)
- Diagnosis results (stored locally)
- Model inference (fully on-device)

### Local Storage of Past Diagnoses
```typescript
await AsyncStorage.setItem('diagnosis_history',
  JSON.stringify({ timestamp, disease, confidence, imageUri })
);
```

---

## 11. Testing the Model in the App

### Before Integration — Verify TFLite File
```bash
python -c "
import tensorflow as tf, numpy as np
interp = tf.lite.Interpreter('deploy/cattlecare_v1.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print('Input :', inp['shape'], inp['dtype'])
print('Output:', out['shape'], out['dtype'])
dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
interp.set_tensor(inp['index'], dummy)
interp.invoke()
print('Output (dummy):', interp.get_tensor(out['index']))
print('Model OK')
"
```

### In-App Testing Checklist
- [ ] Model loads in < 500ms on startup
- [ ] Healthy cow photo → predicts Healthy_Cow > 70%
- [ ] LSD cow photo → predicts LSD > 70%
- [ ] Blurry/dark photo → shows UNCERTAIN (< 60% conf)
- [ ] Non-cow photo → confidence < 60% (shows UNCERTAIN)
- [ ] Works in airplane mode (fully offline)
- [ ] Memory usage < 150MB during inference

---

## 12. Implementation Checklist

- [x] Train model
- [x] Evaluate — confusion matrix + Grad-CAM
- [x] Export to TFLite — int8 quantized
- [ ] Create diseases.json knowledge database
- [ ] Create deploy/ folder (tflite + labels + diseases.json)
- [ ] Set up React Native Expo project
- [ ] Install react-native-fast-tflite + expo-image-manipulator
- [ ] Write inference.ts — initModel() + predict()
- [ ] Build camera capture screen with guide overlay
- [ ] Build diagnosis result screen (disease card + treatment)
- [ ] Build diagnosis history screen
- [ ] Add confidence threshold warning UI
- [ ] Add emergency vet contact / call feature
- [ ] Add cow health map (GPS + history)
- [ ] Offline mode testing
- [ ] Firebase ML model update (optional, for v2)

---

*CattleCare AI — Mobile Integration Reference | 2026-04-28*
