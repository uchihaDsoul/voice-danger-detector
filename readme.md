# 🛡️  voice-danger-detector — Two-Stage Voice Danger Detection

> Detect danger in real-time from voice audio using deep learning.  
> Built with CNN-BiLSTM and CNN-LSTM models trained on three speech emotion datasets.

---

## 📌 What Is This?

VoiceGuard (AP 2.0) is a machine learning system that listens to a voice recording and determines whether the speaker might be in danger. It does this in two stages:

1. **Recognise the emotion** in the voice (e.g. angry, fearful, happy)
2. **Classify that emotion** as either `DANGER` or `CALM`

If someone sounds afraid, angry, disgusted, or startled — the system raises a **DANGER flag**.

---

## ✨ Features

- 🎙️ **Real-time audio analysis** — pass any `.wav` file and get instant results
- 🧠 **Two-stage deep learning pipeline** — fine-grained emotion recognition feeds a binary danger detector
- 📊 **Full emotion breakdown** — confidence scores for all 8 emotion classes
- 🔢 **Configurable danger threshold** — tune precision vs recall for your use case (default: `0.55`)
- ⚡ **Parallel feature extraction** — uses all CPU cores via `ProcessPoolExecutor`
- 📈 **Built-in visualisation** — waveform + emotion probability bar chart per audio file
- 💾 **Exportable models** — save and reload trained models without retraining
- 🔁 **Data augmentation support** — noise injection, time-stretching, pitch-shifting

---

## 🗂️ Emotion → Danger Mapping

| Emotion | Category | Reason |
|---|---|---|
| 😨 Fear | **DANGER** | Perceived threat |
| 😠 Angry | **DANGER** | Aggressive or hostile state |
| 🤢 Disgust | **DANGER** | Often linked to confrontational situations |
| 😲 Surprised | **DANGER** | Reaction to sudden/unexpected event |
| 😐 Neutral | CALM | No distress |
| 😌 Calm | CALM | Relaxed baseline |
| 😊 Happy | CALM | Positive state |
| 😢 Sad | CALM | Low arousal, non-threatening |

---

## 🛠️ Tech Stack

### Language & Platform
| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Google Colab | Training environment with GPU support |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |

### Audio Processing
| Library | Version | Purpose |
|---|---|---|
| `librosa` | 0.10.1 | Feature extraction, augmentation, display |
| `soundfile` | latest | Audio I/O |

### Machine Learning
| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building and training |
| `scikit-learn` | Data splitting, scaling, evaluation metrics |

### Concurrency
| Tool | Purpose |
|---|---|
| `concurrent.futures.ProcessPoolExecutor` | Parallel feature extraction across all CPU cores |

---

## 🔬 How It Works

### Step 1 — Load & Label Data

Three datasets are loaded and combined into a single DataFrame:

- **RAVDESS** — Studio-quality recordings, 8 emotions, parsed from filename
- **TESS** — Female speakers, 7 emotions, parsed from folder name
- **CREMA-D** — Diverse demographics, 6 emotions, parsed from filename

Each audio file gets two labels: its **emotion** (e.g. `fear`) and its **danger category** (`DANGER` or `CALM`).

---

### Step 2 — Feature Extraction

Each `.wav` file is loaded at **22,050 Hz**, padded or trimmed to exactly **3 seconds**, and converted into an **84-dimensional feature vector**:

```
MFCCs (40 coefficients × mean + std)  →  80 features
Zero-Crossing Rate (mean + std)        →   2 features
RMS Energy (mean + std)                →   2 features
─────────────────────────────────────────────────────
Total                                  →  84 features
```

> MFCCs (Mel-Frequency Cepstral Coefficients) capture the timbral texture of speech — they are the most informative features for emotion recognition.

Optional **data augmentation** is also supported:
- 🔊 Gaussian noise injection
- ⏩ Time-stretching (rate: 0.85–1.15×)
- 🎵 Pitch-shifting (±3 semitones)

---

### Step 3 — Parallel Extraction

Instead of extracting features file-by-file (slow), AP 2.0 uses Python's `ProcessPoolExecutor` to distribute work across **all available CPU cores** simultaneously — dramatically reducing pre-processing time on large datasets.

---

### Step 4 — Normalisation & Splitting

- Data is split **80% train / 10% validation / 10% test** with stratification (class proportions preserved)
- A single `StandardScaler` is fitted on the training set and reused across all splits — no data leakage
- The scaler is saved to `scaler.pkl` for consistent inference later

---

### Step 5 — Stage 1: Emotion Model (CNN-BiLSTM)

The 84-dimensional vector is reshaped to `(84, 1)` and passed through:

```
Input (84,)
    ↓
Reshape → (84, 1)
    ↓
Conv1D [128 filters, kernel 5] → BatchNorm → MaxPool → Dropout
Conv1D [256 filters, kernel 3] → BatchNorm → MaxPool → Dropout
Conv1D [128 filters, kernel 3] → BatchNorm → Dropout
    ↓
Bidirectional LSTM [128 units, return_sequences=True] → Dropout
Bidirectional LSTM [64 units] → Dropout
    ↓
Dense 256 (ReLU, L2 reg) → BatchNorm → Dropout
Dense 128 (ReLU) → Dropout
    ↓
Dense 8 (Softmax) → emotion probabilities
```

**Why CNN-BiLSTM?**
- CNN layers capture local spectral patterns in the feature sequence
- Bidirectional LSTM reads the sequence both forward and backward, capturing full temporal context
- Together they model both *what* is in the audio and *how it evolves*

**Output:** Probability scores for all 8 emotions + the most likely emotion label

---

### Step 6 — Stage 2: Danger Model (CNN-LSTM)

Uses the same CNN backbone but a unidirectional LSTM and a single sigmoid output:

```
Input (84,)
    ↓
[Same 3 Conv1D blocks as Stage 1]
    ↓
LSTM [128 units, return_sequences=True] → Dropout
LSTM [64 units] → Dropout
    ↓
Dense 128 (ReLU) → BatchNorm → Dropout
Dense 64 (ReLU) → Dropout
    ↓
Dense 1 (Sigmoid) → danger score (0.0 – 1.0)
```

**Why a separate model?**
The danger model is trained directly on the binary target, allowing it to optimise for **Precision and Recall** independently — without being constrained by the 8-class emotion objective.

**Output:** A `danger_score` between 0 and 1. If `danger_score ≥ threshold (0.55)` → 🚨 **DANGER**

---

### Step 7 — Training

Both models are trained with identical callback strategies:

| Callback | Setting |
|---|---|
| EarlyStopping | Monitor `val_accuracy`, patience 30, min_delta 1e-4 |
| ModelCheckpoint | Save best weights only |
| ReduceLROnPlateau | Halve LR when `val_loss` stalls for 10 epochs, min LR = 1e-7 |

- **Optimiser:** Adam (lr = 0.001)
- **Batch size:** 100
- **Max epochs:** 100

---

### Step 8 — Inference

The `VoiceDangerDetector` class ties everything together:

```python
detector = VoiceDangerDetector(threshold=0.55)
result = detector.analyse("audio.wav")

print(result['emotion'])       # e.g. "fear"
print(result['danger_score'])  # e.g. 0.83
print(result['status'])        # "DANGER" or "CALM"
print(result['message'])       # Human-readable summary
```

**Under the hood:**
1. Load `.wav` → pad/trim to 3 s
2. Extract 84 features with `extract_fast()`
3. Scale with fitted `StandardScaler`
4. Run Stage-1 model → emotion + confidence
5. Run Stage-2 model → danger score
6. Align score direction (handles LabelEncoder ordering)
7. Compare against threshold → raise flag

---

## 📁 Saved Artefacts

After training, all files are zipped into `voice_danger_models.zip`:

| File | Description |
|---|---|
| `emotion_model.keras` | Trained CNN-BiLSTM emotion classifier |
| `danger_model.keras` | Trained CNN-LSTM binary danger detector |
| `scaler.pkl` | Fitted StandardScaler |
| `le_em.pkl` | LabelEncoder for emotion classes |
| `le_dg.pkl` | LabelEncoder for danger classes |

---

## 🔄 End-to-End Flow

```
.wav file
    │
    ▼
librosa load → pad/trim (3 s @ 22050 Hz)
    │
    ▼
extract_fast() → 84-dim feature vector
    │
    ▼
StandardScaler.transform()
    │
    ├──► Stage 1: CNN-BiLSTM ──► emotion label + confidence scores
    │
    └──► Stage 2: CNN-LSTM   ──► danger score → DANGER / CALM flag
```

---

## ⚖️ Strengths & Limitations

**Strengths**
- Two-stage design keeps emotion reasoning transparent and interpretable
- Parallel extraction scales well to large datasets
- Configurable threshold for domain-specific tuning
- Self-contained detector class — easy to integrate anywhere

**Limitations**
- Fixed 3-second window may lose information in very short or long clips
- CREMA-D does not include `surprised` or `calm` — slight class imbalance
- `extract_fast` omits Chroma, Mel, Contrast, and Tonnetz — potential accuracy ceiling
- Future improvement: attention layers or transformer-based encoders (e.g. wav2vec 2.0)

---

*Built with ❤️ using TensorFlow, librosa, and scikit-learn*
