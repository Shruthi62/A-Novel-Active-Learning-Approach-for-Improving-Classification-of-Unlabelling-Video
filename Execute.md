# 🎬 Active Learning Video Classification (ResNet-18 + LSTM + Uncertainty Sampling)

## 🚀 How to Run the Project

### Step 1: Navigate to Project Directory

```bash
cd C:\Users\hp5cd\Downloads\YouTube_DataSet_Annotated\action_youtube_naudio
```

### Step 2: Activate Virtual Environment

```bash
.venv\Scripts\Activate.ps1
```

### Step 3: Launch the Web Application

```bash
streamlit run final_complete_app.py
```

### Step 4: Open in Browserstreamlit 

Navigate to: **http://localhost:8501**

---

## 📊 What You'll See

The web application has **4 main sections:**

1. **About & Overview** - Project details and benefits
2. **Interactive Classifier** - Upload videos and get instant predictions
3. **Visualizations** - View 3 performance graphs:
   - Active Learning Progress (84.5% → 92.4%)
   - Data Efficiency Comparison
   - Uncertainty-Based Sample Selection
4. **Training** - Run the active learning pipeline

---

## ⚡ Run Full Training (Optional)

To see the complete training process with real-time progress:

```bash
python train_standalone.py
```

**Output:**

- Generates 3 graphs automatically
- Creates trained model (`best_model.pth`)
- Shows accuracy improvement across iterations
- Execution time: ~15-20 minutes

---

## 🧠 Model Architecture (with RNN)

```
Input Video
    │
    ▼
ResNet-18 (frozen, pre-trained)
    │  extracts 512-d features per frame
    ▼
LSTM (hidden=256, 1 layer)  ← RNN temporal modeling
    │  processes frame sequence to capture motion
    ▼
Dropout (0.5)
    │
    ▼
Linear (256 → 11 classes)
    │
    ▼
Softmax → Predicted Action
```

**Why LSTM?**  
Averaging frames (old approach) loses motion order. LSTM reads frames in sequence so it learns *how* actions evolve over time (e.g. the wind-up before a swing).

**Query Strategy:** Uncertainty Sampling (least confidence) — picks unlabeled videos the model is most confused about.

---

## 📁 Key Files

- `final_complete_app.py` - Main web application
- `novel_active_learning.py` - Active learning algorithm (ResNet-18 + LSTM + Uncertainty Sampling)
- `train_standalone.py` - Training script
- `best_model.pth` - Pre-trained model
- Video folders - Dataset (11 action classes, 274 videos)

---

## 🎯 Key Results

| Metric                       | Value         |
| ---------------------------- | ------------- |
| **Accuracy**           | 92.4%         |
| **Samples Used**       | 52 out of 274 |
| **Labeling Reduction** | 81%           |
| **Action Classes**     | 11            |
| **Feature Extractor**  | ResNet-18     |
| **Temporal Model**     | LSTM (RNN)    |
| **Query Strategy**     | Uncertainty Sampling |

---

## ❌ Troubleshooting

**Port already in use?**

```bash
streamlit run final_complete_app.py --server.port 8502
```

**Missing dependencies?**

```bash
pip install -r requirements.txt
```

**CUDA/GPU not available?**
The app will automatically use CPU (slower but works)

---

## ✅ Done!

Your application is now running at **http://localhost:8501** 🎉
