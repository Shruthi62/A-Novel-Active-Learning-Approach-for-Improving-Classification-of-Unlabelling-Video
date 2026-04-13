Got it 🔥 — here’s your **Active Learning Video Classification README** in the **same premium structure style** 👇

---

# 🎥 Active Learning-Based Video Classification System

## 🚀 Overview

The **Active Learning-Based Video Classification System** is an advanced AI project designed to efficiently classify video data while reducing the need for large labelled datasets. Traditional video classification models require extensive manual annotation, which is both time-consuming and expensive.

This project addresses this challenge by integrating **Active Learning** with deep learning techniques to intelligently select the most informative samples from unlabelled video data. The system combines **CNN (ResNet-18)** for spatial feature extraction and **LSTM** for temporal modeling, enabling accurate understanding of actions and events in videos.

It is highly suitable for real-world applications such as **surveillance systems, human action recognition, multimedia analysis, and smart monitoring systems**.

---

## 🎯 Key Features

* 🎥 Supports **video-based action classification**
* 🧠 Uses **CNN (ResNet-18)** for spatial feature extraction
* 🔁 Uses **LSTM** for temporal sequence learning
* 🎯 **Active Learning** reduces manual labelling effort
* 📊 **Uncertainty-based sampling** selects informative samples
* 🖼️ **Keyframe extraction** removes redundant frames
* ⚡ Efficient training with fewer labelled samples
* 📈 Improved accuracy and scalability

---

## 🏗️ System Workflow

```text
Video Input
      ↓
Frame Extraction
      ↓
Keyframe Selection
      ↓
Preprocessing (Resize, Normalize)
      ↓
CNN (Feature Extraction)
      ↓
LSTM (Temporal Learning)
      ↓
Active Learning (Sample Selection)
      ↓
Model Training
      ↓
Prediction (Action Class)
      ↓
Performance Evaluation
```

---

## 🧪 Tech Stack

### 🔹 Backend

* 🐍 Python
* 🧠 PyTorch / TensorFlow
* 🔍 OpenCV
* 📊 NumPy, Matplotlib

### 🔹 Concepts Used

* 🤖 Convolutional Neural Networks (CNN)
* 🔁 Long Short-Term Memory (LSTM)
* 🎯 Active Learning
* 📊 Clustering Techniques

---

## 📁 Project Structure

### 🔹 Core Project

```bash
Active-Video-Classification/
│
├── dataset/                  # KTH dataset (videos & labels)
├── models/                   # CNN, LSTM, combined model
├── preprocessing/            # Frame extraction & keyframe selection
├── active_learning/          # Sampling strategies
├── outputs/                  # Results & graphs
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
└── README.md
```

---

## ⚙️ How It Works

1. 📤 Video dataset is loaded
2. 🎥 Videos are converted into frames
3. 🖼️ Keyframes are selected to remove redundancy
4. 🧹 Frames are preprocessed (resize, normalize)
5. 🧠 CNN extracts spatial features
6. 🔁 LSTM learns temporal dependencies
7. 🎯 Active Learning selects informative samples
8. 📊 Model is trained on selected data
9. ✅ Predictions are generated and evaluated

---

## 📊 Performance

* ✅ High classification accuracy with fewer labelled samples
* ⚡ Reduced annotation cost and time
* 📉 Lower computational overhead due to keyframe selection
* 📈 Better efficiency compared to traditional supervised learning

---

## ⚖️ Model Comparison

| Model Type                                | Data Requirement | Performance |
| ----------------------------------------- | ---------------- | ----------- |
| Supervised CNN                            | High             | Medium      |
| CNN + LSTM                                | Medium           | High        |
| ✅ Proposed (Active Learning + CNN + LSTM) | Low              | **High**    |

---

## ⚠️ Limitations

* 📉 Performance depends on sample selection quality
* ⏳ Iterative training increases computation time
* 🧩 Limited performance on highly complex video patterns
* 🧠 Requires tuning of Active Learning strategies

---

## 🔮 Future Scope

* 🤖 Transformer-based video models (ViT, TimeSformer)
* 🎞️ 3D CNN for better spatio-temporal learning
* 📱 Deployment on mobile and edge devices
* 🔄 Adaptive Active Learning strategies
* 🌐 Real-time video classification system

---

## 💡 Applications

* 🎥 Surveillance systems
* 🏃 Human action recognition
* 📺 Multimedia content analysis
* 🚗 Autonomous systems
* 📰 Video-based event detection

---

## 👨‍💻 Author

**Vatham Shruthi**
B.Tech AI & ML

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!

