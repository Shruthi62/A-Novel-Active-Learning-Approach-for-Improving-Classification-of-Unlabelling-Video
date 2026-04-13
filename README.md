# Novel Active Learning Video Classification

This project implements a novel active learning approach for classifying unlabeled videos using deep learning techniques. It analyzes 11 action classes from the YouTube Action Dataset.

## Features

- **Deep Feature Extraction**: Uses pre-trained ResNet-18 for robust video feature extraction
- **Active Learning**: Implements uncertainty sampling (least confidence) to iteratively improve model performance
- **Video Classification**: Classifies videos into 11 action categories
- **Scalable Training**: Starts with small labeled dataset and grows through active learning

## Dataset

The system works with the YouTube Action Dataset containing 11 action classes:
- Basketball shooting
- Biking
- Diving
- Golf swing
- Horse riding
- Soccer juggling
- Swing
- Tennis swing
- Trampoline jumping
- Volleyball spiking
- Walking (with dog)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the dataset in the correct folder structure.

## Usage

### Training with Active Learning

Run the main script to train the model:
```bash
python novel_active_learning.py
```

This will:
- Load the dataset
- Perform active learning iterations
- Save the best model as `best_model.pth`

### Classifying New Videos

To classify a new video, use the `classify_video()` function:

```python
from novel_active_learning import classify_video

# Classify a video
predicted_action, confidence = classify_video("path/to/your/video.avi")
```

Or run classification on an example video:
```python
python -c "from novel_active_learning import classify_video; classify_video('path/to/video.avi')"
```

## Configuration

Modify the `Config` class in `novel_active_learning.py` to adjust:
- Number of initial labeled samples
- Query size per iteration
- Number of active learning iterations
- Training hyperparameters

## Output

The classification function returns:
- Predicted action class
- Confidence score
- Top 3 predictions with probabilities

## Performance

The active learning approach should show improving accuracy as more samples are labeled, demonstrating the efficiency of uncertainty-based sample selection over random sampling.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenCV
- NumPy
- Scikit-learn
- tqdm