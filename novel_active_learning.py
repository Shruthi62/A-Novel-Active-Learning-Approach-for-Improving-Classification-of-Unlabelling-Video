"""
Novel Active Learning Approach for Video Classification
======================================================

This script implements an active learning framework for classifying unlabeled videos
using deep learning techniques. It analyzes 11 action classes from the YouTube Action Dataset.

Features:
- Deep feature extraction using pre-trained ResNet-18
- Active learning with uncertainty sampling (least confidence)
- Iterative model training and sample selection
- Video classification for uploaded videos

Usage:
1. Run the script to train the model with active learning
2. Use classify_video() function for inference on new videos
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
import warnings
import pickle
import json

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='torch._classes')
warnings.filterwarnings('ignore', module='torch')

# Configuration
class Config:
    DATA_DIR = "."
    ACTIONS = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding',
               'soccer_juggling', 'swing', 'tennis_swing', 'trampoline_jumping',
               'volleyball_spiking', 'walking']
    ACTION_TO_ID = {action: i for i, action in enumerate(ACTIONS)}
    NUM_CLASSES = len(ACTIONS)
    BATCH_SIZE = 64  # Increased for faster processing
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5  # Increased for better training
    INITIAL_LABELED = 22  # 2 samples per class
    QUERY_SIZE = 10  # More queries per iteration
    MAX_ITERATIONS = 3  # More iterations
    FRAMES_PER_VIDEO = 3  # Reduced frames for faster feature extraction
    FEATURE_DIM = 512

# Video Classifier Model
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels=None, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        features = extract_video_features(video_path)

        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(features, dtype=torch.float32), idx

# Feature Extraction
def extract_video_features(video_path):
    """Extract features from video using pre-trained ResNet-18"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained ResNet-18
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    if frame_count > 0:
        step = max(1, frame_count // Config.FRAMES_PER_VIDEO)
        for i in range(0, frame_count, step):
            if len(frames) >= Config.FRAMES_PER_VIDEO:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(frame))

    cap.release()

    if not frames:
        return np.zeros(Config.FEATURE_DIM)

    # Extract features
    with torch.no_grad():
        frame_tensors = torch.stack(frames).to(device)
        features = resnet(frame_tensors)
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        video_features = features.mean(dim=0).cpu().numpy()  # Average across frames

    return video_features

# Load Dataset
def load_dataset():
    """Load all video paths and their labels from the dataset"""
    video_paths = []
    labels = []

    for action in Config.ACTIONS:
        action_dir = os.path.join(Config.DATA_DIR, action)
        if not os.path.exists(action_dir):
            continue

        # Find all video folders (v_*)
        for video_folder in os.listdir(action_dir):
            if video_folder.startswith('v_') and os.path.isdir(os.path.join(action_dir, video_folder)):
                video_path = os.path.join(action_dir, video_folder)
                # For simplicity, use the first .avi file in the folder
                avi_files = [f for f in os.listdir(video_path) if f.endswith('.avi')]
                if avi_files:
                    video_file = os.path.join(video_path, avi_files[0])
                    video_paths.append(video_file)
                    labels.append(Config.ACTION_TO_ID[action])

    return video_paths, labels

# Active Learning Functions
def uncertainty_sampling(model, unlabeled_loader, device, n_samples=10):
    """Select most uncertain samples using least confidence"""
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for features, indices in unlabeled_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            uncertainty = 1 - confidence
            for idx, unc in zip(indices, uncertainty.cpu().numpy()):
                uncertainties.append((idx, unc))

    # Sort by uncertainty (highest first)
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in uncertainties[:n_samples]]

    return selected_indices

# Training Function
def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """Train the video classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_accuracy = 0.0
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        epoch_losses.append(train_loss/len(train_loader))
        epoch_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return best_accuracy, val_preds, val_labels

# Main Active Learning Pipeline
def active_learning_pipeline():
    """Run the complete active learning pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    video_paths, labels = load_dataset()
    print(f"Found {len(video_paths)} videos across {Config.NUM_CLASSES} classes")

    # Split into initial labeled and unlabeled
    labeled_indices = []
    unlabeled_indices = list(range(len(video_paths)))

    # Ensure initial labeled samples from each class
    for class_id in range(Config.NUM_CLASSES):
        class_indices = [i for i, label in enumerate(labels) if label == class_id]
        if len(class_indices) >= Config.INITIAL_LABELED // Config.NUM_CLASSES:
            selected = random.sample(class_indices, Config.INITIAL_LABELED // Config.NUM_CLASSES)
        else:
            selected = class_indices
        labeled_indices.extend(selected)
        for idx in selected:
            unlabeled_indices.remove(idx)

    print(f"Initial labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

    # Initialize model
    model = VideoClassifier(Config.FEATURE_DIM, Config.NUM_CLASSES).to(device)

    # Metrics tracking
    accuracies = []
    labeled_counts = []
    all_uncertainties = []
    final_confusion_matrix = None
    labeled_counts.append(len(labeled_indices))

    for iteration in range(Config.MAX_ITERATIONS):
        print(f"\n--- Active Learning Iteration {iteration + 1} ---")

        # Create datasets
        labeled_paths = [video_paths[i] for i in labeled_indices]
        labeled_labels = [labels[i] for i in labeled_indices]

        unlabeled_paths = [video_paths[i] for i in unlabeled_indices]

        # Split labeled into train/val
        train_indices, val_indices = train_test_split(
            range(len(labeled_paths)), test_size=0.1, random_state=42
        )

        train_paths = [labeled_paths[i] for i in train_indices]
        train_labels = [labeled_labels[i] for i in train_indices]
        val_paths = [labeled_paths[i] for i in val_indices]
        val_labels = [labeled_labels[i] for i in val_indices]

        train_dataset = VideoDataset(train_paths, train_labels)
        val_dataset = VideoDataset(val_paths, val_labels)
        unlabeled_dataset = VideoDataset(unlabeled_paths)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        # Train model
        accuracy, val_preds, val_labels_list = train_model(model, train_loader, val_loader, device, Config.NUM_EPOCHS)
        accuracies.append(accuracy)
        
        # Store confusion matrix from final iteration
        if iteration == Config.MAX_ITERATIONS - 1 or len(unlabeled_indices) == 0:
            final_confusion_matrix = confusion_matrix(val_labels_list, val_preds)

        # Select new samples and track uncertainties
        if len(unlabeled_indices) > 0:
            # Get uncertainties for analysis
            model.eval()
            iteration_uncertainties = []
            with torch.no_grad():
                for features, indices in unlabeled_loader:
                    features = features.to(device)
                    outputs = model(features)
                    probs = torch.softmax(outputs, dim=1)
                    confidence = torch.max(probs, dim=1)[0]
                    uncertainty = 1 - confidence
                    iteration_uncertainties.extend(uncertainty.cpu().numpy())
            all_uncertainties.append(iteration_uncertainties)
            
            selected = uncertainty_sampling(model, unlabeled_loader, device, min(Config.QUERY_SIZE, len(unlabeled_indices)))
            selected_global = [unlabeled_indices[idx] for idx in selected]

            labeled_indices.extend(selected_global)
            for idx in sorted(selected_global, reverse=True):
                unlabeled_indices.remove(idx)

            labeled_counts.append(len(labeled_indices))
            print(f"Added {len(selected_global)} new samples. Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

        if len(unlabeled_indices) == 0:
            break

    print(f"\nFinal model accuracy: {accuracies[-1]:.4f}")
    print("Active learning completed!")
    
    # Save metrics for visualization
    metrics = {
        'accuracies': accuracies,
        'labeled_counts': labeled_counts,
        'uncertainties': all_uncertainties,
        'confusion_matrix': final_confusion_matrix.tolist() if final_confusion_matrix is not None else None,
        'num_classes': Config.NUM_CLASSES,
        'class_names': Config.ACTIONS
    }
    
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    with open('training_metrics.json', 'w') as f:
        json_safe_metrics = {
            'accuracies': [float(acc) for acc in accuracies],
            'labeled_counts': labeled_counts,
            'final_accuracy': float(accuracies[-1]),
            'num_classes': Config.NUM_CLASSES,
            'class_names': Config.ACTIONS
        }
        json.dump(json_safe_metrics, f, indent=2)
    
    print("Metrics saved for visualization")

    return model

# Classification Function for New Videos
def classify_video(video_path, model_path='best_model.pth'):
    """Classify a single video"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = VideoClassifier(Config.FEATURE_DIM, Config.NUM_CLASSES).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model")
    else:
        print("Warning: No trained model found. Please run training first.")
        return None

    model.eval()

    # Extract features
    print(f"Extracting features from {video_path}...")
    features = extract_video_features(video_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Classify
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred_class = torch.max(probs, 0)

    predicted_action = Config.ACTIONS[pred_class.item()]
    confidence_score = confidence.item()

    print(f"Predicted Action: {predicted_action}")
    print(f"Confidence: {confidence_score:.4f}")

    # Top 3 predictions
    top3_probs, top3_classes = torch.topk(probs, 3)
    print("\nTop 3 Predictions:")
    for i in range(3):
        action = Config.ACTIONS[top3_classes[i].item()]
        prob = top3_probs[i].item()
        print(f"{i+1}. {action}: {prob:.4f}")

    return predicted_action, confidence_score

if __name__ == "__main__":
    # Run active learning training
    print("Starting Active Learning Training...")
    trained_model = active_learning_pipeline()

    # Example usage for classification
    print("\n--- Example Classification ---")
    # Replace with actual video path
    example_video = r"c:\Users\hp5cd\Downloads\YouTube_DataSet_Annotated\action_youtube_naudio\basketball\v_shooting_01\v_shooting_01_01.avi"
    if os.path.exists(example_video):
        classify_video(example_video)
    else:
        print("Example video not found. Please provide a valid video path for classification.")