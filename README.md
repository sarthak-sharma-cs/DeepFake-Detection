# Deepfake Detection using Deep Learning

This project implements a deep learning-based system to classify images as **real or fake (deepfake)**. It uses Convolutional Neural Networks (CNNs) and transfer learning models such as MobileNetV2 and EfficientNetB0 for improved performance.

---

## Overview

The project trains multiple models on a dataset containing real and fake face images. It includes:

* Basic CNN model
* Transfer learning using MobileNetV2
* Transfer learning using EfficientNetB0
* Fine-tuning of pretrained models
* Performance evaluation using confusion matrix and classification report
* Grad-CAM visualization for interpretability

---

## Dataset Structure

The dataset should be organized as follows:

```
real_and_fake_face/
│── training_real/
│── training_fake/
```

---

## Features

* Image preprocessing and visualization
* Binary classification (Real vs Fake)
* CNN-based model training
* Transfer learning with pretrained models
* Model fine-tuning
* Performance evaluation (accuracy, confusion matrix, classification report)
* Grad-CAM for model interpretability
* Class imbalance handling using class weights

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* PIL (Python Imaging Library)

---

## Model Architectures

### 1. Custom CNN

* Multiple convolution + pooling layers
* Fully connected layers
* Sigmoid output for binary classification

### 2. MobileNetV2

* Pretrained on ImageNet
* Frozen base model
* Fine-tuned last layers

### 3. EfficientNetB0

* Pretrained architecture
* Better performance and efficiency
* Fine-tuned with class weights

---

## Installation

Clone the repository:

```
git clone https://github.com/sarthak-sharma-cs/DeepFake-Detection
```

Navigate to the project folder:

```
cd 
```

---

## Usage

Run the script:

```
python your_script_name.py
```

Make sure your dataset is placed in the correct directory before running.

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## Model Output

The trained model is saved as:

```
deepfake_model_eff.keras
```

---

## Future Improvements

* Add real-time detection using webcam
* Improve dataset size and diversity
* Deploy as a web application
* Optimize model for faster inference

---

## Author

Sarthak Sharma
