# Malaria Parasite Detection AI Model

## Overview

This repository contains an AI model for detecting malaria parasites in blood samples using Convolutional Neural Networks (CNNs). The model is trained on a dataset of blood sample images and is designed to assist in the automated identification of malaria-infected cells.

## 1. Dataset

The dataset used for this project consists of labeled blood sample images, where each image is categorized as either containing malaria parasites or being parasite-free. The dataset should be organized into two folders: `infected` and `uninfected`.

## 2. Data Preprocessing

Before training the model, the dataset undergoes preprocessing to ensure consistency and improve the model's performance. Common preprocessing steps include resizing images, normalizing pixel values, and data augmentation.

## 3. Data Splitting

The dataset is split into training, validation, and test sets. The recommended split is often 70-80% for training, 10-15% for validation, and 10-15% for testing. This ensures that the model is trained on diverse data and evaluated on unseen samples.
Images were split into training and testing sets (80% and 20%, respectively).

## 4. Data Visualization

Visualizations, such as histograms or sample images, can be created to understand the dataset's distribution and characteristics. This step helps in identifying potential biases or imbalances in the data. A barplot was used for this project.

## 5. Model Building

Build a sequential model with convolutional layers for feature extraction.
Utilize max-pooling layers to downsample feature maps.
Implement dense layers for classification, including dropout for regularization.
Use the sigmoid activation function for binary classification.

## 6. Image Generation

Data augmentation is applied to the training set to increase model robustness. Techniques such as rotation, flipping, and zooming are commonly used. Image generators are set up to apply these transformations on the fly during training.

## 7. Model Training

The model is trained using the training set, and its performance is monitored on the validation set. Training parameters, such as learning rate and batch size, are fine-tuned to achieve optimal results. Early stopping is implemented to prevent overfitting.

## 8. Evaluation

After training, the model's performance is evaluated on the test set using metrics such as accuracy, precision, recall, and F1 score. Confusion matrices and ROC curves can also be generated to assess the model's behavior.

## 9. Usage

To use the trained model for inference, load the saved model weights and apply the model to new blood sample images. The predictions can be thresholded to determine the presence of malaria parasites.

```python
# Example code for inference
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('malaria_detection_model.h5')

# Load and preprocess a new blood sample image
image = cv2.imread('new_sample.png')
image = cv2.resize(image, (224, 224))
image = image / 255.0  # Normalize pixel values

# Perform inference
prediction = model.predict(np.expand_dims(image, axis=0))

# Threshold the prediction
if prediction > 0.5:
    print("Malaria Parasite Detected")
else:
    print("No Malaria Parasite Detected")
```

## 10. Dependencies

Ensure you have the required dependencies installed. You can use the following command to install them:

```bash
pip install -r requirements.txt
```

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Acknowledgments**
The dataset used in this project is sourced from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria.