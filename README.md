🩺 AI-Powered Pneumonia Detection from Chest X-Ray Images
🔍 Deep Learning (VGG19) + GradCAM Explainability + Gradio Web App

This project is an end-to-end medical imaging AI system that detects Pneumonia from chest X-ray images using a fine-tuned VGG19 convolutional neural network.
It also generates GradCAM heatmaps to highlight the infected lung regions, making predictions interpretable for medical professionals.

The model is deployed through a Gradio web app, allowing users to upload X-rays and receive predictions instantly.

🚀 Features

VGG19 Deep Learning Model (Transfer Learning on Chest X-Ray Dataset)

Real-Time Pneumonia Prediction from X-Ray images

GradCAM Explainability → highlights areas contributing to the prediction

Interactive Gradio Web App for easy use

High validation accuracy (93–100%)

Confidence Score for each prediction

Clean, modular code ready for improvements & research work

🧠 Tech Stack
Component	Technology
Model	PyTorch, TorchVision (VGG19 pretrained)
Explainability	GradCAM, OpenCV
Web App	Gradio
Data Handling	PIL, NumPy
Training	GPU-accelerated
📊 Dataset

This model is trained on the Chest X-Ray Images (Pneumonia) dataset, containing:

Normal chest X-rays

Pneumonia cases (bacterial & viral)

Dataset splits:

Train

Validation

Test

🧪 Model Performance
Metric	Value
Train Accuracy	~98%
Validation Accuracy	93–100%
Loss Function	CrossEntropyLoss
Optimizer	Adam (LR = 1e-4)
Image Size	224x224
🔥 How It Works
1️⃣ User uploads a chest X-ray
2️⃣ Model predicts Normal or Pneumonia
3️⃣ GradCAM generates a heatmap showing areas of interest
4️⃣ App displays:

Prediction

Confidence score

GradCAM heatmap
