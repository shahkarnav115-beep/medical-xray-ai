# app/app.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from app.gradcam import generate_gradcam    # IMPORT FIXED

labels = ["Normal", "Pneumonia"]

def predict(image):
    image.save("uploaded.jpg")
    heatmap, pred, confidence = generate_gradcam("uploaded.jpg")
    return labels[pred], f"{confidence*100:.2f}%", heatmap

ui = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Image(label="GradCAM Heatmap")
    ],
    title="Medical X-Ray Diagnosis AI",
    description="Upload a Chest X-ray to detect Pneumonia and view heatmap."
)

if __name__ == "__main__":
    ui.launch()
