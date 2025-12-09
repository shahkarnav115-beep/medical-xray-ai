import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from app.gradcam import generate_gradcam

labels = ["Normal", "Pneumonia"]

def predict(image):
    # --- FIX 1: HANDLE NONE INPUT ---
    if image is None:
        return "No Image", "0.00%", None

    image.save("uploaded.jpg")
    heatmap_path, pred, confidence = generate_gradcam("uploaded.jpg")
    
    # Return the path to the heatmap image for Gradio to display
    return labels[pred], f"{confidence*100:.2f}%", heatmap_path

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