# app/gradcam.py
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
from app.model import build_model      # <—— IMPORTANT!

def generate_gradcam(image_path, model_path="models/best_model_vgg19.pt"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    from torch.hub import load_state_dict_from_url

    model_url = "https://github.com/shahkarnav115-beep/medical-xray-ai/releases/download/v1.0.0/best_model_vgg19.pt"
    state_dict = load_state_dict_from_url(model_url, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    inp = transform(img).unsqueeze(0).to(device)
    inp.requires_grad = True

    features = []
    def hook(model, i, o): features.append(o)
    model.features[-1].register_forward_hook(hook)

    out = model(inp)
    pred = out.argmax()
    score = out[0][pred]
    score.backward()

    fmap = features[0][0].detach().cpu().numpy()
    grads = model.features[-1].weight.grad.detach().cpu().numpy()

    cam = np.maximum((fmap * grads).sum(axis=0), 0)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = (cam - cam.min())/(cam.max()-cam.min())

    heatmap = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    output_path = "gradcam_result.jpg"
    cv2.imwrite(output_path, overlay[:, :, ::-1])
    return output_path, int(pred), float(torch.softmax(out,1)[0][pred])
