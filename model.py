# app/model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=2):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
