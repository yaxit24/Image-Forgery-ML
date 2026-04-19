# src/cnn_classifier.py
"""
MobileNetV2-based forgery classifier using transfer learning.
Frozen backbone (ImageNet weights) + trainable classification head.
Designed for CPU inference on HuggingFace Spaces free tier.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Inference transform (no augmentation)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Training transforms (heavy augmentation for small dataset)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2),
])


def build_model(num_classes=2):
    """
    Build MobileNetV2 with frozen backbone + trainable head.

    Head architecture:
        Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 2)

    Returns the model with only the classifier head set to requires_grad=True.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    return model


def load_model(model_path, device="cpu"):
    """Load a trained ForgeryClassifier from disk."""
    model = build_model()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, pil_img, device="cpu"):
    """
    Run inference on a single PIL image.

    Returns:
        (label_str, confidence_float)
        e.g. ("forged", 0.87) or ("authentic", 0.93)
    """
    tensor = INFERENCE_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    # index 0 = authentic, index 1 = forged
    forged_prob = probs[1].item()

    if forged_prob >= 0.5:
        return "forged", forged_prob
    else:
        return "authentic", 1.0 - forged_prob
