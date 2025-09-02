
import argparse
import json
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from utils import load_class_index

def get_model(num_classes, model_name='resnet50'):
    if model_name.lower() == 'resnet50':
        net = models.resnet50(weights=None)
        in_features = net.fc.in_features
        net.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features, num_classes)
        )
    elif model_name.lower() == 'mobilenet_v3':
        net = models.mobilenet_v3_large(weights=None)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model.")
    return net

def preprocess(img, img_size):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    return tfm(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    args = parser.parse_args()

    pkg = torch.load(args.ckpt, map_location="cpu")
    class_names = load_class_index(pkg["class_index_path"])
    model = get_model(len(class_names), model_name=pkg["model_name"])
    model.load_state_dict(pkg["model"])
    model.eval()

    img = cv2.imread(args.image)[:, :, ::-1]  # BGR->RGB
    tensor = preprocess(img, pkg["img_size"]).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0].numpy()
    idx = int(prob.argmax())
    print(f"Pred: {class_names[idx]}  prob={prob[idx]:.4f}")
    for i, p in enumerate(prob):
        print(f"{class_names[i]}: {p:.4f}")

if __name__ == "__main__":
    main()
