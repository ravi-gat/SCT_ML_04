
import argparse
import cv2
import torch
import numpy as np
from torchvision import models, transforms
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

def preprocess(frame, img_size):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    return tfm(frame)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    pkg = torch.load(args.ckpt, map_location="cpu")
    class_names = load_class_index(pkg["class_index_path"])
    model = get_model(len(class_names), model_name=pkg["model_name"])
    model.load_state_dict(pkg["model"])
    model.eval()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb, pkg["img_size"]).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(prob.argmax())
        label = f"{class_names[idx]} ({prob[idx]:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
