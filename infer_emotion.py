# infer_emotion.py
import cv2, yaml, numpy as np, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils import get_device

def build_transform(img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_model(weights_path: str, num_classes: int, device=None):
    device = device or get_device()
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    class_names = ckpt['config']['class_names']
    return model, device, class_names

def detect_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2]*f[3])

def main():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    weights_path = 'models/resnet18_fer2013_best.pth'
    model, device, class_names = load_model(weights_path, num_classes=cfg['num_classes'])
    tfm = build_transform(cfg['image_size'])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam at index 0. Close other apps (FaceTime/Teams/Zoom), then try again.")
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            box = detect_face(gray, face_cascade)
            label_text = 'No face'
            if box is not None:
                x,y,w,h = box
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                face = rgb[y:y+h, x:x+w]
                pil = Image.fromarray(face).convert('RGB')
                x_t = tfm(pil).unsqueeze(0).to(device)
                logits = model(x_t)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(np.argmax(probs))
                label_text = f"{class_names[pred]} ({probs[pred]*100:.1f}%)"

            cv2.putText(frame, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('Emotion Inference (q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

