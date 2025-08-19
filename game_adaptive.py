# game_adaptive.py
import time, random, cv2, yaml, numpy as np, torch, torch.nn as nn, pygame
from torchvision import models, transforms
from PIL import Image
from utils import get_device, EmotionSmoother

WINDOW_W, WINDOW_H = 960, 600
PLAYER_W, PLAYER_H = 60, 20
ENEMY_W, ENEMY_H   = 40, 20

COLOR_BG     = (15, 18, 30)
COLOR_PLAYER = (80, 180, 255)
COLOR_ENEMY  = (255, 80, 110)
COLOR_TEXT   = (230, 230, 230)

DEFAULT_EMOTION_DIFFICULTY = {
    'happy': 1.30,
    'surprise': 1.20,
    'disgust': 1.10,
    'neutral': 1.00,
    'sad': 0.85,
    'angry': 0.85,
    'fear': 0.80,
}

class EmotionPredictor:
    def __init__(self, weights_path, img_size=224, device=None):
        self.device = device or get_device()
        self.model  = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 7))
        ckpt = torch.load(weights_path, map_location='cpu')
        self.class_names = ckpt['config']['class_names']
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval().to(self.device)

        mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam at index 0. Close other apps (FaceTime/Teams/Zoom), then try again.")

    def read_emotion(self):
        ok, frame = self.cap.read()
        if not ok:
            return frame, None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
        if len(faces) == 0:
            return frame, None, None
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        x_t = self.tfm(Image.fromarray(face_rgb).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
        return frame, self.class_names[pred], probs[pred]

    def release(self):
        self.cap.release()

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WINDOW_W//2 - PLAYER_W//2, WINDOW_H - 80, PLAYER_W, PLAYER_H)
        self.speed = 360
    def update(self, dt, keys):
        vx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        self.rect.x += int(vx * self.speed * dt)
        self.rect.x = max(0, min(WINDOW_W - PLAYER_W, self.rect.x))
    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_PLAYER, self.rect, border_radius=6)

class Enemy:
    def __init__(self, y=-ENEMY_H, speed=120):
        self.rect = pygame.Rect(random.randint(0, WINDOW_W-ENEMY_W), y, ENEMY_W, ENEMY_H)
        self.speed = speed
    def update(self, dt, speed_mult):
        self.rect.y += int(self.speed * speed_mult * dt)
    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_ENEMY, self.rect, border_radius=4)

class Game:
    def __init__(self, predictor: EmotionPredictor):
        pygame.init()
        pygame.display.set_caption('Emotion-Aware Adaptive Game')
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Menlo', 20)

        self.predictor = predictor
        self.smoother = EmotionSmoother(window=7)

        # Build difficulty map using whatever class names were used during training
        self.diff_map = {}
        for name in self.predictor.class_names:
            k = name.lower()
            self.diff_map[name] = DEFAULT_EMOTION_DIFFICULTY.get(k, 1.0)

        self.player = Player()
        self.enemies = []
        self.base_enemy_speed = 130
        self.spawn_timer = 0.0
        self.spawn_interval = 0.8

        self.current_emotion = 'Neutral'
        self.current_conf = 0.0
        self.speed_mult = 1.0
        self.last_pred_time = 0.0
        self.pred_every_sec = 0.35
        self.running = True

    def update_difficulty(self, target_emotion: str):
        if not target_emotion:
            target_emotion = 'Neutral'
        target = self.diff_map.get(target_emotion, 1.0)
        self.speed_mult += (target - self.speed_mult) * 0.1

    def spawn_enemy(self):
        self.enemies.append(Enemy(speed=self.base_enemy_speed))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False

            now = time.time()
            if now - self.last_pred_time > self.pred_every_sec:
                frame, label, conf = self.predictor.read_emotion()
                self.last_pred_time = now
                if label is not None:
                    label = self.smoother.update(label)
                    self.current_emotion = label
                    self.current_conf = conf or 0.0
                    self.update_difficulty(label)

            self.player.update(dt, keys)

            self.spawn_timer += dt
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_timer = 0.0
                self.spawn_enemy()

            for e in self.enemies:
                e.update(dt, self.speed_mult)

            self.enemies = [e for e in self.enemies if e.rect.y < WINDOW_H]
            for e in self.enemies:
                if e.rect.colliderect(self.player.rect):
                    self.speed_mult = max(0.7, self.speed_mult * 0.8)

            self.screen.fill(COLOR_BG)
            self.player.draw(self.screen)
            for e in self.enemies:
                e.draw(self.screen)

            hud = f"Emotion: {self.current_emotion:>7}   diff× {self.speed_mult:.2f}   enemies: {len(self.enemies)}   FPS: {self.clock.get_fps():.0f}"
            self.screen.blit(self.font.render(hud, True, COLOR_TEXT), (16, 12))
            self.screen.blit(self.font.render("Esc to quit", True, (170,170,170)), (16, 36))
            pygame.display.flip()

        self.predictor.release()
        pygame.quit()

def main():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    predictor = EmotionPredictor('models/resnet18_fer2013_best.pth', img_size=cfg['image_size'])
    Game(predictor).run()

if __name__ == '__main__':
    main()

