# dataset_fer2013.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FER2013CSV(Dataset):
    """
    Reads fer2013.csv with columns: [emotion, pixels, Usage].
    Produces 3-channel images (replicated grayscale) for ResNet.
    """
    def __init__(self, csv_path, usage='Training', transform=None):
        self.df = pd.read_csv(csv_path)
        # 'Usage' values are usually: Training, PublicTest, PrivateTest
        self.df = self.df[self.df['Usage'] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emotion = int(row['emotion'])
        # Pixels are space-separated values for a 48x48 grayscale image
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
        img = pixels.reshape(48, 48)
        img = Image.fromarray(img, mode='L')  # grayscale
        img = img.convert('RGB')              # replicate to 3 channels for ResNet
        if self.transform:
            img = self.transform(img)
        return img, emotion

EMOTION_ID2NAME = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

