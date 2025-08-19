# Emotion-Aware Adaptive Game 
An interactive facial emotion recognition project that adapts gameplay based on real-time user emotions. It can detect 7 emotions including Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise
Built with **PyTorch**, **OpenCV**, and **Pygame**, trained on the **FER2013 dataset** using a **ResNet18** deep learning model.

## Setup
```bash
conda create -n emo-game python=3.11 -y
conda activate emo-game
python -m pip install -r requirements.txt
```

## Dataset

Download FER2013 via Kaggle and place it under data/ as described in the docs.

## Train
```bash
python train_resnet18.py
```

## Run
```bash
python infer_emotion.py
python game_adaptive.py
```
## Performance

Training on FER2013 for 25 epochs with ResNet18:

- Best train accuracy: **~99.82%**
- Best validation accuracy: **67.3%**
- Test accuracy: **67.64%**
