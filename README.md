

# 🔊 Audio Similarity Detection using Wav2Vec2 + Silero VAD

This project compares the pronunciation similarity of two audio samples recorded through a microphone using a fine-tuned Wav2Vec2-based contrastive learning model. It uses **Silero VAD** to clean the audio, and **cosine similarity** between embeddings to compute how similar the two inputs are.

---

##  Features

- Records two audio clips using microphone
- Applies Silero Voice Activity Detection (VAD) to remove silence/noise
- Extracts embeddings using a custom fine-tuned Wav2Vec2 contrastive model
- Compares the audio using cosine similarity
- Prints whether the audios are **SAME** or **DIFFERENT**
- Lightweight and easy to run

---

## 🛠 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/audio-similarity-wav2vec2.git
cd audio-similarity-wav2vec2
```

## 🛠️ Install Required Packages

```bash
pip install torch torchaudio transformers numpy pyaudio scikit-learn
pip install git+https://github.com/snakers4/silero-vad.git
```
