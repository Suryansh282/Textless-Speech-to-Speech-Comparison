import os
import random
import wave
import torch
import pyaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

# ==================== Set Global Constants ====================
SAMPLING_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
RECORD_SECONDS = 2
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# ==================== Reproducibility ====================
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ==================== Load Feature Extractor ====================
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")


# ==================== Load Contrastive Model ====================
class ContrastiveAudioModel(nn.Module):
    def __init__(self, base_model_name='facebook/wav2vec2-large-xlsr-53', freeze_ratio=0.8):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = False

        num_layers = self.encoder.config.num_hidden_layers
        freeze_layers = int(num_layers * freeze_ratio)
        for layer in self.encoder.encoder.layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_values, attention_mask):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = self.projector(pooled)
        return F.normalize(projected, p=2, dim=1)


model = ContrastiveAudioModel()
model.load_state_dict(torch.load("april_11_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ==================== Silero VAD Setup ====================
from silero_vad import (load_silero_vad, get_speech_timestamps, collect_chunks)

vad_model = load_silero_vad(onnx=False)


# ==================== Audio Recording + VAD ====================
def record_audio_with_silero_vad(filename_prefix="audio", save_dir="recordings", rate=SAMPLING_RATE):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{filename_prefix}.wav")

    # Record audio using PyAudio
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=CHUNK)

    print(f"🎤 Recording '{filename_prefix}' for {RECORD_SECONDS} seconds...")
    frames = [stream.read(CHUNK) for _ in range(int(rate / CHUNK * RECORD_SECONDS))]

    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
    print("🛑 Recording complete.")

    # Convert and normalize audio
    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.tensor(audio_np).unsqueeze(0)

    # Apply VAD
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=rate,
        threshold=0.2, min_speech_duration_ms=70, min_silence_duration_ms=30
    )

    if not speech_timestamps:
        print("❌ No speech detected.")
        return None

    speech_tensor = collect_chunks(speech_timestamps, audio_tensor.squeeze(0))
    speech_np = speech_tensor.squeeze().numpy()

    # Save processed audio
    scaled_speech = np.clip(speech_np * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(scaled_speech.tobytes())

    print(f"✅ Cleaned audio saved to: {filepath}")
    return speech_np


# ==================== Similarity Prediction ====================
@torch.no_grad()
def predict_similarity_from_mic(model, feature_extractor, device, threshold=0.9):
    print("🔴 Record first audio sample...")
    audio1 = record_audio_with_silero_vad(filename_prefix="audio1")
    if audio1 is None:
        return

    print("🔵 Record second audio sample...")
    audio2 = record_audio_with_silero_vad(filename_prefix="audio2")
    if audio2 is None:
        return

    input1 = feature_extractor([audio1], sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    input2 = feature_extractor([audio2], sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)

    input1 = {k: v.to(device) for k, v in input1.items()}
    input2 = {k: v.to(device) for k, v in input2.items()}

    emb1 = model(input1["input_values"], input1["attention_mask"])
    emb2 = model(input2["input_values"], input2["attention_mask"])

    similarity = cosine_similarity(emb1.cpu(), emb2.cpu())[0][0]
    print(f"\n🔍 Cosine Similarity: {similarity:.4f}")

    if similarity >= threshold:
        print("✅ SAME audio (similar meaning or pronunciation)")
    else:
        print("❌ DIFFERENT audio")

    return similarity


# ==================== Run the Tool ====================
if __name__ == "__main__":
    similarity_score = predict_similarity_from_mic(model, feature_extractor, DEVICE)
    print(f"🔁 Final Similarity Score: {similarity_score}")
