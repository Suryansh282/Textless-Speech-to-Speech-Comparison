# In this code we are taking the wav2vec model and fine - tune it on or datasets 
# also we are adding projection layer on top of that .

import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve

# Clear CUDA cache for safety
torch.cuda.empty_cache()


# ---------------------- Dataset Definition ---------------------- #
class AudioContrastiveDataset(Dataset):
    """
    Dataset for building positive and negative audio pairs for contrastive learning.
    """

    def __init__(self, df, max_length=16000 * 5, silence_duration=0.1):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.silence_duration = silence_duration

        self.group_to_indices = self.df.groupby('Group').apply(lambda x: list(x.index)).to_dict()

        self.pairs = []
        for group, indices in self.group_to_indices.items():
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        self.pairs.append((indices[i], indices[j], 1))  # positive pair

        self.all_indices = list(self.df.index)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        # Randomly flip to a negative sample
        if random.random() < 0.5:
            idx2 = random.choice(self.all_indices)
            while self.df.loc[idx2, 'Group'] == self.df.loc[idx1, 'Group']:
                idx2 = random.choice(self.all_indices)
            label = 0

        audio1 = self.load_segment(self.df.loc[idx1])
        audio2 = self.load_segment(self.df.loc[idx2])

        return audio1, audio2, torch.tensor(label, dtype=torch.float32)

    def load_segment(self, row):
        path = row['Path']
        start, end = eval(row['Boundary']) if isinstance(row['Boundary'], str) else row['Boundary']

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]  # keep mono

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            sr = 16000

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = waveform[:, start_sample:end_sample]

        silence = torch.zeros((1, int(self.silence_duration * sr)))
        segment = torch.cat([silence, segment, silence], dim=1)

        if segment.shape[1] > self.max_length:
            segment = segment[:, :self.max_length]
        elif segment.shape[1] < self.max_length:
            pad = torch.zeros((1, self.max_length - segment.shape[1]))
            segment = torch.cat([segment, pad], dim=1)

        return segment.squeeze()


# ---------------------- Model Definition ---------------------- #
class ContrastiveAudioModel(nn.Module):
    """
    Wav2Vec2-based encoder model with a projection head for contrastive learning.
    """

    def __init__(self, base_model_name='facebook/wav2vec2-large-xlsr-53', freeze_ratio=0.8):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze layers
        for param in self.encoder.feature_extractor.parameters():
            param.requires_grad = False

        num_layers = self.encoder.config.num_hidden_layers
        freeze_layers = int(num_layers * freeze_ratio)
        for layer in self.encoder.encoder.layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_values):
        outputs = self.encoder(input_values)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        projected = self.projector(hidden_states)
        return F.normalize(projected, p=2, dim=1)


# ---------------------- Contrastive Loss ---------------------- #
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2):
        batch_size = emb1.size(0)
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        embeddings = torch.cat([emb1, emb2], dim=0)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_matrix.masked_fill_(torch.eye(2 * batch_size, device=emb1.device).bool(), -1e4)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(emb1.device)
        return F.cross_entropy(sim_matrix, labels)


# ---------------------- Trainer Class ---------------------- #
class Trainer:
    def __init__(self, model, feature_extractor, device, save_path="best_model_kn_april21.pt"):
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.device = device
        self.criterion = InfoNCELoss()
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        self.scaler = GradScaler()
        self.save_path = save_path
        self.best_auc = 0.0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for audio1, audio2, _ in tqdm(loader, desc="Training", leave=False):
            inputs1 = self.feature_extractor(audio1.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs2 = self.feature_extractor(audio2.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

            with autocast():
                emb1 = self.model(inputs1['input_values'].to(self.device))
                emb2 = self.model(inputs2['input_values'].to(self.device))
                loss = self.criterion(emb1, emb2)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        all_labels, all_scores = [], []

        with torch.no_grad():
            for audio1, audio2, labels in tqdm(loader, desc="Evaluating", leave=False):
                inputs1 = self.feature_extractor(audio1.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs2 = self.feature_extractor(audio2.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

                emb1 = self.model(inputs1['input_values'].to(self.device))
                emb2 = self.model(inputs2['input_values'].to(self.device))

                sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
                all_scores.extend(sims)
                all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        roc_auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        eer = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        preds = (all_scores >= eer).astype(int)

        f1 = f1_score(all_labels, preds)
        prec = precision_score(all_labels, preds)
        rec = recall_score(all_labels, preds)
        acc = accuracy_score(all_labels, preds)

        return roc_auc, eer, acc, prec, rec, f1

    def train(self, train_loader, val_loader, num_epochs=5):
        for epoch in range(1, num_epochs + 1):
            print(f"\n📘 Epoch {epoch}/{num_epochs}")
            loss = self.train_epoch(train_loader)

            roc_auc, eer, acc, prec, rec, f1 = self.evaluate(val_loader)
            print(f"Loss: {loss:.4f} | ROC-AUC: {roc_auc:.4f} | EER: {eer:.4f} | "
                  f"Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

            if roc_auc > self.best_auc:
                self.best_auc = roc_auc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ New best model saved to {self.save_path} (ROC-AUC: {roc_auc:.4f})")


# ---------------------- Main Entry ---------------------- #
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Load data
    df = pd.read_csv("/hdd_storage/users/sld_tool/Suryansh/data/small_data_kb_hi2.csv")

    # Split into train/val
    df_train_val, df_test = train_test_split(df, test_size=0.5, random_state=42)
    valid_groups = df_train_val['Group'].value_counts()[lambda x: x >= 2].index
    df_valid = df_train_val[df_train_val['Group'].isin(valid_groups)].reset_index(drop=True)

    df_train, df_val = train_test_split(df_valid, test_size=0.5, random_state=42)
    print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

    # Load datasets
    train_dataset = AudioContrastiveDataset(df_train)
    val_dataset = AudioContrastiveDataset(df_val)

    def collate_fn(batch):
        audio1, audio2, label = zip(*batch)
        return torch.stack(audio1), torch.stack(audio2), torch.stack(label)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = ContrastiveAudioModel().to(device)
    trainer = Trainer(model, feature_extractor, device)

    trainer.train(train_loader, val_loader, num_epochs=15)

    torch.save(model.state_dict(), "april_21_hn.pt")
    print("✅ Final model saved as april_21_hn.pt")
