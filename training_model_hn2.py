"""
Evaluate embeddings from intermediate layers of a pre-trained Conformer-CTC model
by computing DTW distance between them for speaker/word similarity tasks.

"""

import os
import random
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from dtw_code import dtw  # You must have this defined in dtw_code.py

# ---------------------------- Dataset Class ---------------------------- #

class AudioContrastiveDataset(Dataset):
    """
    Dataset for generating contrastive audio pairs.
    Each audio is either paired with another from the same group (positive) or from a different group (negative).
    """

    def __init__(self, df, max_length=16000 * 5, silence_duration=0.1):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.silence_duration = silence_duration
        self.group_to_indices = self.df.groupby('Group').apply(lambda x: list(x.index)).to_dict()

        # Precompute all valid positive pairs
        self.pairs = []
        for group, indices in self.group_to_indices.items():
            if len(indices) >= 2:
                self.pairs += [(i, j, 1) for i in indices for j in indices if i < j]

        self.all_indices = list(self.df.index)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        # Randomly replace second audio with a negative pair (different group)
        if random.random() < 0.5:
            idx2 = random.choice(self.all_indices)
            while self.df.loc[idx2, 'Group'] == self.df.loc[idx1, 'Group']:
                idx2 = random.choice(self.all_indices)
            label = 0

        audio1 = self._load_segment(self.df.loc[idx1])
        audio2 = self._load_segment(self.df.loc[idx2])
        return audio1, audio2, torch.tensor(label, dtype=torch.float32)

    def _load_segment(self, row):
        path, boundary_str = row['Path'], row['Boundary']
        try:
            start, end = eval(boundary_str)
        except:
            start, end = map(float, boundary_str.split(','))

        waveform, sr = torchaudio.load(path)
        waveform = waveform[:1, :]  # Mono
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        segment = waveform[:, int(start * 16000):int(end * 16000)]

        # Pad with silence at beginning and end
        silence = torch.zeros((1, int(self.silence_duration * 16000)))
        segment = torch.cat([silence, segment, silence], dim=1)

        # Clip or pad to max_length
        segment = segment[:, :self.max_length] if segment.shape[1] > self.max_length else \
                  torch.cat([segment, torch.zeros((1, self.max_length - segment.shape[1]))], dim=1)

        return segment.squeeze()


# ---------------------------- Feature Extraction via Hook ---------------------------- #

def register_conformer_layer_hook(model, target_layer_output, layer_index=9):
    def hook_fn(module, input, output):
        target_layer_output[0] = output.detach()
    return model.encoder.layers[layer_index].register_forward_hook(hook_fn)


# ---------------------------- Custom Audio Model ---------------------------- #

class CustomAudioModel(nn.Module):
    """
    Wraps a pre-trained NeMo Conformer-CTC model and extracts intermediate layer embeddings.
    """

    def __init__(self, restore_path, freeze_ratio=0.8):
        super().__init__()
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=restore_path)

    def forward(self, input_values):
        self.asr_model.eval()
        if hasattr(self.asr_model, "spec_augmentation"):
            self.asr_model.spec_augmentation = None

        input_values = input_values.unsqueeze(0)  # Add batch dim
        input_len = torch.tensor([input_values.shape[1]], device=input_values.device)

        # Hook to grab intermediate layer output
        layer_output = [None]
        hook = register_conformer_layer_hook(self.asr_model, layer_output, layer_index=9)
        _ = self.asr_model(input_signal=input_values, input_signal_length=input_len)
        hook.remove()

        features = layer_output[0]  # shape: [B, T, D]
        if features is None:
            raise RuntimeError("Hook did not capture output.")
        return F.normalize(features, p=2, dim=-1)  # Normalize each timestep embedding


# ---------------------------- Cosine Distance ---------------------------- #

def cosine_distance(x, y):
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1.0 - (dot / (norm_x * norm_y)) if norm_x != 0 and norm_y != 0 else 1.0


# ---------------------------- Plotting Functions ---------------------------- #

def plot_full_similarity_analysis(y_true, y_scores, save_prefix="eval", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Histogram
    same_sims = y_scores[y_true == 1]
    diff_sims = y_scores[y_true == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(same_sims, bins=30, alpha=0.6, label='Same', color='green')
    plt.hist(diff_sims, bins=30, alpha=0.6, label='Different', color='red')
    plt.axvline(np.mean(same_sims), linestyle='--', color='blue')
    plt.axvline(np.mean(diff_sims), linestyle='--', color='black')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_hist.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_scores):.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_roc.png"))
    plt.close()


# ---------------------------- Evaluation Pipeline ---------------------------- #

@torch.no_grad()
def evaluate_model_custom_conf(model, dataset, device, dataset_name="CustomConf"):
    model.eval()
    y_true, y_scores = [], []

    for audio1, audio2, label in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
        emb1 = model(audio1.to(device)).squeeze(0).cpu().numpy()
        emb2 = model(audio2.to(device)).squeeze(0).cpu().numpy()
        dist, *_ = dtw(emb1, emb2, dist=cosine_distance)
        y_true.append(int(label.item()))
        y_scores.append(-dist)  # Convert to similarity

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Threshold by EER
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    optimal_threshold = thresholds[eer_idx]
    y_pred = (y_scores >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "Dataset": dataset_name,
        "TPR": tp / (tp + fn),
        "FPR": fp / (fp + tn),
        "TNR": tn / (tn + fp),
        "FNR": fn / (fn + tp),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_scores),
        "Threshold": optimal_threshold
    }

    # Save leaderboard
    leaderboard_path = f"leaderboard_{dataset_name}.csv"
    df = pd.DataFrame([metrics])
    df.to_csv(leaderboard_path, index=False)
    print(f"📊 Saved results to {leaderboard_path}")

    # Save plots
    plot_full_similarity_analysis(y_true, y_scores, save_prefix=dataset_name)

    return metrics


# ---------------------------- Main Execution ---------------------------- #

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    restore_path = "/hdd_storage/data/ASR_datasets/NeMo_models/hindi/stt_hi_conformer_ctc_medium.nemo"
    model = CustomAudioModel(restore_path=restore_path).to(device)

    df = pd.read_csv("/hdd_storage/users/sld_tool/Suryansh/data/small_data_kb_hi2.csv")
    df_train_val, df_test = train_test_split(df, test_size=0.5, random_state=42)

    valid_groups = df_train_val['Group'].value_counts()[lambda x: x >= 2].index
    df_valid = df_train_val[df_train_val['Group'].isin(valid_groups)].reset_index(drop=True)
    df_train, df_val = train_test_split(df_valid, test_size=0.5, random_state=42)

    print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

    val_dataset = AudioContrastiveDataset(df_val)
    results, _ = evaluate_model_custom_conf(model, val_dataset, device)
    print(results)
