# Training HuBERT model for acoustic word embedding (paper implementation for English)

#%%
from huggingface_hub import login
# Login to Hugging Face Hub
login()

#%%
import os
import random
import math
import ast
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve , precision_recall_curve, average_precision_score
from transformers import HubertModel, AutoProcessor, AutoFeatureExtractor
from tqdm import tqdm
from dtw_code import dtw
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import nemo.collections.asr as nemo_asr
import jiwer
from sklearn.metrics import confusion_matrix, classification_report





#%%
def register_hubert_layer_hook(model, target_layer_output, layer_index=9):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # extract actual tensor if it's wrapped in a tuple
        target_layer_output[0] = output.detach()
    return model.encoder.layers[layer_index].register_forward_hook(hook_fn)


class CustomHuBERTModel(nn.Module):
    
    def __init__(self, model_name='facebook/hubert-base-ls960', layer_index=9, device=None):
        
      super().__init__()
      self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(self.device).float()

      self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
      self.layer_index = layer_index
      
    def forward(self, input_waveform, sampling_rate=16000):
        
      
      # to ensure the batch dim
      if input_waveform.dim() == 1:
            input_waveform = input_waveform.unsqueeze(0)
            
    #   print(f'input_wavform : {input_waveform.shape} ')
    #   inputs = self.feature_extractor(input_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
      if isinstance(input_waveform, torch.Tensor):
        input_waveform = input_waveform.cpu().numpy()

      if input_waveform.ndim == 2:
        input_waveform = [x for x in input_waveform]  # list of 1D arrays
    #   print(f'input_wavform after numpy : {len(input_waveform)} ')
    #   print(f' one input_wavform  : {input_waveform[0].shape} ')
      
      inputs = self.feature_extractor(input_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

      inputs = {k: v.to(self.device) for k, v in inputs.items()}

      layer_output = [None]
      hook = register_hubert_layer_hook(self.model, layer_output, self.layer_index)
    #   shape_inp = inputs['input_values'].shape
    #   print(f'inputs value : {shape_inp} ')
      input_values = inputs['input_values']
      with torch.no_grad():
          _ = self.model(input_values)
      hook.remove()

      features = layer_output[0]
      if features is None:
          raise RuntimeError("Hook did not capture any output.")

      normed = F.normalize(features, p=2, dim=-1)
      pooled = normed.mean(dim=1)
    #   pooled, _ = normed.min(dim=1)
    #   print(f'pooled shape : {pooled.shape} ')
      
      plt.plot(input_values[0].cpu().numpy())
      plt.title("Input Values After FeatureExtractor")
      plt.xlabel("Time (padded)")
      plt.ylabel("Value")
      plt.grid(True)
      plt.show()
      
      plt.plot(pooled[0].cpu().numpy())
      plt.title("Pooled Values After Model output")
      plt.xlabel("DIM")
      plt.ylabel("Value")
      plt.grid(True)
      plt.show()

      return pooled

  
    


#%%

class AudioContrastiveDataset(Dataset):
    def __init__(self, df, max_length=16000*5, silence_duration=0.1):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.silence_duration = silence_duration

        self.group_to_indices = self.df.groupby('Group').apply(lambda x: list(x.index)).to_dict()

        self.pairs = []
        for group, indices in self.group_to_indices.items():
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        self.pairs.append((indices[i], indices[j], 1))

        self.all_indices = list(self.df.index)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]

        if random.random() < 0.5:
            idx2 = random.choice(self.all_indices)
            while self.df.loc[idx2, 'Group'] == self.df.loc[idx1, 'Group']:
                idx2 = random.choice(self.all_indices)
            label = 0

        row1 = self.df.loc[idx1]
        row2 = self.df.loc[idx2]

        audio1 = self.load_segment(row1)
        audio2 = self.load_segment(row2)

        return audio1, audio2, torch.tensor(label, dtype=torch.float32)

    def load_segment(self, row):
        path = row['Path']
        boundary_str = row['Boundary']
        try:
            start, end = eval(boundary_str)
        except:
            start, end = map(float, boundary_str.split(','))

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = waveform[:, start_sample:end_sample]

        silence_samples = int(self.silence_duration * sr)
        silence = torch.zeros((1, silence_samples))
        segment = torch.cat([silence, segment, silence], dim=1)

        if segment.shape[1] > self.max_length:
            segment = segment[:, :self.max_length]
        elif segment.shape[1] < self.max_length:
            pad = torch.zeros((1, self.max_length - segment.shape[1]))
            segment = torch.cat([segment, pad], dim=1)

        return segment.squeeze()


# Example usage
#%%

model = CustomHuBERTModel()

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

# testing on just two audio 

waveform1, sr1 = torchaudio.load("/hdd_storage/users/sld_tool/Suryansh/code/eng_audio/audio1.wav")

waveform2, sr2 = torchaudio.load("/hdd_storage/users/sld_tool/Suryansh/code/eng_audio/audio2.wav")


embedding1 = model(waveform1, sampling_rate=sr1)
embedding2 = model(waveform2, sampling_rate = sr2)
# print("Pooled Embedding Shape:", embedding.shape)
cosine_sim = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())
print(f'cosine_similarity : {cosine_sim}')

if(cosine_sim < 0.73):
    print(f'❌ DIFFERENT')
else:
    print("✅ SAME")

# %%

# Load and split data
df = pd.read_csv("../data/small_data_kb_hi2.csv")
df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['Group'], random_state=42)
group_counts = df_train_val['Group'].value_counts()
valid_groups = group_counts[group_counts >= 2].index
print('Valid groups:', len(valid_groups))


#%%

df_valid = df_train_val[df_train_val['Group'].isin(valid_groups)].reset_index(drop=True)
df_train, df_val = train_test_split(df_valid, test_size=0.4, stratify=df_valid['Group'], random_state=42)
print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

#%%

####### Trying to create smaller validation datasets ################
    
# Count the number of samples per group.
group_counts2 = df_val["Group"].value_counts()

# Filter out groups that have fewer than 2 samples.
valid_groups2 = group_counts2[group_counts2 >= 2].index
df_valid2 = df_val[df_val["Group"].isin(valid_groups2)].reset_index(drop=True)

# split the train into train and val
df_val2, df_val3 = train_test_split(df_valid2, test_size=0.4, random_state=42)


#%%

train_dataset = AudioContrastiveDataset(df_train)
val_dataset = AudioContrastiveDataset(df_val)
val_dataset3 = AudioContrastiveDataset(df_val3)

#%%

def collate_fn(batch):
    audio1, audio2, label = zip(*batch)
    return torch.stack(audio1), torch.stack(audio2), torch.stack(label)

#%%

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)


#%%

###############################################################################
###############################################################################

########## plotting the roc and histogram ##########

def plot_full_similarity_analysis(y_true, y_scores, save_prefix="april_11_again_eval", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # === Separate scores
    same_sims = y_scores[y_true == 1]
    diff_sims = y_scores[y_true == 0]

    mean_same = np.mean(same_sims)
    mean_diff = np.mean(diff_sims)

    # === 1. Similarity Histogram ===
    plt.figure(figsize=(8, 4))
    plt.hist(same_sims, bins=30, alpha=0.6, label='Same Word', color='green')
    plt.hist(diff_sims, bins=30, alpha=0.6, label='Different Word', color='red')
    plt.axvline(mean_same, color='blue', linestyle='--', label=f'Mean Same = {mean_same:.2f}')
    plt.axvline(mean_diff, color='darkred', linestyle='--', label=f'Mean Diff = {mean_diff:.2f}')
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    hist_path = os.path.join(save_dir, f"{save_prefix}_histogram.png")
    plt.savefig(hist_path)
    plt.close()

    # === 2. ROC Curve ===
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    roc_path = os.path.join(save_dir, f"{save_prefix}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # === 3. Precision-Recall Curve ===
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color='orange')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    pr_path = os.path.join(save_dir, f"{save_prefix}_precision_recall.png")
    plt.savefig(pr_path)
    plt.close()
    
    # === 1. Confusion Matrix ===
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    # disp.plot(cmap="Blues", values_format='d')
    # plt.title("Confusion Matrix")
    # plt.savefig(os.path.join(save_dir, f"{save_prefix}_conf_matrix.png"))
    # plt.close()

    print(f"✅ All plots saved to '{save_dir}/'")

#%%

##### Calculating the leaderboard for hubert
@torch.no_grad()
def evaluate_model_hubert(model, dataset, dataset_name="april_24_hubert_lar_hi"):
    model.eval()
    y_true, y_scores = [], []
    transcripts = []
    cer_scores = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {dataset_name}"):
        audio1, audio2, label = dataset[i]
        label = int(label.item())


        with torch.no_grad():
            emb1 = model(audio1)
            emb2 = model(audio2)

        # print(f'emb1 shape : {emb1.shape} ')
        # print(f'emb2 shape : {emb2.shape} ')
        sim = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
        y_true.append(label)
        y_scores.append(sim)

        # Optionally collect CER
        # transcription1 = "<audio>"
        # transcription2 = "<audio>"
        # cer = jiwer.cer(transcription1, transcription2)
        # cer_scores.append(cer)
        # transcripts.append((transcription1, transcription2, cer))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Determine threshold via EER
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    optimal_threshold = thresholds[eer_idx]

    y_pred = (y_scores >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
    Precision = precision_score(y_true, y_pred)
    Recall = recall_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)

    leaderboard_row = {
        "Dataset": dataset_name,
        "TPR": round(TPR, 4),
        "FPR": round(FPR, 4),
        "TNR": round(TNR, 4),
        "FNR": round(FNR, 4),
        "Precision": round(Precision, 4),
        "Recall": round(Recall, 4),
        "F1": round(F1, 4),
        "Optimal_Threshold": round(optimal_threshold, 4),
        "Num_Val_Pairs": len(y_true)
    }

    # Save leaderboard
    leaderboard_path = "leaderboard_results_april_24_hubert_lar_hi.csv"
    try:
        leaderboard_df = pd.read_csv(leaderboard_path)
    except FileNotFoundError:
        leaderboard_df = pd.DataFrame(columns=list(leaderboard_row.keys()))

    leaderboard_df = pd.concat([leaderboard_df, pd.DataFrame([leaderboard_row])], ignore_index=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"✅ Leaderboard saved to: {leaderboard_path}")

    # 🔍 Save similarity histogram + ROC
    plot_full_similarity_analysis(
        y_true=y_true,
        y_scores=y_scores,
        save_prefix='april_24_hubert_lar_hi',
        save_dir="plots"
    )

    return leaderboard_row, transcripts

#%%

# The above code is likely calling a function named `evaluate_model_hubert` with the arguments `model`
# and `val_dataset3`. This function is likely used to evaluate the performance of a model using a
# validation dataset.
# evaluate_model_hubert(model,val_dataset3)

#%%

# this code is taking the output from the selected layer of the models and then
# it will calculate the cosine similarity between the same and different
# word embeddings and then take the average so the we can visualize
# which layer has maximum difference between same or diff word embeddings.

def plot_layerwise_cosine_with_labels(model, data_loader, device, num_batches=10):
    model.eval()
    selected_layers = list(range(0, 12, 2))  
    cos_same = [[] for _ in selected_layers]
    cos_diff = [[] for _ in selected_layers]

    with torch.no_grad():
        for batch_idx, (audio1, audio2, labels) in enumerate(tqdm(data_loader, total=num_batches, desc="Processing Batches")):
            if batch_idx >= num_batches:
                break

            labels = labels.to(device)

            for idx, layer_idx in enumerate(selected_layers):
                # Update model's internal layer index for hook
                model.layer_index = layer_idx

                emb1 = model(audio1.to(device))  # [B, D]
                emb2 = model(audio2.to(device))  # [B, D]

                for j in range(len(labels)):
                    v1 = emb1[j].cpu().numpy()
                    v2 = emb2[j].cpu().numpy()
                    sim = cosine_similarity([v1], [v2])[0][0]

                    if labels[j] == 1:
                        cos_same[idx].append(sim)
                    else:
                        cos_diff[idx].append(sim)

    # Average values
    avg_cos_same = [np.mean(s) if s else 0 for s in cos_same]
    avg_cos_diff = [np.mean(s) if s else 0 for s in cos_diff]
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(selected_layers, avg_cos_same, label="Same Word (lower is better)", marker='o')
    plt.plot(selected_layers, avg_cos_diff, label="Different Word (higher is better)", marker='x')
    plt.title("Layer-wise Coisne Sim")
    plt.xlabel("Layer Index (0 = Conv, 1–24 = Transformer)")
    plt.ylabel("Average Cosine Sim")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join("plots", f"layerwise_hubert_base_hi.png")
    plt.savefig(pr_path)
    plt.show()
#%%

# calling the function to get the layerwise cossine difference value
plot_layerwise_cosine_with_labels(model,val_loader,device,num_batches=100)
#%%