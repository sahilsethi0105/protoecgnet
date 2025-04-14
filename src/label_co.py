import os
import numpy as np
import torch
from ecg_utils import get_dataloaders

print("Initializing train_loader...")

train_loader, _, _, _ = get_dataloaders(
        batch_size=128, 
        mode='1D',
        sampling_rate=100, 
        label_set="3",
        work_num=4,
        return_sample_ids=False,
        custom_groups=True, 
        standardize=False,
        remove_baseline=True,
)

print("Loaded train_loader. Computing label co-occurence...")

all_labels = []
for _, labels in train_loader:
    all_labels.append(labels)
all_labels = torch.cat(all_labels, dim=0).numpy()

# Compute co-occurrence matrix
def compute_label_cooccurrence_matrix(Y, method="jaccard"):
    Y = Y.astype(np.float32)
    intersection = np.dot(Y.T, Y)
    if method == "jaccard":
        union = np.clip(Y.sum(axis=0, keepdims=True) + Y.sum(axis=0, keepdims=True).T - intersection, a_min=1e-6, a_max=None)
        cooc = intersection / union
    else:
        raise ValueError("Unsupported method")
    return cooc

cooc = compute_label_cooccurrence_matrix(all_labels)
save_path = os.path.join('/gpfs/data/bbj-lab/users/sethis/experiments/preprocessing', "label_cooccur_Cat3.pt")
torch.save(torch.tensor(cooc, dtype=torch.float32), save_path)
print(f"Label co-occurence matrix saved to: {save_path}")
