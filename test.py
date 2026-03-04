import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os
import torch
import torch.nn as nn
import tensorboard
import gc
from sklearn.model_selection import train_test_split
from mae_visualize_utils import *
from mae_visualize import *
from util.datasets import *
from util.pos_embed import *
from models_mae import *
from torch.utils.data import Subset
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.cm as cm
import random

### initialization
patch_size = [1, 60]
normalization = False
data_path = "/niddk-data-central/mae_hr/rise_moca_4AP_20s_transition"
test_data = True
data = "RISE"
transform = False
RISE_hz = 30
alt = True
mask_ratio = 0.75
use_transition_sub_label = True
RISE_bin_label = False
device = 'cuda'

### Data Load
dataset, labels, activity_labels = load_dataset(
    data_path=data_path, data=data, test_data=test_data, alt=alt,
    normalization=normalization, transform=transform,
    use_transition_sub_label=use_transition_sub_label, RISE_bin_label=RISE_bin_label
)

patch_num = None
patch_size1, patch_size2 = patch_size
input_size = [dataset[0][0].shape[1], dataset[0][0].shape[2]]
if int(input_size[1] / patch_size2) != patch_num:
    patch_num = int(input_size[1] / patch_size2)

in_chans     = dataset[0][0].shape[0]
input_length = input_size[1]
print(f"input_size: {input_size}, patch_size: {patch_size}, patch_num: {patch_num}, in_chans: {in_chans}")
print("dataset shape", dataset.shape)

### Dataset Subsetting
use_subset = False

if use_subset:
    target_ids  = [200046, 200532, 200267, 200444]
    indices     = np.where(np.isin(dataset.id, target_ids))[0].tolist()
    dataset_sub = Subset(dataset, indices)
    sampler     = torch.utils.data.SequentialSampler(dataset_sub)
    data_loader_sub = torch.utils.data.DataLoader(
        dataset_sub, sampler=sampler, batch_size=256,
        num_workers=4, drop_last=False, pin_memory=True
    )
else:
    indices     = list(range(len(dataset)))
    sampler     = torch.utils.data.SequentialSampler(dataset)
    data_loader_sub = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=256,
        num_workers=4, drop_last=False, pin_memory=True
    )

print(f"Number of samples: {len(indices)}")

### checkpoints
finetuned_checkpoint_path = "/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_augp/123(RISE_FT_20s_augp_trans)/checkpoint-5.pth"
model_mae_classifier = load_model_ft(
    checkpoint_path=finetuned_checkpoint_path, LP=False, nb_classes=7, in_chans=in_chans,
    input_size=input_size, patch_size=patch_size, finetune_interpolate_patches=False,
    mask_finetune=0, device='cuda', global_pool=False
)
print("finetuned checkpoint loaded")

# parallel GPUs
model_mae_classifier = nn.DataParallel(model_mae_classifier, device_ids=[0, 1, 2])
model_mae_classifier = model_mae_classifier.to(device)
model_mae_classifier.eval()

cls_list = []
with torch.no_grad():
    for batch in iter(data_loader_sub):
        x       = batch[0].to(device, non_blocking=True)
        outcome = model_mae_classifier.module.forward_features(x)  # .module to access underlying model
        cls_list.append(outcome.detach().cpu())

print("latent representation loaded")
del model_mae_classifier
gc.collect()
torch.cuda.empty_cache()

cls_all  = torch.cat(cls_list, dim=0)   # (N, 768)
cls_flat = cls_all.numpy()
print(f"CLS token shape: {cls_all.shape}, {cls_flat.shape}")


# Data
ids    = np.array(dataset.id)[indices]
dates = pd.to_datetime(np.array(dataset.time)[indices].astype(int), unit='s').date

visits = np.array(dataset.visit)[indices]
labels = np.array(dataset.y)[indices]

trt_arms  = pd.read_csv("/niddk-data-central/P2/CHAP2.0_support_files/P2_3arm_rand.csv")
id_to_arm = dict(zip(trt_arms["study_id"], trt_arms["rand_assignment"]))
print(f"Unique treatment arms: {trt_arms['rand_assignment'].unique()}")

# color
lab_to_color = {0: "steelblue", 1: "darkorange", 2: "red", 3: "darkblue",
                4: "purple",    5: "darkgreen",   6: "lightgreen"}
lab_to_name  = {0: "Sedentary", 1: "Standing", 2: "Stepping", 3: "Lying",
                4: "Seated Transport", 5: "Transition-active", 6: "Transition-mixed"}
trt_to_color = {0: 'darkblue', 1: 'orange', 2: 'darkgreen'}

### UMAP with labels
reducer   = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
embedding = reducer.fit_transform(cls_flat)

unique_labels = np.unique(labels)
fig, ax = plt.subplots(figsize=(10, 8))
for val in unique_labels:
    mask = labels == val
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               color=lab_to_color[val], label=lab_to_name.get(val, str(val)), s=5, alpha=0.6)
ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3, fontsize=8)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP for Label Classification (visit1 + visit2)')
plt.tight_layout()
plt.savefig('/niddk-data-central/mae_hr/RISE_PH/plots/umap_labels.png', dpi=400, bbox_inches='tight')
plt.show()

### Person-day analysis
feat_cols = list(range(cls_flat.shape[1]))   # 0..767

cls_df = pd.DataFrame(cls_flat)
cls_df['id']    = ids
cls_df['date']  = dates
cls_df['visit'] = visits

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: MEAN + STD
# ══════════════════════════════════════════════════════════════════════════════
mean_df = cls_df.groupby(['id', 'date', 'visit'])[feat_cols].mean()
std_df  = cls_df.groupby(['id', 'date', 'visit'])[feat_cols].std().fillna(0)
mean_df.columns = [f'mean_{c}' for c in feat_cols]
std_df.columns  = [f'std_{c}'  for c in feat_cols]

person_day_meanstd = pd.concat([mean_df, std_df], axis=1).reset_index()
person_day_meanstd['trt'] = person_day_meanstd['id'].apply(lambda sid: id_to_arm.get(int(sid), np.nan))
print(f"Person-day embeddings (mean+std): {person_day_meanstd.shape}")
print(person_day_meanstd[['id', 'date', 'visit', 'trt']].head())

meanstd_feat_cols  = [c for c in person_day_meanstd.columns if str(c).startswith('mean_') or str(c).startswith('std_')]
embeddings_meanstd = person_day_meanstd[meanstd_feat_cols].values
emb_scaled         = StandardScaler().fit_transform(embeddings_meanstd)
embedding_2d       = umap.UMAP(n_components=2, random_state=42, n_jobs=-1).fit_transform(emb_scaled)

person_day_meanstd['umap1'] = embedding_2d[:, 0]
person_day_meanstd['umap2'] = embedding_2d[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, visit_val, title in zip(axes, [0, 1], ['Visit 0 (Baseline)', 'Visit 1 (Follow-up)']):
    sub = person_day_meanstd[person_day_meanstd['visit'] == visit_val]
    for trt in person_day_meanstd['trt'].dropna().unique():
        mask = sub['trt'] == trt
        ax.scatter(sub.loc[mask, 'umap1'], sub.loc[mask, 'umap2'],
                   color=trt_to_color.get(trt, 'gray'), label=str(trt), s=15, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(title='Intervention Group', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=9)
plt.suptitle('Person-Day Embeddings (Mean+Std) by Intervention Group', fontsize=14)
plt.tight_layout()
plt.savefig('/niddk-data-central/mae_hr/RISE_PH/plots/umap_mean_std_by_arm.png', dpi=450, bbox_inches='tight')
plt.show()
print("Method1 done")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: BAG OF WORDS / K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
K  = 32
km = MiniBatchKMeans(n_clusters=K, random_state=1004, n_init=10)
km.fit(cls_flat)
print(f"K-means inertia: {km.inertia_:.2f}")

cls_df['cluster'] = km.predict(cls_flat)

person_day_bow = (cls_df.groupby(['id', 'date', 'visit'])['cluster']
                        .value_counts(normalize=True)
                        .unstack(fill_value=0)
                        .reset_index())

for k in range(K):
    if k not in person_day_bow.columns:
        person_day_bow[k] = 0.0
meta_cols    = ['id', 'date', 'visit']
cluster_cols = sorted([c for c in person_day_bow.columns if c not in meta_cols])
person_day_bow = person_day_bow[meta_cols + cluster_cols]
person_day_bow['trt'] = person_day_bow['id'].apply(lambda sid: id_to_arm.get(int(sid), np.nan))
print(f"Person-day embeddings (BoW): {person_day_bow.shape}")
print(person_day_bow[['id', 'date', 'visit', 'trt']].head())

embeddings_bow   = person_day_bow[cluster_cols].values
emb_scaled_bow   = StandardScaler().fit_transform(embeddings_bow)
embedding_2d_bow = umap.UMAP(n_components=2, random_state=42, n_jobs=-1).fit_transform(emb_scaled_bow)

person_day_bow['umap1'] = embedding_2d_bow[:, 0]
person_day_bow['umap2'] = embedding_2d_bow[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, visit_val, title in zip(axes, [0, 1], ['Visit 0 (Baseline)', 'Visit 1 (Follow-up)']):
    sub = person_day_bow[person_day_bow['visit'] == visit_val]
    for trt in person_day_bow['trt'].dropna().unique():
        mask = sub['trt'] == trt
        ax.scatter(sub.loc[mask, 'umap1'], sub.loc[mask, 'umap2'],
                   color=trt_to_color.get(trt, 'gray'), label=str(trt), s=15, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=9)
plt.suptitle('Person-Day Embeddings (BoW K-means) by Treatment', fontsize=14)
plt.tight_layout()
plt.savefig('/niddk-data-central/mae_hr/RISE_PH/plots/umap_bow_by_trt.png', dpi=450, bbox_inches='tight')
plt.show()
print("Method2 done")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: ATTENTION POOLING
# ══════════════════════════════════════════════════════════════════════════════
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=0)   # (T, 1)
        return (w * x).sum(dim=0)                 # (D,)

dim       = cls_flat.shape[1]
attn_pool = AttentionPooling(dim=dim)
attn_pool.eval()

cls_tensor = torch.tensor(cls_flat, dtype=torch.float32)
group_keys = list(zip(ids, dates, visits))
group_df   = pd.DataFrame({'group': group_keys})
groups     = group_df.groupby('group').groups

person_day_attn_rows = []
with torch.no_grad():
    for (sid, date, visit), row_idx in groups.items():
        x   = cls_tensor[list(row_idx)]
        emb = attn_pool(x).numpy()
        row = {'id': sid, 'date': date, 'visit': visit}
        row.update({i: emb[i] for i in range(dim)})
        person_day_attn_rows.append(row)

person_day_attn = pd.DataFrame(person_day_attn_rows)
person_day_attn['trt'] = person_day_attn['id'].apply(lambda sid: id_to_arm.get(int(sid), np.nan))
print(f"Person-day embeddings (attention pooling): {person_day_attn.shape}")
print(person_day_attn[['id', 'date', 'visit', 'trt']].head())

attn_feat_cols    = list(range(dim))
embeddings_attn   = person_day_attn[attn_feat_cols].values
emb_scaled_attn   = StandardScaler().fit_transform(embeddings_attn)
embedding_2d_attn = umap.UMAP(n_components=2, random_state=42, n_jobs=-1).fit_transform(emb_scaled_attn)

person_day_attn['umap1'] = embedding_2d_attn[:, 0]
person_day_attn['umap2'] = embedding_2d_attn[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, visit_val, title in zip(axes, [0, 1], ['Visit 0 (Baseline)', 'Visit 1 (Follow-up)']):
    sub = person_day_attn[person_day_attn['visit'] == visit_val]
    for trt in person_day_attn['trt'].dropna().unique():
        mask = sub['trt'] == trt
        ax.scatter(sub.loc[mask, 'umap1'], sub.loc[mask, 'umap2'],
                   color=trt_to_color.get(trt, 'gray'), label=str(trt), s=15, alpha=0.7)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=9)
plt.suptitle('Person-Day Embeddings (Attention Pooling) by Treatment', fontsize=14)
plt.tight_layout()
plt.savefig('/niddk-data-central/mae_hr/RISE_PH/plots/umap_attn_by_trt.png', dpi=450, bbox_inches='tight')
plt.show()
print("Method3 done")