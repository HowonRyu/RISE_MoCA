# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import random
import os
import PIL
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
import gc


from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class UCIHAR(Dataset):
    def __init__(self, data_path, alt, is_test=False, is_all_data=False, pre_mix_up = False, normalization_chan=False,
                 normalization=False, transform=False, padding_transform=False, mix_up=True, nb_classes=7):

        # only for off-line mix_up in pretrain stage
        if pre_mix_up:
            x_train_dir = os.path.join(data_path, 'X_train_aug_all.pt') #(N, H, W, C)
            self.X = torch.tensor(torch.load(x_train_dir),dtype=torch.float32)
            if alt:
                self.X = self.X.unsqueeze(dim=3)
            else:
                self.X = self.X.unsqueeze(dim=3).permute(0,3,2,1)

            # dummy variable
            self.y = torch.zeros(self.X.shape[0], dtype=torch.long)

        else:
            X_train_dir = os.path.join(data_path, 'X_train_all.pt')
            y_train_dir = os.path.join(data_path, 'y_train_all_mode.pt')
            X_test_dir = os.path.join(data_path, 'X_test_all.pt')
            y_test_dir = os.path.join(data_path, 'y_test_all_mode.pt')
            X_all_dir = os.path.join(data_path, 'X_all_all.pt')
            y_all_dir = os.path.join(data_path, 'y_all_all_mode.pt')

            # Load data based on whether it's a test set or training set
            if is_all_data:
                self.X = torch.tensor(torch.load(X_all_dir), dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_all_dir), dtype=torch.long)

            elif is_test:
                self.X = torch.tensor(torch.load(X_test_dir),dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_test_dir), dtype=torch.long)
            else:
                self.X = torch.tensor(torch.load(X_train_dir),dtype=torch.float32)
                self.y = torch.tensor(torch.load(y_train_dir), dtype=torch.long)


            if normalization_chan:
                per_channel_mean = self.X.mean(dim=(0,1,2), keepdim=True)
                per_channel_std = self.X.std(dim=(0,1,2), keepdim=True)
                self.X = (self.X - per_channel_mean) / (per_channel_std)


            if alt:
                self.X = self.X.permute(0, 3, 2, 1) # (N, H', W, C') = (N, C, W, H) = (N, 6, L, 1)


            if nb_classes==6:
                # Filter out samples with label 0 and adjust labels 1-6 to 0-5
                mask = self.y != 0
                self.X = self.X[mask]
                self.y = self.y[mask] -1  

        self.X = self.X.permute(0,3,1,2) # conform to (N, 1, H, W) = (N, 1, 6, L)
        self.normalization = normalization
        self.transform = transform
        self.padding_transform = padding_transform
        self.mix_up = mix_up
        self.shape = [self.X.shape, self.y.shape]
        print(padding_transform)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        # Apply normalization if transform is set
        if self.normalization:
            sample_X = (sample_X - sample_X.mean(dim=1,keepdim=True)) / sample_X.std(dim=1,keepdim=True)

        if self.mix_up:
            # Randomly select another sample
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            sample_X = self.mix_time_series(sample_X, sample_X2)

        # only for vit_base line
        if self.transform:
            sample_X = sample_X.unsqueeze(0)
            #sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
            sample_X = F.interpolate(sample_X, size=(96, 160), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
        return sample_X, sample_y

        if self.padding_transform:
            pad_right = 224 - 200  # 24
            pad_bottom = 224 - 6   # 218
            print("sample x before shape:", sample_X.shape)
            sample_padded = F.pad(sample_X, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
            sample_X = sample_padded.unsqueeze(0).repeat(3, 1, 1)
            print("padding_transform shape:", sample_X.shape)
        return sample_X, sample_y

    def mix_time_series(self, sample1, sample2):

        ts_len = sample1.shape[1]
        lambada = torch.distributions.Uniform(0, 0.5).sample().item()
        sample1_size = int(ts_len * lambada)
        sample2_size = ts_len - sample1_size 

        chunk1 = sample1[:, :sample1_size, :]  
        chunk2 = sample2[:, sample1_size:, :]  
        result = torch.cat((chunk1, chunk2), dim=1)

        return result


class RISE(Dataset):
    def __init__(self, data_path, alt, is_test=False, normalization_chan=False, use_transition_sub_label = False, RISE_bin_label = False,
                 normalization=False, transform=False, mix_up=False, RISE_hz = 30, hz_adjustment=False, subject_level_analysis=False, active_aug=False, aug_method=None):

        if is_test:
            prefix = "test"
            n_split = 2
            time_df_path = os.path.join(data_path, f"{n_split-1}/time_dist_df_{prefix}.csv")
            time_dist_df = pd.read_csv(time_df_path)
            total_samples = np.sum(time_dist_df['num_inputs_aftex'].values)
        else:
            prefix = "train"
            n_split = 3
            time_df_path = os.path.join(data_path, f"{n_split-1}/time_dist_df_{prefix}.csv")
            time_dist_df = pd.read_csv(time_df_path)
            total_samples = np.sum(time_dist_df['num_inputs_aftex'].values)          
        
        X_train_2 = torch.load(os.path.join(data_path, f"2/X_train.pt"))
        
        if not subject_level_analysis:
            self.X = torch.empty((total_samples, 1, X_train_2.shape[2], 3), dtype=torch.float32)
        self.y = torch.empty((total_samples, 1), dtype=torch.long)
        self.labels = torch.empty((total_samples, X_train_2.shape[2]))
        self.time = torch.empty((total_samples), dtype=torch.long)
        self.visit = torch.empty((total_samples,1), dtype=torch.long)
        self.id = torch.empty((total_samples,1), dtype=torch.long)
        self.sleeping = torch.empty((total_samples,1), dtype=torch.long)

        idx = 0
        for i in range(n_split):
            print(f"{i}/{n_split} split loading")
            
            if not subject_level_analysis:
                X_path = os.path.join(data_path, f"{i}/X_{prefix}.pt")
                X_part = torch.load(X_path)
                n = X_part.shape[0]
                self.X[idx:idx+n] = X_part
                del X_part
            
            
            for name in ["y", "labels", "time", "visit", "id", "sleeping"]:
                path = os.path.join(data_path, f"{i}/{name}_{prefix}.pt")
                part = torch.load(path)
                n = part.shape[0]
                getattr(self, name)[idx:idx+n] = part
                del part

            gc.collect()
            idx += n
            print(f"{i}/{n_split} split done")

        self.y = self.y.squeeze()
        print(f"Data loading all done: # X {prefix} samples = {total_samples}")


        # add treatment label features

        # exclusion rules according to {0: "sedentary", 1:"standing", 2:"stepping", 3:"cycling", 4:"sleeping", 5:"lying", 6:"seated_transport", -9:"transition"}


        # Compute indices to exclude
        keep_mask = ~( (self.y == -1) | (self.y == 3) | (self.y == 4) |
                      ((self.sleeping[:, 0] == 1) & (self.y != 4)) |
                      (self.labels == -1).any(dim=1) |
                      (self.labels == 3).any(dim=1) |
                      (self.labels == 4).any(dim=1) )

        keep_idx = keep_mask.nonzero(as_tuple=True)[0]



        if not subject_level_analysis:
            self.X = self.X[keep_idx]
        
        for attr in ["y", "labels", "time", "visit", "id", "sleeping"]:
            t = getattr(self, attr)
            setattr(self, attr, t[keep_idx])
            del t

        gc.collect()
        #print(f"masking done")
        print(f"After Exclusion: # X {prefix} samples = {len(self.y)}")
    
        # re-define the mapping
        self.y[self.y == -9 ] = 7  # transition is now 7 w/ {0: "sedentary", 1:"standing", 2:"stepping", 5:"lying", 6:"seated_transport", 7:"transition"}
        self.y[self.y > 2] -= 2  # now {0: "sedentary", 1:"standing", 2:"stepping", 3:"lying", 4:"seated_transport", 5:"transition"}

        self.labels[self.labels > 2] -= 2

        if use_transition_sub_label:
            transition_mask = (self.y == 5)
            trans_labels_org = self.labels[transition_mask]
            trans_labels_org[trans_labels_org > 2] -= 2 #{0: "sedentary", 1:"standing", 2:"stepping", 3:"lying", 4:"seated_transport"}

            #print(np.unique(trans_labels_org))
            _, _, _, _, categories_4, categories_2  = self.classify_sequences(trans_labels_org.long())
             
            cat2_mapping = torch.tensor([5,6]).long() #cat4_mapping = [5,6,7,8]
            self.y[transition_mask] = cat2_mapping[categories_2] # for now use cat2
            pre_designated_labels = [1, 2, 5, 6]  # for active augmentation
            if RISE_bin_label:
                trans_bin_mapping =  torch.tensor([0, 1, 1, 0, 0, 1, 2]).long()
                self.y = trans_bin_mapping[self.y]

        else:
            if RISE_bin_label:
                bin_mapping =  torch.tensor([0, 1, 1, 0, 0, 1]).long()
                self.y = bin_mapping[self.y]
            else:
                pre_designated_labels = [1, 2, 5]


        print(f"unique y: {np.unique(self.y)}") # sanity check
        print(f"y shape: {self.y.shape}") # sanity check



        if normalization_chan:
            per_channel_mean = self.X.mean(dim=(0,1,2), keepdim=True)
            per_channel_std = self.X.std(dim=(0,1,2), keepdim=True)
            self.X = (self.X - per_channel_mean) / (per_channel_std)

        if alt &  (not subject_level_analysis):
            self.X = self.X.permute(0, 3, 2, 1)  # (N, H', W, C') = (N, C, W, H) = (N, 3, L, 1)

        if not subject_level_analysis:
            self.X = self.X.permute(0,3,1,2) # conform to (N, 1, H, W) = (N, 1, 3, L)
            

        if active_aug & (not subject_level_analysis):
            print(f"starting data augmentation for labels: {pre_designated_labels}")
            print(f"# samples before augmentation = {len(self.X)}")
            self.augmented_methods = []
            target_ratio = 0.5  
            
            # Count per label
            unique, counts = np.unique(self.y.numpy(), return_counts=True)
            label_counts = dict(zip(unique, counts))
            max_count = max(counts)

            print(f"# augmenting labels to {target_ratio} of max label {np.argmax(counts)} ({max_count})")

            X_aug_list = []
            y_aug_list = []

            for label in pre_designated_labels:
                current_count = label_counts.get(label, 0)
                n_to_generate = int(max_count * target_ratio - current_count)
                print(f"n_to_generate for label {label}= {n_to_generate}")
                label_idx = (self.y == label).nonzero(as_tuple=True)[0]
                
                for i in range(n_to_generate):
                    idx_sample = label_idx[i % len(label_idx)]
                    X_sample = self.X[idx_sample]
                    
                    X_aug, method = self.augment_tensor(X_sample, seed=(i+10), method=aug_method)
                    
                    X_aug_list.append(X_aug.unsqueeze(0))
                    y_aug_list.append(self.y[idx_sample].unsqueeze(0))
                    self.augmented_methods.append(method)

            if len(X_aug_list) > 0:
                self.X = torch.cat([self.X] + X_aug_list, dim=0)
                self.y = torch.cat([self.y] + y_aug_list, dim=0)

            print(f"After stochastic augmentation: # X samples = {self.X.shape[0]}")



        self.normalization = normalization
        self.transform = transform
        self.mix_up = mix_up
        if not subject_level_analysis:
            self.shape = [self.X.shape, self.y.shape]
        else:
            self.shape = self.y.shape
        self.RISE_hz = RISE_hz
        self.hz_adjustment = hz_adjustment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]

        # Apply normalization if transform is set
        if self.normalization:
            sample_X = (sample_X - sample_X.mean(dim=1,keepdim=True)) / sample_X.std(dim=1,keepdim=True)

        if self.mix_up:
            # Randomly select another sample
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            sample_X = self.mix_time_series(sample_X, sample_X2)

        # only for vit_base line
        if self.transform:
            sample_X = sample_X.unsqueeze(0)
            sample_X = F.interpolate(sample_X, size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).squeeze(0)
        if self.hz_adjustment:
            new_length = int((sample_X.shape[-1]/self.RISE_hz) * 50 ) 
            sample_X = F.interpolate(sample_X, size=new_length, mode='linear', align_corners=True)
        return sample_X, sample_y

    def mix_time_series(self, sample1, sample2):

        ts_len = sample1.shape[1]
        lambada = torch.distributions.Uniform(0, 0.5).sample().item()
        sample1_size = int(ts_len * lambada)
        sample2_size = ts_len - sample1_size 

        chunk1 = sample1[:, :sample1_size, :]  
        chunk2 = sample2[:, sample1_size:, :]  
        result = torch.cat((chunk1, chunk2), dim=1)

        return result

    def augment_tensor(self, X, seed=None, method=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if method is None:
            augmentation_methods = ["mixup", "time_warp"]   #"rotation", 
            aug_random = random.Random(seed) 
            method = aug_random.choice(augmentation_methods)
    
        X_aug = X.clone()
        _, C, L = X.shape
        

        if method == "rotation":
            # Rotate channels: shuffle 3 channels
            perm = torch.randperm(C)
            X_aug = X_aug[:, perm, :]
            method_0 = 0

        elif method == "mixup":     
            mix_idx = torch.randint(0, len(self.X), (1,)).item()
            sample_X2 = self.X[mix_idx]
            
            lambada = torch.distributions.Uniform(0, 0.5).sample().item()
            sample1_size = int(L * lambada)

            chunk1 = X[:, :, :sample1_size]  
            chunk2 = sample_X2[:, :, sample1_size:]  
            X_aug = torch.cat((chunk1, chunk2), dim=2)
            method_0 = 1

        elif method == "time_warp":
            scale = np.random.uniform(0.7, 1.3)
            new_L = int(L * scale)
            X_aug = torch.nn.functional.interpolate(
                X_aug, size=new_L, mode="linear", align_corners=True)
            if new_L > L:
                X_aug = X_aug[:, :, :L]
            elif new_L < L:
                pad = L - new_L
                X_aug = torch.nn.functional.pad(X_aug, (0, pad))
            method_0 = 2
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

        return X_aug, method_0

    def classify_sequences(self, x):
        N = x.shape[0]
        categories_2 = torch.zeros(N, dtype=torch.long)
        categories_4 = torch.zeros(N, dtype=torch.long)
        n_bouts_tensor = torch.zeros(N, dtype=torch.long)
        n_unique_acts_tensor = torch.zeros(N, dtype=torch.long)
        n_bouts_bin_tensor = torch.zeros(N, dtype=torch.long)
        n_unique_acts_bin_tensor = torch.zeros(N, dtype=torch.long)

        if False:
            mapping = torch.tensor([0, 1, 1, 1, 0, 0, 0], dtype=torch.long)  #([-1,  0,  1,  2,  3,  4,  5,  6])
        else:
            mapping = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)  # 0: "sedentary", 1:"standing", 2:"stepping", 3:"lying", 4:"seated_transport"

        x_bin = mapping[x]

        for i in range(N):

            # original data
            seq = x[i].cpu().numpy()
            simplified = seq[np.concatenate(([True], seq[1:] != seq[:-1]))]
            
            n_bouts = len(simplified)
            n_bouts_tensor[i] = n_bouts
            n_unique_acts = len(np.unique(simplified))
            n_unique_acts_tensor[i] = n_unique_acts
            
            # binary
            seq_bin = x_bin[i].cpu().numpy()
            simplified_bin = seq_bin[np.concatenate(([True], seq_bin[1:] != seq_bin[:-1]))]
            n_bouts_bin = len(simplified_bin)
            n_bouts_bin_tensor[i] = n_bouts_bin
            n_unique_acts_bin = len(np.unique(simplified_bin))
            n_unique_acts_bin_tensor[i] = n_unique_acts_bin


            # one binary bout: no transition across sitting/non-sitting
            if n_bouts_bin == 1:
                if simplified_bin[0] == 0:  #sedentary -> sedentary
                    cat_2 = 0
                    cat_4 = 0
                elif simplified_bin[0] == 1:  #active -> active
                    cat_2 = 0
                    cat_4 = 0

            # two binary bouts: one time change across sitting/non-sitting
            elif n_bouts_bin == 2: 
                if simplified_bin[0] == 0:
                    cat_2 = 1
                    cat_4 = 1
                elif simplified_bin[0] == 1:
                    cat_2 = 1
                    cat_4 = 2
            # more than two binary bouts:         
            elif n_bouts_bin > 2: 
                cat_2 = 1
                cat_4 = 3

            categories_4[i] = cat_4

            categories_2[i] = cat_2

        return n_bouts_tensor, n_bouts_bin_tensor, n_unique_acts_tensor, n_unique_acts_bin_tensor, categories_4, categories_2


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

