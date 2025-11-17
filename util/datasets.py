# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

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
                 normalization=False, transform=False, mix_up=False, RISE_hz = 30, hz_adjustment=False, subject_level_analysis=False):

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
            print(f"{i}/{last_split} split loading")
            
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
            print(f"{i}/{last_split} split done")

        print(f"Data loading all done: -------- # X {prefix} samples = {total_samples}")


        # add treatment label features

        # exclusion rules according to {0: "sedentary", 1:"standing", 2:"stepping", 3:"cycling", 4:"sleeping", 5:"lying", 6:"seated_transport", -9:"transition"}
        excl_mask = (self.y.squeeze()==-1) |   (self.y.squeeze()==3) | (self.y.squeeze()==4)
        excl_34_mask = ((self.sleeping==1) & (self.y.squeeze() != 4)   )
        excl_org_labs_mask = (self.labels == -1).any(dim=1) | (self.labels == 3).any(dim=1) | (self.labels == 4).any(dim=1) 

        final_excl_mask = (excl_mask | excl_34_mask | excl_org_labs_mask)

        self.X = self.y[~final_excl_mask]
        self.y = self.y[~final_excl_mask]
        self.labels = self.labels[~final_excl_mask]
        self.time = self.time[~final_excl_mask]
        self.visit = self.y[~final_excl_mask]
        self.id = self.id[~final_excl_mask]
        self.sleeping = self.sleeping[~final_excl_mask]

        print(f"( After Exclusion )# X {prefix} samples = {len(self.y)}")
    
        # re-define the mapping
        self.y[self.y.squeeze == -9 ] = 7  # transition is now 7 w/ {0: "sedentary", 1:"standing", 2:"stepping", 5:"lying", 6:"seated_transport", 7:"transition"}
        self.y[self.y.squeeze() > 2] =- 2  # now {0: "sedentary", 1:"standing", 2:"stepping", 3:"lying", 4:"seated_transport", 5:"transition"}


        if use_transition_sub_label:
            transition_mask = (self.y.squeeze() == 5)
            trans_labels_org = self.labels[transition_mask]
            _, _, _, _, categories_4, categories_2  = self.classify_sequences(trans_labels_org.long())
             
            cat2_mapping = [5,6] #cat4_mapping = [5,6,7,8]
            trans_bin_mapping = [0, 1, 1, 0, 0, 1, 2]
            self.y[transition_mask] = cat2_mapping[categories_2] # for now use cat2

            if RISE_bin_label:
                trans_bin_mapping = [0, 1, 1, 0, 0, 1, 2]
                self.y = trans_bin_mapping[self.y.squeeze()]
        else:
            if RISE_bin_label:
                bin_mapping = [0, 1, 1, 0, 0, 1]
                self.y = bin_mapping[self.y.squeeze()]


        print(f"unique y: {np.unique(self.y.squeeze())}") # sanity check




        if normalization_chan:
            per_channel_mean = self.X.mean(dim=(0,1,2), keepdim=True)
            per_channel_std = self.X.std(dim=(0,1,2), keepdim=True)
            self.X = (self.X - per_channel_mean) / (per_channel_std)

        if alt:
            self.X = self.X.permute(0, 3, 2, 1)  # (N, H', W, C') = (N, C, W, H) = (N, 3, L, 1)


        self.X = self.X.permute(0,3,1,2) # conform to (N, 1, H, W) = (N, 1, 3, L)
        self.normalization = normalization
        self.transform = transform
        self.mix_up = mix_up
        self.shape = [self.X.shape, self.y.shape]
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
            mapping = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)  #{0: "SEDENTARY", 1: "STANDING", 2: "STEPPING", 3: "LYING", 4: "SEATED_TRANSPORT", 5:"TRANSITION"}

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

