import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os
import gc

import random
from sklearn.model_selection import train_test_split
from mae_visualize_utils import *
from mae_visualize import *
from util.datasets import *
from util.pos_embed import *
from models_mae import *
from torch.utils.data import Subset
import umap
import matplotlib.pyplot as plt



from sklearn.metrics import balanced_accuracy_score

def get_balanced_accuracy(dataset, model_nickname, nb_classes, finetuned_checkpoint_path, device, input_size=[3, 600], patch_size=[1, 60], in_chans=1):

    #finetuned_checkpoint_path = "/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_augp/123(RISE_FT_20s_augp_trans)/checkpoint-5.pth"
    model_mae_classifier = load_model_ft(
        checkpoint_path=finetuned_checkpoint_path, LP=False, nb_classes=nb_classes, in_chans=in_chans,
        input_size=input_size, patch_size=patch_size, finetune_interpolate_patches=False,
        mask_finetune=0, device='cuda', global_pool=False
    )
    model_mae_classifier = model_mae_classifier.to(device)
    model_mae_classifier.eval()

    preds_list, targets_list, acc1 = mae_classification_pass(dataset, model_mae_classifier, batch_size=100, device='cuda', num_workers=1)
    global_bal_acc = balanced_accuracy_score(targets_list, preds_list)

    print(f'{model_nickname}-- balanced accuracy:{global_bal_acc*100}, accuracy: {acc1}')
    
    output_dir = "/niddk-data-central/mae_hr/RISE_PH/plots/balanced_accuracy"
    output_path = os.path.join(output_dir, f'{model_nickname}_acc.txt')

    with open(output_path, 'w') as f:
        f.write(f"Model:              {model_nickname}\n")
        f.write(f"{'─' * 50}\n")
        f.write(f"Accuracy:           {acc1:.4f}\n")
        f.write(f"Balanced Accuracy:  {global_bal_acc*100:.4f}\n")
        f.write(f"{'─' * 50}\n")
    
    del model_mae_classifier
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Results saved to: {output_path}")



def main():
    ########### specify options
    patch_size = [1, 60]
    normalization=False
    data_path ="/niddk-data-central/mae_hr/rise_moca_4AP_20s_transition"
    test_data = True
    data = "RISE"
    transform=False
    RISE_hz = 30
    alt=True
    mask_ratio=0.75
    use_transition_sub_label = True
    RISE_bin_label = False
    device='cuda'
    ###########

    # load data
    dataset, labels, activity_labels = load_dataset(data_path=data_path, data=data, test_data=test_data, alt=alt, normalization=normalization, transform=transform,
                                                    use_transition_sub_label=use_transition_sub_label, RISE_bin_label=RISE_bin_label)

    patch_num = None
    patch_size1, patch_size2 = patch_size

    input_size = [dataset[0][0].shape[1], dataset[0][0].shape[2]]

    if int(input_size[1]/patch_size2) !=patch_num:
        patch_num = int(input_size[1]/patch_size2)


    in_chans = dataset[0][0].shape[0]
    print(f"input_size: {input_size}, patch_size:{patch_size}, patch_num: {patch_num}, in_chans: {in_chans}")
    print("dataset shape", dataset.shape)


    # multiple labels
    get_balanced_accuracy(dataset=dataset, model_nickname="1)augp", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_augp/123(RISE_FT_20s_augp_trans)/checkpoint-5.pth",
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1)
    get_balanced_accuracy(dataset=dataset, model_nickname="2)warp", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_warp/141(RISE_FT_20s_warp_trans)/checkpoint-5.pth",
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1)
    get_balanced_accuracy(dataset=dataset, model_nickname="3)mixup", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_mixup/139(RISE_FT_20s_mixup_trans)/checkpoint-5.pth",
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1)
    get_balanced_accuracy(dataset=dataset, model_nickname="4)none", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_none/64(RISE_FT_20s_none_trans)/checkpoint-5.pth",
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1) 
    get_balanced_accuracy(dataset=dataset, model_nickname="5)augp_wl1", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_augp_wl1/146(Final_RISE_FT_20s_augp_wl1)/checkpoint-5.pth",
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1)
    get_balanced_accuracy(dataset=dataset, model_nickname="6)augp_wl2", nb_classes=7,
                          finetuned_checkpoint_path="/niddk-data-central/mae_hr/RISE_PH/RISE_MoCA/experiments/Final_RISE_FT_20s_augp_wl2/132(Final_RISE_FT_20s_augp_wl2)/checkpoint-5.pth", 
                          device = device, input_size=[3, 600], patch_size=[1, 60], in_chans=1)

    return (f"DONE")


if __name__ == "main":
    main()