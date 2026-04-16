import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from mae_visualize_utils import load_dataset, load_model_ft, mae_classification_pass


def process_gt_pred_pair_data(patch_size=[1, 60], data_path="/niddk-data-central/mae_hr/rise_moca_4AP_20s_transition",
                            test_data=True, RISE_bin_label=False, device="cuda",
                            finetuned_checkpoint_path=None, produce_plots=False,
                            output_dir="/niddk-data-central/mae_hr/RISE_PH/plots"):
    
    data="RISE"
    global_pool=False
    normalization=False
    transform=False
    RISE_hz=30
    alt=True
    mask_ratio=0.75
    use_transition_sub_label=True
    batch_size=200
    active_aug=False
    finetune_interpolate_patches=False
    mask_finetune=0
    LP=False

    if test_data:
        split = "test"
    else:
        split = "train"

    if RISE_bin_label:
        nb_classes = 3
    else:
        nb_classes = 7
    
    ##### data load
    dataset, labels, activity_labels = load_dataset(
        data_path=data_path,
        data=data,
        test_data=test_data,
        alt=alt,
        normalization=normalization,
        transform=transform,
        use_transition_sub_label=use_transition_sub_label,
        active_aug=active_aug,
        RISE_bin_label=RISE_bin_label,
    )

    ##### pass through the model to get pred and gt pairs
    patch_size1, patch_size2 = patch_size
    input_size = [dataset[0][0].shape[1], dataset[0][0].shape[2]]
    patch_num = int(input_size[1] / patch_size2)
    in_chans = dataset[0][0].shape[0]
    input_length = input_size[1]

    print(f"input_size: {input_size}, patch_size: {patch_size}, patch_num: {patch_num}, in_chans: {in_chans}")
    print("dataset shape", dataset.shape)

    model = load_model_ft(
        checkpoint_path=finetuned_checkpoint_path,
        LP=LP,
        nb_classes=nb_classes,
        in_chans=in_chans,
        input_size=input_size,
        patch_size=patch_size,
        finetune_interpolate_patches=finetune_interpolate_patches,
        mask_finetune=mask_finetune,
        device=device,
        global_pool=global_pool,
    )
    model = model.to(device)
    model.eval()

    preds_list, targets_list, acc1 = mae_classification_pass(
        dataset, model, batch_size=batch_size, device=device
    )

    with open(f"{data_path}/pred_{split}.pkl", "wb") as f:
        pickle.dump(preds_list, f)
    with open(f"{data_path}/target_{split}.pkl", "wb") as f:
        pickle.dump(targets_list, f)

    print(f"Saved pred_{split}.pkl and target_{split}.pkl to {data_path}")


    # calculating statistical features
    X = dataset.X.squeeze(1)   # (N, 3, T)
    X_np = X.numpy()

    # Vector magnitude
    vm = np.sqrt(X_np[:, 0, :]**2 + X_np[:, 1, :]**2 + X_np[:, 2, :]**2)

    pred_gt_df = pd.DataFrame(
        np.column_stack([list(preds_list), list(targets_list)]),
        columns=["pred", "truth"],
    )

    pred_gt_df["mean_x"]        = X[:, 0, :].mean(dim=1).numpy()
    pred_gt_df["mean_y"]        = X[:, 1, :].mean(dim=1).numpy()
    pred_gt_df["mean_z"]        = X[:, 2, :].mean(dim=1).numpy()
    pred_gt_df["mean_vm"]       = vm.mean(axis=1)
    pred_gt_df["std_x"]         = X[:, 0, :].std(dim=1).numpy()
    pred_gt_df["std_y"]         = X[:, 1, :].std(dim=1).numpy()
    pred_gt_df["std_z"]         = X[:, 2, :].std(dim=1).numpy()
    pred_gt_df["std_vm"]        = vm.std(axis=1)
    pred_gt_df["min_x"]         = X[:, 0, :].min(dim=1).values.numpy()
    pred_gt_df["min_y"]         = X[:, 1, :].min(dim=1).values.numpy()
    pred_gt_df["min_z"]         = X[:, 2, :].min(dim=1).values.numpy()
    pred_gt_df["min_vm"]        = vm.min(axis=1)
    pred_gt_df["max_x"]         = X[:, 0, :].max(dim=1).values.numpy()
    pred_gt_df["max_y"]         = X[:, 1, :].max(dim=1).values.numpy()
    pred_gt_df["max_z"]         = X[:, 2, :].max(dim=1).values.numpy()
    pred_gt_df["max_vm"]        = vm.max(axis=1)
    pred_gt_df["median_x"]      = X[:, 0, :].median(dim=1).values.numpy()
    pred_gt_df["median_y"]      = X[:, 1, :].median(dim=1).values.numpy()
    pred_gt_df["median_z"]      = X[:, 2, :].median(dim=1).values.numpy()
    pred_gt_df["median_vm"]     = np.median(vm, axis=1)
    pred_gt_df["rms_x"]         = (X[:, 0, :] ** 2).mean(dim=1).sqrt().numpy()
    pred_gt_df["rms_y"]         = (X[:, 1, :] ** 2).mean(dim=1).sqrt().numpy()
    pred_gt_df["rms_z"]         = (X[:, 2, :] ** 2).mean(dim=1).sqrt().numpy()
    pred_gt_df["rms_vm"]        = np.sqrt((vm**2).mean(axis=1))
    pred_gt_df["skew_x"]        = skew(X_np[:, 0, :], axis=1)
    pred_gt_df["skew_y"]        = skew(X_np[:, 1, :], axis=1)
    pred_gt_df["skew_z"]        = skew(X_np[:, 2, :], axis=1)
    pred_gt_df["skew_vm"]       = skew(vm, axis=1)
    pred_gt_df["kurt_x"]        = kurtosis(X_np[:, 0, :], axis=1)
    pred_gt_df["kurt_y"]        = kurtosis(X_np[:, 1, :], axis=1)
    pred_gt_df["kurt_z"]        = kurtosis(X_np[:, 2, :], axis=1)
    pred_gt_df["kurt_vm"]       = kurtosis(vm, axis=1)
    pred_gt_df["iqr_x"]         = np.percentile(X_np[:, 0, :], 75, axis=1) - np.percentile(X_np[:, 0, :], 25, axis=1)
    pred_gt_df["iqr_y"]         = np.percentile(X_np[:, 1, :], 75, axis=1) - np.percentile(X_np[:, 1, :], 25, axis=1)
    pred_gt_df["iqr_z"]         = np.percentile(X_np[:, 2, :], 75, axis=1) - np.percentile(X_np[:, 2, :], 25, axis=1)
    pred_gt_df["iqr_vm"]        = np.percentile(vm, 75, axis=1) - np.percentile(vm, 25, axis=1)
    pred_gt_df["mad_x"]         = np.abs(X_np[:, 0, :] - X_np[:, 0, :].mean(axis=1, keepdims=True)).mean(axis=1)
    pred_gt_df["mad_y"]         = np.abs(X_np[:, 1, :] - X_np[:, 1, :].mean(axis=1, keepdims=True)).mean(axis=1)
    pred_gt_df["mad_z"]         = np.abs(X_np[:, 2, :] - X_np[:, 2, :].mean(axis=1, keepdims=True)).mean(axis=1)
    pred_gt_df["mad_vm"]        = np.abs(vm - vm.mean(axis=1, keepdims=True)).mean(axis=1)
    pred_gt_df["specenergy_x"]  = (np.abs(np.fft.rfft(X_np[:, 0, :], axis=1))**2).sum(axis=1)
    pred_gt_df["specenergy_y"]  = (np.abs(np.fft.rfft(X_np[:, 1, :], axis=1))**2).sum(axis=1)
    pred_gt_df["specenergy_z"]  = (np.abs(np.fft.rfft(X_np[:, 2, :], axis=1))**2).sum(axis=1)
    pred_gt_df["specenergy_vm"] = (np.abs(np.fft.rfft(vm, axis=1))**2).sum(axis=1)



    ##### pred_gt_df_summary
    feature_cols = [
        "mean_x",      "mean_y",      "mean_z",      "mean_vm",
        "std_x",       "std_y",       "std_z",       "std_vm",
        "min_x",       "min_y",       "min_z",       "min_vm",
        "max_x",       "max_y",       "max_z",       "max_vm",
        "median_x",    "median_y",    "median_z",    "median_vm",
        "rms_x",       "rms_y",       "rms_z",       "rms_vm",
        "skew_x",      "skew_y",      "skew_z",      "skew_vm",
        "kurt_x",      "kurt_y",      "kurt_z",      "kurt_vm",
        "iqr_x",       "iqr_y",       "iqr_z",       "iqr_vm",
        "mad_x",       "mad_y",       "mad_z",       "mad_vm",
        "specenergy_x","specenergy_y","specenergy_z","specenergy_vm",
    ]

    pred_gt_pair_sum_df = pred_gt_df.groupby(["truth", "pred"])[feature_cols].mean().reset_index()



    #### Save the outputs
    pred_gt_df_path = f"{data_path}/pred_gt_df/pred_gt_df_{split}.csv"
    pred_gt_pair_sum_df_path = f"{data_path}/pred_gt_df/pred_gt_pair_sum_df_{split}.csv"

    pred_gt_df.to_csv(pred_gt_df_path, index=False)
    pred_gt_pair_sum_df.to_csv(pred_gt_pair_sum_df_path, index=False)

    print(f"Saved {pred_gt_df_path}")
    print(f"Saved {pred_gt_pair_sum_df_path}")
    

    if produce_plots:
        class_labels = sorted(pred_gt_pair_sum_df['truth'].unique())

        stat_groups = {
            'Mean':        ['mean_x',       'mean_y',       'mean_z',       'mean_vm'],
            'Std':         ['std_x',        'std_y',        'std_z',        'std_vm'],
            'Min':         ['min_x',        'min_y',        'min_z',        'min_vm'],
            'Max':         ['max_x',        'max_y',        'max_z',        'max_vm'],
            'Median':      ['median_x',     'median_y',     'median_z',     'median_vm'],
            'RMS':         ['rms_x',        'rms_y',        'rms_z',        'rms_vm'],
            'Skew':        ['skew_x',       'skew_y',       'skew_z',       'skew_vm'],
            'Kurt':        ['kurt_x',       'kurt_y',       'kurt_z',       'kurt_vm'],
            'IQR':         ['iqr_x',        'iqr_y',        'iqr_z',        'iqr_vm'],
            'MAD':         ['mad_x',        'mad_y',        'mad_z',        'mad_vm'],
            'Spec Energy': ['specenergy_x', 'specenergy_y', 'specenergy_z', 'specenergy_vm'],
        }
        lab_to_name = {0: "Sedentary", 1: "Standing", 2: "Stepping", 3: "Lying", 4: "Seated Transport", 5: "Transition-active", 6: "Transition-mixed"}
        n_stats = len(stat_groups)
        n_axes  = 4

        fig, axes = plt.subplots(n_stats, n_axes, figsize=(28, n_stats * 6))

        for row, (stat_name, cols) in enumerate(stat_groups.items()):
            for col, feat in enumerate(cols):
                ax = axes[row, col]
                pivot = pred_gt_pair_sum_df.pivot(index='truth', columns='pred', values=feat)
                pivot = pivot.reindex(index=class_labels, columns=class_labels)

                im = ax.imshow(pivot.values, cmap='Blues', aspect='auto')
                plt.colorbar(im, ax=ax, label=feat)

                for i in range(len(class_labels)):
                    for j in range(len(class_labels)):
                        val = pivot.values[i, j]
                        text = f'{val:.3f}' if not np.isnan(val) else 'N/A'
                        ax.text(j, i, text, ha='center', va='center', fontsize=7,
                                color='white' if val > np.nanmax(pivot.values) * 0.7 else 'black')

                ax.set_xticks(range(len(class_labels)))
                ax.set_yticks(range(len(class_labels)))
                ax.set_xticklabels([lab_to_name[l] for l in class_labels], rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels([lab_to_name[l] for l in class_labels], fontsize=8)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_title(feat, color='black', fontsize=15)

            axes[row, 0].set_ylabel(f'{stat_name}\nTrue Label', fontsize=9, fontweight='bold')

        plt.suptitle('Signal Features by Truth-Prediction Pairs', fontsize=16, y=1.01)
        plt.tight_layout()
        plot_path = f"{output_dir}/pred_truth_table_all_{split}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved {plot_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GT/pred pair DataFrames with statistical features")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[1, 60])
    parser.add_argument("--data_path", type=str, default="/niddk-data-central/mae_hr/rise_moca_4AP_20s_transition")
    parser.add_argument("--test_data", action="store_true", default=False)
    parser.add_argument("--RISE_bin_label", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetuned_checkpoint_path", type=str, default=None)
    args = parser.parse_args()


    process_gt_pred_pair_data(
        patch_size=args.patch_size,
        data_path=args.data_path,
        test_data=args.test_data,
        RISE_bin_label=args.RISE_bin_label,
        device=args.device,
        finetuned_checkpoint_path=args.finetuned_checkpoint_path
    )
