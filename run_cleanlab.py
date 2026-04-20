import random
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz

from cleanlab import Datalab


def get_label_issues(SEED, test_data, data_path, from_saved=False):
    print("Starting Analysis")
    if test_data:
        split = "test"
    else:
        split = "train"

    np.random.seed(SEED)
    random.seed(SEED)

    NUM_CV_FOLDS = 5

    LABEL_NAMES = {
        0: "Sedentary",
        1: "Standing",
        2: "Stepping",
        3: "Lying",
        4: "Seated_Transport",
        5: "Transition-active",
        6: "Transition-mixed",
    }

    df_full = pd.read_csv(os.path.join(data_path, "stat_feat_df", f"stat_feat_df_{split}.csv"))
    print(f"Full dataset shape: {df_full.shape}")

    FEATURE_COLS = [c for c in df_full.columns if c not in ("pred", "truth")]

    X_raw  = df_full[FEATURE_COLS].values
    labels = df_full["truth"].values.astype(int)

    # Check for NaNs
    n_nan = np.isnan(X_raw).sum()
    rows_w_nan = np.isnan(X_raw).any(axis=1)
    n_rows_w_nan = rows_w_nan.sum()
    print(f"Total NaN values in features: {n_nan}")
    print(f"Rows with NaN values in features: {n_rows_w_nan}")

    X = X_raw[rows_w_nan == False]
    labels = labels[rows_w_nan == False]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"X shape : {X.shape}")
    print(f"labels  : {labels.shape}  unique={np.unique(labels)}")

    # HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(random_state=SEED)
    cv  = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED)

    print(f"Running {NUM_CV_FOLDS}-fold cross-validation...")
    pred_probs = cross_val_predict(
        clf, X, labels,
        cv=cv,
        method="predict_proba",
    )

    print(f"pred_probs shape : {pred_probs.shape}")
    print(f"Row-sum check    : {pred_probs.sum(axis=1)[:5].round(4)}")

    if from_saved:
        from scipy.sparse import load_npz
        knn_graph = load_npz(os.path.join(data_path, 'stat_feat_df', f'knn_graph_{split}.npz'))
        pred_probs = np.load(os.path.join(data_path, 'stat_feat_df', f'pred_probs_{split}.npy'))
        print(f"Loaded knn_graph_{split}.npz and pred_probs_{split}.npy")
    else:
        knn = NearestNeighbors(metric="euclidean")
        knn.fit(X)
        knn_graph = knn.kneighbors_graph(mode="distance")

        # save in case the job fails
        save_npz(os.path.join(data_path, 'stat_feat_df', f'knn_graph_{split}.npz'), knn_graph)
        np.save(os.path.join(data_path, 'stat_feat_df', f'pred_probs_{split}.npy'), pred_probs)
        print(f"KNN graph shape: {knn_graph.shape}")
        print(f"Saved knn_graph_{split}.npz and pred_probs_{split}.npy")


    lab = Datalab(data={"X": X, "y": labels}, label_name="y")

    lab.find_issues(
        pred_probs=pred_probs,
        knn_graph=knn_graph,
    )

    MoCA_pred = df_full['pred'].values.astype(int)
    label_issues = lab.get_issues("label").copy()
    label_issues['MoCA_pred'] = MoCA_pred
    label_issues['given_label_chr'] = label_issues['given_label'].map(LABEL_NAMES)
    label_issues['MoCA_pred_chr'] = label_issues['MoCA_pred'].map(LABEL_NAMES)
    label_issues['predicted_label_chr'] = label_issues['predicted_label'].map(LABEL_NAMES)
    label_issues.reset_index()

    label_issues.to_csv(os.path.join(data_path, f"stat_feat_df/cleanlab_df_{split}.csv"), index=False)

    return label_issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run cleanlab label issue detection")
    parser.add_argument("--SEED", type=int, default=1994)
    parser.add_argument("--test_data", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default="/niddk-data-central/mae_hr/rise_moca_4AP_20s_transition")
    parser.add_argument("--from_saved", action="store_true", default=False)
    args = parser.parse_args()

    get_label_issues(SEED=args.SEED, test_data=args.test_data, data_path=args.data_path, from_saved=args.from_saved)


if __name__ == "__main__":
    main()

