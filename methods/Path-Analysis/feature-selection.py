# Feature Selection

# This script selects features from training and test datasets based on:

# 1. Mutual information with the target.
# 2. Rare but high-signal features (mostly active in positives).

# Running Command: python feature-selection.py path/to/train.csv path/to/test.csv [output.csv]

import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def supervised_feature_selection(
    train_csv,
    test_csv,
    min_positive_only_ratio=0.9,
    min_mi_threshold=0.001,
):
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Separate labels and features
    y_train = train_df["GROUNDTRUTH"]
    X_train = train_df.drop(columns=["GROUNDTRUTH"])
    X_test = test_df.drop(columns=["GROUNDTRUTH"])

    # Keep only shared feature columns
    shared_features = list(set(X_train.columns) & set(X_test.columns))
    if "Drug_Target" in shared_features:
        shared_features.remove("Drug_Target")

    X_train = X_train[shared_features]
    X_test = X_test[shared_features]

    # STEP 1. Mutual Information filtering
    print("Calculating mutual information...")
    mi_scores = mutual_info_classif(X_train.fillna(0), y_train)
    mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi_scores})

    top_mi_features = mi_df[mi_df["mi"] >= min_mi_threshold]["feature"].tolist()

    # STEP 2. Rare but high-signal features (mostly active in positives)
    print("Checking for rare, high-signal features...")
    positive_mask = y_train == 1
    rare_positive_features = []

    for col in X_train.columns:
        non_zero_total = (X_train[col] != 0).sum()
        non_zero_in_positives = (X_train[col][positive_mask] != 0).sum()

        if non_zero_total > 0:
            ratio = non_zero_in_positives / non_zero_total
            if ratio >= min_positive_only_ratio:
                rare_positive_features.append(col)

    # Combine and deduplicate
    selected_features = sorted(set(top_mi_features + rare_positive_features))

    print(f"\n Total selected features: {len(selected_features)}")
    print(
        f"From Mutual Information: {len(top_mi_features)} | Rare Positive-Only: {len(rare_positive_features)}"
    )

    return selected_features


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python feature-selection.py train.csv test.csv [output.csv]")
        sys.exit(1)

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else None

    selected = supervised_feature_selection(train_csv, test_csv)
    print("Selected features:", selected)

    if output_csv:
        pd.DataFrame({"selected_features": selected}).to_csv(output_csv, index=False)
        print(f"Selected features saved to {output_csv}")
