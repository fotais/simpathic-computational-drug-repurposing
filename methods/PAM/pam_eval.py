from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import Literal, Union, Tuple


def get_rank(
    cur_target_probas: np.ndarray,
    correct_drug_index: int,
    strategy: Literal["random", "pessimistic", "optimistic", "balanced"] = "random",
) -> int:
    """Helper function for calculating the rank.
    Args:
        cur_target_probas (np.ndarray): Array with probabilities per cell.
        correct_drug_index (int): The index of the wanted item.
        strategy ["random", "pessimistic", "optimistic", "balanced"]): How to handle ties. Defaults to "random".
    Raises:
        NotImplementedError: In case the strategy is not understood.
    Returns:
        int: The true rank
    """

    rank = -1
    if strategy == "random":
        rank = (
            np.argsort(cur_target_probas)[::-1].tolist().index(correct_drug_index) + 1
        )
    else:
        correct_drug_value = cur_target_probas[correct_drug_index]
        if strategy == "pessimistic":
            rank = (cur_target_probas >= correct_drug_value).sum()
        elif strategy == "optimistic":
            rank = (cur_target_probas > correct_drug_value).sum() + 1
        elif strategy == "balanced":
            rank = (
                (cur_target_probas > correct_drug_value).sum()
                + 1
                + (cur_target_probas >= correct_drug_value).sum()
            ) / 2
        else:
            raise NotImplementedError(f"{strategy} not understood")
    return rank


def get_ranks(
    cur_y_probas: np.ndarray,
    targets_test: Union[pd.DataFrame, pd.Series],
    drugs_test: Union[pd.DataFrame, pd.Series],
    strategy: Literal["random", "pessimistic", "optimistic", "balanced"] = "balanced",
) -> np.ndarray:
    """Helper function to calculate all ranks.

    Args:
        cur_y_probas (np.ndarray): The probabilities for all pairs.
        targets_test (Union[pd.DataFrame, pd.Series]): The targets per test sample.
        drugs_test (Union[pd.DataFrame, pd.Series]): The drugs per test sample.
         strategy ["random", "pessimistic", "optimistic", "balanced"]): How to handle ties. Defaults to "random"..

    Returns:
        np.ndarray: The array with the ranks per target.
    """
    targets_2_test_indices = (
        targets_test.reset_index(drop=True)
        .reset_index()
        .groupby("pair")["index"]
        .agg(list)
        .to_dict()
    )
    ranks = []
    for pair_index, target in enumerate(targets_test):
        pred_indices = targets_2_test_indices[target]
        correct_drug = drugs_test.values[pair_index]
        cur_target_probas = cur_y_probas[np.array(pred_indices)][:, 1]
        correct_drug_index = (
            drugs_test.reset_index(drop=True)
            .values[pred_indices]
            .tolist()
            .index(correct_drug)
        )
        rank = get_rank(cur_target_probas, correct_drug_index, strategy)
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks


def get_rank_metrics(
    ranks: np.ndarray, ks=[1, 5, 10, 100], print_=True, return_vals=False
) -> Tuple[float, float, dict[int, float]]:
    """Calculated the metric ranks

    Args:
        ranks ( np.ndarray ): The actual ranks.
        ks (list, optional): The cut-off values for H@K. Defaults to [1, 5, 10, 100].
        print_ (bool, optional): Whether to print the scores as well. Defaults to True.
        return_vals (bool, optional): Whether to return the scores as well. Defaults to False.

    Returns:
        Tuple[float, float, dict[int, float]]: MR, MRR and H@K dict(k,score).
    """
    mr = np.mean(ranks)
    mrr = np.mean(1 / ranks)

    hits_dict = {}
    for k in ks:
        hits = (ranks <= k).mean()
        hits_dict[k] = hits
    if print_:
        print(f"Mean rank: {mr:.2f}")
        print(f"MRR: {mrr:.4f}")
        for k, hits in hits_dict.items():
            print(f"H@{k} : {100*hits:.2f} %")
    if return_vals:
        return mr, mrr, hits_dict
    else:
        return 0, 0, {}


def get_ranks_grouped(
    cur_y_probas: np.ndarray,
    drugs_test: Union[list, np.ndarray],
    wanted_test_drugs: Union[list, np.ndarray],
    num_unique_drugs: int,
    strategy="random",
    targets_test=[],
    top_k=10,
) -> tuple[np.ndarray, list]:
    """Calculating the ranks when the predictions are batch-grouped.
       Also prints per disease the top_k predictions if wanted.

    Args:
        cur_y_probas (np.ndarray): The cutrent group predictions.
        drugs_test (Union[list, np.ndarray]): The names/indices of the drugs in test.
        wanted_test_drugs (Union[list, np.ndarray]): The wanted test drugs.
        num_unique_drugs (int): The number of unique drugs to do the batching.
        strategy (str, optional): Strategy for tie-breaks in ranking. Defaults to "random".
        targets_test (list, optional): The specific names of the targets in test -- for printing. Defaults to [].
        top_k (int, optional): The number of top-k prediction -- for printing. Defaults to 10.

    Returns:
        tuple[np.ndarray, list]: The actual ranks, and the accumulated prediction details.
    """
    ranks = []
    to_print = False

    preds = []
    if len(targets_test) > 0:
        to_print = True
    for pair_index, correct_drug in enumerate(wanted_test_drugs):
        range_of_current_probas = (
            pair_index * num_unique_drugs,
            pair_index * num_unique_drugs + num_unique_drugs,
        )
        cur_drug_probas = cur_y_probas[
            range_of_current_probas[0] : range_of_current_probas[1]
        ]
        cur_drug_names = drugs_test[
            range_of_current_probas[0] : range_of_current_probas[1]
        ]

        try:
            correct_drug_index = cur_drug_names.tolist().index(correct_drug)
        except ValueError as e:
            print(e)
            print(
                f"!!!! THIS IS AN ERROR AND SHOYLDNT HAPPEN. IT MEANS THE WANTED DRUG IS NOT IN THE UNIVERSE OF DRUGS !!!!"
            )
        rank = get_rank(cur_drug_probas, correct_drug_index, strategy=strategy)
        ranks.append(rank)

        if to_print:
            print(f"Rank of correct: {rank}")
            print(
                f"For target - drug pair: {targets_test[pair_index*num_unique_drugs]} - {correct_drug} we predicted:"
            )
            cur_drug_indices_argsort = np.argsort(cur_drug_probas)[::-1]
            softmaxed_scores = softmax(cur_drug_probas)
            for drug_order, drug_index in enumerate(cur_drug_indices_argsort[:top_k]):
                preds.append(
                    (
                        targets_test[pair_index * num_unique_drugs],
                        cur_drug_names[drug_index],
                        softmaxed_scores[drug_index],
                    )
                )
                str_ = f"({drug_order}) Drug: {cur_drug_names[drug_index]} ({softmaxed_scores[drug_index]:.6f})"
                if rank == drug_order + 1:
                    str_ += " (CORRECT)"
                print(str_)

            print("\n")

    ranks = np.array(ranks)
    return ranks, preds


random_state = 42
np.random.seed(random_state)


# Load groundtruth

df_groundtruth = pd.read_csv("./data/drug_disease_interactions.csv")
df_groundtruth["Drug"] = df_groundtruth["Drug_Target"].apply(lambda x: x.split("_")[0])
df_groundtruth["Target"] = df_groundtruth["Drug_Target"].apply(
    lambda x: x.split("_")[1]
)


target2drug = (
    df_groundtruth[df_groundtruth["GROUNDTRUTH"] == 1]
    .groupby("Target")["Drug"]
    .agg(list)
    .to_dict()
)

unique_drugs = np.sort(df_groundtruth["Drug"].unique())
num_unique_drugs = len(unique_drugs)

unique_targets = np.sort(df_groundtruth["Target"].unique())

# Load test set
cur_test_df = pd.read_csv("./data/test.csv", header=None, sep="\t")
cur_test_df.columns = ["Target", "rel", "Drug"]
cur_test_df["pair"] = cur_test_df["Drug"] + "_" + cur_test_df["Target"]
created_test = False


wanted_test_drugs = cur_test_df["Drug"]

cur_y_test = []

for row_i, row in cur_test_df.iterrows():
    cur_drugs_probas = np.zeros(len(unique_drugs))
    try:
        cur_drugs_probas[unique_drugs.tolist().index(row["Drug"])] = 1
    except ValueError as e:
        print(e)
        print(
            f"!!!! THIS IS AN ERROR AND SHOULDN'T HAPPEN. IT MEANS THE WANTED DRUG IS NOT IN THE UNIVERSE OF DRUGS !!!!"
        )

    cur_y_test.append(cur_drugs_probas)

cur_y_test = np.concatenate(cur_y_test)


drugs_test = np.concatenate([unique_drugs] * len(cur_test_df))
targets_test = np.repeat(cur_test_df["Target"].values, repeats=len(unique_drugs))

cur_test_pairs = [
    f"{drug}_{target}" for (drug, target) in zip(drugs_test, targets_test)
]


# Setup models to run

models = {
    "BoP_All": {"clf": "catboost"},
}


if any(["BoP" in model for model in models]):
    pairs_df = pd.read_csv("./feats/PAM_path_feats.csv")
    pairs = pairs_df.set_index("pair")
    nodes_df = pd.read_csv("./feats/PAM_node_feats.csv").set_index("CUI")
    print(f"Loaded nodes and pairs feats...")


res = []
filtered_metrics = True  # Whether to filter out known-positives

## CATBOOST DETAILS ##
neg_sample_ratio = 10  # Default good values for negative sampling
scale_pos_weight = 10  # Default good values for negative sampling
device = "GPU"  # for faster training of CatBoost


root_out = "./results"

print_per_item = True  # for predicting prediction per disease
all_preds = []


for model_name, model_kwargs in models.items():

    if "BoP" in model_name:
        if not created_test:

            try:
                targets_np = nodes_df.loc[targets_test].values
            except KeyError:
                targets_np = np.zeros((len(targets_test), nodes_df.shape[1]))
                print(f"Some test targets have no node features")

            try:
                drugs_np = nodes_df.loc[drugs_test].values
            except KeyError:
                drugs_np = np.zeros((len(drugs_test), nodes_df.shape[1]))
                print(f"Some test drugs have no node features")

            try:
                pairs_np = pairs.loc[cur_test_pairs].values
            except KeyError:
                pairs_np = np.zeros((len(cur_test_pairs), pairs.shape[1]))
                print(f"Some test pairs have no pair features")

            cur_X_test = np.hstack((targets_np, drugs_np, pairs_np))

            subset_train = df_groundtruth[
                ~df_groundtruth["Drug_Target"].isin(cur_test_pairs)
            ]

            num_negs = (subset_train["GROUNDTRUTH"] == 0).sum()

            neg_sample_number = min(
                int(neg_sample_ratio * subset_train["GROUNDTRUTH"].sum()),
                subset_train.shape[0]
                - subset_train[subset_train["GROUNDTRUTH"] == 1].shape[0],
            )
            subset_train = pd.concat(
                (
                    subset_train[subset_train["GROUNDTRUTH"] == 1],
                    subset_train[subset_train["GROUNDTRUTH"] == 0].sample(
                        neg_sample_number, replace=False, random_state=random_state
                    ),
                )
            )
            cur_train_pairs = subset_train["Drug_Target"].values

            try:
                targets_np = nodes_df.loc[subset_train["Target"]].values
            except KeyError:
                targets_np = np.zeros((len(subset_train["Target"]), nodes_df.shape[1]))
                print(f"Some train targets have no node features")

            try:
                drugs_np = nodes_df.loc[subset_train["Drug"]].values
            except KeyError:
                drugs_np = np.zeros((len(subset_train["Drug"]), nodes_df.shape[1]))
                print(f"Some train drugs have no node features")

            try:
                pairs_np = pairs.loc[cur_train_pairs].values
            except KeyError:
                pairs_np = np.zeros((len(cur_train_pairs), pairs.shape[1]))
                print(f"Some train pairs have no node features")

            cur_X_train = np.hstack((targets_np, drugs_np, pairs_np))

            cur_y_train = subset_train["GROUNDTRUTH"].values

            created_test = True

        print(f"Finished creating train / test feature vectors... ")

        cur_X_train_to_use = cur_X_train
        cur_X_test_to_use = cur_X_test

        if model_kwargs["clf"] == "catboost":
            (
                cur_X_train_to_use_final,
                cur_X_val_to_use,
                cur_y_train_final,
                cur_y_val,
            ) = train_test_split(
                cur_X_train_to_use,
                cur_y_train,
                test_size=0.1,
                stratify=cur_y_train,
                random_state=42,
            )
            counts_train = np.bincount(cur_y_train)
            print(f"\n Neg/Pos train counts: {counts_train.tolist()}\n ")

            clf = CatBoostClassifier(
                thread_count=30,
                task_type=device,
                devices="1",
                random_state=42,
                early_stopping_rounds=50,
                verbose=0,
                allow_writing_files=False,
                scale_pos_weight=scale_pos_weight * counts_train[0] / counts_train[1],
                use_best_model=True,
                iterations=1000,
            )
            print(f"Training catboost...")

            clf.fit(
                cur_X_train_to_use_final,
                cur_y_train_final,
                eval_set=(cur_X_val_to_use, cur_y_val),
                verbose=0,
            )
            cur_y_probas = clf.predict_proba(cur_X_test_to_use)
        else:
            raise NotImplementedError(f"{model_kwargs} does not work..")

        (num_samples_train, num_feats_train) = cur_X_train_to_use.shape

        del clf
        del (
            cur_X_train,
            cur_X_train_to_use,
            cur_X_train_to_use_final,
            cur_X_val_to_use,
            cur_X_test,
            cur_X_test_to_use,
            subset_train,
            targets_np,
            drugs_np,
            pairs_np,
        )

    print("Generated scores")

    if filtered_metrics:
        for pair_index, pair_row in cur_test_df.iterrows():
            range_of_current_probas = np.arange(
                pair_index * num_unique_drugs,
                pair_index * num_unique_drugs + num_unique_drugs,
            )
            cur_drug_names = drugs_test[range_of_current_probas]
            interacting_drugs = [
                d for d in target2drug[pair_row["Target"]] if d != pair_row["Drug"]
            ]
            interacting_indices = np.where(np.isin(cur_drug_names, interacting_drugs))[
                0
            ]
            cur_y_probas[range_of_current_probas[interacting_indices], :] = [1, 0]
        print("Filtered scores")

    cur_y_pred = np.argmax(cur_y_probas, axis=1)

    print(f"\n\n")
    if print_per_item:
        to_print = targets_test
    else:
        to_print = []
    ranks, preds = get_ranks_grouped(
        cur_y_probas[:, 1],
        drugs_test,
        wanted_test_drugs,
        num_unique_drugs,
        strategy="random",
        targets_test=to_print,
    )

    mr, mrr, hits_dict = get_rank_metrics(ranks, return_vals=True, print_=False)

    all_preds.extend(preds)

    res.append(
        [
            model_name,
            num_samples_train,
            num_feats_train,
            mr,
            mrr,
        ]
        + list(hits_dict.values())
    )

print()
df_res = pd.DataFrame(
    res,
    columns=[
        "model",
        "num_train_samples",
        "num_train_feats",
        "mr",
        "mrr",
    ]
    + list(hits_dict.keys()),
)


df_res.to_csv(f"{root_out}/results.csv", index=False)

# df_res.to_csv(path_to_save, index=False)
print(df_res.groupby("model")[df_res.columns[2:]].mean().sort_values("mrr"))


preds = pd.DataFrame(all_preds, columns=["Target", "Drug", "Score"])
preds.to_csv(f"{root_out}/per_disease_preds.csv", index=False)
