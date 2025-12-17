from itertools import product

import click
import numpy as np
import pandas as pd
from utils_pam import create_pam_matrices
from scipy.sparse import csr_array, hstack

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)


def len_u(x: list) -> int:
    """Helper wrapper function to return the length of unique elements in list.

    Args:
        x (list): list

    Returns:
        (int): Length of the set of the list.
    """
    return len(set(x))


def generate_path_features(
    pam_powers: list[csr_array],
    node2id: dict[str, int],
    pairs: list[str or tuple] = [],
    drugs_and_targets: list[str] = [],
    max_node_feats: int = 1000,
    min_df=5,
    max_df=0.95,
) -> bool:
    """
    Function to generate path features for the pairs given. Saves the files in {root}/feats/...
    (Or all pairs if no list of pairs is given).

    Args:
        pam_powers (list[csr_array]): List of sparse arrays.
        node2id (dict[str, int]): Node name to integer in the adjacency matrix.
        pairs (list[str or tuple], optional): Pair names for which to generate the features. If none given
        all are generated. Defaults to []
        drugs_and_targets (list[str], optional): List of the unique drugs and targets . Defaults to [].
        max_node_feats (int, optional): Number of feats to keep. Defaults to 1000.
        min_df (int, optional): Minimum frequency of paths kept (sklearn-based). Defaults to 5.
        max_df (float, optional): Minimum frequency of paths kept (sklearn-based). Defaults to 0.95.

    Returns:
        bool: Nothing, the files are generated and saved
    """

    heads, tails = (
        zip(*pairs)
        if isinstance(pairs[0], tuple)
        else zip(*[pair.split("_") for pair in pairs])
    )

    dt_2_index = {name: index for index, name in enumerate(drugs_and_targets)}

    head_indices_subset = [*map(dt_2_index.get, heads)]
    tail_indices_subset = [*map(dt_2_index.get, tails)]

    head_indices_global = [*map(node2id.get, heads)]
    tail_indices_global = [*map(node2id.get, tails)]

    # These are the path features
    print("Wanted nodes", len(dt_2_index))
    features = []
    for pam_index, pam_power in enumerate(pam_powers):
        print(
            f"Generated feats from PAM-{pam_index} with {int(pam_power.nnz/10**6)}M nnz elements..."
        )
        if pam_power.shape[1] == len(drugs_and_targets):
            straight = pam_power[head_indices_subset, tail_indices_subset]
            reverse = pam_power[tail_indices_subset, head_indices_subset]
        else:
            straight = pam_power[head_indices_subset, tail_indices_global]
            reverse = pam_power[tail_indices_subset, head_indices_global]
        features.append(straight)
        features.append(reverse)

    features = np.array(features).T
    print("Path Features", features.shape)

    feats = np.array(features)
    feats = pd.DataFrame(
        feats,
        columns=[
            f"val@{k+1}_{direction}"
            for k in range(len(pam_powers))
            for direction in ["ori", "rev"]
        ],
    )
    feats["pair"] = (
        pairs if isinstance(pairs[0], str) else ["_".join(pair) for pair in pairs]
    )

    feats_path = f"./feats/PAM_path_feats.csv"
    feats.to_csv(feats_path, index=False)

    if max_node_feats > 0:

        # These are the node features
        node_features = []
        drugs_and_targets_indices_local = [*map(dt_2_index.get, drugs_and_targets)]
        drugs_and_targets_indices_global = [*map(node2id.get, drugs_and_targets)]

        for pam_index, pam_power in enumerate(pam_powers):
            straight = pam_power[drugs_and_targets_indices_local, :]
            if pam_power.shape[1] == len(drugs_and_targets):
                reverse = pam_power[:, drugs_and_targets_indices_local].T
            else:
                reverse = pam_power[:, drugs_and_targets_indices_global].T
            node_features.append(straight)
            node_features.append(reverse)
            print(f"Generated node feats from PAM-{pam_index}")

        stacked = hstack((node_features)).tocsr()
        stacked.eliminate_zeros()
        print("Stacked")

        node_feats2 = [node.data for node in stacked.tocsr()]
        vocab = np.unique(stacked.data)
        # vocab = reduce(np.union1d, node_feats2)

        vect = CountVectorizer(
            vocabulary=vocab,
            token_pattern=None,
            tokenizer=lambda x: x,
            lowercase=False,
            preprocessor=lambda x: x,
        )
        node_feats = vect.fit_transform(node_feats2)
        print(f"BoP for nodes...")
        occurs = node_feats.sum(axis=0).A1

        mask = (occurs >= min_df) & (occurs / len(node_feats2) <= max_df)
        wanted_feats = np.where(mask)[0]
        if max_node_feats > 0:
            wanted_feats_max_indices = np.argsort(occurs[wanted_feats])[::-1][
                :max_node_feats
            ]
            wanted_feats = wanted_feats[wanted_feats_max_indices]
            wanted_feats_names = vect.get_feature_names_out()[wanted_feats_max_indices]
        else:
            wanted_feats_names = vect.get_feature_names_out()
        node_feats = node_feats[:, wanted_feats]
        print(f"Filtered, will do Tfidf..")

        vect_tf = TfidfTransformer()
        node_feats = vect_tf.fit_transform(node_feats).toarray()

        print(f"Node features shape: {node_feats.shape} vocab: {len(vocab)}")

        node_feats_df = pd.DataFrame(node_feats, columns=wanted_feats_names)
        node_feats_df["CUI"] = drugs_and_targets

        node_feats_df.to_csv(
            f"./feats/PAM_node_feats.csv",
            index=False,
        )
    return True


root = "./data"


@click.command()
@click.option(
    "--kg_file",
    "-kg",
    default=f"{root}/knowledge_graph.csv",
    help="Path to the knowledge graph csv.",
)

# We use both the train and test interactions here to make sure we generate
# features for both train and test drugs/diseases.
@click.option(
    "--dt_file",
    "-dt",
    default=f"{root}/drug_disease_interactions.csv",
    help="Path to the drug-target pairs csv.",
)
@click.option(
    "--max_order",
    "-k",
    default=2,
    help="Number of hops to calculate.",
)
@click.option(
    "--max_node_feats",
    "-feat",
    default=500,
    help="Whether to append head and tail features to each path feature vector. If negative will not append.",
)
@click.option(
    "--all_pairs",
    is_flag=True,
    show_default=True,
    default=True,
    help="Whether to create pairs for all possible drug_pairs..",
)
def pam_path_generation(
    kg_file: str,
    dt_file: str,
    max_order: int = 5,
    max_node_feats: int = -1,
    all_pairs: bool = False,
):
    """Wrapper function to generate paths"""

    #### READ the KG ####
    df = pd.read_csv(kg_file, dtype="str")

    unique_nodes = set(df["head"].unique().tolist() + df["tail"].unique().tolist())
    unique_rels = sorted(df["relation"].unique().tolist())
    print(f"Original (N,R,E): ({len(unique_nodes)}, {len(unique_rels)}, {df.shape[0]})")

    print(
        df["relation"]
        .value_counts(normalize=True)
        .sort_values(ascending=False)
        .apply(lambda x: f"{100*x:.2f} %")
    )

    # Read the Drug-Disease Pairs
    df_groundtruth = pd.read_csv(dt_file)
    df_groundtruth["Drug"] = df_groundtruth["Drug_Target"].apply(
        lambda x: x.split("_")[0]
    )
    df_groundtruth["Target"] = df_groundtruth["Drug_Target"].apply(
        lambda x: x.split("_")[1]
    )

    # REMOVE DIRECT TREATS OF GROUNDTRUTH FROM KG
    dt_index = df_groundtruth.set_index(["Drug", "Target"]).index
    td_index = df_groundtruth.set_index(["Target", "Drug"]).index
    df_index = df.set_index(["head", "tail"]).index
    overlap_mask = (df_index.isin(dt_index) | df_index.isin(td_index)) & df[
        "relation"
    ].str.contains("TREAT")
    df = df[~overlap_mask]

    drugs_diseases = pd.read_csv(dt_file)
    drugs_diseases["Drug"] = drugs_diseases["Drug_Target"].apply(
        lambda x: x.split("_")[0]
    )
    drugs_diseases["Target"] = drugs_diseases["Drug_Target"].apply(
        lambda x: x.split("_")[1]
    )

    # KEEP ONLY THOSE IN KG
    print(f"Will keep only pairs that are in KG. Before {drugs_diseases.shape[0]}")
    drugs_diseases = drugs_diseases[
        drugs_diseases["Drug"].isin(unique_nodes)
        & drugs_diseases["Target"].isin(unique_nodes)
    ]
    print(f"After {drugs_diseases.shape[0]}")

    pairs = drugs_diseases["Drug_Target"]
    # assert no duplicates
    assert len(pairs) == len(set(pairs))

    freq = df["tail"].value_counts(normalize=False).sort_values(ascending=False)
    print(f"Total tails: {len(freq)}")

    wanted_nodes_from_pairs = unique_nodes

    # Keep only triples where:
    # It is either an edge starting/ending in the wanted drug, gene nodes
    wanted_triples_from_pairs = df["head"].isin(wanted_nodes_from_pairs) | df[
        "tail"
    ].isin(wanted_nodes_from_pairs)

    print(f"Dropped {sum(~wanted_triples_from_pairs)} triples...")
    df = df[wanted_triples_from_pairs]

    unique_nodes = set(df["head"].unique().tolist() + df["tail"].unique().tolist())
    unique_rels = sorted(df["relation"].unique().tolist())
    print(f"(N,R,E): ({len(unique_nodes)}, {len(unique_rels)}, {df.shape[0]})")

    drugs, targets, pairs_final, kepts = [], [], [], []
    for pair in pairs.apply(lambda x: x.split("_")).values.tolist():
        if pair[0] in unique_nodes and pair[1] in unique_nodes:
            drugs.append(pair[0])
            targets.append(pair[1])
            pairs_final.append("_".join(pair))
            kept = 1
        else:
            kept = 0
        kepts.append(kept)
    drugs_diseases["kept"] = kepts
    print(f"{drugs_diseases.groupby(['GROUNDTRUTH', 'kept']).size().to_string()}")

    # No drugs are targets and vice-versa
    assert len(set(drugs).intersection(targets)) == 0
    # All drugs and targets are in the graph
    assert (
        len(set(drugs).difference(unique_nodes))
        == len(set(targets).difference(unique_nodes))
        == 0
    )

    print(
        f"We have {len_u(pairs_final)}/{len_u(pairs)} ({len_u(drugs)} drugs X {len_u(targets)} targets)"
    )

    drugs = list(set(drugs))
    targets = list(set(targets))
    drugs_and_targets = list(set(drugs + targets))

    if all_pairs:

        unq_drugs = set(drugs)
        unq_targets = set(targets)
        print(
            f"Will generate {len(unq_drugs)} (Drugs) X {len(unq_targets)} (Targets) = {len(unq_drugs)*len(unq_targets)} pairs.."
        )
        pairs_final = list(product(unq_drugs, unq_targets))

    (
        _,
        pam_powers,
        node2id,
        _,
        _,
    ) = create_pam_matrices(
        df,
        max_order=max_order,
        method="plus_times",
        use_log=True,
        eliminate_diagonal=True,
        spacing_strategy="step_10000",
        check_error_lossless=False,
        wanted_nodes_start=drugs_and_targets,
        wanted_nodes_end=drugs_and_targets,
    )

    generate_path_features(
        pam_powers,
        pairs=pairs_final,
        node2id=node2id,
        drugs_and_targets=drugs_and_targets,
        max_node_feats=max_node_feats,
    )
    print("FINISHED")
    return True


if __name__ == "__main__":
    import time

    tic = time.time()
    pam_path_generation()
    took = time.time() - tic
    print(f"Took {took:.1f} secs ({took/60:.2f} mins)")
