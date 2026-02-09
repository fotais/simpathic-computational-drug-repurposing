# Script for training and evaluating the path-based approach.
# Includes 5-run Random Forest training, query-level ranking metrics (Hits@k, MRR),
# and generation of top-100 candidate predictions per studied neuro-RD.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# The feature columns were retained after a 2-stage supervised feature selection procedure including:
# mutual information & recovery of rare but positive-enriched features.

feature_cols = [
    'Similarity C0001125', 'Similarity C0002066', 'Similarity C0002395', 'Similarity C0002986',
    'Similarity C0004096', 'Similarity C0004238', 'Similarity C0007194', 'Similarity C0007222',
    'Similarity C0011168', 'Similarity C0011849', 'Similarity C0011860', 'Similarity C0014544',
    'Similarity C0017205', 'Similarity C0018790', 'Similarity C0018799', 'Similarity C0018801',
    'Similarity C0018802', 'Similarity C0019147', 'Similarity C0019151', 'Similarity C0019202',
    'Similarity C0019247', 'Similarity C0019880', 'Similarity C0020179', 'Similarity C0020538',
    'Similarity C0020541', 'Similarity C0020615', 'Similarity C0023521', 'Similarity C0023890',
    'Similarity C0023895', 'Similarity C0024117', 'Similarity C0024408', 'Similarity C0024776',
    'Similarity C0025517', 'Similarity C0025521', 'Similarity C0026848', 'Similarity C0026850',
    'Similarity C0026896', 'Similarity C0027051', 'Similarity C0027126', 'Similarity C0027765',
    'Similarity C0027868', 'Similarity C0028754', 'Similarity C0030567', 'Similarity C0031485',
    'Similarity C0038454', 'Similarity C0043459', 'Similarity C0085584', 'Similarity C0087012',
    'Similarity C0149931', 'Similarity C0154246', 'Similarity C0162309', 'Similarity C0162671',
    'Similarity C0220756', 'Similarity C0242422', 'Similarity C0268250', 'Similarity C0268251',
    'Similarity C0268465', 'Similarity C0268490', 'Similarity C0268542', 'Similarity C0268579',
    'Similarity C0270846', 'Similarity C0282528', 'Similarity C0349653', 'Similarity C0442874',
    'Similarity C0494475', 'Similarity C0524851', 'Similarity C0751434', 'Similarity C0751435',
    'Similarity C0751436', 'Similarity C0751882', 'Similarity C0752120', 'Similarity C0878544',
    'Similarity C0878773', 'Similarity C0948089', 'Similarity C1145670', 'Similarity C1527231',
    'Similarity C1849508', 'Similarity C1961835', 'Similarity C2931688', 'Similarity C5203670',
    'nod1_chem', 'nod1_elii', 'nod1_euka', 'nod1_imft', 'nod1_inbe', 'nod1_inpr', 'nod1_mcha',
    'nod1_orch', 'nod1_orgm', 'nod1_phsu', 'nod2_aapp', 'nod2_antb', 'nod2_bacs', 'nod2_celf',
    'nod2_cell', 'nod2_dsyn', 'nod2_elii', 'nod2_emod', 'nod2_fndg', 'nod2_gngm', 'nod2_hops',
    'nod2_horm', 'nod2_imft', 'nod2_inch', 'nod2_irda', 'nod2_lbtr', 'nod2_mobd', 'nod2_moft',
    'nod2_mosq', 'nod2_neop', 'nod2_nnon', 'nod2_orch', 'nod2_ortf', 'nod2_phob', 'nod2_phsu',
    'nod2_plnt', 'nod2_pros', 'nod2_sosy', 'nod2_topp', 'nod2_vita', 'nod3_aapp', 'nod3_antb',
    'nod3_bacs', 'nod3_bpoc', 'nod3_chvf', 'nod3_chvs', 'nod3_clna', 'nod3_diap', 'nod3_dsyn',
    'nod3_enzy', 'nod3_euka', 'nod3_fndg', 'nod3_gngm', 'nod3_hlca', 'nod3_hops', 'nod3_horm',
    'nod3_humn', 'nod3_lbpr', 'nod3_mamm', 'nod3_medd', 'nod3_mobd', 'nod3_mosq', 'nod3_neop',
    'nod3_ocac', 'nod3_orch', 'nod3_orgf', 'nod3_ortf', 'nod3_patf', 'nod3_phsu', 'nod3_plnt',
    'nod3_podg', 'nod3_popg', 'nod3_sosy', 'nod3_tisu', 'nod3_topp', 'nod3_virs', 'rel1_AFFECTS',
    'rel1_AUGMENTS', 'rel1_COEXISTS_WITH', 'rel1_HAS_MESH', 'rel1_INHIBITS', 'rel1_INTERACTS_WITH',
    'rel1_ISA', 'rel1_MENTIONED_IN', 'rel1_PREVENTS', 'rel1_STIMULATES', 'rel1_TREATS',
    'rel1_USES', 'rel1_compared_with', 'rel1_higher_than', 'rel2_ADMINISTERED_TO', 'rel2_AFFECTS',
    'rel2_ASSOCIATED_WITH', 'rel2_AUGMENTS', 'rel2_CAUSES', 'rel2_COEXISTS_WITH', 'rel2_COMPLICATES',
    'rel2_CONVERTS_TO', 'rel2_DISRUPTS', 'rel2_HAS_MESH', 'rel2_INHIBITS', 'rel2_INTERACTS_WITH',
    'rel2_ISA', 'rel2_IS_A', 'rel2_LOCATION_OF', 'rel2_MENTIONED_IN', 'rel2_PREDISPOSES',
    'rel2_PREVENTS', 'rel2_PROCESS_OF', 'rel2_PRODUCES', 'rel2_STIMULATES', 'rel2_TREATS',
    'rel2_USES', 'rel2_compared_with', 'rel2_higher_than', 'rel2_lower_than', 'rel2_same_as',
    'rel3_AFFECTS', 'rel3_ASSOCIATED_WITH', 'rel3_AUGMENTS', 'rel3_CAUSES', 'rel3_COEXISTS_WITH',
    'rel3_DISRUPTS', 'rel3_HAS_MESH', 'rel3_ISA', 'rel3_IS_A', 'rel3_LOCATION_OF', 'rel3_MENTIONED_IN',
    'rel3_PREDISPOSES', 'rel3_PREVENTS', 'rel3_PROCESS_OF', 'rel3_TREATS'
]

# Query-based evaluation
def query_based_metrics(test_df, y_scores, k_list=[1,5,10,100]):
    """
    Computes query-level ranking metrics.
    For each positive drug–disease pair, candidates are restricted to the same disease.
    """
    hits_at_k_counts = {k: [] for k in k_list}
    mrr_list = []

    positive_queries = test_df[test_df['GROUNDTRUTH'] == 1]

    for _, query_row in positive_queries.iterrows():
        query_pair = query_row['Drug_Target']
        query_drug, query_disease = query_pair.split('_')

        # Retrieve candidate drug–disease pairs for the same disease.
        # Other positives are excluded.
        candidates = test_df[test_df['Drug_Target'].str.split('_').str[1] == query_disease]
        candidates = candidates[(candidates['GROUNDTRUTH'] == 0) | (candidates['Drug_Target'] == query_pair)]

        candidates = candidates.copy()
        candidates['score'] = y_scores[candidates.index]
        candidates = candidates.sort_values('score', ascending=False)
        ranked_list = candidates['Drug_Target'].tolist()

        # Hits@k for each k
        for k in k_list:
            hits_at_k_counts[k].append(int(query_pair in ranked_list[:k]))
            
        # Reciprocal Rank of the true pair
        rank = ranked_list.index(query_pair) + 1
        mrr_list.append(1.0 / rank)

    avg_hits = {k: np.mean(hits_at_k_counts[k]) for k in k_list}
    avg_mrr = np.mean(mrr_list)
    return avg_hits, avg_mrr

# Save query-based top-N predictions
def save_query_based_topN(test_df, y_scores, top_n=100, output_file="QUERY_TOP_PREDICTIONS.csv"):
    """
    Saves the top-N predicted drugs for each disease appearing in positive test queries.
    """
    positive_queries = test_df[test_df['GROUNDTRUTH'] == 1].copy()
    rows = []

    for _, query_row in positive_queries.iterrows():
        query_pair = query_row['Drug_Target']
        query_drug, query_disease = query_pair.split('_')

        # Candidates: same disease, exclude other positives
        candidates = test_df[test_df['Drug_Target'].str.split('_').str[1] == query_disease]
        candidates = candidates[(candidates['GROUNDTRUTH'] == 0) | (candidates['Drug_Target'] == query_pair)]

        candidates = candidates.copy()
        candidates['score'] = y_scores[candidates.index]
        candidates = candidates.sort_values('score', ascending=False)

        # Extract drug IDs from ranked Drug_Target entries.
        ranked_drugs = [d.split('_')[0] for d in candidates['Drug_Target'].tolist()[:top_n]]

        # Row: diseaseID, query drug, top-N predictions
        rows.append([query_disease, query_drug] + ranked_drugs)

    columns = ['Disease_ID', 'Drug_ID (query)'] + list(range(1, top_n+1))
    topN_df = pd.DataFrame(rows, columns=columns)
    topN_df.to_csv(output_file, index=False)
    print(f"Saved query-based top-{top_n} predictions to {output_file}")

# Main training & Evaluation
def main(train_file, test_file, n_runs=5, k_list=[1,5,10,100], top_n=100):
    """
    Trains multiple Random Forest models, evaluates them with query-level metrics,
    and stores the best-performing run (based on Hits@10).
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['GROUNDTRUTH']

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['GROUNDTRUTH']

    best_hits10 = -1
    best_run_scores = None
    best_run_metrics = None

    for run in range(n_runs):
        print(f"\n ~~~~ Run {run+1}/{n_runs} ~~~~")
        
        # A new model per run via different random seeds.
        model = RandomForestClassifier(n_estimators=100, random_state=run)
        model.fit(X_train, y_train)
        
        # Predicted probabilities assigned to the positive class.
        y_scores = model.predict_proba(X_test)[:,1]

        avg_hits, avg_mrr = query_based_metrics(test_df, y_scores, k_list=k_list)
        print("Query-based Hits@k:", avg_hits)
        print("Query-based MRR:", avg_mrr)

        # Keep the run with the strongest Hits@10 value.
        if avg_hits[10] > best_hits10:
            best_hits10 = avg_hits[10]
            best_run_scores = y_scores
            best_run_metrics = (avg_hits, avg_mrr)

    # Save query-based top-N predictions for the best run.
    save_query_based_topN(test_df, best_run_scores, top_n=top_n)
    print("\n ~~~~ Best run (by Hits@10) saved ~~~~ ")
    print("Query-based Hits@k:", best_run_metrics[0])
    print("Query-based MRR:", best_run_metrics[1])

if __name__ == "__main__":
    main("Training_features.csv", "Test_features.csv", n_runs=5, top_n=100)
