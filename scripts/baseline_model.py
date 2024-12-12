import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
import mlflow

def recommendations(df, filename, rows='user_session', cols='product_id', quantity='product_quantity', top=11):
    """
    Generate product recommendations using cosine similarity.
    Tracks metrics and artifacts with MLflow.
    """
    with mlflow.start_run(run_name="Cosine Similarity Recommendations"):
        try:
            # Start time tracking
            start_time = time.time()

            # Prepare data
            orders = list(sorted(set(df[rows])))
            products = list(sorted(set(df[cols])))
            quantities = list(df[quantity])

            rs = pd.Categorical(df[rows], categories=orders).codes
            cs = pd.Categorical(df[cols], categories=products).codes

            # Create sparse matrix
            sparse_matrix = csr_matrix((quantities, (rs, cs)), shape=(len(orders), len(products)))

            # Log sparsity
            matrix_size = sparse_matrix.shape[0] * sparse_matrix.shape[1]
            num_purchases = len(sparse_matrix.nonzero()[0])
            sparsity = round(100 * (1 - (float(num_purchases) / matrix_size)), 2)
            mlflow.log_metric("sparsity", sparsity)

            # Compute cosine similarity
            similarities = cosine_similarity(sparse_matrix.T)
            df_sim = pd.DataFrame(similarities, index=products, columns=products)
            mlflow.log_metric("cosine_similarity_calculation_time", round(time.time() - start_time, 2))

            # Generate recommendations
            start_time = time.time()
            df_match = pd.DataFrame(index=products, columns=[f'Rec {i}' for i in range(1, top)])
            df_score = pd.DataFrame(index=products, columns=[f'Score {i}' for i in range(1, top)])

            for i in range(len(products)):
                top_recs = df_sim.iloc[:, i].sort_values(ascending=False)
                top_recs = top_recs[top_recs.index != df_sim.index[i]]
                num_recs = min(top - 1, len(top_recs), df_match.shape[1])

                df_match.iloc[i, :num_recs] = top_recs.iloc[:num_recs].index
                df_score.iloc[i, :num_recs] = top_recs.iloc[:num_recs].values

            # Combine recommendations and scores
            df_new = df_match.merge(df_score, how="inner", left_index=True, right_index=True)
            df_new.index.names = ['product_id']

            # Save recommendations to file
            df_new.to_csv(filename)
            mlflow.log_artifact(filename)

            print("Cosine similarity recommendations generated.")
            return df_new

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise e

def apriori_rules(df, min_support=0.001):
    """
    Generate association rules using the Apriori algorithm.
    Tracks metrics and artifacts with MLflow.
    """
    with mlflow.start_run(run_name="Apriori Rules"):
        try:
            # Prepare basket data
            basket = (
                df.groupby(['user_session', 'product_id'])['product_quantity']
                .sum().unstack().reset_index().fillna(0).set_index('user_session')
            )

            # Encode as binary
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0).astype(bool)

            # Generate frequent itemsets
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
            frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 2]
            frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

            # Save results to files
            frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
            rules.to_csv("apriori_rules.csv", index=False)
            mlflow.log_artifact("frequent_itemsets.csv")
            mlflow.log_artifact("apriori_rules.csv")

            print("Apriori rules generated.")
            return frequent_itemsets, rules

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise e

