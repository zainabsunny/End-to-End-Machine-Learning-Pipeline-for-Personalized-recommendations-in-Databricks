import pandas as pd

def check_cosine_sim_recs_output(df_recs: pd.DataFrame, top: int = 11):
    """
    Perform basic quality checks on the output from generate_cosine_sim_recs().
    :param df_recs: Pandas DataFrame of recommendations, 
                    with index = product_id 
                    and columns = [Rec 1..Rec top-1, Score 1..Score top-1]
    :param top: Number of total columns for Rec+Score pairs (by default 11 -> top-1 = 10 rec/score pairs)
    """
    print("\n=== Cosine Sim Recommendations Output Checks ===")

    # 1. Check if DataFrame is not empty
    if df_recs.empty:
        print("[CHECK] ❌ The recommendation DataFrame is empty!")
        return
    else:
        print(f"[CHECK] ✅ DataFrame has {df_recs.shape[0]} rows and {df_recs.shape[1]} columns.")

    # 2. Check index (product_id) for nulls or duplicates
    if df_recs.index.hasnans:
        print("[CHECK] ❌ The product_id index has null values.")
    else:
        print("[CHECK] ✅ No null values in the product_id index.")

    duplicates_in_index = df_recs.index.duplicated().sum()
    if duplicates_in_index > 0:
        print(f"[CHECK] ❌ Found {duplicates_in_index} duplicate product_id(s) in index.")
    else:
        print("[CHECK] ✅ No duplicate product_id in the index.")

    # 3. Check columns: 
    #    Expecting Rec 1..Rec top-1, Score 1..Score top-1
    expected_rec_cols = [f"Rec {i}" for i in range(1, top)]
    expected_score_cols = [f"Score {i}" for i in range(1, top)]
    missing_cols = [col for col in (expected_rec_cols + expected_score_cols) if col not in df_recs.columns]
    
    if missing_cols:
        print(f"[CHECK] ❌ Missing columns in the output DataFrame: {missing_cols}")
    else:
        print("[CHECK] ✅ All expected Rec / Score columns are present.")

    # 4. Check that Rec columns have no nulls (optional, depends on your needs)
    for rc in expected_rec_cols:
        if rc in df_recs.columns:
            null_count = df_recs[rc].isnull().sum()
            if null_count > 0:
                print(f"[CHECK] ⚠️ Column '{rc}' has {null_count} null values.")
    
    # 5. Check that Score columns are numeric
    for sc in expected_score_cols:
        if sc in df_recs.columns:
            if not pd.api.types.is_numeric_dtype(df_recs[sc]):
                print(f"[CHECK] ❌ Column '{sc}' is not numeric.")
            else:
                print(f"[CHECK] ✅ Column '{sc}' is numeric.")

    print("=== Cosine Sim Recommendations Output Checks Complete ===\n")


def check_fp_growth_output(frequent_itemsets: pd.DataFrame, association_rules: pd.DataFrame):
    """
    Perform basic quality checks on the outputs from FP-Growth.
    :param frequent_itemsets: Pandas DataFrame of frequent itemsets (items, freq)
    :param association_rules: Pandas DataFrame of association rules (antecedent, consequent, confidence, etc.)
    """
    print("\n=== FP-Growth Output Checks ===")

    # --- Check frequent_itemsets ---
    print("Checking frequent_itemsets...")
    if frequent_itemsets.empty:
        print("[CHECK] ❌ frequent_itemsets DataFrame is empty!")
    else:
        print(f"[CHECK] ✅ frequent_itemsets has {frequent_itemsets.shape[0]} rows and {frequent_itemsets.shape[1]} columns.")

        # Example: check expected columns
        expected_cols_fi = {"items", "freq"}
        missing_cols_fi = expected_cols_fi - set(frequent_itemsets.columns)
        if missing_cols_fi:
            print(f"[CHECK] ❌ Missing columns in frequent_itemsets: {missing_cols_fi}")
        else:
            print("[CHECK] ✅ All expected columns in frequent_itemsets are present.")

    # --- Check association_rules ---
    print("Checking association_rules...")
    if association_rules.empty:
        print("[CHECK] ❌ association_rules DataFrame is empty!")
    else:
        print(f"[CHECK] ✅ association_rules has {association_rules.shape[0]} rows and {association_rules.shape[1]} columns.")

        # Example: check expected columns
        expected_cols_ar = {"antecedent", "consequent", "confidence"}
        missing_cols_ar = expected_cols_ar - set(association_rules.columns)
        if missing_cols_ar:
            print(f"[CHECK] ❌ Missing columns in association_rules: {missing_cols_ar}")
        else:
            print("[CHECK] ✅ All expected columns in association_rules are present.")

    print("=== FP-Growth Output Checks Complete ===\n")


def check_als_output(model, user_recs: pd.DataFrame, item_recs: pd.DataFrame):
    """
    Perform basic quality checks on the ALS model output.
    :param model: Trained ALS model (Spark's ALSModel)
    :param user_recs: Pandas DataFrame of user-based recommendations
    :param item_recs: Pandas DataFrame of item-based recommendations
    """
    print("\n=== ALS Recommender Output Checks ===")

    # Check that model is not None
    if model is None:
        print("[CHECK] ❌ The ALS model is None!")
    else:
        print("[CHECK] ✅ ALS model is present.")

    # --- Check user_recs ---
    print("\nChecking user_recs DataFrame...")
    if user_recs.empty:
        print("[CHECK] ❌ user_recs DataFrame is empty!")
    else:
        print(f"[CHECK] ✅ user_recs has {user_recs.shape[0]} rows and {user_recs.shape[1]} columns.")

        # Example: check columns
        expected_cols_ur = {"user_session_index", "recommendations"}
        missing_cols_ur = expected_cols_ur - set(user_recs.columns)
        if missing_cols_ur:
            print(f"[CHECK] ❌ Missing columns in user_recs: {missing_cols_ur}")
        else:
            print("[CHECK] ✅ All expected columns in user_recs are present.")

    # --- Check item_recs ---
    print("\nChecking item_recs DataFrame...")
    if item_recs.empty:
        print("[CHECK] ❌ item_recs DataFrame is empty!")
    else:
        print(f"[CHECK] ✅ item_recs has {item_recs.shape[0]} rows and {item_recs.shape[1]} columns.")

        # Example: check columns
        expected_cols_ir = {"cosmetic_product_id", "recommendations"}
        missing_cols_ir = expected_cols_ir - set(item_recs.columns)
        if missing_cols_ir:
            print(f"[CHECK] ❌ Missing columns in item_recs: {missing_cols_ir}")
        else:
            print("[CHECK] ✅ All expected columns in item_recs are present.")

    print("\n=== ALS Recommender Output Checks Complete ===\n")