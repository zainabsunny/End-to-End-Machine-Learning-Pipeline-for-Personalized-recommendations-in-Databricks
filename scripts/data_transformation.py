from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col

# -----------------------------
# 1) Transform Structured Data
# -----------------------------

def transform_cosmetic_data(cosmetic_df):
    """
    Transform structured interaction data (e.g., scaling prices).
    """
    # Use the correct column name
    price_col = "cosmetic_price"
    
    # Check if `cosmetic_price` column exists
    if price_col not in cosmetic_df.columns:
        raise ValueError(f"Column '{price_col}' not found in cosmetic_df. Available columns: {cosmetic_df.columns}")

    # Scale `cosmetic_price`
    assembler = VectorAssembler(inputCols=[price_col], outputCol="price_vec")
    cosmetic_df = assembler.transform(cosmetic_df)
    
    scaler = StandardScaler(inputCol="price_vec", outputCol="price_scaled")
    cosmetic_df = scaler.fit(cosmetic_df).transform(cosmetic_df)
    
    return cosmetic_df

# -----------------------------
# 2) Transform Review Data
# -----------------------------

def transform_reviews_data(reviews_df):
    """
    Transform unstructured sentiment data by casting numerical columns to appropriate types.
    """
    reviews_df = reviews_df.withColumn("mrp", reviews_df["mrp"].cast(DoubleType())) \
                           .withColumn("review_price", reviews_df["review_price"].cast(DoubleType())) \
                           .withColumn("product_rating", reviews_df["product_rating"].cast(DoubleType())) \
                           .withColumn("product_rating_count", reviews_df["product_rating_count"].cast(IntegerType()))
    return reviews_df


