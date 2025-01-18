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
    # Scale `cosmetic_price`
    assembler = VectorAssembler(inputCols=["cosmetic_price"], outputCol="price_vec")
    cosmetic_df = assembler.transform(cosmetic_df)
    
    scaler = StandardScaler(inputCol="price_vec", outputCol="price_scaled")
    cosmetic_df = scaler.fit(cosmetic_df).transform(cosmetic_df)
    
    return cosmetic_df

# -----------------------------
# 2) Transform Mapping Data
# -----------------------------

def transform_mapping_data(mapping_df):
    """
    Transform product mapping data.
    """
    from pyspark.sql.functions import mean, stddev

    # Cast columns to IntegerType
    mapping_df = mapping_df.withColumn("cosmetic_product_id", mapping_df["cosmetic_product_id"].cast(IntegerType()))
    mapping_df = mapping_df.withColumn("review_product_id", mapping_df["review_product_id"].cast(IntegerType()))
    
    # Calculate mean and standard deviation for filtering outliers
    stats = mapping_df.select(
        mean("review_product_id").alias("mean_reviews"),
        stddev("review_product_id").alias("stddev_reviews")
    ).collect()[0]
    
    mean_reviews = stats["mean_reviews"]
    stddev_reviews = stats["stddev_reviews"]
    
    # Filter out outliers
    mapping_df = mapping_df.filter(
        (col("review_product_id") > mean_reviews - 3 * stddev_reviews) &
        (col("review_product_id") < mean_reviews + 3 * stddev_reviews)
    )
    
    return mapping_df

# -----------------------------
# 3) Transform Review Data
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


