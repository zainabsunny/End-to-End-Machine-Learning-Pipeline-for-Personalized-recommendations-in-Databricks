from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType
from pyspark.sql.functions import when, concat_ws, col
import re
from pyspark.sql import functions as F

# -----------------------------
# 1) Clean Text
# -----------------------------

# Utility function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
        return text
    return text

# Register as Spark UDF
clean_text_udf = udf(clean_text, StringType())

# -----------------------------
# 2) Clean Cosemtic Data 
# -----------------------------

def clean_cosmetic_df(cosmetic_df):
    return (
        cosmetic_df
        .withColumn(
            "cosmetic_product_title_clean", 
            when(col("category_code").isNotNull(), col("category_code"))  # Use category_code if available
            .otherwise(col("brand"))  # Fallback to brand name
        )
        .withColumnRenamed("product_id", "cosmetic_product_id")
        .withColumnRenamed("price", "cosmetic_price")
        .filter(col('cosmetic_price') > 0)  # Remove invalid prices
    )

# -----------------------------
# 3) Clean Reviews Data
# -----------------------------

def clean_reviews_df(reviews_df):
    """
    Cleans the reviews dataset and ensures a standardized product title column.
    """
    return (reviews_df
        .withColumn("review_product_title_clean", clean_text_udf(col("product_title")))  # Ensure column exists
        .withColumnRenamed("product_id", "review_product_id")
        .withColumnRenamed("price", "review_price")
    )
