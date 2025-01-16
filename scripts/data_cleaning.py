from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
import re


# Utility function for text cleaning, for dunzhang/stella_en_400M_v5 only need to clean up whitespaces
def clean_text(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ')  # Remove new lines
        text = text.replace('\t', ' ')  # Remove tabs
        text = text.replace('\r', ' ')  # Remove returns
        return text.strip()
    return text 

# Register the clean_text function as a UDF
clean_text_udf = udf(clean_text, StringType())

def clean_cosmetic_df(cosmetic_df: DataFrame) -> DataFrame:
    # Standardize column names
    cosmetic_df = cosmetic_df.withColumnRenamed("product_id", "cosmetic_product_id") \
                             .withColumnRenamed("price", "cosmetic_price") 
    
    # Filter out rows with invalid prices
    cosmetic_df = cosmetic_df.filter(cosmetic_df['cosmetic_price'] > 0)
    
    # Filter for valid events in multi-classification
    valid_events = ["view", "cart", "remove_from_cart", "purchase"]
    cosmetic_df = cosmetic_df.filter(cosmetic_df.event_type.isin(valid_events))
    
    return cosmetic_df

def clean_mapping_df(mapping_df: DataFrame) -> DataFrame:
    # Standardize column names
    mapping_df = mapping_df.withColumnRenamed("product_id_events", "cosmetic_product_id") \
                           .withColumnRenamed("product_id_reviews", "review_product_id")
    
    # Drop rows with missing values
    mapping_df = mapping_df.na.drop()
    return mapping_df

def clean_reviews_df(reviews_df: DataFrame) -> DataFrame:
    # Standardize column names
    reviews_df = reviews_df.withColumnRenamed("product_id", "review_product_id") \
                           .withColumnRenamed("price", "review_price")
    
    # Drop unnecessary columns and handle missing values
    reviews_df = reviews_df.drop("product_tags")
    reviews_df = reviews_df.fillna({
        'review_text': 'No review', 
        'brand_name': 'Unknown', 
        'review_label': 'No Label',
        'product_title': 'Unknown Title'
    })
    
    # Apply cleaning to text columns
    reviews_df = reviews_df.withColumn("review_title_clean", clean_text_udf(reviews_df["review_title"]))
    reviews_df = reviews_df.withColumn("review_text_clean", clean_text_udf(reviews_df["review_text"]))
    
    return reviews_df

