from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Utility function for text cleaning
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.replace('\n', ' ')  # Remove new lines
        text = text.replace('\t', ' ')  # Remove tabs
        text = text.replace('\r', ' ')  # Remove returns
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphanumeric characters except spaces
        words = text.split()  # Tokenize the text
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        text = ' '.join(words)  # Combine words back into text
        return text.strip()
    return text  # Return original value if not a string (e.g., NaN)

# Register the clean_text function as a UDF
clean_text_udf = udf(clean_text, StringType())

def clean_cosmetic_df(cosmetic_df: DataFrame) -> DataFrame:
    # Standardize column names
    cosmetic_df = cosmetic_df.withColumnRenamed("product_id", "cosmeticProductId") \
                             .withColumnRenamed("price", "cosmetic_price") 
    
    # Filter out rows with invalid prices
    cosmetic_df = cosmetic_df.filter(cosmetic_df['cosmetic_price'] > 0)
    
    # Filter for valid events in multi-classification
    valid_events = ["view", "cart", "remove_from_cart", "purchase"]
    cosmetic_df = cosmetic_df.filter(cosmetic_df.event_type.isin(valid_events))
    
    return cosmetic_df

def clean_mapping_df(mapping_df: DataFrame) -> DataFrame:
    # Standardize column names
    mapping_df = mapping_df.withColumnRenamed("product_id_events", "cosmeticProductId") \
                           .withColumnRenamed("product_id_reviews", "reviewProductId")
    
    # Drop rows with missing values
    mapping_df = mapping_df.na.drop()
    return mapping_df

def clean_reviews_df(reviews_df: DataFrame) -> DataFrame:
    # Standardize column names
    reviews_df = reviews_df.withColumnRenamed("product_id", "reviewProductId") \
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

