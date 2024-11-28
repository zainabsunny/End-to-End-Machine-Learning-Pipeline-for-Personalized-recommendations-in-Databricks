from pyspark.sql.functions import udf, col, datediff, current_date, min, max, countDistinct
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
import spacy

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Unstructured Data: Text Cleaning and NLP Features
def process_text_spacy(text):
    """Process text for stemming and lemmatization using SpaCy."""
    try:
        if isinstance(text, str) and text:
            doc = nlp(text)
            lemmatized = " ".join([token.lemma_ for token in doc])  # Lemmatization
            stemmed = " ".join([token.text[:3] for token in doc])   # Simulated "stemming"
            return stemmed, lemmatized
        else:
            return "", ""  # Handle invalid input
    except Exception as e:
        print(f"Error processing text: {text} | Error: {e}")
        return "", ""

# Register UDFs
stem_udf = udf(lambda text: process_text_spacy(text)[0], StringType())
lemmatize_udf = udf(lambda text: process_text_spacy(text)[1], StringType())

def process_reviews_df(reviews_df):
    """Clean and process reviews DataFrame with stemming and lemmatization."""
    reviews_df = reviews_df.fillna({
        "review_title_clean": "No title",
        "review_text_clean": "No review"
    })
    reviews_df = reviews_df.withColumn("stemmed_title", stem_udf(reviews_df["review_title_clean"]))
    reviews_df = reviews_df.withColumn("lemmatized_title", lemmatize_udf(reviews_df["review_title_clean"]))
    reviews_df = reviews_df.withColumn("stemmed_text", stem_udf(reviews_df["review_text_clean"]))
    reviews_df = reviews_df.withColumn("lemmatized_text", lemmatize_udf(reviews_df["review_text_clean"]))
    return reviews_df

# Structured Data: Outcome Variable (Y) and Predictor Variables (X)
def add_customer_engagement(cosmetic_df):
    """Encode event_type to map engagement levels."""
    print("Updated add_customer_engagement function is running...")
    if "event_type_index" in cosmetic_df.columns:
        cosmetic_df = cosmetic_df.drop("event_type_index")
    indexer = StringIndexer(inputCol="event_type", outputCol="event_type_index")
    cosmetic_df = indexer.fit(cosmetic_df).transform(cosmetic_df)
    return cosmetic_df

def add_predictor_features(cosmetic_df):
    """Add structured predictor features to the DataFrame."""
    # Recency
    cosmetic_df = cosmetic_df.withColumn("recency", datediff(current_date(), col("event_time")))

    # Frequency
    frequency_df = cosmetic_df.groupBy("user_session").count().withColumnRenamed("count", "frequency")
    cosmetic_df = cosmetic_df.join(frequency_df, on="user_session", how="left")

    # Product Popularity
    product_popularity = (
        cosmetic_df.filter(col("event_type") == "purchase")
        .groupBy("cosmeticProductId")
        .count()
        .withColumnRenamed("count", "popularity")
    )
    cosmetic_df = cosmetic_df.join(product_popularity, on="cosmeticProductId", how="left")

    # Session Diversity
    session_diversity = (
        cosmetic_df.groupBy("user_session")
        .agg(countDistinct("category_code").alias("session_diversity"))
    )
    cosmetic_df = cosmetic_df.join(session_diversity, on="user_session", how="left")

    return cosmetic_df
