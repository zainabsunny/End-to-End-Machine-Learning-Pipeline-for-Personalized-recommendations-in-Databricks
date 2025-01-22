from pyspark.sql.types import (
    StructType, StructField, TimestampType, StringType,
    IntegerType, LongType, DoubleType
)

expected_cosmetic_schema = StructType([
    StructField("event_time", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("category_id", LongType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("user_session", StringType(), True),
])

expected_reviews_schema = StructType([
    StructField("product_id", IntegerType(), True),
    StructField("brand_name", StringType(), True),
    StructField("review_id", IntegerType(), True),
    StructField("review_title", StringType(), True),
    StructField("review_text", StringType(), True),
    StructField("author", StringType(), True),
    StructField("review_date", StringType(), True),
    StructField("review_rating", StringType(), True),
    StructField("is_a_buyer", StringType(), True),
    StructField("pro_user", StringType(), True),
    StructField("review_label", StringType(), True),
    StructField("product_title", StringType(), True),
    StructField("mrp", StringType(), True),
    StructField("price", StringType(), True),
    StructField("product_rating", StringType(), True),
    StructField("product_rating_count", StringType(), True),
    StructField("product_tags", StringType(), True),
    StructField("product_url", StringType(), True),
])

expected_mapping_schema = StructType([
    StructField("product_id_events", IntegerType(), True),
    StructField("product_id_reviews", IntegerType(), True),
])