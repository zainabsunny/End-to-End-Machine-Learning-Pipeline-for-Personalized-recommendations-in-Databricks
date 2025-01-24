from pyspark.sql.functions import col, size, when, lit, array
from pyspark.sql.types import ArrayType, FloatType

def validate_embeddings(df, embedding_col="aggregated_embedding", expected_dim=384):
    """
    Validate the embeddings in the DataFrame by checking their size and existence.

    Args:
        df (DataFrame): The input Spark DataFrame with embeddings.
        embedding_col (str): The name of the column containing embeddings.
        expected_dim (int): The expected dimensionality of the embeddings.

    Returns:
        DataFrame: A DataFrame with a validation column indicating if the embedding is valid.
    """
    return df.withColumn(
        "is_valid_embedding",
        when(
            (col(embedding_col).isNotNull()) & (size(col(embedding_col)) == expected_dim),
            True
        ).otherwise(False)
    )


def replace_invalid_embeddings(df, embedding_col="aggregated_embedding", default_embedding=None, expected_dim=384):
    """
    Replace invalid or missing embeddings with a default embedding.

    Args:
        df (DataFrame): The input Spark DataFrame with embeddings.
        embedding_col (str): The name of the column containing embeddings.
        default_embedding (list): The default embedding to replace invalid or missing ones.

    Returns:
        DataFrame: A DataFrame with invalid embeddings replaced by the default.
    """
    if default_embedding is None:
        default_embedding = [0.0] * expected_dim  # Default to a zero vector

    # Convert the default embedding into an array of literals
    default_embedding_array = array(*[lit(value) for value in default_embedding])

    return df.withColumn(
        embedding_col,
        when(
            col("is_valid_embedding") == False,  # Replace if invalid
            default_embedding_array
        ).otherwise(col(embedding_col))
    ).drop("is_valid_embedding")


def clean_aggregated_embeddings(aggregated_embeddings_df, embedding_col="aggregated_embedding", expected_dim=384):
    """
    Clean the aggregated embeddings by validating and replacing invalid entries.

    Args:
        aggregated_embeddings_df (DataFrame): The DataFrame with aggregated embeddings.
        embedding_col (str): The name of the column containing embeddings.
        expected_dim (int): The expected dimensionality of the embeddings.

    Returns:
        DataFrame: A cleaned DataFrame with valid embeddings.
    """
    # Validate embeddings
    validated_df = validate_embeddings(
        aggregated_embeddings_df,
        embedding_col=embedding_col,
        expected_dim=expected_dim
    )

    # Replace invalid embeddings
    cleaned_df = replace_invalid_embeddings(
        validated_df,
        embedding_col=embedding_col,
        default_embedding=[0.0] * expected_dim,
        expected_dim=expected_dim
    )

    return cleaned_df