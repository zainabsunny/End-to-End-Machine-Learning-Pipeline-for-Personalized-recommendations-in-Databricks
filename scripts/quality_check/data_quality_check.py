from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

def check_schema(df: DataFrame, expected_schema: StructType) -> bool:
    """
    Compare the actual DataFrame schema to the expected schema.
    Return True if they match exactly (in the same order and types).
    Otherwise, return False.
    """
    actual_fields = df.schema.fields
    expected_fields = expected_schema.fields
    
    if len(actual_fields) != len(expected_fields):
        print("[SCHEMA CHECK] ❌ Number of fields do not match.")
        return False
    
    for actual_field, expected_field in zip(actual_fields, expected_fields):
        if (actual_field.name != expected_field.name or
            type(actual_field.dataType) != type(expected_field.dataType) or
            actual_field.nullable != expected_field.nullable):
            print(f"[SCHEMA CHECK] ❌ Mismatch found: {actual_field} vs. {expected_field}")
            return False
    
    print("[SCHEMA CHECK] ✅ Schema matches expected.")
    return True


def check_null_values(df: DataFrame, threshold: float = 0.5):
    """
    Checks for columns with a high proportion of nulls.
    threshold: if > threshold fraction of rows in a column are null, flag it.
    """
    total_rows = df.count()
    if total_rows == 0:
        print("[NULL VALUES CHECK] ⚠️ DataFrame is empty, cannot check null values.")
        return
    
    for col in df.columns:
        null_count = df.filter(df[col].isNull()).count()
        fraction_null = null_count / total_rows
        if fraction_null > threshold:
            print(f"[NULL VALUES CHECK] ⚠️ Column '{col}' has high null fraction: {fraction_null:.2f}")


def check_duplicate_rows(df: DataFrame, subset_columns=None):
    """
    Checks for duplicate rows in the entire DataFrame or a subset of columns.
    If subset_columns is None, checks duplicates across all columns.
    """
    if subset_columns is None:
        subset_columns = df.columns
    
    total_count = df.count()
    unique_count = df.dropDuplicates(subset_columns).count()
    
    if total_count != unique_count:
        print(f"[DUPLICATES CHECK] ❌ Found duplicates. {total_count - unique_count} duplicate rows.")
    else:
        print("[DUPLICATES CHECK] ✅ No duplicates found.")


def check_data_types(df: DataFrame, expected_schema: StructType):
    """
    Checks if each column in the DataFrame has the correct data type
    as specified in the expected_schema.
    """
    for field in expected_schema.fields:
        col_name = field.name
        expected_type = field.dataType
        # Find the actual field in df
        actual_field = next((f for f in df.schema.fields if f.name == col_name), None)
        if actual_field is None:
            print(f"[DATA TYPES CHECK] ❌ Missing column '{col_name}'.")
        else:
            if type(actual_field.dataType) != type(expected_type):
                print(f"[DATA TYPES CHECK] ❌ Column '{col_name}' has type {actual_field.dataType}, expected {expected_type}.")


def check_row_count(df: DataFrame, min_count: int = 1):
    """
    Ensure that the DataFrame has at least min_count rows.
    """
    count_ = df.count()
    if count_ < min_count:
        print(f"[ROW COUNT CHECK] ❌ DataFrame has only {count_} rows, expected at least {min_count}.")
    else:
        print(f"[ROW COUNT CHECK] ✅ DataFrame has {count_} rows (min expected was {min_count}).")