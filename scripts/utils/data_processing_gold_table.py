import os
import glob
import pyspark
import pyspark.sql.functions as F

from tqdm import tqdm

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, NumericType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def read_silver_table(table, silver_db, snapshot_date_str, spark):
    """
    Helper function to read silver table - handles both static and monthly data
    """
    if table in ["attributes", "financials"]:
        # Static data - read from static file
        partition_name = 'silver_' + table + '_static.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        if not os.path.exists(filepath):
            print(f"Static silver {table} file not found: {filepath}")
            return None
        df = spark.read.option("header", "true").parquet(filepath)
        return df
    else:
        # Monthly data - read from monthly partition
        partition_name = 'silver_' + table + '_mthly_' + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = os.path.join(silver_db, table, partition_name)
        df = spark.read.option("header", "true").parquet(filepath)
        return df

############################
# Label Store
############################
def build_label_store(mob, dpd, df):
    """
    Function to build label store
    """
    ####################
    # Create labels
    ####################

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    return df

############################
# Feature Store
############################
def one_hot_encoder(df, category_col):
    """
    Utility function for one hot encoding
    """
    # Get label encoding
    indexer = StringIndexer(inputCol=category_col, outputCol=f"{category_col}_index", handleInvalid="keep")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)

    # Transform into one hot encoding
    encoder = OneHotEncoder(inputCol=f"{category_col}_index", outputCol=f"{category_col}_ohe", dropLast=False)
    df = encoder.fit(df).transform(df)
    vector_to_array_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{category_col}_array", vector_to_array_udf(f"{category_col}_ohe"))

    # Split into columns
    categories = [cat.lower() for cat in indexer_model.labels]

    for i, cat in enumerate(categories):
        df = df.withColumn(f"{category_col}_{cat}", df[f"{category_col}_array"][i])
        df = df.withColumn(f"{category_col}_{cat}", col(f"{category_col}_{cat}").cast(IntegerType()))

    # Optional: drop intermediate columns
    df = df.drop(category_col, f"{category_col}_index", f"{category_col}_ohe", f"{category_col}_array")
    return df

def build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_lms, df_label):
    #############
    # Use df_lms filtered to mob=6 as the anchor table
    #############
    df_lms_mob6 = df_lms.filter(F.col("mob") == 6)

    # Join static and monthly features
    df_joined = df_lms_mob6 \
        .join(df_attributes.drop("snapshot_date"), ["customer_id"], "left") \
        .join(df_financials.drop("snapshot_date"), ["customer_id"], "left") \
        .join(df_loan_type.drop("snapshot_date"), ["customer_id"], "left")
    
    # Drop unnecessary columns
    df_joined = df_joined.drop("name", "ssn", "type_of_loan", "credit_history_age", "type_of_loan")

    # Merge credit history age into one column
    df_joined = df_joined.withColumn("credit_history_age_month", F.col("credit_history_age_year") * 12 + F.col("credit_history_age_month"))
    df_joined = df_joined.drop("credit_history_age_year")

    print("1. Joined dataframes")

    #############
    # Impute mean into null numeric variables
    #############
    numeric_columns = [c.name for c in df_joined.schema.fields if isinstance(c.dataType, NumericType)]
    imputable_columns = []
    for col_name in numeric_columns:
        non_null_count = df_joined.select(col_name).na.drop().count()
        if non_null_count > 0:
            imputable_columns.append(col_name)
        else:
            print(f"Warning: Column '{col_name}' is entirely null and will be skipped by the imputer.")
    if imputable_columns:
        imputer = Imputer(inputCols=imputable_columns, outputCols=imputable_columns)
        df_joined = imputer.fit(df_joined).transform(df_joined)
    print("2. Imputed mean into numeric variables")

    #############
    # Turn categorical variables into one hot encoded columns
    #############
    df_joined = one_hot_encoder(df_joined, "occupation")
    df_joined = one_hot_encoder(df_joined, "payment_of_min_amount")
    df_joined = one_hot_encoder(df_joined, "credit_mix")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_spent")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_value")
    print("3. Performed one-hot encoding")

    #############
    # Aggregate mean clickstream data for each user and snapshot_date
    #############
    df_clickstream_agg = df_clickstream.groupBy("customer_id", "snapshot_date").agg(
        *[F.avg(f'fe_{i}').alias(f"avg_fe_{i}") for i in range(1, 21)]
    )
    print("4. Processed clickstream data")

    #############
    # Join clickstream data with the rest of the features
    #############
    df_joined = df_joined.join(df_clickstream_agg.drop("snapshot_date"), ["customer_id"], how="left")
    print("5. Joined clickstream data with the rest of the features")

    return df_joined

############################
# Pipeline
############################

def process_gold_table(silver_db, gold_db, snapshot_date_str):
    """
    Wrapper function to build all gold tables
    """
    spark = pyspark.sql.SparkSession.builder \
    .appName("gold_table") \
    .master("local[*]") \
    .getOrCreate()

    try:
        # Read silver tables
        df_attributes = read_silver_table('attributes', silver_db, snapshot_date_str, spark)
        df_clickstream = read_silver_table('clickstream', silver_db, snapshot_date_str, spark)
        df_financials = read_silver_table('financials', silver_db, snapshot_date_str, spark)
        df_loan_type = read_silver_table('loan_type', silver_db, snapshot_date_str, spark)
        df_lms = read_silver_table('lms', silver_db, snapshot_date_str, spark)

        # Check if static data exists
        if df_attributes is None or df_financials is None:
            print("Static data (attributes or financials) not found. Skipping gold table processing.")
            return

        # Build label store
        print("Building label store...")
        df_label = build_label_store(6, 30, df_lms)
        
        # Build features
        print("Building features...")
        df_features = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_lms, df_label)

        # Partition and save features
        partition_name_1 = 'gold_feature_store_' + snapshot_date_str.replace('-','_') + '.parquet'
        feature_filepath = os.path.join(gold_db, 'feature_store', partition_name_1)
        df_features.filter(col('snapshot_date')==snapshot_date_str).write.mode('overwrite').parquet(feature_filepath)

        # Partition and save labels
        partition_name_2 = 'gold_label_store_' + snapshot_date_str.replace('-','_') + '.parquet'
        label_filepath = os.path.join(gold_db, 'label_store', partition_name_2)
        df_label.filter(col('snapshot_date')==snapshot_date_str).write.mode('overwrite').parquet(label_filepath)

    finally:
        spark.stop()