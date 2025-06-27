import os
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, DateType, FloatType, StructType, StructField

def process_silver_table(table_name, snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Wrapper for silver table processing. Handles both static and monthly tables.
    Args:
        table_name: str, one of ['clickstream', 'lms', 'attributes', 'financials']
        snapshot_date_str: str, e.g. '2023-01-01'
        bronze_dir: str, path to bronze directory
        silver_dir: str, path to silver directory
        spark: SparkSession
    """
    table_name = table_name.lower()
    # Standardize all join keys to lower case for consistency
    def standardize_columns(df):
        return df.toDF(*[c.lower() for c in df.columns])

    if table_name == 'clickstream':
        # Monthly clickstream
        bronze_path = os.path.join(bronze_dir, 'clickstream', f'bronze_clks_mthly_{snapshot_date_str.replace("-", "_")}.csv')
        silver_path = os.path.join(silver_dir, 'clickstream', f'silver_clickstream_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        df = standardize_columns(df)
        df = df.withColumn("customer_id", F.col("customer_id").cast(StringType()))
        df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date")))
        for i in range(1, 21):
            df = df.withColumn(f"fe_{i}", F.col(f"fe_{i}").cast(IntegerType()))
        df = df.na.drop(subset=["customer_id", "snapshot_date"])
        df.write.mode("overwrite").parquet(silver_path)
        print(f"Saved {silver_path}")
        return df
    elif table_name == 'lms':
        # Monthly lms
        bronze_path = os.path.join(bronze_dir, 'lms', f'bronze_loan_daily_{snapshot_date_str.replace("-", "_")}.csv')
        silver_path = os.path.join(silver_dir, 'lms', f'silver_lms_mthly_{snapshot_date_str.replace("-", "_")}.parquet')
        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        df = standardize_columns(df)
        column_type_map = {
            "loan_id": StringType(),
            "customer_id": StringType(),
            "loan_start_date": DateType(),
            "tenure": IntegerType(),
            "installment_num": IntegerType(),
            "loan_amt": FloatType(),
            "due_amt": FloatType(),
            "paid_amt": FloatType(),
            "overdue_amt": FloatType(),
            "balance": FloatType(),
            "snapshot_date": DateType(),
        }
        for column, new_type in column_type_map.items():
            df = df.withColumn(column, F.col(column).cast(new_type))
        df = df.withColumn("mob", F.col("installment_num").cast(IntegerType()))
        df = df.withColumn("installments_missed", F.ceil(F.col("overdue_amt") / F.col("due_amt")).cast(IntegerType())).fillna(0)
        df = df.withColumn("first_missed_date", F.when(F.col("installments_missed") > 0, F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed"))).cast(DateType()))
        df = df.withColumn("dpd", F.when(F.col("overdue_amt") > 0.0, F.datediff(F.col("snapshot_date"), F.col("first_missed_date"))).otherwise(0).cast(IntegerType()))
        df.write.mode("overwrite").parquet(silver_path)
        print(f"Saved {silver_path}")
        return df
    elif table_name == 'attributes':
        # Static attributes
        bronze_path = os.path.join(bronze_dir, 'attributes', 'bronze_attr_static.csv')
        silver_path = os.path.join(silver_dir, 'attributes', 'silver_attributes_static.parquet')
        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        # --- Begin full cleaning logic for attributes ---
        invalid_occ = ["_______", "", None]
        df_clean = df.withColumn("Occupation", F.trim(F.col("Occupation"))) \
            .withColumn("Occupation", F.when(F.col("Occupation").isin(invalid_occ), None).otherwise(F.col("Occupation")))
        df_clean = df_clean.withColumn("Name", F.trim(F.col("Name")))
        df_clean = df_clean.withColumn("ssn_valid", F.when(F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), 1).otherwise(0)) \
                     .withColumn("occupation_known", F.when(F.col("Occupation").isNull(), 0).otherwise(1)) \
                     .withColumn("age_valid", F.when((F.col("Age").cast("int").isNotNull()) & (F.col("Age") >= 15), 1).otherwise(0))
        df_clean = df_clean.withColumn("Age", F.col("Age").cast(IntegerType()))
        name_counts = df_clean.groupBy("Name").agg(F.count("*").alias("name_shared_count"))
        df_with_name_count = df_clean.join(name_counts, on="Name", how="left")
        df_with_name_count = df_with_name_count.withColumn("is_name_shared", F.when(F.col("name_shared_count") > 1, 1).otherwise(0))
        df_with_name_count = df_with_name_count.drop("Name", "SSN",)
        df_with_name_count = standardize_columns(df_with_name_count)
        df_with_name_count.write.mode("overwrite").parquet(silver_path)
        print(f"Saved {silver_path}")
        return df_with_name_count
    elif table_name == 'financials':
        # Static financials
        bronze_path = os.path.join(bronze_dir, 'financials', 'bronze_fin_static.csv')
        silver_path = os.path.join(silver_dir, 'financials', 'silver_financials_static.parquet')
        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        # --- Begin full cleaning logic for financials ---
        df = df.withColumn('Annual_Income', F.regexp_replace('Annual_Income', '[^0-9.]', '').cast(FloatType()))
        numeric_cols = ['Monthly_Inhand_Salary', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', \
                        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', \
                        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
        for col in numeric_cols:
            df = df.withColumn(col, F.col(col).cast(FloatType()))
        count_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan']
        for col in count_cols:
            df = df.withColumn(col, F.col(col).cast(IntegerType()))
        df = df.withColumn('Payment_of_Min_Amount', F.when(F.col('Payment_of_Min_Amount') == 'Yes', 1).otherwise(0))
        df = df.withColumn(
            'Credit_History_Years', F.regexp_extract('Credit_History_Age', r'(\d+) Years', 1).cast(IntegerType())
        ).withColumn(
            'Credit_History_Months', F.regexp_extract('Credit_History_Age', r'(\d+) Months', 1).cast(IntegerType())
        ).withColumn(
            'Credit_History_Total_Months', F.col('Credit_History_Years') * 12 + F.coalesce(F.col('Credit_History_Months'), F.lit(0))
        )
        loan_types = (
            df
            .filter(F.col("Type_of_Loan").isNotNull())
            .select(F.explode(F.split(F.col("Type_of_Loan"), ",\\s*")).alias("Loan"))
            .select(
                F.trim(F.regexp_replace(F.col("Loan"), ",", "")).alias("Loan")
            )
            .filter(~F.col("Loan").rlike("(?i)^and\\b.*"))
            .distinct()
            .orderBy("Loan")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        for loan in loan_types:
            col_name = loan.replace(" ", "_").replace("-", "_").lower()
            df = df.withColumn(
                col_name,
                F.when(F.col("Type_of_Loan").contains(loan), F.lit(1)).otherwise(F.lit(0))
            )
        df = df.withColumn('Credit_Mix', F.when(F.col('Credit_Mix') == '_', None).otherwise(F.col('Credit_Mix')))
        df = df.withColumn("credit_mix_good",     F.when(F.col("Credit_Mix") == "Good", 1).otherwise(0)) \
                .withColumn("credit_mix_bad",      F.when(F.col("Credit_Mix") == "Bad", 1).otherwise(0)) \
                .withColumn("credit_mix_standard", F.when(F.col("Credit_Mix") == "Standard", 1).otherwise(0)) \
                .withColumn("valid_credit_mix",    F.when(F.col("Credit_Mix").isin(['Good','Standard','Bad']), 1).otherwise(0))
        df = df.withColumn('Type_of_Loan', F.when(F.col('Type_of_Loan').isin(['NULL', 'Not Specified']), None).otherwise(F.col('Type_of_Loan')))
        df = df.withColumn("Payment_Behaviour", F.lower(F.col("Payment_Behaviour")))        
        df = df.withColumn("Payment_Behaviour", F.when(F.col("Payment_Behaviour") == "!@9#%8", 0).otherwise(F.col("Payment_Behaviour")))        
        df = df.withColumn("has_valid_payment_behavior", F.col("Payment_Behaviour").isNotNull().cast("int"))
        payment_behaviours = (df.select("Payment_Behaviour").filter(F.col("Payment_Behaviour").isNotNull()).distinct().orderBy("Payment_Behaviour").rdd.flatMap(lambda x: x).collect())
        for behaviour in payment_behaviours:
            df = df.withColumn(f"pb_{behaviour}", (F.col("Payment_Behaviour") == behaviour).cast("int"))
        df = standardize_columns(df)
        df = df.drop("credit_history_age","type_of_loan")
        df.write.mode("overwrite").parquet(silver_path)
        print(f"Saved {silver_path}")
        return df
    else:
        raise ValueError(f"Unknown table_name: {table_name}") 