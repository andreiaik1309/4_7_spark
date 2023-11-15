from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd


def main():
    spark = SparkSession.builder \
        .appName("task 4.7") \
        .master("local[*]") \
        .enableHiveSupport() \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .getOrCreate()
    
    customSchema = StructType([ \
        StructField("InvoiceNo",StringType(),True), \
        StructField("StockCode",StringType(),True), \
        StructField("Description",StringType(),True), \
        StructField("Quantity", DoubleType(), True), \
        StructField("InvoiceDate", TimestampType(), True), \
        StructField("UnitPrice", DoubleType(), True), \
        StructField("CustomerID", StringType(), True), \
        StructField("Country", StringType(), True)
    ])
    
    df_raw = spark.read.format("csv").options(header="true").schema(customSchema).load("file:///home/prophet/conda/yakovlev/1t_data/online_retail.csv")

    # Количество строк в файле
    number_rows = df_raw.count()
    # Количество уникальных клиентов
    unique_customers = df_raw.select("CustomerID").distinct().count()
    # В какой стране совершается большинство покупок
    most_purchased_country = df_raw.groupBy("Country").count().orderBy(F.col("count").desc()).first()["Country"]
    # Даты самой ранней и самой последней покупки на платформе
    min_date = df_raw.agg(F.min('InvoiceDate')).collect()[0][0]
    max_date = df_raw.agg(F.max('InvoiceDate')).collect()[0][0]

    df_raw = df_raw.filter(F.col('CustomerID').isNotNull())
    number_rows_whisout_null_custid = df_raw.count()

    # Посчитаем Recency 
    df_rfm = df_raw.groupBy("CustomerID").agg(F.max("InvoiceDate").alias("MaxPurchaseDate"))
    max_date_str = max_date.strftime("%Y-%m-%d %H:%M:%S") 
    df_rfm = df_rfm.withColumn('Recency', F.datediff(F.lit(max_date_str), F.col('MaxPurchaseDate')))
    # Посчитаем Frequency
    df_rfm_f = df_raw.groupBy('CustomerID').agg(F.countDistinct('InvoiceNo').alias("Frequency"))
    # Посчитаем столбец Monetary
    df_rfm_m = df_raw.groupBy("CustomerID").agg(F.sum(F.col("UnitPrice") * F.col("Quantity")).alias("TotalAmount"))

    df_rfm = df_rfm.join(df_rfm_f, "CustomerID").join(df_rfm_m, "CustomerID")
    df_rfm = df_rfm.withColumn("Monetary", F.round(F.col("TotalAmount") / F.col('Frequency'), 2))
    df_rfm = df_rfm.select('CustomerID', 'Recency', 'Frequency', 'Monetary')

    # Функция для определения группы в соответствии с квантилями
    def assign_group_recency(value, quantiles):
        if value <= quantiles[0]:
            return 'A'
        elif quantiles[0] < value <= quantiles[1]:
            return 'B'
        else:
            return 'C'

    def assign_group_other(value, quantiles):
        if value <= quantiles[0]:
            return 'C'
        elif quantiles[0] < value <= quantiles[1]:
            return 'B'
        else:
            return 'A'
        
    # Применение функции к каждому столбцу
    for column in df_rfm.columns[1:]:
        quantiles = df_rfm.approxQuantile(column, [0.333, 0.667], 0.01)
        if column == 'Recency':
            assign_group_udf = spark.udf.register(f"assign_group_{column}", lambda x: assign_group_recency(x, quantiles))
        else:
            assign_group_udf = spark.udf.register(f"assign_group_{column}", lambda x: assign_group_other(x, quantiles))

        df_rfm = df_rfm.withColumn(f"{column}_Group", assign_group_udf(F.col(column)))
    # Добавляем столбец с общим Score
    df_rfm = df_rfm.withColumn('TotalScore', F.concat(F.col('Recency_Group'), F.col('Frequency_Group'), F.col('Monetary_Group')))
    df_rfm = df_rfm.filter(F.col('TotalScore') == 'AAA')

    output_path = '/home/prophet/conda/yakovlev/1t_data/result_task_4_7.csv'
    #df_rfm.write.option("header",True).csv(output_path)
    df_rfm.toPandas().to_csv(output_path, index = False)

    print('Количество строк в файле ', number_rows)
    print('Количество уникальных клиентов ', unique_customers)
    print(f'В {most_purchased_country} совершается больше всего покупок')
    print('Даты самой ранней покупки на платформе ', min_date)
    print('Даты последней покупки на платформе ', max_date)
    print('Количество строк в файле, где CustomerID is not null ', number_rows_whisout_null_custid)

    print('################## END ##############################')

if __name__ == '__main__':
    main()