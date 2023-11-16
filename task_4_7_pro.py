from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import rand, randn, col
from pyspark.sql.types import *
import random
import numpy as np
import pandas as pd
import os

def main(num_rows: int, num_columns: int, data_types: list, numeric_columns: dict,
         string_columns: dict, batch_size: int):

    spark = SparkSession.builder \
        .appName("task 4.7 pro") \
        .enableHiveSupport() \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .getOrCreate()

    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    # Создание схемы данных
    columns = [f'col_{i}' for i in range(num_columns)]

    # Генерация данных внутри Spark DataFrame
    default_row = Row(*([0.0] * num_columns))  # Assuming default values are 0.0 for numeric columns
    df = spark.createDataFrame([default_row]).toDF(*columns)

    for i in range(num_rows):
        row_data = []
        for col_idx in range(num_columns):
            data_type = data_types[col_idx % len(data_types)]
            if data_type == 'double' or data_type == 'int':
                mean = numeric_columns[data_type]['mean']
                stddev = numeric_columns[data_type]['stddev']
                value = float(random.gauss(mean, stddev))
                if data_type == 'int':
                    value = int(value)
            elif data_type == 'string':
                values = string_columns[data_type]['values']
                probabilities = string_columns[data_type]['probabilities']
                value = str(np.random.choice(values, p=probabilities))
            row_data.append(value)

        # Добавление строки к DataFrame
        current_row = Row(*row_data)
        df = df.union(spark.createDataFrame([current_row]))

        if (i + 1) % batch_size == 0:
            logger.info(f"Generated {i} rows")
            # Запись текущего батча в DataFrame
            pass


    # Remove the default row before displaying final results
    #df = df.filter(col(columns[0]) != 0.0)
    output_path = "/user/hadoop/synthetic_data.parquet"
    df.write.mode("overwrite").parquet(output_path)
    # Отображение информации о сгенерированных данных
    df.show()
    df.printSchema()

    print('################## END ##############################')


if __name__ == '__main__':
    # Параметры генерации
    num_rows = 10000000  # Общее количество строк
    num_columns = 5  # Количество столбцов
    data_types = ['string', 'double', 'int']  # Типы данных
    # Настройки для числовых столбцов
    numeric_columns = {
        'double': {'mean': 0, 'stddev': 1},
        'int': {'mean': 50, 'stddev': 10}
    }
    # Настройки для строковых столбцов
    string_columns = {
        'string': {'values': ['A', 'B', 'C'], 'probabilities': [0.4, 0.4, 0.2]}
    }
    batch_size = 100000

    main(num_rows, num_columns, data_types, numeric_columns, string_columns, batch_size)
