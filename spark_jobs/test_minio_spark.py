from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("test-minio-spark")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "admin")
    .config("spark.hadoop.fs.s3a.secret.key", "admin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .getOrCreate()
)

data = [
    ("2026-03-13 00:00:00", 83000.0),
    ("2026-03-13 01:00:00", 83250.5),
    ("2026-03-13 02:00:00", 83110.2),
]

df = spark.createDataFrame(data, ["open_time", "close"])

df.write.mode("overwrite").parquet("s3a://btc-bronze/test_spark_minio")

df_leido = spark.read.parquet("s3a://btc-bronze/test_spark_minio")
df_leido.show()

print("OK: Spark escribió y leyó datos en MinIO")

spark.stop()
