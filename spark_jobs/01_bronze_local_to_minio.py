from pyspark.sql import SparkSession

# =========================================================
# RUTAS
# =========================================================
LOCAL_BRONZE_PATH = "/opt/spark/work-dir/data/bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026.parquet"
MINIO_BRONZE_OUTPUT = (
    "s3a://btc-bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026"
)

# =========================================================
# SESION SPARK
# =========================================================
spark = (
    SparkSession.builder.appName("bronze-local-to-minio")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "admin")
    .config("spark.hadoop.fs.s3a.secret.key", "admin123")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Leyendo Bronze local...")
df = spark.read.parquet(LOCAL_BRONZE_PATH)

print("Schema del Bronze:")
df.printSchema()

print("Primeras filas:")
df.show(5, truncate=False)

print("Total de filas:")
print(df.count())

print("Escribiendo Bronze en MinIO...")
(df.write.mode("overwrite").parquet(MINIO_BRONZE_OUTPUT))

print("Verificando lectura desde MinIO...")
df_minio = spark.read.parquet(MINIO_BRONZE_OUTPUT)

print("Total de filas leídas desde MinIO:")
print(df_minio.count())

print("Primeras filas desde MinIO:")
df_minio.show(5, truncate=False)

print("OK: Bronze local fue copiado a MinIO correctamente.")

spark.stop()
