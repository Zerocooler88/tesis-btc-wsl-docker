from pyspark.sql import SparkSession
import argparse


def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("spark://spark-master:7077")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )
    return spark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", required=True, help="Ruta parquet local dentro del contenedor Spark")
    parser.add_argument("--s3-path", required=True, help="Ruta destino s3a://bucket/ruta")
    parser.add_argument("--app-name", default="subir_parquet_local_a_minio")
    args = parser.parse_args()

    spark = build_spark(args.app_name)

    print(f"\nLeyendo local: {args.local_path}")
    df = spark.read.parquet(args.local_path)

    total = df.count()
    print(f"Filas leídas: {total}")
    print("Schema:")
    df.printSchema()

    print(f"\nEscribiendo en MinIO: {args.s3_path}")
    df.write.mode("overwrite").parquet(args.s3_path)

    print("\nProceso terminado correctamente.")
    spark.stop()


if __name__ == "__main__":
    main()
