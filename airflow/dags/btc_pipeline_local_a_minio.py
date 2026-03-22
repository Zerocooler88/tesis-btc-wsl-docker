from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="btc_pipeline_local_a_minio",
    start_date=datetime(2026, 3, 13),
    schedule=None,
    catchup=False,
    tags=["btc", "spark", "minio", "medallion"],
) as dag:

    bronze_task = BashOperator(
        task_id="bronze_local_a_minio",
        bash_command="""
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026.parquet \
        --s3-path s3a://btc-bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026 \
        --app-name airflow_bronze_local_a_minio
        """
    )

    silver_task = BashOperator(
        task_id="silver_local_a_minio",
        bash_command="""
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/silver/spot/BTCUSDT/1m/btcusdt_spot_1m_silver_2019_2026.parquet \
        --s3-path s3a://btc-silver/spot/BTCUSDT/1m/btcusdt_spot_1m_silver_2019_2026 \
        --app-name airflow_silver_local_a_minio
        """
    )

    gold_task = BashOperator(
        task_id="gold_local_a_minio",
        bash_command="""
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026.parquet \
        --s3-path s3a://btc-gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026 \
        --app-name airflow_gold_local_a_minio
        """
    )

    bronze_task >> silver_task >> gold_task