from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="btc_pipeline_full_ingesta_a_minio",
    start_date=datetime(2026, 3, 14),
    schedule=None,
    catchup=False,
    tags=["btc", "binance", "spark", "minio", "medallion"],
) as dag:

    ingest_binance_historical = BashOperator(
        task_id="ingest_binance_historical",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO INGESTA HISTORICA BINANCE =====' &&
        python /opt/airflow/src/01_download_binance_btcusdt_1m_monthly.py &&
        echo '===== FIN INGESTA HISTORICA BINANCE ====='
        """
    )

    build_bronze = BashOperator(
        task_id="build_bronze",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO BUILD BRONZE =====' &&
        python /opt/airflow/src/02_construir_bronze_parquet.py &&
        echo '===== FIN BUILD BRONZE ====='
        """
    )

    bronze_to_minio = BashOperator(
        task_id="bronze_to_minio",
        bash_command="""
        echo '===== INICIO BRONZE TO MINIO =====' &&
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026.parquet \
        --s3-path s3a://btc-bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026 \
        --app-name airflow_bronze_local_a_minio &&
        echo '===== FIN BRONZE TO MINIO ====='
        """
    )

    build_silver = BashOperator(
        task_id="build_silver",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO BUILD SILVER =====' &&
        python /opt/airflow/src/03_construir_silver_1m_limpio.py &&
        echo '===== FIN BUILD SILVER ====='
        """
    )

    silver_to_minio = BashOperator(
        task_id="silver_to_minio",
        bash_command="""
        echo '===== INICIO SILVER TO MINIO =====' &&
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/silver/spot/BTCUSDT/1m/btcusdt_spot_1m_silver_2019_2026.parquet \
        --s3-path s3a://btc-silver/spot/BTCUSDT/1m/btcusdt_spot_1m_silver_2019_2026 \
        --app-name airflow_silver_local_a_minio &&
        echo '===== FIN SILVER TO MINIO ====='
        """
    )

    build_gold = BashOperator(
        task_id="build_gold",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO BUILD GOLD =====' &&
        python /opt/airflow/src/04_construir_gold_1h.py &&
        echo '===== FIN BUILD GOLD ====='
        """
    )

    gold_to_minio = BashOperator(
        task_id="gold_to_minio",
        bash_command="""
        echo '===== INICIO GOLD TO MINIO =====' &&
        docker exec spark-master /opt/spark/bin/spark-submit \
        /opt/spark/work-dir/spark_jobs/06_subir_parquet_local_a_minio.py \
        --local-path /opt/spark/work-dir/data/gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026.parquet \
        --s3-path s3a://btc-gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026 \
        --app-name airflow_gold_local_a_minio &&
        echo '===== FIN GOLD TO MINIO ====='
        """
    )

    prepare_model_input = BashOperator(
        task_id="prepare_model_input",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO PREPARE MODEL INPUT =====' &&
        python /opt/airflow/src/07_preparar_model_input_gold_1h.py &&
        echo '===== FIN PREPARE MODEL INPUT ====='
        """
    )

    generate_windows_trimestrales = BashOperator(
        task_id="generate_windows_trimestrales",
        bash_command="""
        cd /opt/airflow &&
        echo '===== INICIO GENERATE WINDOWS TRIMESTRALES =====' &&
        python /opt/airflow/src/08_generar_windows_trimestrales_1h.py &&
        echo '===== FIN GENERATE WINDOWS TRIMESTRALES ====='
        """
    )

    ingest_binance_historical >> build_bronze >> bronze_to_minio >> build_silver >> silver_to_minio >> build_gold >> gold_to_minio >> prepare_model_input >> generate_windows_trimestrales