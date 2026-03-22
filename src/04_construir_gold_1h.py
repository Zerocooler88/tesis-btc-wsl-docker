from __future__ import annotations  # Permite usar anotaciones de tipos modernas

from pathlib import Path  # Permite manejar rutas y carpetas fácilmente
from datetime import datetime  # Permite registrar fecha y hora en el resumen

import polars as pl  # Librería principal para leer, transformar y guardar datos


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
SIMBOLO = "BTCUSDT"  # Símbolo del mercado
INTERVALO_ORIGEN = "1m"  # Intervalo del dataset de entrada
INTERVALO_GOLD = "1h"  # Intervalo final del dataset Gold

RUTA_SILVER = Path(
    "data/silver/spot/BTCUSDT/1m/btcusdt_spot_1m_silver_2019_2026.parquet"
)  # Ruta del archivo Silver 1m

CARPETA_GOLD = Path("data/gold/spot/BTCUSDT/1h")  # Carpeta donde se guardará Gold 1h
CARPETA_GOLD.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

CARPETA_LOGS = Path("logs")  # Carpeta de logs
CARPETA_LOGS.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

RUTA_GOLD = (
    CARPETA_GOLD / "btcusdt_spot_1h_gold_2019_2026.parquet"
)  # Ruta del archivo Gold final

RUTA_RESUMEN = (
    CARPETA_LOGS / "resumen_gold_1h_2019_2026.txt"
)  # Ruta del resumen de auditoría

FILAS_ESPERADAS_POR_HORA = 60  # Una hora completa debe tener 60 velas de 1 minuto


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def cargar_silver() -> pl.DataFrame:  # Lee el archivo Silver
    if not RUTA_SILVER.exists():  # Si el archivo Silver no existe
        raise FileNotFoundError(
            f"No existe el archivo Silver: {RUTA_SILVER}"
        )  # Lanza un error claro

    print("=" * 70)  # Imprime separador visual
    print("INICIO DE CONSTRUCCION DE LA CAPA GOLD 1H")  # Muestra el inicio del proceso
    print(f"Leyendo Silver desde: {RUTA_SILVER}")  # Muestra la ruta del archivo Silver

    df = pl.read_parquet(RUTA_SILVER)  # Lee el archivo Parquet Silver
    print(
        f"Filas leídas desde Silver: {df.height:,}"
    )  # Muestra cuántas filas se cargaron
    print(
        f"Columnas leídas desde Silver: {df.width}"
    )  # Muestra cuántas columnas se cargaron

    return df  # Devuelve el DataFrame Silver leído


def ordenar_silver(df: pl.DataFrame) -> pl.DataFrame:  # Ordena el Silver por tiempo
    df = df.sort("open_datetime_utc")  # Ordena por fecha/hora de apertura
    return df  # Devuelve el DataFrame ordenado


def agregar_de_1m_a_1h(df: pl.DataFrame) -> pl.DataFrame:
    """
    Agrupa velas de 1 minuto en velas de 1 hora.

    Reglas OHLC por hora:
    - open  = primer open de la hora
    - high  = máximo de la hora
    - low   = mínimo de la hora
    - close = último close de la hora
    - volume = suma del volumen
    """
    df_1h = (
        df.group_by_dynamic(
            index_column="open_datetime_utc",  # Columna temporal sobre la que se agrupa
            every="1h",  # Cada grupo será de 1 hora
            period="1h",  # El periodo del grupo es de 1 hora
            closed="left",  # La ventana incluye el inicio y excluye el final
            label="left",  # La etiqueta de la hora será el inicio de la ventana
            start_by="window",  # Las ventanas empiezan alineadas por hora
        )
        .agg(
            [
                pl.first("open").alias("open"),  # Primer open de la hora
                pl.max("high").alias("high"),  # Máximo high de la hora
                pl.min("low").alias("low"),  # Mínimo low de la hora
                pl.last("close").alias("close"),  # Último close de la hora
                pl.sum("volume").alias("volume"),  # Suma del volumen
                pl.sum("quote_asset_volume").alias(
                    "quote_asset_volume"
                ),  # Suma del volumen cotizado
                pl.sum("number_of_trades").alias("number_of_trades"),  # Suma de trades
                pl.sum("taker_buy_base_volume").alias(
                    "taker_buy_base_volume"
                ),  # Suma volumen comprador base
                pl.sum("taker_buy_quote_volume").alias(
                    "taker_buy_quote_volume"
                ),  # Suma volumen comprador cotizado
                pl.len().alias(
                    "filas_en_hora"
                ),  # Cuenta cuántas filas de 1m cayeron en esa hora
            ]
        )
        .sort("open_datetime_utc")  # Ordena por hora
    )

    return df_1h  # Devuelve el DataFrame agregado a 1 hora


def marcar_y_filtrar_horas_completas(
    df_1h: pl.DataFrame,
) -> tuple[pl.DataFrame, int, int]:
    df_1h = df_1h.with_columns(
        [
            (pl.col("filas_en_hora") == FILAS_ESPERADAS_POR_HORA).alias(
                "hora_completa"
            ),  # Marca si la hora tiene exactamente 60 filas
        ]
    )

    horas_totales = df_1h.height  # Cuenta cuántas horas agregadas existen en total
    horas_completas = df_1h.filter(
        pl.col("hora_completa") == True
    ).height  # Cuenta cuántas horas completas hay
    horas_incompletas = (
        horas_totales - horas_completas
    )  # Calcula cuántas horas quedaron incompletas

    df_1h = df_1h.filter(
        pl.col("hora_completa") == True
    )  # Conserva solo las horas completas
    df_1h = df_1h.sort("open_datetime_utc")  # Reordena por seguridad

    return (
        df_1h,
        horas_totales,
        horas_incompletas,
    )  # Devuelve Gold limpio y métricas de completitud


def crear_variables_derivadas(df_1h: pl.DataFrame) -> pl.DataFrame:
    """
    Crea variables útiles para análisis y modelado.

    Importante:
    En Polars no conviene usar una columna recién creada dentro del mismo
    with_columns, por eso aquí lo hacemos en dos pasos.
    """

    # ---------------------------------------------------------
    # PASO 1: crear columnas base derivadas
    # ---------------------------------------------------------
    df_1h = df_1h.with_columns(
        [
            (pl.col("high") - pl.col("low")).alias(
                "rango_hl"
            ),  # Rango high-low de la hora
            (pl.col("close") - pl.col("open")).alias(
                "cuerpo_oc"
            ),  # Diferencia close-open
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias(
                "return_1h"
            ),  # Retorno porcentual simple
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .alias("log_return_1h"),  # Retorno logarítmico
        ]
    )

    # ---------------------------------------------------------
    # PASO 2: crear columnas que dependen de las anteriores
    # ---------------------------------------------------------
    df_1h = df_1h.with_columns(
        [
            pl.col("close")
            .rolling_mean(window_size=24)
            .alias("sma_24h"),  # Media móvil simple 24 horas
            pl.col("close")
            .rolling_mean(window_size=168)
            .alias("sma_168h"),  # Media móvil simple 168 horas (7 días)
            pl.col("log_return_1h")
            .rolling_std(window_size=24)
            .alias("volatilidad_24h"),  # Volatilidad 24 horas
            pl.col("volume")
            .rolling_mean(window_size=24)
            .alias("volumen_sma_24h"),  # Media móvil de volumen 24 horas
        ]
    )

    return df_1h  # Devuelve el DataFrame con variables derivadas


def etiquetar_regimen_mercado(df_1h: pl.DataFrame) -> pl.DataFrame:
    """
    Etiqueta cada fila por régimen temporal.
    Criterio usado:
    - pre_covid:   antes de 2020-03-01
    - covid:       desde 2020-03-01 hasta 2023-05-31 23:59:59
    - post_covid:  desde 2023-06-01 en adelante
    """
    fecha_inicio_covid = datetime(2020, 3, 1, 0, 0, 0)  # Inicio del periodo COVID
    fecha_fin_covid = datetime(2023, 5, 31, 23, 59, 59)  # Fin del periodo COVID

    df_1h = df_1h.with_columns(
        [
            pl.when(
                pl.col("open_datetime_utc") < pl.lit(fecha_inicio_covid)
            )  # Si está antes del COVID
            .then(pl.lit("pre_covid"))  # Etiqueta pre_covid
            .when(
                (pl.col("open_datetime_utc") >= pl.lit(fecha_inicio_covid))
                & (pl.col("open_datetime_utc") <= pl.lit(fecha_fin_covid))
            )  # Si está dentro del periodo COVID
            .then(pl.lit("covid"))  # Etiqueta covid
            .otherwise(pl.lit("post_covid"))  # Si no, es post_covid
            .alias("regimen_mercado"),  # Guarda la etiqueta final
        ]
    )

    return df_1h  # Devuelve el DataFrame con la etiqueta de régimen


def agregar_columnas_calendario(df_1h: pl.DataFrame) -> pl.DataFrame:
    df_1h = df_1h.with_columns(
        [
            pl.col("open_datetime_utc").dt.year().alias("anio"),  # Extrae año
            pl.col("open_datetime_utc").dt.month().alias("mes"),  # Extrae mes
            pl.col("open_datetime_utc").dt.day().alias("dia"),  # Extrae día
            pl.col("open_datetime_utc").dt.hour().alias("hora"),  # Extrae hora
        ]
    )

    return df_1h  # Devuelve el DataFrame con columnas calendario


def contar_regimenes(df_1h: pl.DataFrame) -> dict[str, int]:
    conteo = (
        df_1h.group_by("regimen_mercado")  # Agrupa por régimen
        .agg(pl.len().alias("filas"))  # Cuenta filas por régimen
        .sort("regimen_mercado")  # Ordena por nombre de régimen
    )

    resultado = {}  # Diccionario donde se guardará el conteo
    for fila in conteo.iter_rows(named=True):  # Recorre cada fila del conteo
        resultado[fila["regimen_mercado"]] = fila["filas"]  # Guarda régimen -> cantidad

    return resultado  # Devuelve el diccionario final


def guardar_gold(df_1h: pl.DataFrame) -> None:
    print("-" * 70)  # Imprime separador visual
    print("Guardando archivo Parquet Gold 1h...")  # Muestra mensaje de guardado

    df_1h.write_parquet(
        RUTA_GOLD,  # Ruta final del archivo Gold
        compression="zstd",  # Usa compresión zstd
    )

    tamano_mb = RUTA_GOLD.stat().st_size / (
        1024 * 1024
    )  # Calcula el tamaño del archivo en MB

    print(
        f"Archivo Gold guardado en: {RUTA_GOLD}"
    )  # Muestra la ruta del archivo generado
    print(
        f"Tamaño aproximado del Parquet Gold: {tamano_mb:,.2f} MB"
    )  # Muestra el tamaño del archivo


def guardar_resumen(
    filas_silver_iniciales: int,
    horas_totales_agregadas: int,
    horas_incompletas: int,
    filas_gold_finales: int,
    min_fecha: str,
    max_fecha: str,
    conteo_regimenes: dict[str, int],
) -> None:
    contenido = []  # Lista donde se construirá el texto del resumen

    contenido.append("RESUMEN PROCESO GOLD BTCUSDT 1H")  # Título del resumen
    contenido.append("=" * 60)  # Separador visual
    contenido.append(
        f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}"
    )  # Fecha del proceso
    contenido.append(
        f"Filas iniciales desde Silver 1m: {filas_silver_iniciales:,}"
    )  # Filas Silver 1m
    contenido.append(
        f"Horas totales agregadas: {horas_totales_agregadas:,}"
    )  # Total de horas agregadas
    contenido.append(
        f"Horas incompletas eliminadas: {horas_incompletas:,}"
    )  # Horas incompletas eliminadas
    contenido.append(
        f"Filas finales Gold 1h: {filas_gold_finales:,}"
    )  # Filas finales Gold
    contenido.append(f"Fecha mínima Gold: {min_fecha}")  # Fecha mínima
    contenido.append(f"Fecha máxima Gold: {max_fecha}")  # Fecha máxima
    contenido.append(
        "Conteo por régimen de mercado:"
    )  # Título de la parte de regímenes
    for regimen, filas in conteo_regimenes.items():  # Recorre conteos por régimen
        contenido.append(f"  - {regimen}: {filas:,}")  # Escribe cada conteo
    contenido.append("=" * 60)  # Separador final

    RUTA_RESUMEN.write_text(
        "\n".join(contenido), encoding="utf-8"
    )  # Guarda el resumen en archivo txt


# =========================================================
# FUNCION PRINCIPAL
# =========================================================
def main() -> None:
    df = cargar_silver()  # Lee el archivo Silver
    filas_silver_iniciales = df.height  # Guarda cuántas filas tenía Silver

    df = ordenar_silver(df)  # Ordena Silver por tiempo
    df_1h = agregar_de_1m_a_1h(df)  # Agrega de 1m a 1h
    df_1h, horas_totales_agregadas, horas_incompletas = (
        marcar_y_filtrar_horas_completas(df_1h)
    )  # Marca y filtra horas completas
    df_1h = crear_variables_derivadas(df_1h)  # Crea variables derivadas útiles
    df_1h = etiquetar_regimen_mercado(df_1h)  # Etiqueta pre_covid, covid y post_covid
    df_1h = agregar_columnas_calendario(df_1h)  # Agrega columnas calendario

    min_fecha = str(df_1h["open_datetime_utc"].min())  # Obtiene la fecha mínima
    max_fecha = str(df_1h["open_datetime_utc"].max())  # Obtiene la fecha máxima
    conteo_regimenes = contar_regimenes(df_1h)  # Cuenta filas por régimen

    print("-" * 70)  # Imprime separador visual
    print(
        f"Filas iniciales desde Silver 1m: {filas_silver_iniciales:,}"
    )  # Muestra filas iniciales
    print(
        f"Horas totales agregadas: {horas_totales_agregadas:,}"
    )  # Muestra horas agregadas
    print(
        f"Horas incompletas eliminadas: {horas_incompletas:,}"
    )  # Muestra horas incompletas eliminadas
    print(f"Filas finales Gold 1h: {df_1h.height:,}")  # Muestra filas finales Gold
    print(f"Fecha mínima Gold: {min_fecha}")  # Muestra fecha mínima
    print(f"Fecha máxima Gold: {max_fecha}")  # Muestra fecha máxima
    print("Conteo por régimen de mercado:")  # Título de regímenes
    for regimen, filas in conteo_regimenes.items():  # Recorre regímenes
        print(f"  - {regimen}: {filas:,}")  # Muestra conteo por régimen

    guardar_gold(df_1h)  # Guarda el archivo Gold
    guardar_resumen(
        filas_silver_iniciales=filas_silver_iniciales,
        horas_totales_agregadas=horas_totales_agregadas,
        horas_incompletas=horas_incompletas,
        filas_gold_finales=df_1h.height,
        min_fecha=min_fecha,
        max_fecha=max_fecha,
        conteo_regimenes=conteo_regimenes,
    )  # Guarda el resumen

    print("=" * 70)  # Imprime separador visual final
    print("PROCESO GOLD 1H TERMINADO")  # Muestra el fin del proceso
    print(f"Archivo Gold: {RUTA_GOLD}")  # Muestra la ruta del archivo Gold
    print(f"Resumen Gold: {RUTA_RESUMEN}")  # Muestra la ruta del resumen
    print(
        "Siguiente paso: análisis del close y preparación del dataset DL"
    )  # Indica el siguiente paso lógico
    print("=" * 70)  # Imprime separador visual final


if __name__ == "__main__":
    main()  # Ejecuta el proceso principal
