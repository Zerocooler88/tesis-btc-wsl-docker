from __future__ import (
    annotations,
)  # Permite usar anotaciones de tipos modernas de forma más flexible

import re  # Sirve para buscar patrones dentro del nombre de archivo
from pathlib import Path  # Sirve para manejar rutas y carpetas de forma ordenada

import polars as pl  # Librería rápida para leer CSV, transformar datos y guardar Parquet


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
SIMBOLO = "BTCUSDT"  # Símbolo del mercado que estamos trabajando
INTERVALO = "1m"  # Intervalo de velas: 1 minuto

CARPETA_CSV = Path(
    "data/raw_csv/spot/BTCUSDT/1m"
)  # Carpeta donde están los CSV extraídos desde los ZIP
CARPETA_BRONZE = Path(
    "data/bronze/spot/BTCUSDT/1m"
)  # Carpeta donde se guardará la capa Bronze en Parquet
CARPETA_BRONZE.mkdir(
    parents=True, exist_ok=True
)  # Crea la carpeta Bronze si todavía no existe

ARCHIVO_PARQUET_BRONZE = (
    CARPETA_BRONZE / "btcusdt_spot_1m_bronze_2019_2026.parquet"
)  # Nombre del archivo Parquet final

COLUMNAS_BINANCE = [  # Nombres oficiales de las 12 columnas del CSV de Binance
    "open_time",  # Timestamp de apertura de la vela
    "open",  # Precio de apertura
    "high",  # Precio máximo
    "low",  # Precio mínimo
    "close",  # Precio de cierre
    "volume",  # Volumen negociado
    "close_time",  # Timestamp de cierre de la vela
    "quote_asset_volume",  # Volumen en el activo cotizado
    "number_of_trades",  # Número de trades
    "taker_buy_base_volume",  # Volumen comprador taker en activo base
    "taker_buy_quote_volume",  # Volumen comprador taker en activo cotizado
    "ignore",  # Columna final que Binance incluye pero que normalmente no se usa
]

COLUMNAS_DECIMALES = [  # Columnas que deben convertirse a números decimales
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

COLUMNAS_ENTERAS = [  # Columnas que deben convertirse a números enteros
    "open_time",
    "close_time",
    "number_of_trades",
]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def obtener_archivos_csv(
    carpeta_csv: Path,
) -> list[Path]:  # Busca todos los archivos CSV dentro de la carpeta indicada
    archivos = sorted(
        carpeta_csv.glob("*.csv")
    )  # Busca archivos .csv y los ordena alfabéticamente
    return archivos  # Devuelve la lista de archivos encontrados


def extraer_periodo_desde_nombre(
    nombre_archivo: str,
) -> tuple[
    str | None, int | None, int | None
]:  # Extrae periodo, año y mes desde el nombre del archivo
    patron = r"BTCUSDT-1m-(\d{4})-(\d{2})\.csv"  # Patrón esperado, por ejemplo BTCUSDT-1m-2019-01.csv
    coincidencia = re.search(
        patron, nombre_archivo
    )  # Busca el patrón en el nombre del archivo

    if coincidencia is None:  # Si no encontró el patrón esperado
        return None, None, None  # Devuelve valores vacíos

    anio = int(coincidencia.group(1))  # Extrae el año del nombre del archivo
    mes = int(coincidencia.group(2))  # Extrae el mes del nombre del archivo
    periodo = f"{anio}-{mes:02d}"  # Construye un texto tipo 2019-01

    return periodo, anio, mes  # Devuelve periodo, año y mes


def leer_un_csv_binance(
    ruta_csv: Path,
) -> pl.DataFrame:  # Lee un CSV individual de Binance y devuelve un DataFrame de Polars
    print(f"Leyendo archivo: {ruta_csv.name}")  # Muestra qué archivo se está leyendo

    periodo_archivo, anio_archivo, mes_archivo = extraer_periodo_desde_nombre(
        ruta_csv.name
    )  # Extrae metadata desde el nombre del archivo

    df = pl.read_csv(  # Lee el archivo CSV con Polars
        source=ruta_csv,  # Ruta del archivo CSV
        has_header=False,  # Indica que el CSV no trae cabecera
        new_columns=COLUMNAS_BINANCE,  # Asigna nombres correctos a las columnas
        infer_schema_length=1000,  # Usa las primeras 1000 filas para inferir tipos iniciales
        ignore_errors=True,  # Si encuentra líneas problemáticas, las ignora para no romper el proceso
    )

    df = df.with_columns(  # Convierte columnas a tipos adecuados y agrega metadata del archivo
        [
            pl.col("open_time").cast(
                pl.Int64, strict=False
            ),  # Convierte open_time a entero de 64 bits
            pl.col("close_time").cast(
                pl.Int64, strict=False
            ),  # Convierte close_time a entero de 64 bits
            pl.col("number_of_trades").cast(
                pl.Int64, strict=False
            ),  # Convierte number_of_trades a entero
            pl.col("open").cast(pl.Float64, strict=False),  # Convierte open a decimal
            pl.col("high").cast(pl.Float64, strict=False),  # Convierte high a decimal
            pl.col("low").cast(pl.Float64, strict=False),  # Convierte low a decimal
            pl.col("close").cast(pl.Float64, strict=False),  # Convierte close a decimal
            pl.col("volume").cast(
                pl.Float64, strict=False
            ),  # Convierte volume a decimal
            pl.col("quote_asset_volume").cast(
                pl.Float64, strict=False
            ),  # Convierte quote_asset_volume a decimal
            pl.col("taker_buy_base_volume").cast(
                pl.Float64, strict=False
            ),  # Convierte taker_buy_base_volume a decimal
            pl.col("taker_buy_quote_volume").cast(
                pl.Float64, strict=False
            ),  # Convierte taker_buy_quote_volume a decimal
            pl.col("ignore").cast(
                pl.Float64, strict=False
            ),  # Convierte ignore a decimal por uniformidad
            pl.lit(ruta_csv.name).alias(
                "archivo_origen"
            ),  # Agrega una columna con el nombre del archivo origen
            pl.lit(periodo_archivo).alias(
                "periodo_archivo"
            ),  # Agrega una columna con el periodo YYYY-MM del archivo
            pl.lit(anio_archivo)
            .cast(pl.Int32, strict=False)
            .alias("anio_archivo"),  # Agrega el año del archivo
            pl.lit(mes_archivo)
            .cast(pl.Int32, strict=False)
            .alias("mes_archivo"),  # Agrega el mes del archivo
        ]
    )

    df = df.filter(  # Elimina filas claramente dañadas en campos esenciales
        pl.col("open_time").is_not_null()  # Exige que open_time exista
        & pl.col("close_time").is_not_null()  # Exige que close_time exista
        & pl.col("open").is_not_null()  # Exige que open exista
        & pl.col("high").is_not_null()  # Exige que high exista
        & pl.col("low").is_not_null()  # Exige que low exista
        & pl.col("close").is_not_null()  # Exige que close exista
    )

    return df  # Devuelve el DataFrame del archivo ya leído y tipado mínimamente


def construir_bronze_desde_csv() -> (
    pl.DataFrame
):  # Lee todos los CSV, los une y construye el dataset Bronze
    archivos_csv = obtener_archivos_csv(
        CARPETA_CSV
    )  # Busca todos los CSV en la carpeta raw_csv

    if not archivos_csv:  # Si no encontró archivos
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en la carpeta: {CARPETA_CSV}"
        )  # Lanza un error claro

    print("=" * 70)  # Imprime separador visual
    print("INICIO DE CONSTRUCCION DE LA CAPA BRONZE")  # Muestra el título del proceso
    print(f"Carpeta origen CSV: {CARPETA_CSV}")  # Muestra la carpeta origen
    print(
        f"Cantidad de archivos CSV encontrados: {len(archivos_csv)}"
    )  # Muestra cuántos archivos encontró
    print("=" * 70)  # Imprime separador visual

    partes = []  # Lista donde se guardarán los DataFrames de cada archivo
    total_filas = 0  # Contador total de filas acumuladas

    for ruta_csv in archivos_csv:  # Recorre archivo por archivo
        df_archivo = leer_un_csv_binance(ruta_csv)  # Lee un archivo CSV individual
        filas_archivo = (
            df_archivo.height
        )  # Cuenta cuántas filas tiene el archivo ya leído
        total_filas += filas_archivo  # Suma esas filas al total acumulado

        print(
            f"  -> Filas válidas en {ruta_csv.name}: {filas_archivo:,}"
        )  # Muestra las filas válidas del archivo actual
        partes.append(df_archivo)  # Guarda el DataFrame en la lista

    print("-" * 70)  # Imprime separador visual
    print(
        f"Total de filas acumuladas antes de unir: {total_filas:,}"
    )  # Muestra el total acumulado antes de concatenar
    print(
        "Uniendo todos los archivos en un solo DataFrame..."
    )  # Informa que ahora va a unir todo

    df_bronze = pl.concat(
        partes, how="vertical_relaxed"
    )  # Une todos los DataFrames en uno solo
    df_bronze = df_bronze.sort(
        ["open_time", "archivo_origen"]
    )  # Ordena por tiempo de apertura y por nombre de archivo

    print(
        f"Total de filas final en Bronze: {df_bronze.height:,}"
    )  # Muestra el total final de filas en Bronze
    print(
        f"Cantidad de columnas en Bronze: {df_bronze.width}"
    )  # Muestra cuántas columnas tiene el DataFrame final
    print(
        f"Primer open_time bruto: {df_bronze['open_time'].min()}"
    )  # Muestra el timestamp mínimo bruto
    print(
        f"Ultimo open_time bruto: {df_bronze['open_time'].max()}"
    )  # Muestra el timestamp máximo bruto

    if (
        df_bronze.height > 3_000_000
    ):  # Verifica si el dataset supera los 3 millones de registros
        print("SI cumple con mas de 3 millones de filas.")  # Muestra confirmación
    else:  # Si no llega a 3 millones
        print("NO cumple con mas de 3 millones de filas.")  # Muestra advertencia

    return df_bronze  # Devuelve el DataFrame Bronze completo


def guardar_bronze_parquet(
    df_bronze: pl.DataFrame,
) -> None:  # Guarda el DataFrame Bronze en formato Parquet
    print("-" * 70)  # Imprime separador visual
    print("Guardando archivo Parquet Bronze...")  # Muestra mensaje de guardado

    df_bronze.write_parquet(  # Escribe el DataFrame en formato Parquet
        ARCHIVO_PARQUET_BRONZE,  # Ruta de salida del archivo Parquet
        compression="zstd",  # Usa compresión zstd para ahorrar espacio sin perder datos
    )

    tamano_mb = ARCHIVO_PARQUET_BRONZE.stat().st_size / (
        1024 * 1024
    )  # Calcula el tamaño del archivo en MB

    print(
        f"Archivo Bronze guardado en: {ARCHIVO_PARQUET_BRONZE}"
    )  # Muestra la ruta del archivo guardado
    print(
        f"Tamaño aproximado del Parquet: {tamano_mb:,.2f} MB"
    )  # Muestra el tamaño del archivo generado


# =========================================================
# FUNCION PRINCIPAL
# =========================================================
def main() -> None:  # Función principal que controla todo el flujo del script
    df_bronze = (
        construir_bronze_desde_csv()
    )  # Construye la capa Bronze leyendo y uniendo todos los CSV
    guardar_bronze_parquet(df_bronze)  # Guarda el resultado en formato Parquet

    print("=" * 70)  # Imprime separador visual final
    print("PROCESO BRONZE TERMINADO")  # Muestra mensaje de fin del proceso
    print(
        f"Total filas Bronze: {df_bronze.height:,}"
    )  # Muestra el total final de filas
    print(
        "Siguiente paso: construir la capa Silver"
    )  # Informa cuál es el siguiente paso lógico
    print("=" * 70)  # Imprime separador visual final


if __name__ == "__main__":  # Verifica que este archivo se ejecuta directamente
    main()  # Llama a la función principal para iniciar el proceso
