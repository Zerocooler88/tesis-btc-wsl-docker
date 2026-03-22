from __future__ import annotations  # Permite usar anotaciones de tipos modernas

from pathlib import Path  # Permite manejar rutas y carpetas fácilmente
from datetime import datetime  # Permite registrar fecha y hora en el resumen

import polars as pl  # Librería principal para leer, limpiar y guardar datos


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
SIMBOLO = "BTCUSDT"  # Símbolo del mercado
INTERVALO = "1m"  # Intervalo original de los datos

RUTA_BRONZE = Path(
    "data/bronze/spot/BTCUSDT/1m/btcusdt_spot_1m_bronze_2019_2026.parquet"
)  # Ruta del archivo Bronze

CARPETA_SILVER = Path("data/silver/spot/BTCUSDT/1m")  # Carpeta donde se guardará Silver
CARPETA_SILVER.mkdir(parents=True, exist_ok=True)  # Crea la carpeta Silver si no existe

CARPETA_LOGS = Path("logs")  # Carpeta de logs
CARPETA_LOGS.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de logs si no existe

RUTA_SILVER = (
    CARPETA_SILVER / "btcusdt_spot_1m_silver_2019_2026.parquet"
)  # Archivo Parquet Silver final

RUTA_RESUMEN = (
    CARPETA_LOGS / "resumen_silver_1m_2019_2026.txt"
)  # Archivo de resumen del proceso Silver

UMBRAL_MICROSEGUNDOS = 100_000_000_000_000  # Umbral para distinguir timestamps en microsegundos de milisegundos


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def cargar_bronze() -> pl.DataFrame:  # Lee el archivo Bronze
    if not RUTA_BRONZE.exists():  # Si el archivo Bronze no existe
        raise FileNotFoundError(
            f"No existe el archivo Bronze: {RUTA_BRONZE}"
        )  # Lanza un error claro

    print("=" * 70)  # Imprime separador visual
    print("INICIO DE CONSTRUCCION DE LA CAPA SILVER")  # Muestra el inicio del proceso
    print(f"Leyendo Bronze desde: {RUTA_BRONZE}")  # Muestra la ruta del archivo Bronze

    df = pl.read_parquet(RUTA_BRONZE)  # Lee el archivo Parquet Bronze
    print(
        f"Filas leídas desde Bronze: {df.height:,}"
    )  # Muestra cuántas filas se cargaron
    print(
        f"Columnas leídas desde Bronze: {df.width}"
    )  # Muestra cuántas columnas se cargaron

    return df  # Devuelve el DataFrame Bronze leído


def asegurar_tipos_basicos(
    df: pl.DataFrame,
) -> pl.DataFrame:  # Reafirma tipos por seguridad
    df = df.with_columns(
        [
            pl.col("open_time").cast(
                pl.Int64, strict=False
            ),  # Convierte open_time a entero largo
            pl.col("close_time").cast(
                pl.Int64, strict=False
            ),  # Convierte close_time a entero largo
            pl.col("number_of_trades").cast(
                pl.Int64, strict=False
            ),  # Convierte number_of_trades a entero largo
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
        ]
    )

    return df  # Devuelve el DataFrame con tipos reafirmados


def convertir_timestamps_mixtos(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte timestamps mezclados en milisegundos y microsegundos a datetime.

    Importante:
    - No usamos .replace_time_zone("UTC") para evitar errores de tzdata en Windows.
    - Conceptualmente estos tiempos siguen representando UTC porque vienen así desde Binance.
    - Por eso mantenemos los nombres open_datetime_utc y close_datetime_utc.
    """
    df = df.with_columns(
        [
            pl.when(
                pl.col("open_time") >= UMBRAL_MICROSEGUNDOS
            )  # Si parece microsegundos
            .then(
                pl.from_epoch(pl.col("open_time"), time_unit="us")
            )  # Convierte usando microsegundos
            .otherwise(
                pl.from_epoch(pl.col("open_time"), time_unit="ms")
            )  # Si no, usa milisegundos
            .alias("open_datetime_utc"),  # Guarda la fecha/hora de apertura
            pl.when(
                pl.col("close_time") >= UMBRAL_MICROSEGUNDOS
            )  # Si parece microsegundos
            .then(
                pl.from_epoch(pl.col("close_time"), time_unit="us")
            )  # Convierte usando microsegundos
            .otherwise(
                pl.from_epoch(pl.col("close_time"), time_unit="ms")
            )  # Si no, usa milisegundos
            .alias("close_datetime_utc"),  # Guarda la fecha/hora de cierre
        ]
    )

    return df  # Devuelve el DataFrame con columnas datetime reales


def eliminar_nulos_esenciales(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    filas_antes = df.height  # Guarda cuántas filas había antes de limpiar

    df = df.filter(
        pl.col("open_time").is_not_null()  # open_time no debe ser nulo
        & pl.col("close_time").is_not_null()  # close_time no debe ser nulo
        & pl.col(
            "open_datetime_utc"
        ).is_not_null()  # open_datetime_utc no debe ser nulo
        & pl.col(
            "close_datetime_utc"
        ).is_not_null()  # close_datetime_utc no debe ser nulo
        & pl.col("open").is_not_null()  # open no debe ser nulo
        & pl.col("high").is_not_null()  # high no debe ser nulo
        & pl.col("low").is_not_null()  # low no debe ser nulo
        & pl.col("close").is_not_null()  # close no debe ser nulo
        & pl.col("volume").is_not_null()  # volume no debe ser nulo
    )

    filas_despues = df.height  # Guarda cuántas filas quedaron después de limpiar
    eliminadas = filas_antes - filas_despues  # Calcula cuántas filas se eliminaron

    return df, eliminadas  # Devuelve el DataFrame limpio y el número de eliminadas


def ordenar_por_tiempo(
    df: pl.DataFrame,
) -> pl.DataFrame:  # Ordena el dataset por tiempo real
    df = df.sort(
        ["open_datetime_utc", "archivo_origen"]
    )  # Ordena por fecha de apertura y archivo origen
    return df  # Devuelve el DataFrame ordenado


def eliminar_duplicados_por_minuto(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    filas_antes = df.height  # Guarda las filas antes de quitar duplicados

    df = df.unique(
        subset=["open_datetime_utc"],  # Usa la fecha/hora de apertura como clave única
        keep="first",  # Si hay repetidos, se queda con el primero
        maintain_order=True,  # Mantiene el orden original
    )

    filas_despues = df.height  # Guarda cuántas filas quedaron
    duplicados_eliminados = (
        filas_antes - filas_despues
    )  # Calcula cuántos duplicados se eliminaron

    return (
        df,
        duplicados_eliminados,
    )  # Devuelve el DataFrame limpio y el número de duplicados eliminados


def validar_ohlc(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    condicion_valida = (
        (pl.col("high") >= pl.col("open"))  # high debe ser mayor o igual que open
        & (pl.col("high") >= pl.col("close"))  # high debe ser mayor o igual que close
        & (pl.col("high") >= pl.col("low"))  # high debe ser mayor o igual que low
        & (pl.col("low") <= pl.col("open"))  # low debe ser menor o igual que open
        & (pl.col("low") <= pl.col("close"))  # low debe ser menor o igual que close
        & (pl.col("low") <= pl.col("high"))  # low debe ser menor o igual que high
        & (pl.col("volume") >= 0)  # volume no debe ser negativo
    )

    filas_antes = df.height  # Guarda las filas antes de validar
    df = df.filter(condicion_valida)  # Conserva solo las filas con OHLC válido
    filas_despues = df.height  # Guarda las filas después de validar
    invalidas_ohlc = filas_antes - filas_despues  # Calcula cuántas se eliminaron

    return (
        df,
        invalidas_ohlc,
    )  # Devuelve el DataFrame validado y el número de filas inválidas


def calcular_huecos_temporales(df: pl.DataFrame) -> tuple[int, int | None]:
    df_diferencias = df.select(
        [
            pl.col("open_datetime_utc"),  # Conserva la fecha/hora de apertura
            pl.col("open_datetime_utc")
            .diff()
            .alias("delta_tiempo"),  # Diferencia temporal entre filas consecutivas
        ]
    )

    huecos = df_diferencias.filter(
        pl.col("delta_tiempo").is_not_null()  # Excluye la primera fila
        & (
            pl.col("delta_tiempo") != pl.duration(minutes=1)
        )  # Busca saltos distintos a 1 minuto
    )

    cantidad_huecos = huecos.height  # Cuenta cuántos huecos existen

    if cantidad_huecos > 0:  # Si encontró huecos
        max_segundos = int(
            huecos.select(pl.col("delta_tiempo").dt.total_seconds().max()).item()
        )  # Calcula el mayor hueco encontrado en segundos
    else:
        max_segundos = None  # No hay máximo hueco

    return (
        cantidad_huecos,
        max_segundos,
    )  # Devuelve cantidad de huecos y el máximo hueco


def guardar_silver(df: pl.DataFrame) -> None:  # Guarda el dataset Silver en Parquet
    print("-" * 70)  # Imprime separador visual
    print("Guardando archivo Parquet Silver...")  # Muestra mensaje de guardado

    df.write_parquet(
        RUTA_SILVER,  # Ruta final del archivo Silver
        compression="zstd",  # Usa compresión zstd
    )

    tamano_mb = RUTA_SILVER.stat().st_size / (
        1024 * 1024
    )  # Calcula el tamaño del archivo en MB

    print(
        f"Archivo Silver guardado en: {RUTA_SILVER}"
    )  # Muestra la ruta del archivo generado
    print(
        f"Tamaño aproximado del Parquet Silver: {tamano_mb:,.2f} MB"
    )  # Muestra el tamaño del archivo


def guardar_resumen(
    filas_iniciales: int,
    filas_finales: int,
    nulos_eliminados: int,
    duplicados_eliminados: int,
    invalidas_ohlc: int,
    cantidad_huecos: int,
    max_segundos_hueco: int | None,
    min_fecha: str,
    max_fecha: str,
) -> None:
    contenido = []  # Lista donde se irá construyendo el texto del resumen

    contenido.append("RESUMEN PROCESO SILVER BTCUSDT 1m")  # Título del resumen
    contenido.append("=" * 60)  # Separador visual
    contenido.append(
        f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}"
    )  # Fecha y hora del proceso
    contenido.append(
        f"Filas iniciales desde Bronze: {filas_iniciales:,}"
    )  # Filas iniciales
    contenido.append(
        f"Filas eliminadas por nulos esenciales: {nulos_eliminados:,}"
    )  # Nulos eliminados
    contenido.append(
        f"Duplicados eliminados por minuto: {duplicados_eliminados:,}"
    )  # Duplicados eliminados
    contenido.append(
        f"Filas eliminadas por OHLC inválido: {invalidas_ohlc:,}"
    )  # OHLC inválidos eliminados
    contenido.append(f"Filas finales en Silver: {filas_finales:,}")  # Filas finales
    contenido.append(f"Fecha mínima open_datetime_utc: {min_fecha}")  # Fecha mínima
    contenido.append(f"Fecha máxima open_datetime_utc: {max_fecha}")  # Fecha máxima
    contenido.append(
        f"Cantidad de huecos temporales distintos a 1 minuto: {cantidad_huecos:,}"
    )  # Cantidad de huecos
    contenido.append(
        f"Mayor hueco temporal en segundos: {max_segundos_hueco}"
    )  # Tamaño máximo del hueco
    contenido.append("=" * 60)  # Separador visual final

    RUTA_RESUMEN.write_text(
        "\n".join(contenido), encoding="utf-8"
    )  # Escribe el resumen en un archivo de texto


# =========================================================
# FUNCION PRINCIPAL
# =========================================================
def main() -> None:  # Función principal que controla todo el proceso Silver
    df = cargar_bronze()  # Lee Bronze
    filas_iniciales = df.height  # Guarda cuántas filas tenía Bronze al iniciar

    df = asegurar_tipos_basicos(df)  # Reafirma tipos por seguridad
    df = convertir_timestamps_mixtos(
        df
    )  # Convierte open_time y close_time a fecha/hora real
    df, nulos_eliminados = eliminar_nulos_esenciales(
        df
    )  # Elimina filas con nulos críticos
    df = ordenar_por_tiempo(df)  # Ordena por tiempo real
    df, duplicados_eliminados = eliminar_duplicados_por_minuto(
        df
    )  # Elimina duplicados por minuto
    df, invalidas_ohlc = validar_ohlc(df)  # Elimina velas con OHLC incoherente

    cantidad_huecos, max_segundos_hueco = calcular_huecos_temporales(
        df
    )  # Calcula huecos temporales

    min_fecha = str(
        df["open_datetime_utc"].min()
    )  # Obtiene la fecha mínima del dataset limpio
    max_fecha = str(
        df["open_datetime_utc"].max()
    )  # Obtiene la fecha máxima del dataset limpio

    print("-" * 70)  # Imprime separador visual
    print(
        f"Filas iniciales desde Bronze: {filas_iniciales:,}"
    )  # Muestra filas iniciales
    print(
        f"Nulos esenciales eliminados: {nulos_eliminados:,}"
    )  # Muestra nulos eliminados
    print(
        f"Duplicados eliminados por minuto: {duplicados_eliminados:,}"
    )  # Muestra duplicados eliminados
    print(
        f"Filas eliminadas por OHLC inválido: {invalidas_ohlc:,}"
    )  # Muestra OHLC inválidos eliminados
    print(f"Filas finales en Silver: {df.height:,}")  # Muestra filas finales
    print(f"Fecha mínima limpia: {min_fecha}")  # Muestra fecha mínima
    print(f"Fecha máxima limpia: {max_fecha}")  # Muestra fecha máxima
    print(
        f"Huecos temporales distintos a 1 minuto: {cantidad_huecos:,}"
    )  # Muestra cuántos huecos hay
    print(
        f"Mayor hueco temporal en segundos: {max_segundos_hueco}"
    )  # Muestra el mayor hueco encontrado

    guardar_silver(df)  # Guarda el archivo Silver en Parquet

    guardar_resumen(
        filas_iniciales=filas_iniciales,
        filas_finales=df.height,
        nulos_eliminados=nulos_eliminados,
        duplicados_eliminados=duplicados_eliminados,
        invalidas_ohlc=invalidas_ohlc,
        cantidad_huecos=cantidad_huecos,
        max_segundos_hueco=max_segundos_hueco,
        min_fecha=min_fecha,
        max_fecha=max_fecha,
    )  # Guarda resumen de auditoría

    print("=" * 70)  # Imprime separador visual final
    print("PROCESO SILVER TERMINADO")  # Muestra el fin del proceso
    print(f"Archivo Silver: {RUTA_SILVER}")  # Muestra la ruta del archivo Silver
    print(f"Resumen Silver: {RUTA_RESUMEN}")  # Muestra la ruta del resumen
    print(
        "Siguiente paso: construir la capa Gold en 1 hora"
    )  # Indica el siguiente paso lógico
    print("=" * 70)  # Imprime separador visual final


if __name__ == "__main__":  # Verifica que el archivo se esté ejecutando directamente
    main()  # Inicia el proceso Silver
