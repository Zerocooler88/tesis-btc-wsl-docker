from __future__ import annotations  # Habilita anotaciones de tipos modernas.

from pathlib import Path  # Permite gestionar rutas de archivos y carpetas.
from datetime import datetime  # Permite registrar fecha y hora del proceso.
import polars as pl  # Librería empleada para lectura y transformación de datos.


# =========================================================
# CONFIGURACION
# =========================================================
RUTA_GOLD = Path("data/gold/spot/BTCUSDT/1h/btcusdt_spot_1h_gold_2019_2026.parquet")  # Define la ruta del archivo Gold de entrada.

CARPETA_SALIDA = Path("data/model_input/spot/BTCUSDT/1h")  # Define la carpeta de salida del dataset preparado.
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe previamente.

RUTA_MODELO_INPUT = CARPETA_SALIDA / "btcusdt_spot_1h_model_input_2019_2026.parquet"  # Define la ruta final del archivo model_input.

CARPETA_LOGS = Path("logs")  # Define la carpeta donde se guardarán los resúmenes.
CARPETA_LOGS.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de logs si no existe.

RUTA_RESUMEN = CARPETA_LOGS / "resumen_model_input_gold_1h_2019_2026.txt"  # Define la ruta del archivo resumen.


# =========================================================
# FUNCIONES
# =========================================================
def cargar_gold() -> pl.DataFrame:  # Declara la función que carga el dataset Gold.
    if not RUTA_GOLD.exists():  # Verifica si el archivo Gold existe.
        raise FileNotFoundError(f"No existe Gold: {RUTA_GOLD}")  # Lanza un error si el archivo no está disponible.

    print("=" * 70)  # Imprime una línea separadora en consola.
    print("INICIO PREPARACION MODEL INPUT DESDE GOLD 1H")  # Informa el inicio del proceso.
    print(f"Leyendo Gold desde: {RUTA_GOLD}")  # Muestra la ruta del archivo leído.

    df = pl.read_parquet(RUTA_GOLD)  # Lee el archivo Parquet y lo carga en un DataFrame.

    print(f"Filas leidas: {df.height:,}")  # Muestra el número de filas cargadas.
    print(f"Columnas leidas: {df.width}")  # Muestra el número de columnas cargadas.

    return df  # Devuelve el DataFrame cargado.


def agregar_campos_entrenamiento(df: pl.DataFrame) -> pl.DataFrame:  # Declara la función que crea variables útiles para entrenamiento.
    df = df.sort("open_datetime_utc")  # Ordena el dataset cronológicamente por fecha y hora.

    df = df.with_columns(  # Agrega nuevas columnas derivadas al DataFrame.
        [
            pl.col("open_datetime_utc").dt.quarter().alias("trimestre"),  # Extrae el trimestre numérico de cada registro.
            (  # Inicia la construcción de una etiqueta trimestre-año.
                pl.col("open_datetime_utc").dt.year().cast(pl.Utf8)  # Extrae el año y lo convierte a texto.
                + pl.lit("-Q")  # Añade el separador textual "-Q".
                + pl.col("open_datetime_utc").dt.quarter().cast(pl.Utf8)  # Extrae el trimestre y lo convierte a texto.
            ).alias("year_quarter"),  # Guarda la etiqueta final como year_quarter.
            pl.col("close").alias("target_close"),  # Duplica la variable close como variable objetivo de predicción.
        ]
    )

    return df  # Devuelve el DataFrame con las nuevas variables.


def etiquetar_temporada_precio(df: pl.DataFrame) -> pl.DataFrame:  # Declara la función que clasifica niveles de precio.
    q1 = df["close"].quantile(0.25)  # Calcula el primer cuartil del precio close.
    q3 = df["close"].quantile(0.75)  # Calcula el tercer cuartil del precio close.

    df = df.with_columns(  # Agrega una columna categórica al DataFrame.
        [
            pl.when(pl.col("close") <= q1)  # Evalúa si el precio pertenece al tramo bajo.
            .then(pl.lit("baja"))  # Asigna la etiqueta "baja".
            .when(pl.col("close") >= q3)  # Evalúa si el precio pertenece al tramo alto.
            .then(pl.lit("alta"))  # Asigna la etiqueta "alta".
            .otherwise(pl.lit("media"))  # Asigna "media" al resto de observaciones.
            .alias("temporada_precio")  # Guarda la clasificación en la columna temporada_precio.
        ]
    )

    return df  # Devuelve el DataFrame etiquetado.


def seleccionar_columnas(df: pl.DataFrame) -> pl.DataFrame:  # Declara la función que organiza las columnas finales.
    columnas_finales = [  # Define la lista de columnas requeridas para modelado.
        "open_datetime_utc",  # Conserva la marca temporal de cada observación.
        "anio",  # Conserva el año.
        "mes",  # Conserva el mes.
        "dia",  # Conserva el día.
        "hora",  # Conserva la hora.
        "trimestre",  # Conserva el trimestre numérico.
        "year_quarter",  # Conserva la etiqueta año-trimestre.
        "regimen_mercado",  # Conserva el régimen de mercado.
        "temporada_precio",  # Conserva la categoría de nivel de precio.
        "open",  # Conserva el precio de apertura.
        "high",  # Conserva el precio máximo.
        "low",  # Conserva el precio mínimo.
        "close",  # Conserva el precio de cierre.
        "volume",  # Conserva el volumen operado.
        "quote_asset_volume",  # Conserva el volumen en activo cotizado.
        "number_of_trades",  # Conserva el número de transacciones.
        "taker_buy_base_volume",  # Conserva el volumen comprador en activo base.
        "taker_buy_quote_volume",  # Conserva el volumen comprador en activo cotizado.
        "rango_hl",  # Conserva el rango entre high y low.
        "cuerpo_oc",  # Conserva la diferencia entre open y close.
        "return_1h",  # Conserva el retorno simple de una hora.
        "log_return_1h",  # Conserva el retorno logarítmico de una hora.
        "sma_24h",  # Conserva la media móvil simple de 24 horas.
        "sma_168h",  # Conserva la media móvil simple de 168 horas.
        "volatilidad_24h",  # Conserva la volatilidad móvil de 24 horas.
        "volumen_sma_24h",  # Conserva la media móvil del volumen en 24 horas.
        "target_close",  # Conserva la variable objetivo para predicción.
    ]

    faltantes = [c for c in columnas_finales if c not in df.columns]  # Identifica columnas esperadas que no existen.
    if faltantes:  # Verifica si hay columnas ausentes.
        raise ValueError(f"Faltan columnas esperadas en Gold: {faltantes}")  # Lanza un error si faltan columnas requeridas.

    return df.select(columnas_finales)  # Devuelve únicamente las columnas finales seleccionadas.


def guardar_salida(df: pl.DataFrame) -> None:  # Declara la función que guarda el dataset final.
    df.write_parquet(RUTA_MODELO_INPUT, compression="zstd")  # Guarda el DataFrame en formato Parquet comprimido.

    size_mb = RUTA_MODELO_INPUT.stat().st_size / 1024 / 1024  # Calcula el tamaño del archivo en megabytes.
    print("-" * 70)  # Imprime una línea separadora en consola.
    print(f"Archivo model_input guardado en: {RUTA_MODELO_INPUT}")  # Informa la ruta del archivo generado.
    print(f"Tamaño aproximado: {size_mb:,.2f} MB")  # Muestra el tamaño aproximado del archivo.


def guardar_resumen(df: pl.DataFrame) -> None:  # Declara la función que genera un resumen textual del proceso.
    conteo_q = (  # Inicia el cálculo del conteo por trimestre.
        df.group_by("year_quarter")  # Agrupa las observaciones por año-trimestre.
        .agg(pl.len().alias("filas"))  # Cuenta la cantidad de filas por grupo.
        .sort("year_quarter")  # Ordena los resultados cronológicamente.
    )

    lineas = []  # Crea una lista para almacenar el contenido del resumen.
    lineas.append("RESUMEN MODEL INPUT DESDE GOLD 1H")  # Agrega el título del resumen.
    lineas.append("=" * 60)  # Agrega una línea separadora.
    lineas.append(f"Fecha de ejecucion: {datetime.now().isoformat(timespec='seconds')}")  # Registra la fecha y hora de ejecución.
    lineas.append(f"Filas finales: {df.height:,}")  # Registra el total de filas finales.
    lineas.append(f"Fecha minima: {df['open_datetime_utc'].min()}")  # Registra la fecha mínima del dataset.
    lineas.append(f"Fecha maxima: {df['open_datetime_utc'].max()}")  # Registra la fecha máxima del dataset.
    lineas.append("Columnas:")  # Agrega el encabezado de columnas.
    for c in df.columns:  # Recorre todas las columnas del DataFrame.
        lineas.append(f"  - {c}")  # Añade cada columna al resumen.
    lineas.append("Conteo por trimestre:")  # Agrega el encabezado de conteo trimestral.
    for fila in conteo_q.iter_rows(named=True):  # Recorre cada fila del conteo por trimestre.
        lineas.append(f"  - {fila['year_quarter']}: {fila['filas']:,}")  # Añade el trimestre y su cantidad de registros.
    lineas.append("=" * 60)  # Agrega una línea separadora final.

    RUTA_RESUMEN.write_text("\n".join(lineas), encoding="utf-8")  # Escribe el resumen en un archivo de texto.
    print(f"Resumen guardado en: {RUTA_RESUMEN}")  # Informa la ruta del resumen generado.


def main() -> None:  # Declara la función principal del script.
    df = cargar_gold()  # Carga el dataset Gold.
    df = agregar_campos_entrenamiento(df)  # Agrega variables de apoyo para entrenamiento.
    df = etiquetar_temporada_precio(df)  # Clasifica el precio en baja, media o alta.
    df = seleccionar_columnas(df)  # Conserva únicamente las columnas finales requeridas.

    print("-" * 70)  # Imprime una línea separadora en consola.
    print(f"Filas finales model_input: {df.height:,}")  # Muestra el total de filas del dataset final.
    print(f"Fecha minima: {df['open_datetime_utc'].min()}")  # Muestra la fecha mínima del dataset final.
    print(f"Fecha maxima: {df['open_datetime_utc'].max()}")  # Muestra la fecha máxima del dataset final.
    print(f"Total de trimestres: {df.select(pl.col('year_quarter').n_unique()).item()}")  # Muestra el número de trimestres únicos.

    guardar_salida(df)  # Guarda el dataset final preparado para modelado.
    guardar_resumen(df)  # Guarda el resumen descriptivo del proceso.

    print("=" * 70)  # Imprime una línea separadora final.
    print("MODEL INPUT LISTO")  # Informa que la preparación terminó correctamente.
    print("=" * 70)  # Imprime una línea separadora final.


if __name__ == "__main__":  # Verifica que el archivo se ejecute de forma directa.
    main()  # Ejecuta la función principal del script.