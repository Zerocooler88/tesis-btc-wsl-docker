from __future__ import annotations  # Habilita anotaciones modernas de tipos para mayor claridad del código.

from pathlib import Path  # Permite gestionar rutas y carpetas de forma robusta y portable.
from datetime import datetime  # Permite registrar la fecha y hora de ejecución del proceso.
import polars as pl  # Librería principal para lectura, transformación y escritura de datos tabulares.


# =========================================================
# CONFIGURACION
# =========================================================
RUTA_ENTRADA = Path(  # Define la ruta del archivo de entrada para el proceso trimestral.
    "data/model_input/spot/BTCUSDT/1h/btcusdt_spot_1h_model_input_2019_2026.parquet"  # Archivo model_input consolidado a 1 hora.
)

CARPETA_SALIDA = Path("data/windows_trimestrales/spot/BTCUSDT/1h")  # Define la carpeta donde se guardarán los archivos por trimestre.
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de salida si todavía no existe.

CARPETA_LOGS = Path("logs")  # Define la carpeta donde se guardarán los reportes de auditoría.
CARPETA_LOGS.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de logs si todavía no existe.

RUTA_RESUMEN = CARPETA_LOGS / "resumen_windows_trimestrales_1h_2019_2026.txt"  # Define el archivo de resumen del proceso.


# =========================================================
# COLUMNAS CLAVE
# =========================================================
COLUMNAS_MODELO = [  # Lista de variables predictoras y objetivo para el entrenamiento.
    "open",  # Precio de apertura de la vela horaria.
    "high",  # Precio máximo de la vela horaria.
    "low",  # Precio mínimo de la vela horaria.
    "close",  # Precio de cierre de la vela horaria.
    "volume",  # Volumen transado del activo base.
    "quote_asset_volume",  # Volumen transado expresado en activo cotizado.
    "number_of_trades",  # Número de transacciones realizadas en la hora.
    "taker_buy_base_volume",  # Volumen comprador agresor en activo base.
    "taker_buy_quote_volume",  # Volumen comprador agresor en activo cotizado.
    "rango_hl",  # Rango intrahorario calculado como high - low.
    "cuerpo_oc",  # Cuerpo de la vela calculado como close - open.
    "return_1h",  # Retorno simple respecto a la hora previa.
    "log_return_1h",  # Retorno logarítmico respecto a la hora previa.
    "sma_24h",  # Media móvil simple de 24 horas.
    "sma_168h",  # Media móvil simple de 168 horas.
    "volatilidad_24h",  # Volatilidad estimada sobre una ventana de 24 horas.
    "volumen_sma_24h",  # Media móvil del volumen sobre 24 horas.
    "target_close",  # Variable objetivo que representa el close a modelar.
]

COLUMNAS_CONTROL = [  # Lista de columnas de control y segmentación analítica.
    "open_datetime_utc",  # Marca temporal de cada observación.
    "year_quarter",  # Identificador de trimestre en formato YYYY-Qn.
    "regimen_mercado",  # Régimen temporal del mercado: pre_covid, covid o post_covid.
    "temporada_precio",  # Nivel relativo del precio: baja, media o alta.
]

COLUMNAS_REQUERIDAS = COLUMNAS_CONTROL + COLUMNAS_MODELO  # Une columnas de control y columnas requeridas por el modelo.


# =========================================================
# FUNCIONES
# =========================================================
def cargar_model_input() -> pl.DataFrame:  # Carga el archivo consolidado de entrada.
    if not RUTA_ENTRADA.exists():  # Verifica que el archivo exista antes de continuar.
        raise FileNotFoundError(f"No existe el archivo de entrada: {RUTA_ENTRADA}")  # Lanza un error claro si falta el archivo.

    print("=" * 70)  # Imprime un separador visual para el inicio del proceso.
    print("INICIO GENERACION DE WINDOWS TRIMESTRALES 1H")  # Muestra el nombre del proceso en ejecución.
    print(f"Leyendo archivo desde: {RUTA_ENTRADA}")  # Informa la ruta exacta del archivo de entrada.

    df = pl.read_parquet(RUTA_ENTRADA)  # Lee el archivo Parquet en memoria.

    print(f"Filas leidas: {df.height:,}")  # Muestra el total de filas cargadas.
    print(f"Columnas leidas: {df.width}")  # Muestra el total de columnas cargadas.

    return df  # Devuelve el DataFrame cargado para las siguientes etapas.


def validar_columnas(df: pl.DataFrame) -> None:  # Verifica que todas las columnas requeridas estén presentes.
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]  # Identifica columnas esperadas que no existen en el dataset.
    if faltantes:  # Evalúa si existen columnas faltantes.
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")  # Lanza un error explícito con el detalle de columnas ausentes.


def limpiar_nulos_modelo(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:  # Elimina filas con nulos en columnas necesarias para modelado.
    filas_antes = df.height  # Guarda el número de filas antes de la limpieza.

    condicion: pl.Expr = pl.lit(True)  # Inicializa la condición lógica acumulativa.
    for col in COLUMNAS_REQUERIDAS:  # Recorre todas las columnas obligatorias.
        condicion = condicion & pl.col(col).is_not_null()  # Acumula la condición lógica entre todas las columnas.

    df = df.filter(condicion).sort("open_datetime_utc")  # Filtra las filas completas y ordena cronológicamente.

    filas_despues = df.height  # Guarda el número de filas después de la limpieza.
    eliminadas = filas_antes - filas_despues  # Calcula cuántas filas fueron eliminadas.

    return df, eliminadas  # Devuelve el DataFrame limpio y el total de filas eliminadas.


def obtener_trimestres(df: pl.DataFrame) -> list[str]:  # Extrae la lista ordenada de trimestres disponibles.
    trimestres = (  # Inicia el bloque de extracción de trimestres únicos.
        df.select("year_quarter")  # Selecciona únicamente la columna de trimestre.
        .unique()  # Conserva solo los valores únicos.
        .sort("year_quarter")  # Ordena los trimestres cronológicamente.
        .to_series()  # Convierte el resultado a serie.
        .to_list()  # Convierte la serie a lista de Python.
    )
    return trimestres  # Devuelve la lista final de trimestres.


def guardar_archivos_trimestrales(df: pl.DataFrame) -> list[dict]:  # Genera y guarda un archivo independiente por trimestre.
    resumen_trimestres = []  # Inicializa la lista donde se almacenará el resumen de cada archivo generado.

    trimestres = obtener_trimestres(df)  # Obtiene todos los trimestres presentes en el dataset.

    for trimestre in trimestres:  # Recorre cada trimestre para procesarlo individualmente.
        df_trim = (  # Inicia la construcción del subconjunto trimestral.
            df.filter(pl.col("year_quarter") == trimestre)  # Filtra únicamente las filas del trimestre actual.
            .sort("open_datetime_utc")  # Ordena el subconjunto por fecha y hora.
        )

        nombre_archivo = f"btcusdt_spot_1h_{trimestre}_model_input.parquet"  # Construye el nombre del archivo trimestral.
        ruta_salida = CARPETA_SALIDA / nombre_archivo  # Define la ruta de salida completa para el trimestre actual.

        df_trim.write_parquet(ruta_salida, compression="zstd")  # Guarda el subconjunto en formato Parquet comprimido.

        resumen_trimestres.append(  # Agrega al resumen la información del archivo recién creado.
            {
                "year_quarter": trimestre,  # Registra el identificador del trimestre.
                "filas": df_trim.height,  # Registra el total de filas contenidas en el archivo trimestral.
                "fecha_min": str(df_trim["open_datetime_utc"].min()),  # Registra la fecha mínima del trimestre.
                "fecha_max": str(df_trim["open_datetime_utc"].max()),  # Registra la fecha máxima del trimestre.
                "ruta": str(ruta_salida),  # Registra la ruta del archivo generado.
            }
        )

        print("-" * 70)  # Imprime un separador visual entre trimestres.
        print(f"Trimestre guardado: {trimestre}")  # Informa el trimestre procesado.
        print(f"Filas: {df_trim.height:,}")  # Informa el total de filas del trimestre.
        print(f"Desde: {df_trim['open_datetime_utc'].min()}")  # Informa la fecha inicial del trimestre.
        print(f"Hasta: {df_trim['open_datetime_utc'].max()}")  # Informa la fecha final del trimestre.
        print(f"Archivo: {ruta_salida}")  # Informa la ubicación del archivo generado.

    return resumen_trimestres  # Devuelve la lista consolidada de resumen trimestral.


def guardar_resumen(
    filas_iniciales: int,  # Recibe el número de filas antes de limpiar.
    filas_finales: int,  # Recibe el número de filas después de limpiar.
    nulos_eliminados: int,  # Recibe el total de filas eliminadas por nulos.
    resumen_trimestres: list[dict],  # Recibe el detalle de archivos trimestrales generados.
) -> None:
    lineas = []  # Inicializa la lista que almacenará el contenido del reporte.
    lineas.append("RESUMEN GENERACION WINDOWS TRIMESTRALES 1H")  # Escribe el título principal del resumen.
    lineas.append("=" * 60)  # Agrega separador visual superior.
    lineas.append(f"Fecha de ejecucion: {datetime.now().isoformat(timespec='seconds')}")  # Registra fecha y hora de ejecución.
    lineas.append(f"Filas iniciales: {filas_iniciales:,}")  # Registra el total de filas antes de limpiar.
    lineas.append(f"Filas eliminadas por nulos: {nulos_eliminados:,}")  # Registra las filas descartadas por nulos.
    lineas.append(f"Filas finales limpias: {filas_finales:,}")  # Registra el total de filas limpias disponibles.
    lineas.append(f"Total de trimestres generados: {len(resumen_trimestres)}")  # Registra cuántos archivos trimestrales fueron generados.
    lineas.append("-" * 60)  # Agrega separador antes del detalle por trimestre.

    for item in resumen_trimestres:  # Recorre cada elemento del resumen trimestral.
        lineas.append(f"Trimestre: {item['year_quarter']}")  # Registra el identificador del trimestre.
        lineas.append(f"  Filas: {item['filas']:,}")  # Registra el total de filas del trimestre.
        lineas.append(f"  Fecha minima: {item['fecha_min']}")  # Registra la fecha mínima del trimestre.
        lineas.append(f"  Fecha maxima: {item['fecha_max']}")  # Registra la fecha máxima del trimestre.
        lineas.append(f"  Archivo: {item['ruta']}")  # Registra la ruta del archivo generado.
        lineas.append("-" * 60)  # Agrega separador entre bloques trimestrales.

    lineas.append("=" * 60)  # Agrega separador visual final.

    RUTA_RESUMEN.write_text("\n".join(lineas), encoding="utf-8")  # Escribe el resumen completo en archivo de texto.
    print(f"Resumen guardado en: {RUTA_RESUMEN}")  # Informa la ubicación del reporte de salida.


def main() -> None:  # Controla la secuencia principal de ejecución del script.
    df = cargar_model_input()  # Carga el archivo consolidado de model_input.
    filas_iniciales = df.height  # Guarda el total de filas originales antes de la limpieza.

    validar_columnas(df)  # Verifica que el dataset contenga todas las columnas necesarias.
    df, nulos_eliminados = limpiar_nulos_modelo(df)  # Limpia filas con nulos y obtiene el total eliminado.

    print("-" * 70)  # Imprime un separador visual para el resumen intermedio.
    print(f"Filas iniciales: {filas_iniciales:,}")  # Muestra el total de filas de entrada.
    print(f"Nulos eliminados: {nulos_eliminados:,}")  # Muestra cuántas filas fueron eliminadas por nulos.
    print(f"Filas finales limpias: {df.height:,}")  # Muestra el total de filas útiles para particionar.
    print(f"Fecha minima limpia: {df['open_datetime_utc'].min()}")  # Muestra la fecha mínima después de limpiar.
    print(f"Fecha maxima limpia: {df['open_datetime_utc'].max()}")  # Muestra la fecha máxima después de limpiar.

    resumen_trimestres = guardar_archivos_trimestrales(df)  # Genera y guarda los archivos trimestrales.

    guardar_resumen(
        filas_iniciales=filas_iniciales,  # Envía al resumen el total de filas de entrada.
        filas_finales=df.height,  # Envía al resumen el total de filas finales limpias.
        nulos_eliminados=nulos_eliminados,  # Envía al resumen el total de filas eliminadas.
        resumen_trimestres=resumen_trimestres,  # Envía al resumen el detalle de cada trimestre generado.
    )

    print("=" * 70)  # Imprime separador visual de cierre.
    print("WINDOWS TRIMESTRALES LISTOS")  # Informa que el proceso finalizó correctamente.
    print("=" * 70)  # Imprime separador visual final.


if __name__ == "__main__":  # Verifica que el script se ejecute de forma directa.
    main()  # Ejecuta la función principal del proceso.