from __future__ import annotations  # Permite usar anotaciones modernas de tipos

from pathlib import Path  # Sirve para manejar rutas y carpetas
from datetime import datetime  # Sirve para trabajar con fechas y horas

import numpy as np  # Sirve para cálculos numéricos
import polars as pl  # Sirve para leer y transformar datos Parquet
import matplotlib.pyplot as plt  # Sirve para hacer gráficos
import matplotlib.dates as mdates  # Sirve para formatear fechas en ejes
from matplotlib.figure import Figure  # Corrige el tipo Figure para Pylance


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
PROYECTO_ROOT = (
    Path(__file__).resolve().parent.parent
)  # Toma la raíz del proyecto desde la carpeta src

RUTA_SILVER = (
    PROYECTO_ROOT
    / "data"
    / "silver"
    / "spot"
    / "BTCUSDT"
    / "1m"
    / "btcusdt_spot_1m_silver_2019_2026.parquet"
)  # Ruta absoluta de Silver
RUTA_GOLD = (
    PROYECTO_ROOT
    / "data"
    / "gold"
    / "spot"
    / "BTCUSDT"
    / "1h"
    / "btcusdt_spot_1h_gold_2019_2026.parquet"
)  # Ruta absoluta de Gold

CARPETA_REPORTES = (
    PROYECTO_ROOT / "reports" / "analisis_resultados"
)  # Carpeta principal del análisis
CARPETA_TABLAS = CARPETA_REPORTES / "tablas"  # Carpeta para tablas CSV
CARPETA_FIGURAS = CARPETA_REPORTES / "figuras"  # Carpeta para figuras PNG
CARPETA_REPORTES.mkdir(
    parents=True, exist_ok=True
)  # Crea carpeta principal si no existe
CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)  # Crea carpeta de tablas si no existe
CARPETA_FIGURAS.mkdir(
    parents=True, exist_ok=True
)  # Crea carpeta de figuras si no existe

RUTA_RESUMEN_GENERAL = (
    CARPETA_REPORTES / "resumen_analisis_resultados.txt"
)  # Ruta del resumen general

FECHA_CORTE_PRE_COVID = datetime(2020, 3, 11, 0, 0, 0)  # Fin del periodo pre_covid
FECHA_CORTE_POST_COVID = datetime(2022, 1, 1, 0, 0, 0)  # Inicio del periodo post_covid

COLUMNAS_MINIMAS_SILVER = [  # Lista de columnas obligatorias en Silver
    "open_datetime_utc",  # Fecha y hora de apertura
    "open",  # Precio de apertura
    "high",  # Precio máximo
    "low",  # Precio mínimo
    "close",  # Precio de cierre
    "volume",  # Volumen
]

COLUMNAS_MINIMAS_GOLD = [  # Lista de columnas obligatorias en Gold
    "open_datetime_utc",  # Fecha y hora de apertura
    "open",  # Precio de apertura
    "high",  # Precio máximo
    "low",  # Precio mínimo
    "close",  # Precio de cierre
    "volume",  # Volumen
]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def preparar_carpetas() -> None:  # Crea las carpetas de salida
    CARPETA_REPORTES.mkdir(
        parents=True, exist_ok=True
    )  # Crea carpeta principal si no existe
    CARPETA_TABLAS.mkdir(
        parents=True, exist_ok=True
    )  # Crea carpeta de tablas si no existe
    CARPETA_FIGURAS.mkdir(
        parents=True, exist_ok=True
    )  # Crea carpeta de figuras si no existe


def validar_archivo_existe(
    ruta: Path, nombre_logico: str
) -> None:  # Verifica si existe un archivo
    if not ruta.exists():  # Si el archivo no existe
        raise FileNotFoundError(
            f"No existe el archivo {nombre_logico}: {ruta}"
        )  # Lanza error claro


def validar_columnas_obligatorias(  # Revisa columnas mínimas
    df: pl.DataFrame, columnas: list[str], nombre_df: str
) -> None:
    faltantes = [
        col for col in columnas if col not in df.columns
    ]  # Busca columnas faltantes
    if faltantes:  # Si faltan columnas
        raise ValueError(  # Lanza error entendible
            f"En {nombre_df} faltan columnas obligatorias: {faltantes}"  # Muestra cuáles faltan
        )


def asegurar_orden_temporal(df: pl.DataFrame) -> pl.DataFrame:  # Ordena por fecha
    if "open_datetime_utc" in df.columns:  # Si existe la columna de tiempo
        return df.sort("open_datetime_utc")  # Devuelve el DataFrame ordenado
    return df  # Si no existe, devuelve el mismo DataFrame


def enriquecer_gold_si_falta(
    gold: pl.DataFrame,
) -> pl.DataFrame:  # Agrega columnas derivadas si faltan
    if (  # Verifica si falta rango_hl
        "rango_hl" not in gold.columns  # No existe rango_hl
        and "high" in gold.columns  # Existe high
        and "low" in gold.columns  # Existe low
    ):
        gold = gold.with_columns(  # Agrega nueva columna
            (pl.col("high") - pl.col("low")).alias("rango_hl")  # Calcula high menos low
        )

    if (  # Verifica si falta cuerpo_oc
        "cuerpo_oc" not in gold.columns  # No existe cuerpo_oc
        and "open" in gold.columns  # Existe open
        and "close" in gold.columns  # Existe close
    ):
        gold = gold.with_columns(  # Agrega nueva columna
            (pl.col("close") - pl.col("open"))
            .abs()
            .alias("cuerpo_oc")  # Diferencia absoluta close-open
        )

    if (
        "return_1h" not in gold.columns and "close" in gold.columns
    ):  # Si falta return_1h
        gold = gold.with_columns(  # Agrega nueva columna
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias(
                "return_1h"
            )  # Retorno simple 1h
        )

    if (
        "log_return_1h" not in gold.columns and "close" in gold.columns
    ):  # Si falta log_return_1h
        gold = gold.with_columns(  # Agrega nueva columna
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .alias("log_return_1h")  # Retorno logarítmico 1h
        )

    if (
        "volatilidad_24h" not in gold.columns and "log_return_1h" in gold.columns
    ):  # Si falta volatilidad_24h
        gold = gold.with_columns(  # Agrega nueva columna
            pl.col("log_return_1h")  # Toma los retornos logarítmicos
            .rolling_std(
                window_size=24, min_samples=2
            )  # Calcula desviación estándar móvil de 24 horas
            .alias("volatilidad_24h")  # Nombra la columna nueva
        )

    if (
        "regimen_mercado" not in gold.columns and "open_datetime_utc" in gold.columns
    ):  # Si falta régimen
        gold = gold.with_columns(  # Agrega columna de régimen
            pl.when(
                pl.col("open_datetime_utc") < pl.lit(FECHA_CORTE_PRE_COVID)
            )  # Si es antes de covid
            .then(pl.lit("pre_covid"))  # Marca como pre_covid
            .when(
                pl.col("open_datetime_utc") < pl.lit(FECHA_CORTE_POST_COVID)
            )  # Si está entre cortes
            .then(pl.lit("covid"))  # Marca como covid
            .otherwise(pl.lit("post_covid"))  # Lo demás es post_covid
            .alias("regimen_mercado")  # Nombre de la nueva columna
        )

    return gold  # Devuelve Gold enriquecido


def guardar_figura(
    fig: Figure, nombre_archivo: str
) -> None:  # Guarda una figura en PNG
    fig.tight_layout()  # Ajusta espacios automáticamente
    fig.savefig(
        CARPETA_FIGURAS / nombre_archivo, dpi=200, bbox_inches="tight"
    )  # Guarda la imagen
    plt.close(fig)  # Cierra la figura para liberar memoria


def formatear_numero(
    valor: float | int | None, decimales: int = 2
) -> str:  # Da formato bonito a números
    if valor is None:  # Si el valor es nulo
        return "N/A"  # Devuelve texto N/A
    try:  # Intenta formatear
        return f"{valor:.{decimales}f}"  # Devuelve con decimales
    except Exception:  # Si falla
        return str(valor)  # Devuelve el valor como texto


# =========================================================
# FUNCIONES DE CARGA
# =========================================================
def cargar_datasets() -> tuple[pl.DataFrame, pl.DataFrame]:  # Lee Silver y Gold
    validar_archivo_existe(RUTA_SILVER, "Silver")  # Verifica Silver
    validar_archivo_existe(RUTA_GOLD, "Gold")  # Verifica Gold

    print("=" * 70)  # Línea visual
    print(
        "INICIO DEL ANALISIS DE RESULTADOS DEL DATASET Y DEL CLOSE"
    )  # Mensaje principal
    print(f"Leyendo Silver desde: {RUTA_SILVER}")  # Muestra ruta Silver
    print(f"Leyendo Gold desde:   {RUTA_GOLD}")  # Muestra ruta Gold

    silver = pl.read_parquet(RUTA_SILVER)  # Lee Silver
    gold = pl.read_parquet(RUTA_GOLD)  # Lee Gold

    silver = asegurar_orden_temporal(silver)  # Ordena Silver por fecha
    gold = asegurar_orden_temporal(gold)  # Ordena Gold por fecha

    validar_columnas_obligatorias(
        silver, COLUMNAS_MINIMAS_SILVER, "Silver"
    )  # Valida columnas Silver
    validar_columnas_obligatorias(
        gold, COLUMNAS_MINIMAS_GOLD, "Gold"
    )  # Valida columnas Gold

    gold = enriquecer_gold_si_falta(gold)  # Agrega columnas derivadas si faltan

    print(f"Filas Silver: {silver.height:,}")  # Muestra número de filas Silver
    print(f"Filas Gold:   {gold.height:,}")  # Muestra número de filas Gold
    print("=" * 70)  # Línea visual

    return silver, gold  # Devuelve ambos DataFrames


# =========================================================
# FUNCIONES DE RESUMEN DEL DATASET
# =========================================================
def calcular_huecos_silver(
    silver: pl.DataFrame,
) -> tuple[int, int | None]:  # Busca huecos temporales en Silver
    df_huecos = silver.select(  # Crea tabla temporal
        [
            pl.col("open_datetime_utc"),  # Toma la fecha
            pl.col("open_datetime_utc")
            .diff()
            .alias("delta_tiempo"),  # Calcula diferencia entre filas
        ]
    )

    huecos = df_huecos.filter(  # Filtra solo diferencias raras
        pl.col("delta_tiempo").is_not_null()  # Descarta el primer nulo
        & (
            pl.col("delta_tiempo") != pl.duration(minutes=1)
        )  # Busca diferencias distintas a 1 minuto
    )

    cantidad_huecos = huecos.height  # Cuenta cuántos huecos hay

    if cantidad_huecos > 0:  # Si sí hay huecos
        max_segundos = int(  # Convierte a entero
            huecos.select(
                pl.col("delta_tiempo").dt.total_seconds().max()
            ).item()  # Toma el hueco más grande
        )
    else:  # Si no hay huecos
        max_segundos = None  # Deja None

    return cantidad_huecos, max_segundos  # Devuelve cantidad y máximo


def guardar_tabla_resumen_dataset(  # Crea tabla resumen general
    silver: pl.DataFrame, gold: pl.DataFrame
) -> pl.DataFrame:
    cantidad_huecos, max_segundos = calcular_huecos_silver(
        silver
    )  # Calcula huecos en Silver

    filas = [  # Construye filas de la tabla
        {
            "dataset": "silver_1m",  # Nombre lógico del dataset
            "filas": silver.height,  # Número de filas
            "columnas": silver.width,  # Número de columnas
            "fecha_minima": str(silver["open_datetime_utc"].min()),  # Fecha mínima
            "fecha_maxima": str(silver["open_datetime_utc"].max()),  # Fecha máxima
            "huecos_temporales": cantidad_huecos,  # Cantidad de huecos
            "mayor_hueco_segundos": max_segundos,  # Hueco más grande
        },
        {
            "dataset": "gold_1h",  # Nombre lógico del dataset
            "filas": gold.height,  # Número de filas
            "columnas": gold.width,  # Número de columnas
            "fecha_minima": str(gold["open_datetime_utc"].min()),  # Fecha mínima
            "fecha_maxima": str(gold["open_datetime_utc"].max()),  # Fecha máxima
            "huecos_temporales": None,  # No se analiza hueco aquí
            "mayor_hueco_segundos": None,  # No se analiza hueco aquí
        },
    ]

    tabla = pl.DataFrame(filas)  # Convierte lista a DataFrame
    tabla.write_csv(CARPETA_TABLAS / "tabla_resumen_dataset.csv")  # Guarda el CSV
    return tabla  # Devuelve la tabla


def guardar_tabla_estructura_variables(
    silver: pl.DataFrame,
) -> pl.DataFrame:  # Guarda nombres y tipos de Silver
    tabla = pl.DataFrame(  # Construye la tabla
        {
            "variable": silver.columns,  # Nombres de columnas
            "tipo_dato": [
                str(t) for t in silver.dtypes
            ],  # Tipos de dato convertidos a texto
        }
    )

    tabla.write_csv(
        CARPETA_TABLAS / "tabla_estructura_variables_silver.csv"
    )  # Guarda la tabla
    return tabla  # Devuelve la tabla


def guardar_tabla_nulos(
    df: pl.DataFrame, nombre_archivo: str
) -> pl.DataFrame:  # Cuenta nulos por columna
    filas = []  # Lista vacía para construir la tabla

    for columna in df.columns:  # Recorre cada columna
        nulos = df.select(pl.col(columna).is_null().sum()).item()  # Cuenta nulos
        filas.append(
            {"variable": columna, "nulos": int(nulos)}
        )  # Agrega fila a la lista

    tabla = pl.DataFrame(filas)  # Convierte lista a DataFrame
    tabla.write_csv(CARPETA_TABLAS / nombre_archivo)  # Guarda la tabla
    return tabla  # Devuelve la tabla


def guardar_descriptivos_gold(
    gold: pl.DataFrame,
) -> pl.DataFrame:  # Guarda descriptivos de Gold
    columnas_numericas = [  # Lista de columnas que se quieren resumir
        "open",  # Apertura
        "high",  # Máximo
        "low",  # Mínimo
        "close",  # Cierre
        "volume",  # Volumen
        "number_of_trades",  # Número de trades
        "quote_asset_volume",  # Volumen cotizado
        "taker_buy_base_volume",  # Compra agresora base
        "taker_buy_quote_volume",  # Compra agresora quote
        "rango_hl",  # Rango high-low
        "cuerpo_oc",  # Cuerpo open-close
        "return_1h",  # Retorno simple
        "log_return_1h",  # Retorno logarítmico
        "volatilidad_24h",  # Volatilidad móvil
    ]

    columnas_presentes = [
        c for c in columnas_numericas if c in gold.columns
    ]  # Filtra solo columnas existentes

    if not columnas_presentes:  # Si no existe ninguna
        raise ValueError(
            "No se encontraron columnas numéricas esperadas en Gold para descriptivos."
        )  # Lanza error

    tabla = gold.select(
        columnas_presentes
    ).describe()  # Calcula estadísticos descriptivos
    tabla.write_csv(CARPETA_TABLAS / "tabla_descriptivos_gold.csv")  # Guarda el CSV
    return tabla  # Devuelve la tabla


# =========================================================
# FUNCIONES DE ANALISIS DEL CLOSE
# =========================================================
def construir_tabla_mensual_close(
    gold: pl.DataFrame,
) -> pl.DataFrame:  # Resume el close por mes
    expresiones = [  # Lista de agregaciones
        pl.col("close")
        .mean()
        .alias("close_promedio_mensual"),  # Media mensual del close
        pl.col("close").min().alias("close_min_mensual"),  # Mínimo mensual del close
        pl.col("close").max().alias("close_max_mensual"),  # Máximo mensual del close
        pl.col("close")
        .std()
        .alias("close_std_mensual"),  # Desviación estándar del close
        pl.len().alias("filas_mes"),  # Cantidad de filas del mes
    ]

    if "log_return_1h" in gold.columns:  # Si existe log_return_1h
        expresiones.append(
            pl.col("log_return_1h").std().alias("volatilidad_mensual")
        )  # Agrega volatilidad mensual
    else:  # Si no existe
        expresiones.append(pl.lit(None).alias("volatilidad_mensual"))  # Deja nulo

    mensual = (  # Empieza construcción de tabla mensual
        gold.group_by_dynamic(  # Agrupa por periodos de tiempo
            index_column="open_datetime_utc",  # Usa la fecha como índice temporal
            every="1mo",  # Agrupa cada 1 mes
            period="1mo",  # Cada ventana dura 1 mes
            label="left",  # Marca el inicio de la ventana
            closed="left",  # Incluye el borde izquierdo
            start_by="window",  # Empieza por ventana calendario
        )
        .agg(expresiones)  # Aplica agregaciones
        .sort("open_datetime_utc")  # Ordena por fecha
        .with_columns(  # Agrega columnas nuevas
            pl.col("open_datetime_utc")
            .dt.strftime("%Y-%m")
            .alias("periodo")  # Convierte a texto tipo 2024-01
        )
    )

    mensual.write_csv(
        CARPETA_TABLAS / "tabla_close_mensual.csv"
    )  # Guarda la tabla mensual
    return mensual  # Devuelve la tabla


def guardar_top_maximos_minimos_estables(  # Guarda top mensuales
    mensual: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    top_maximos = (  # Construye top máximos
        mensual.sort(
            "close_promedio_mensual", descending=True
        )  # Ordena de mayor a menor
        .select(  # Selecciona columnas útiles
            [
                "periodo",  # Periodo
                "close_promedio_mensual",  # Close promedio
                "close_max_mensual",  # Máximo mensual
                "volatilidad_mensual",  # Volatilidad mensual
                "filas_mes",  # Filas del mes
            ]
        )
        .head(10)  # Toma los primeros 10
    )

    top_minimos = (  # Construye top mínimos
        mensual.sort(
            "close_promedio_mensual", descending=False
        )  # Ordena de menor a mayor
        .select(  # Selecciona columnas útiles
            [
                "periodo",  # Periodo
                "close_promedio_mensual",  # Close promedio
                "close_min_mensual",  # Mínimo mensual
                "volatilidad_mensual",  # Volatilidad mensual
                "filas_mes",  # Filas del mes
            ]
        )
        .head(10)  # Toma los primeros 10
    )

    top_estables = (  # Construye top estables
        mensual.filter(pl.col("volatilidad_mensual").is_not_null())  # Quita nulos
        .sort(
            "volatilidad_mensual", descending=False
        )  # Ordena de menor a mayor volatilidad
        .select(  # Selecciona columnas útiles
            [
                "periodo",  # Periodo
                "close_promedio_mensual",  # Close promedio
                "volatilidad_mensual",  # Volatilidad mensual
                "filas_mes",  # Filas del mes
            ]
        )
        .head(10)  # Toma los primeros 10
    )

    top_maximos.write_csv(
        CARPETA_TABLAS / "tabla_top_10_meses_maximos.csv"
    )  # Guarda top máximos
    top_minimos.write_csv(
        CARPETA_TABLAS / "tabla_top_10_meses_minimos.csv"
    )  # Guarda top mínimos
    top_estables.write_csv(
        CARPETA_TABLAS / "tabla_top_10_meses_estables.csv"
    )  # Guarda top estables

    return top_maximos, top_minimos, top_estables  # Devuelve las 3 tablas


def guardar_tabla_regimenes(
    gold: pl.DataFrame,
) -> pl.DataFrame:  # Resume el close por régimen
    if "regimen_mercado" not in gold.columns:  # Si no existe la columna
        tabla = pl.DataFrame(  # Crea tabla vacía
            {
                "regimen_mercado": [],  # Columna vacía
                "filas": [],  # Columna vacía
                "close_promedio": [],  # Columna vacía
                "close_minimo": [],  # Columna vacía
                "close_maximo": [],  # Columna vacía
                "close_std": [],  # Columna vacía
                "volatilidad_retornos": [],  # Columna vacía
            }
        )
        tabla.write_csv(
            CARPETA_TABLAS / "tabla_close_por_regimen.csv"
        )  # Guarda tabla vacía
        return tabla  # Devuelve tabla vacía

    expresiones = [  # Lista de agregaciones por régimen
        pl.len().alias("filas"),  # Cuenta filas
        pl.col("close").mean().alias("close_promedio"),  # Media del close
        pl.col("close").min().alias("close_minimo"),  # Mínimo del close
        pl.col("close").max().alias("close_maximo"),  # Máximo del close
        pl.col("close").std().alias("close_std"),  # Desviación estándar del close
    ]

    if "log_return_1h" in gold.columns:  # Si existe log_return_1h
        expresiones.append(
            pl.col("log_return_1h").std().alias("volatilidad_retornos")
        )  # Agrega volatilidad
    else:  # Si no existe
        expresiones.append(pl.lit(None).alias("volatilidad_retornos"))  # Deja nulo

    tabla = (  # Construye la tabla final
        gold.group_by("regimen_mercado")  # Agrupa por régimen
        .agg(expresiones)  # Aplica agregaciones
        .sort("regimen_mercado")  # Ordena alfabéticamente
    )

    tabla.write_csv(CARPETA_TABLAS / "tabla_close_por_regimen.csv")  # Guarda la tabla
    return tabla  # Devuelve la tabla


def guardar_tabla_outliers_close_volume(
    gold: pl.DataFrame,
) -> pl.DataFrame:  # Detecta outliers con IQR
    q1_close = float(gold.select(pl.col("close").quantile(0.25)).item())  # Q1 del close
    q3_close = float(gold.select(pl.col("close").quantile(0.75)).item())  # Q3 del close
    iqr_close = q3_close - q1_close  # Rango intercuartílico del close
    lim_inf_close = q1_close - 1.5 * iqr_close  # Límite inferior close
    lim_sup_close = q3_close + 1.5 * iqr_close  # Límite superior close

    q1_vol = float(gold.select(pl.col("volume").quantile(0.25)).item())  # Q1 del volume
    q3_vol = float(gold.select(pl.col("volume").quantile(0.75)).item())  # Q3 del volume
    iqr_vol = q3_vol - q1_vol  # Rango intercuartílico del volume
    lim_inf_vol = q1_vol - 1.5 * iqr_vol  # Límite inferior volume
    lim_sup_vol = q3_vol + 1.5 * iqr_vol  # Límite superior volume

    out_close = gold.filter(  # Filtra outliers de close
        (pl.col("close") < lim_inf_close)
        | (pl.col("close") > lim_sup_close)  # Fuera de límites
    ).height  # Cuenta cuántos hay

    out_vol = gold.filter(  # Filtra outliers de volume
        (pl.col("volume") < lim_inf_vol)
        | (pl.col("volume") > lim_sup_vol)  # Fuera de límites
    ).height  # Cuenta cuántos hay

    tabla = pl.DataFrame(  # Construye tabla de outliers
        [
            {
                "variable": "close",  # Nombre de variable
                "q1": q1_close,  # Primer cuartil
                "q3": q3_close,  # Tercer cuartil
                "iqr": iqr_close,  # Rango intercuartílico
                "limite_inferior": lim_inf_close,  # Límite inferior
                "limite_superior": lim_sup_close,  # Límite superior
                "cantidad_outliers": out_close,  # Cantidad de outliers
            },
            {
                "variable": "volume",  # Nombre de variable
                "q1": q1_vol,  # Primer cuartil
                "q3": q3_vol,  # Tercer cuartil
                "iqr": iqr_vol,  # Rango intercuartílico
                "limite_inferior": lim_inf_vol,  # Límite inferior
                "limite_superior": lim_sup_vol,  # Límite superior
                "cantidad_outliers": out_vol,  # Cantidad de outliers
            },
        ]
    )

    tabla.write_csv(
        CARPETA_TABLAS / "tabla_outliers_close_volume.csv"
    )  # Guarda la tabla
    return tabla  # Devuelve la tabla


# =========================================================
# FUNCIONES DE GRAFICOS
# =========================================================
def grafico_serie_temporal_close(
    gold: pl.DataFrame,
) -> None:  # Gráfico de serie temporal del close
    x = gold["open_datetime_utc"].to_list()  # Convierte fechas a lista
    y = gold["close"].to_list()  # Convierte close a lista

    fig, ax = plt.subplots(figsize=(14, 5))  # Crea figura y eje
    ax.plot(x, y)  # Dibuja línea de tiempo
    ax.set_title("Serie temporal del precio de cierre (close) - BTCUSDT 1h")  # Título
    ax.set_xlabel("Tiempo")  # Etiqueta eje X
    ax.set_ylabel("Precio de cierre")  # Etiqueta eje Y
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Marca años en eje X
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Formato de año
    guardar_figura(fig, "fig_01_serie_temporal_close.png")  # Guarda la figura


def grafico_histograma_close(gold: pl.DataFrame) -> None:  # Histograma del close
    valores = gold["close"].to_numpy()  # Pasa close a arreglo numérico

    fig, ax = plt.subplots(figsize=(10, 5))  # Crea figura y eje
    ax.hist(valores, bins=60)  # Dibuja histograma
    ax.set_title("Distribución del precio de cierre (close)")  # Título
    ax.set_xlabel("Precio de cierre")  # Etiqueta eje X
    ax.set_ylabel("Frecuencia")  # Etiqueta eje Y
    guardar_figura(fig, "fig_02_histograma_close.png")  # Guarda la figura


def grafico_histograma_volume(gold: pl.DataFrame) -> None:  # Histograma del volumen
    valores = gold["volume"].to_numpy()  # Pasa volume a arreglo numérico

    fig, ax = plt.subplots(figsize=(10, 5))  # Crea figura y eje
    ax.hist(valores, bins=60)  # Dibuja histograma
    ax.set_title("Distribución del volumen")  # Título
    ax.set_xlabel("Volumen")  # Etiqueta eje X
    ax.set_ylabel("Frecuencia")  # Etiqueta eje Y
    guardar_figura(fig, "fig_03_histograma_volume.png")  # Guarda la figura


def grafico_close_promedio_mensual(
    mensual: pl.DataFrame,
) -> None:  # Línea del close promedio por mes
    x = mensual["open_datetime_utc"].to_list()  # Fechas mensuales
    y = mensual["close_promedio_mensual"].to_list()  # Valores promedio mensuales

    fig, ax = plt.subplots(figsize=(14, 5))  # Crea figura y eje
    ax.plot(x, y)  # Dibuja línea
    ax.set_title("Precio de cierre promedio mensual")  # Título
    ax.set_xlabel("Mes")  # Etiqueta eje X
    ax.set_ylabel("Close promedio mensual")  # Etiqueta eje Y
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Marca años
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Formato de año
    guardar_figura(fig, "fig_04_close_promedio_mensual.png")  # Guarda la figura


def grafico_top_maximos(
    top_maximos: pl.DataFrame,
) -> None:  # Barras de meses con mayor close promedio
    x = top_maximos["periodo"].to_list()  # Periodos
    y = top_maximos["close_promedio_mensual"].to_list()  # Valores

    fig, ax = plt.subplots(figsize=(12, 5))  # Crea figura y eje
    ax.bar(x, y)  # Dibuja barras
    ax.set_title("Top 10 meses con mayor precio de cierre promedio")  # Título
    ax.set_xlabel("Periodo")  # Etiqueta eje X
    ax.set_ylabel("Close promedio mensual")  # Etiqueta eje Y
    ax.tick_params(axis="x", rotation=45)  # Gira etiquetas del eje X
    guardar_figura(fig, "fig_05_top_10_meses_maximos.png")  # Guarda la figura


def grafico_top_minimos(
    top_minimos: pl.DataFrame,
) -> None:  # Barras de meses con menor close promedio
    x = top_minimos["periodo"].to_list()  # Periodos
    y = top_minimos["close_promedio_mensual"].to_list()  # Valores

    fig, ax = plt.subplots(figsize=(12, 5))  # Crea figura y eje
    ax.bar(x, y)  # Dibuja barras
    ax.set_title("Top 10 meses con menor precio de cierre promedio")  # Título
    ax.set_xlabel("Periodo")  # Etiqueta eje X
    ax.set_ylabel("Close promedio mensual")  # Etiqueta eje Y
    ax.tick_params(axis="x", rotation=45)  # Gira etiquetas del eje X
    guardar_figura(fig, "fig_06_top_10_meses_minimos.png")  # Guarda la figura


def grafico_top_estables(
    top_estables: pl.DataFrame,
) -> None:  # Barras de meses más estables
    x = top_estables["periodo"].to_list()  # Periodos
    y = top_estables["volatilidad_mensual"].to_list()  # Valores de volatilidad

    fig, ax = plt.subplots(figsize=(12, 5))  # Crea figura y eje
    ax.bar(x, y)  # Dibuja barras
    ax.set_title("Top 10 meses más estables según volatilidad mensual")  # Título
    ax.set_xlabel("Periodo")  # Etiqueta eje X
    ax.set_ylabel("Volatilidad mensual de log_return_1h")  # Etiqueta eje Y
    ax.tick_params(axis="x", rotation=45)  # Gira etiquetas del eje X
    guardar_figura(fig, "fig_07_top_10_meses_estables.png")  # Guarda la figura


def grafico_boxplot_close_por_regimen(
    gold: pl.DataFrame,
) -> None:  # Boxplot del close por régimen
    if "regimen_mercado" not in gold.columns:  # Si no existe la columna
        print("No existe 'regimen_mercado'. Se omite fig_08.")  # Muestra aviso
        return  # Sale de la función

    orden = ["pre_covid", "covid", "post_covid"]  # Orden lógico de regímenes
    datos = []  # Lista vacía para los grupos

    for regimen in orden:  # Recorre cada régimen
        serie = gold.filter(pl.col("regimen_mercado") == regimen)[
            "close"
        ].to_numpy()  # Toma close del régimen
        if serie.size > 0:  # Si sí hay datos
            datos.append(serie)  # Agrega la serie real
        else:  # Si no hay datos
            datos.append(np.array([np.nan]))  # Agrega un NaN para no romper el gráfico

    fig, ax = plt.subplots(figsize=(10, 5))  # Crea figura y eje
    ax.boxplot(datos, tick_labels=orden)  # Dibuja boxplot con tick_labels corregido
    ax.set_title("Boxplot del close por régimen de mercado")  # Título
    ax.set_xlabel("Régimen")  # Etiqueta eje X
    ax.set_ylabel("Precio de cierre")  # Etiqueta eje Y
    guardar_figura(fig, "fig_08_boxplot_close_por_regimen.png")  # Guarda la figura


def grafico_boxplot_retornos_por_regimen(
    gold: pl.DataFrame,
) -> None:  # Boxplot de log_return_1h por régimen
    if (
        "regimen_mercado" not in gold.columns or "log_return_1h" not in gold.columns
    ):  # Si faltan columnas
        print(
            "No existen 'regimen_mercado' o 'log_return_1h'. Se omite fig_09."
        )  # Muestra aviso
        return  # Sale de la función

    orden = ["pre_covid", "covid", "post_covid"]  # Orden lógico de regímenes
    datos = []  # Lista vacía para los grupos

    for regimen in orden:  # Recorre cada régimen
        serie = (  # Empieza extracción de serie
            gold.filter(pl.col("regimen_mercado") == regimen)  # Filtra por régimen
            .filter(pl.col("log_return_1h").is_not_null())[
                "log_return_1h"
            ]  # Toma solo retornos no nulos
            .to_numpy()  # Convierte a numpy
        )
        if serie.size > 0:  # Si sí hay datos
            datos.append(serie)  # Agrega la serie real
        else:  # Si no hay datos
            datos.append(np.array([np.nan]))  # Agrega un NaN para no romper el gráfico

    fig, ax = plt.subplots(figsize=(10, 5))  # Crea figura y eje
    ax.boxplot(datos, tick_labels=orden)  # Dibuja boxplot con tick_labels corregido
    ax.set_title("Boxplot de retornos logarítmicos por régimen")  # Título
    ax.set_xlabel("Régimen")  # Etiqueta eje X
    ax.set_ylabel("log_return_1h")  # Etiqueta eje Y
    guardar_figura(fig, "fig_09_boxplot_retornos_por_regimen.png")  # Guarda la figura


def grafico_matriz_correlacion(
    gold: pl.DataFrame,
) -> None:  # Matriz de correlación de variables
    columnas_base = [  # Lista base de columnas
        "open",  # Apertura
        "high",  # Máximo
        "low",  # Mínimo
        "close",  # Cierre
        "volume",  # Volumen
        "number_of_trades",  # Número de trades
        "quote_asset_volume",  # Volumen quote
        "taker_buy_base_volume",  # Compra base
        "taker_buy_quote_volume",  # Compra quote
    ]

    columnas = [
        c for c in columnas_base if c in gold.columns
    ]  # Deja solo las columnas que sí existen

    if len(columnas) < 2:  # Si hay menos de 2 columnas
        print(
            "No hay suficientes columnas numéricas para matriz de correlación."
        )  # Avisa
        return  # Sale de la función

    df_corr = gold.select(columnas).drop_nulls()  # Selecciona columnas y elimina nulos

    if df_corr.height < 2:  # Si hay muy pocas filas
        print(
            "No hay suficientes filas sin nulos para calcular correlaciones."
        )  # Avisa
        return  # Sale de la función

    matriz = df_corr.to_numpy()  # Convierte a matriz numpy
    corr = np.corrcoef(matriz, rowvar=False)  # Calcula correlación entre columnas

    tabla_corr = pl.DataFrame(  # Construye tabla de correlación
        {
            "variable": columnas,  # Primera columna con nombres
            **{
                col: corr[:, i] for i, col in enumerate(columnas)
            },  # Agrega cada columna de la matriz
        }
    )
    tabla_corr.write_csv(
        CARPETA_TABLAS / "tabla_matriz_correlacion_gold.csv"
    )  # Guarda la tabla

    fig, ax = plt.subplots(figsize=(10, 8))  # Crea figura y eje
    imagen = ax.imshow(corr, aspect="auto")  # Dibuja mapa de calor simple
    fig.colorbar(imagen)  # Agrega barra de color
    ax.set_xticks(range(len(columnas)))  # Define posiciones eje X
    ax.set_xticklabels(columnas, rotation=45, ha="right")  # Escribe nombres en eje X
    ax.set_yticks(range(len(columnas)))  # Define posiciones eje Y
    ax.set_yticklabels(columnas)  # Escribe nombres en eje Y
    ax.set_title("Matriz de correlación de variables del dataset Gold")  # Título
    guardar_figura(fig, "fig_10_matriz_correlacion_gold.png")  # Guarda la figura


def grafico_dispersion_close_volume(
    gold: pl.DataFrame,
) -> None:  # Dispersión close vs volume
    muestra = (  # Empieza muestra sistemática
        gold.with_row_index("indice_muestra")  # Crea un índice auxiliar
        .filter((pl.col("indice_muestra") % 50) == 0)  # Toma 1 de cada 50 filas
        .select(["close", "volume"])  # Deja solo close y volume
    )

    if muestra.height == 0:  # Si no hay datos
        print("No hay datos suficientes para la dispersión close vs volume.")  # Avisa
        return  # Sale de la función

    x = muestra["close"].to_numpy()  # Toma close como eje X
    y = muestra["volume"].to_numpy()  # Toma volume como eje Y

    fig, ax = plt.subplots(figsize=(10, 5))  # Crea figura y eje
    ax.scatter(x, y, s=8)  # Dibuja puntos
    ax.set_title("Diagrama de dispersión entre close y volume")  # Título
    ax.set_xlabel("Close")  # Etiqueta eje X
    ax.set_ylabel("Volume")  # Etiqueta eje Y
    guardar_figura(fig, "fig_11_dispersion_close_volume.png")  # Guarda la figura


# =========================================================
# FUNCION DE RESUMEN GENERAL
# =========================================================
def guardar_resumen_general(  # Genera un resumen narrativo en TXT
    silver: pl.DataFrame,  # DataFrame Silver
    gold: pl.DataFrame,  # DataFrame Gold
    tabla_resumen: pl.DataFrame,  # Tabla resumen general
    tabla_regimenes: pl.DataFrame,  # Tabla por régimen
    top_maximos: pl.DataFrame,  # Top máximos
    top_minimos: pl.DataFrame,  # Top mínimos
    top_estables: pl.DataFrame,  # Top estables
    tabla_outliers: pl.DataFrame,  # Tabla de outliers
) -> None:
    lineas: list[str] = []  # Lista vacía para escribir el texto

    lineas.append("RESUMEN GENERAL DEL ANALISIS DE RESULTADOS")  # Título
    lineas.append("=" * 70)  # Línea visual
    lineas.append(
        f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}"
    )  # Fecha actual
    lineas.append("")  # Línea en blanco

    lineas.append("1. ESTADO GENERAL DEL DATASET")  # Sección 1
    lineas.append(f"- Filas Silver 1m: {silver.height:,}")  # Filas Silver
    lineas.append(f"- Filas Gold 1h: {gold.height:,}")  # Filas Gold
    lineas.append(
        f"- Fecha mínima Silver: {silver['open_datetime_utc'].min()}"
    )  # Mínimo Silver
    lineas.append(
        f"- Fecha máxima Silver: {silver['open_datetime_utc'].max()}"
    )  # Máximo Silver
    lineas.append(
        f"- Fecha mínima Gold: {gold['open_datetime_utc'].min()}"
    )  # Mínimo Gold
    lineas.append(
        f"- Fecha máxima Gold: {gold['open_datetime_utc'].max()}"
    )  # Máximo Gold

    huecos_silver = tabla_resumen.filter(
        pl.col("dataset") == "silver_1m"
    )  # Filtra fila Silver
    if huecos_silver.height > 0:  # Si existe esa fila
        lineas.append(
            f"- Huecos temporales en Silver: {huecos_silver['huecos_temporales'][0]}"
        )  # Cantidad de huecos
        lineas.append(  # Agrega línea
            f"- Mayor hueco temporal en Silver (segundos): {huecos_silver['mayor_hueco_segundos'][0]}"  # Mayor hueco
        )
    lineas.append("")  # Línea en blanco

    lineas.append("2. PERIODOS DE MAXIMOS DEL CLOSE")  # Sección 2
    for fila in top_maximos.iter_rows(named=True):  # Recorre cada fila
        lineas.append(  # Agrega línea
            f"- {fila['periodo']}: close promedio mensual = {formatear_numero(fila['close_promedio_mensual'], 2)}"  # Texto
        )
    lineas.append("")  # Línea en blanco

    lineas.append("3. PERIODOS DE MINIMOS DEL CLOSE")  # Sección 3
    for fila in top_minimos.iter_rows(named=True):  # Recorre cada fila
        lineas.append(  # Agrega línea
            f"- {fila['periodo']}: close promedio mensual = {formatear_numero(fila['close_promedio_mensual'], 2)}"  # Texto
        )
    lineas.append("")  # Línea en blanco

    lineas.append("4. PERIODOS MAS ESTABLES")  # Sección 4
    for fila in top_estables.iter_rows(named=True):  # Recorre cada fila
        lineas.append(  # Agrega línea
            f"- {fila['periodo']}: volatilidad mensual = {formatear_numero(fila['volatilidad_mensual'], 6)}"  # Texto
        )
    lineas.append("")  # Línea en blanco

    lineas.append("5. COMPORTAMIENTO POR REGIMEN")  # Sección 5
    if tabla_regimenes.height > 0:  # Si hay datos por régimen
        for fila in tabla_regimenes.iter_rows(named=True):  # Recorre cada fila
            lineas.append(  # Agrega línea
                f"- {fila['regimen_mercado']}: "  # Empieza el texto
                f"filas={fila['filas']:,}, "  # Cantidad de filas
                f"close_promedio={formatear_numero(fila['close_promedio'], 2)}, "  # Media
                f"close_minimo={formatear_numero(fila['close_minimo'], 2)}, "  # Mínimo
                f"close_maximo={formatear_numero(fila['close_maximo'], 2)}, "  # Máximo
                f"volatilidad_retornos={formatear_numero(fila['volatilidad_retornos'], 6)}"  # Volatilidad
            )
    else:  # Si no hay datos por régimen
        lineas.append(
            "- No se encontró la columna regimen_mercado para resumir por régimen."
        )  # Avisa
    lineas.append("")  # Línea en blanco

    lineas.append("6. OUTLIERS DETECTADOS POR IQR")  # Sección 6
    for fila in tabla_outliers.iter_rows(named=True):  # Recorre cada fila
        lineas.append(  # Agrega línea
            f"- {fila['variable']}: "  # Nombre de variable
            f"Q1={formatear_numero(fila['q1'], 4)}, "  # Cuartil 1
            f"Q3={formatear_numero(fila['q3'], 4)}, "  # Cuartil 3
            f"IQR={formatear_numero(fila['iqr'], 4)}, "  # IQR
            f"outliers={fila['cantidad_outliers']}"  # Cantidad de outliers
        )
    lineas.append("")  # Línea en blanco

    lineas.append("7. INTERPRETACION GENERAL")  # Sección 7
    lineas.append(
        "- Silver permite verificar continuidad temporal, estructura del dataset y calidad básica de la ingesta."
    )  # Interpretación
    lineas.append(
        "- Gold permite analizar el comportamiento agregado de BTCUSDT en 1 hora, útil para modelado de series temporales."
    )  # Interpretación
    lineas.append(
        "- El análisis mensual del close ayuda a identificar fases de expansión, contracción y estabilidad relativa del mercado."
    )  # Interpretación
    lineas.append(
        "- La segmentación por regímenes pre_covid, covid y post_covid facilita comparar el comportamiento del precio y la volatilidad en distintos contextos históricos."
    )  # Interpretación
    lineas.append(
        "- Los outliers en precio y volumen son esperables en Bitcoin y no siempre significan error de datos."
    )  # Interpretación
    lineas.append("=" * 70)  # Línea visual final

    RUTA_RESUMEN_GENERAL.write_text(
        "\n".join(lineas), encoding="utf-8"
    )  # Guarda el TXT


# =========================================================
# FUNCION PRINCIPAL
# =========================================================
def main() -> None:  # Función principal del script
    preparar_carpetas()  # Crea carpetas de salida

    silver, gold = cargar_datasets()  # Carga Silver y Gold

    print("Generando tablas del estado del dataset...")  # Mensaje de avance
    tabla_resumen = guardar_tabla_resumen_dataset(silver, gold)  # Crea tabla resumen
    _tabla_variables = guardar_tabla_estructura_variables(
        silver
    )  # Guarda estructura de variables
    _tabla_nulos_silver = guardar_tabla_nulos(
        silver, "tabla_nulos_silver.csv"
    )  # Guarda nulos de Silver
    _tabla_nulos_gold = guardar_tabla_nulos(
        gold, "tabla_nulos_gold.csv"
    )  # Guarda nulos de Gold
    _tabla_descriptivos = guardar_descriptivos_gold(gold)  # Guarda descriptivos de Gold

    print("Generando análisis mensual del close...")  # Mensaje de avance
    mensual = construir_tabla_mensual_close(gold)  # Construye tabla mensual
    top_maximos, top_minimos, top_estables = guardar_top_maximos_minimos_estables(
        mensual
    )  # Guarda tops
    tabla_regimenes = guardar_tabla_regimenes(gold)  # Guarda tabla por régimen
    tabla_outliers = guardar_tabla_outliers_close_volume(gold)  # Guarda outliers

    print("Generando gráficas...")  # Mensaje de avance
    grafico_serie_temporal_close(gold)  # Genera figura 1
    grafico_histograma_close(gold)  # Genera figura 2
    grafico_histograma_volume(gold)  # Genera figura 3
    grafico_close_promedio_mensual(mensual)  # Genera figura 4
    grafico_top_maximos(top_maximos)  # Genera figura 5
    grafico_top_minimos(top_minimos)  # Genera figura 6
    grafico_top_estables(top_estables)  # Genera figura 7
    grafico_boxplot_close_por_regimen(gold)  # Genera figura 8
    grafico_boxplot_retornos_por_regimen(gold)  # Genera figura 9
    grafico_matriz_correlacion(gold)  # Genera figura 10
    grafico_dispersion_close_volume(gold)  # Genera figura 11

    guardar_resumen_general(  # Genera el resumen narrativo
        silver=silver,  # Pasa Silver
        gold=gold,  # Pasa Gold
        tabla_resumen=tabla_resumen,  # Pasa tabla resumen
        tabla_regimenes=tabla_regimenes,  # Pasa tabla por régimen
        top_maximos=top_maximos,  # Pasa top máximos
        top_minimos=top_minimos,  # Pasa top mínimos
        top_estables=top_estables,  # Pasa top estables
        tabla_outliers=tabla_outliers,  # Pasa tabla de outliers
    )

    print("=" * 70)  # Línea visual
    print("ANALISIS TERMINADO")  # Mensaje final
    print(f"Tablas guardadas en:  {CARPETA_TABLAS}")  # Ruta de tablas
    print(f"Figuras guardadas en: {CARPETA_FIGURAS}")  # Ruta de figuras
    print(f"Resumen general:      {RUTA_RESUMEN_GENERAL}")  # Ruta del resumen
    print("=" * 70)  # Línea visual


if __name__ == "__main__":  # Ejecuta solo si el archivo se corre directamente
    main()  # Llama a la función principal
