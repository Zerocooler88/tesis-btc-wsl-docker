from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
PROYECTO_ROOT = Path(__file__).resolve().parent.parent

RUTA_BRONZE = (
    PROYECTO_ROOT
    / "data"
    / "bronze"
    / "spot"
    / "BTCUSDT"
    / "1m"
    / "btcusdt_spot_1m_bronze_2019_2026.parquet"
)

RUTA_SILVER = (
    PROYECTO_ROOT
    / "data"
    / "silver"
    / "spot"
    / "BTCUSDT"
    / "1m"
    / "btcusdt_spot_1m_silver_2019_2026.parquet"
)

CARPETA_REPORTES = PROYECTO_ROOT / "reports" / "analisis_ruido_1m"
CARPETA_TABLAS = CARPETA_REPORTES / "tablas"
CARPETA_FIGURAS = CARPETA_REPORTES / "figuras"

CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)

RUTA_RESUMEN = CARPETA_REPORTES / "resumen_analisis_ruido_1m.txt"

UMBRAL_MICROSEGUNDOS = 100_000_000_000_000


# =========================================================
# FUNCIONES BASICAS
# =========================================================
def validar_archivo_existe(ruta: Path, nombre_logico: str) -> None:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo {nombre_logico}: {ruta}")


def guardar_figura(nombre_archivo: str) -> None:
    plt.tight_layout()
    plt.savefig(CARPETA_FIGURAS / nombre_archivo, dpi=200, bbox_inches="tight")
    plt.close()


def preparar_dataframe_1m(df: pl.DataFrame, nombre_dataset: str) -> pl.DataFrame:
    columnas_float = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    columnas_int = [
        "open_time",
        "close_time",
        "number_of_trades",
    ]

    exprs = []

    for col in columnas_float:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False))

    for col in columnas_int:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Int64, strict=False))

    if exprs:
        df = df.with_columns(exprs)

    if "open_datetime_utc" not in df.columns and "open_time" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("open_time") >= UMBRAL_MICROSEGUNDOS)
            .then(pl.from_epoch(pl.col("open_time"), time_unit="us"))
            .otherwise(pl.from_epoch(pl.col("open_time"), time_unit="ms"))
            .alias("open_datetime_utc")
        )

    if "close_datetime_utc" not in df.columns and "close_time" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("close_time") >= UMBRAL_MICROSEGUNDOS)
            .then(pl.from_epoch(pl.col("close_time"), time_unit="us"))
            .otherwise(pl.from_epoch(pl.col("close_time"), time_unit="ms"))
            .alias("close_datetime_utc")
        )

    if "open_datetime_utc" not in df.columns:
        raise ValueError(
            f"{nombre_dataset}: no existe open_datetime_utc ni open_time para construirla"
        )

    df = df.sort("open_datetime_utc")

    df = df.with_columns(
        [
            pl.col("close").diff().alias("close_diff_1m"),
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1m"),
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return_1m"),
            pl.col("close").log().diff().alias("log_diff_close_1m"),
            (pl.col("high") - pl.col("low")).alias("rango_hl_abs"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("rango_hl_pct"),
            (pl.col("close") - pl.col("open")).abs().alias("cuerpo_abs"),
            ((pl.col("close") - pl.col("open")).abs() / pl.col("open")).alias(
                "cuerpo_pct"
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.col("log_return_1m").abs().alias("abs_log_return_1m"),
            pl.col("log_return_1m")
            .rolling_std(window_size=60)
            .alias("volatilidad_rolling_60m"),
        ]
    )

    return df


def cargar_datasets() -> tuple[pl.DataFrame, pl.DataFrame]:
    validar_archivo_existe(RUTA_BRONZE, "Bronze")
    validar_archivo_existe(RUTA_SILVER, "Silver")

    print("=" * 70)
    print("INICIO DEL ANALISIS DE RUIDO Y VALORES ATIPICOS EN 1 MINUTO")
    print(f"Leyendo Bronze desde: {RUTA_BRONZE}")
    print(f"Leyendo Silver desde: {RUTA_SILVER}")

    bronze = pl.read_parquet(RUTA_BRONZE)
    silver = pl.read_parquet(RUTA_SILVER)

    bronze = preparar_dataframe_1m(bronze, "Bronze")
    silver = preparar_dataframe_1m(silver, "Silver")

    print(f"Filas Bronze: {bronze.height:,}")
    print(f"Filas Silver: {silver.height:,}")
    print("=" * 70)

    return bronze, silver


# =========================================================
# RESUMEN DE ESTADO DEL DATASET
# =========================================================
def calcular_huecos(df: pl.DataFrame) -> tuple[int, int | None]:
    aux = df.select(
        [
            pl.col("open_datetime_utc"),
            pl.col("open_datetime_utc").diff().alias("delta_tiempo"),
        ]
    )

    huecos = aux.filter(
        pl.col("delta_tiempo").is_not_null()
        & (pl.col("delta_tiempo") != pl.duration(minutes=1))
    )

    cantidad_huecos = huecos.height

    if cantidad_huecos > 0:
        max_segundos = int(
            huecos.select(pl.col("delta_tiempo").dt.total_seconds().max()).item()
        )
    else:
        max_segundos = None

    return cantidad_huecos, max_segundos


def construir_tabla_resumen_estado(
    bronze: pl.DataFrame, silver: pl.DataFrame
) -> pl.DataFrame:
    dup_bronze = bronze.select(
        (pl.len() - pl.col("open_datetime_utc").n_unique()).alias("duplicados")
    ).item()
    dup_silver = silver.select(
        (pl.len() - pl.col("open_datetime_utc").n_unique()).alias("duplicados")
    ).item()

    huecos_bronze, max_hueco_bronze = calcular_huecos(bronze)
    huecos_silver, max_hueco_silver = calcular_huecos(silver)

    tabla = pl.DataFrame(
        [
            {
                "dataset": "bronze_1m",
                "filas": bronze.height,
                "columnas": bronze.width,
                "fecha_minima": str(bronze["open_datetime_utc"].min()),
                "fecha_maxima": str(bronze["open_datetime_utc"].max()),
                "duplicados_por_minuto": int(dup_bronze),
                "huecos_temporales": huecos_bronze,
                "mayor_hueco_segundos": max_hueco_bronze,
            },
            {
                "dataset": "silver_1m",
                "filas": silver.height,
                "columnas": silver.width,
                "fecha_minima": str(silver["open_datetime_utc"].min()),
                "fecha_maxima": str(silver["open_datetime_utc"].max()),
                "duplicados_por_minuto": int(dup_silver),
                "huecos_temporales": huecos_silver,
                "mayor_hueco_segundos": max_hueco_silver,
            },
        ]
    )

    tabla.write_csv(CARPETA_TABLAS / "tabla_resumen_estado_1m.csv")
    return tabla


def construir_tabla_nulos(df: pl.DataFrame, nombre: str) -> pl.DataFrame:
    filas = []

    for columna in df.columns:
        nulos = df.select(pl.col(columna).is_null().sum()).item()
        filas.append(
            {
                "dataset": nombre,
                "variable": columna,
                "nulos": int(nulos),
            }
        )

    tabla = pl.DataFrame(filas)
    tabla.write_csv(CARPETA_TABLAS / f"tabla_nulos_{nombre}.csv")
    return tabla


def construir_tabla_descriptivos(df: pl.DataFrame, nombre: str) -> pl.DataFrame:
    columnas = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "number_of_trades",
        "quote_asset_volume",
        "rango_hl_abs",
        "rango_hl_pct",
        "cuerpo_abs",
        "cuerpo_pct",
        "return_1m",
        "log_return_1m",
        "abs_log_return_1m",
        "volatilidad_rolling_60m",
    ]
    columnas = [c for c in columnas if c in df.columns]

    tabla = df.select(columnas).describe()
    tabla.write_csv(CARPETA_TABLAS / f"tabla_descriptivos_{nombre}.csv")
    return tabla


# =========================================================
# METRICAS DE RUIDO
# =========================================================
def calcular_metricas_ruido(df: pl.DataFrame, nombre: str) -> dict:
    p99_abs_return = df.select(pl.col("abs_log_return_1m").quantile(0.99)).item()
    p999_abs_return = df.select(pl.col("abs_log_return_1m").quantile(0.999)).item()
    p99_hl_pct = df.select(pl.col("rango_hl_pct").quantile(0.99)).item()

    metricas = {
        "dataset": nombre,
        "media_abs_log_return_1m": df.select(pl.col("abs_log_return_1m").mean()).item(),
        "mediana_abs_log_return_1m": df.select(
            pl.col("abs_log_return_1m").median()
        ).item(),
        "std_log_return_1m": df.select(pl.col("log_return_1m").std()).item(),
        "p99_abs_log_return_1m": p99_abs_return,
        "p999_abs_log_return_1m": p999_abs_return,
        "media_rango_hl_pct": df.select(pl.col("rango_hl_pct").mean()).item(),
        "p99_rango_hl_pct": p99_hl_pct,
        "media_volatilidad_rolling_60m": df.select(
            pl.col("volatilidad_rolling_60m").mean()
        ).item(),
    }

    return metricas


def construir_tabla_metricas_ruido(
    bronze: pl.DataFrame, silver: pl.DataFrame
) -> pl.DataFrame:
    tabla = pl.DataFrame(
        [
            calcular_metricas_ruido(bronze, "bronze_1m"),
            calcular_metricas_ruido(silver, "silver_1m"),
        ]
    )
    tabla.write_csv(CARPETA_TABLAS / "tabla_metricas_ruido_1m.csv")
    return tabla


def calcular_outliers_iqr(df: pl.DataFrame, columna: str) -> dict:
    serie = df.select(pl.col(columna)).drop_nulls()
    q1 = serie.select(pl.col(columna).quantile(0.25)).item()
    q3 = serie.select(pl.col(columna).quantile(0.75)).item()
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr

    cantidad = df.filter(
        (pl.col(columna) < lim_inf) | (pl.col(columna) > lim_sup)
    ).height

    porcentaje = (cantidad / df.height) * 100 if df.height > 0 else 0.0

    return {
        "variable": columna,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": lim_inf,
        "limite_superior": lim_sup,
        "cantidad_outliers": cantidad,
        "porcentaje_outliers": porcentaje,
    }


def construir_tabla_outliers(
    bronze: pl.DataFrame, silver: pl.DataFrame
) -> pl.DataFrame:
    variables = ["volume", "rango_hl_pct", "log_return_1m", "abs_log_return_1m"]
    filas = []

    for dataset_nombre, df in [("bronze_1m", bronze), ("silver_1m", silver)]:
        for variable in variables:
            fila = calcular_outliers_iqr(df, variable)
            fila["dataset"] = dataset_nombre
            filas.append(fila)

    tabla = pl.DataFrame(filas).select(
        [
            "dataset",
            "variable",
            "q1",
            "q3",
            "iqr",
            "limite_inferior",
            "limite_superior",
            "cantidad_outliers",
            "porcentaje_outliers",
        ]
    )
    tabla.write_csv(CARPETA_TABLAS / "tabla_outliers_ruido_1m.csv")
    return tabla


# =========================================================
# TABLAS MENSUALES DE RUIDO
# =========================================================
def construir_tabla_ruido_mensual(bronze: pl.DataFrame) -> pl.DataFrame:
    mensual = (
        bronze.group_by_dynamic(
            index_column="open_datetime_utc",
            every="1mo",
            period="1mo",
            label="left",
            closed="left",
            start_by="window",
        )
        .agg(
            [
                pl.col("close").mean().alias("close_promedio_mensual"),
                pl.col("log_return_1m").std().alias("volatilidad_1m_mensual"),
                pl.col("abs_log_return_1m").mean().alias("ruido_abs_promedio_mensual"),
                pl.col("rango_hl_pct").mean().alias("rango_hl_pct_promedio_mensual"),
                pl.col("volume").mean().alias("volume_promedio_mensual"),
                pl.len().alias("filas_mes"),
            ]
        )
        .sort("open_datetime_utc")
        .with_columns(pl.col("open_datetime_utc").dt.strftime("%Y-%m").alias("periodo"))
    )

    mensual.write_csv(CARPETA_TABLAS / "tabla_ruido_mensual_bronze.csv")
    return mensual


def construir_top_meses_ruidosos_y_estables(
    mensual: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    top_ruidosos = (
        mensual.filter(pl.col("volatilidad_1m_mensual").is_not_null())
        .sort("volatilidad_1m_mensual", descending=True)
        .select(
            [
                "periodo",
                "close_promedio_mensual",
                "volatilidad_1m_mensual",
                "ruido_abs_promedio_mensual",
                "rango_hl_pct_promedio_mensual",
                "filas_mes",
            ]
        )
        .head(10)
    )

    top_estables = (
        mensual.filter(pl.col("volatilidad_1m_mensual").is_not_null())
        .sort("volatilidad_1m_mensual", descending=False)
        .select(
            [
                "periodo",
                "close_promedio_mensual",
                "volatilidad_1m_mensual",
                "ruido_abs_promedio_mensual",
                "rango_hl_pct_promedio_mensual",
                "filas_mes",
            ]
        )
        .head(10)
    )

    top_ruidosos.write_csv(CARPETA_TABLAS / "tabla_top_10_meses_mas_ruidosos_1m.csv")
    top_estables.write_csv(CARPETA_TABLAS / "tabla_top_10_meses_mas_estables_1m.csv")
    return top_ruidosos, top_estables


# =========================================================
# GRAFICOS
# =========================================================
def grafico_close_1m_muestreado(bronze: pl.DataFrame) -> None:
    muestra = (
        bronze.with_row_index("idx")
        .filter((pl.col("idx") % 240) == 0)
        .select(["open_datetime_utc", "close"])
    )

    x = muestra["open_datetime_utc"].to_list()
    y = muestra["close"].to_list()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y)
    ax.set_title("Serie temporal del close en Bronze 1m (muestreo cada 240 minutos)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Precio de cierre")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    guardar_figura("fig_01_serie_close_bronze_1m_muestreada.png")


def grafico_hist_close(bronze: pl.DataFrame) -> None:
    valores = bronze["close"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valores, bins=60)
    ax.set_title("Distribución del precio de cierre en Bronze 1m")
    ax.set_xlabel("Close")
    ax.set_ylabel("Frecuencia")
    guardar_figura("fig_02_hist_close_bronze_1m.png")


def grafico_hist_volume(bronze: pl.DataFrame) -> None:
    valores = bronze["volume"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valores, bins=60)
    ax.set_title("Distribución del volumen en Bronze 1m")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Frecuencia")
    guardar_figura("fig_03_hist_volume_bronze_1m.png")


def grafico_hist_log_return(bronze: pl.DataFrame) -> None:
    valores = bronze.filter(pl.col("log_return_1m").is_not_null())[
        "log_return_1m"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valores, bins=80)
    ax.set_title("Distribución de log_return_1m en Bronze 1m")
    ax.set_xlabel("log_return_1m")
    ax.set_ylabel("Frecuencia")
    guardar_figura("fig_04_hist_log_return_bronze_1m.png")


def grafico_boxplot_log_return(bronze: pl.DataFrame) -> None:
    valores = bronze.filter(pl.col("log_return_1m").is_not_null())[
        "log_return_1m"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([valores], tick_labels=["log_return_1m"])
    ax.set_title("Boxplot de log_return_1m en Bronze 1m")
    ax.set_ylabel("log_return_1m")
    guardar_figura("fig_05_boxplot_log_return_bronze_1m.png")


def grafico_ruido_mensual(mensual: pl.DataFrame) -> None:
    x = mensual["open_datetime_utc"].to_list()
    y = mensual["volatilidad_1m_mensual"].to_list()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y)
    ax.set_title("Volatilidad mensual de log_return_1m en Bronze 1m")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Volatilidad mensual 1m")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    guardar_figura("fig_06_volatilidad_mensual_bronze_1m.png")


def grafico_top_ruidosos(top_ruidosos: pl.DataFrame) -> None:
    x = top_ruidosos["periodo"].to_list()
    y = top_ruidosos["volatilidad_1m_mensual"].to_list()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, y)
    ax.set_title("Top 10 meses más ruidosos según volatilidad 1m mensual")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Volatilidad 1m mensual")
    plt.xticks(rotation=45)
    guardar_figura("fig_07_top_10_meses_mas_ruidosos_1m.png")


def grafico_top_estables(top_estables: pl.DataFrame) -> None:
    x = top_estables["periodo"].to_list()
    y = top_estables["volatilidad_1m_mensual"].to_list()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, y)
    ax.set_title("Top 10 meses más estables en 1m según volatilidad mensual")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Volatilidad 1m mensual")
    plt.xticks(rotation=45)
    guardar_figura("fig_08_top_10_meses_mas_estables_1m.png")


def grafico_boxplot_rango_hl_pct(bronze: pl.DataFrame) -> None:
    valores = bronze.filter(pl.col("rango_hl_pct").is_not_null())[
        "rango_hl_pct"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([valores], tick_labels=["rango_hl_pct"])
    ax.set_title("Boxplot del rango relativo intravela en Bronze 1m")
    ax.set_ylabel("rango_hl_pct")
    guardar_figura("fig_09_boxplot_rango_hl_pct_bronze_1m.png")


def grafico_dispersion_abs_return_vs_volume(bronze: pl.DataFrame) -> None:
    muestra = (
        bronze.with_row_index("idx")
        .filter((pl.col("idx") % 120) == 0)
        .select(["abs_log_return_1m", "volume"])
        .drop_nulls()
    )

    x = muestra["abs_log_return_1m"].to_numpy()
    y = muestra["volume"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=8)
    ax.set_title("Dispersión entre ruido absoluto 1m y volumen en Bronze")
    ax.set_xlabel("abs_log_return_1m")
    ax.set_ylabel("Volume")
    guardar_figura("fig_10_dispersion_abs_return_vs_volume_bronze_1m.png")


def grafico_volatilidad_rolling(bronze: pl.DataFrame) -> None:
    muestra = (
        bronze.with_row_index("idx")
        .filter((pl.col("idx") % 240) == 0)
        .select(["open_datetime_utc", "volatilidad_rolling_60m"])
        .drop_nulls()
    )

    x = muestra["open_datetime_utc"].to_list()
    y = muestra["volatilidad_rolling_60m"].to_list()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y)
    ax.set_title("Volatilidad rolling de 60 minutos en Bronze 1m")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Volatilidad rolling 60m")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    guardar_figura("fig_11_volatilidad_rolling_60m_bronze_1m.png")


# =========================================================
# RESUMEN NARRATIVO
# =========================================================
def guardar_resumen_narrativo(
    tabla_estado: pl.DataFrame,
    tabla_metricas_ruido: pl.DataFrame,
    tabla_outliers: pl.DataFrame,
    top_ruidosos: pl.DataFrame,
    top_estables: pl.DataFrame,
) -> None:
    estado_bronze = tabla_estado.filter(pl.col("dataset") == "bronze_1m").to_dicts()[0]
    estado_silver = tabla_estado.filter(pl.col("dataset") == "silver_1m").to_dicts()[0]

    ruido_bronze = tabla_metricas_ruido.filter(
        pl.col("dataset") == "bronze_1m"
    ).to_dicts()[0]
    ruido_silver = tabla_metricas_ruido.filter(
        pl.col("dataset") == "silver_1m"
    ).to_dicts()[0]

    out_bronze = tabla_outliers.filter(pl.col("dataset") == "bronze_1m")
    out_silver = tabla_outliers.filter(pl.col("dataset") == "silver_1m")

    lineas = []
    lineas.append("RESUMEN DEL ANALISIS DE RUIDO Y VALORES ATIPICOS EN 1 MINUTO")
    lineas.append("=" * 70)
    lineas.append(f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}")
    lineas.append("")
    lineas.append("1. ESTADO INICIAL DEL DATASET")
    lineas.append(
        f"- Bronze 1m: {estado_bronze['filas']:,} filas, desde {estado_bronze['fecha_minima']} hasta {estado_bronze['fecha_maxima']}."
    )
    lineas.append(
        f"- Silver 1m: {estado_silver['filas']:,} filas, desde {estado_silver['fecha_minima']} hasta {estado_silver['fecha_maxima']}."
    )
    lineas.append(
        f"- Huecos temporales Bronze: {estado_bronze['huecos_temporales']:,}; mayor hueco: {estado_bronze['mayor_hueco_segundos']} segundos."
    )
    lineas.append(
        f"- Huecos temporales Silver: {estado_silver['huecos_temporales']:,}; mayor hueco: {estado_silver['mayor_hueco_segundos']} segundos."
    )
    lineas.append("")
    lineas.append("2. DIAGNOSTICO DE RUIDO DE ALTA FRECUENCIA")
    lineas.append(
        f"- Bronze presenta una media de abs_log_return_1m de {ruido_bronze['media_abs_log_return_1m']:.6f} y una desviación estándar de log_return_1m de {ruido_bronze['std_log_return_1m']:.6f}."
    )
    lineas.append(
        f"- Silver presenta una media de abs_log_return_1m de {ruido_silver['media_abs_log_return_1m']:.6f} y una desviación estándar de log_return_1m de {ruido_silver['std_log_return_1m']:.6f}."
    )
    lineas.append(
        f"- El percentil 99 del ruido absoluto en Bronze es {ruido_bronze['p99_abs_log_return_1m']:.6f}, lo que confirma la existencia de movimientos extremos de corta duración."
    )
    lineas.append(
        f"- El percentil 99 del rango relativo intravela en Bronze es {ruido_bronze['p99_rango_hl_pct']:.6f}, lo que sugiere oscilaciones relevantes dentro de una misma vela de 1 minuto."
    )
    lineas.append("")
    lineas.append("3. VALORES ATIPICOS")
    lineas.append("- Bronze 1m:")
    for fila in out_bronze.iter_rows(named=True):
        lineas.append(
            f"  - {fila['variable']}: {fila['cantidad_outliers']:,} outliers ({fila['porcentaje_outliers']:.2f}%)."
        )
    lineas.append("- Silver 1m:")
    for fila in out_silver.iter_rows(named=True):
        lineas.append(
            f"  - {fila['variable']}: {fila['cantidad_outliers']:,} outliers ({fila['porcentaje_outliers']:.2f}%)."
        )
    lineas.append("")
    lineas.append("4. MESES MAS RUIDOSOS")
    for fila in top_ruidosos.iter_rows(named=True):
        lineas.append(
            f"- {fila['periodo']}: volatilidad_1m_mensual={fila['volatilidad_1m_mensual']:.6f}, ruido_abs_promedio={fila['ruido_abs_promedio_mensual']:.6f}"
        )
    lineas.append("")
    lineas.append("5. MESES MAS ESTABLES")
    for fila in top_estables.iter_rows(named=True):
        lineas.append(
            f"- {fila['periodo']}: volatilidad_1m_mensual={fila['volatilidad_1m_mensual']:.6f}, ruido_abs_promedio={fila['ruido_abs_promedio_mensual']:.6f}"
        )
    lineas.append("")
    lineas.append("6. INTERPRETACION INICIAL")
    lineas.append(
        "- El dataset en frecuencia de 1 minuto presenta heterocedasticidad, colas pesadas y presencia de valores extremos, rasgos coherentes con un mercado financiero altamente volátil."
    )
    lineas.append(
        "- La distribución del volumen exhibe asimetría positiva marcada, lo que indica concentración de observaciones en valores bajos y episodios puntuales de actividad extraordinaria."
    )
    lineas.append(
        "- La distribución de los retornos de 1 minuto y del rango relativo intravela confirma que el nivel de ruido de alta frecuencia es elevado, por lo que el uso directo de la serie 1m para entrenamiento puede dificultar el aprendizaje estable de los modelos."
    )
    lineas.append(
        "- En consecuencia, el diagnóstico inicial respalda la conveniencia metodológica de transformar la serie a una frecuencia analítica más agregada, como 1 hora, para reducir ruido y mejorar la estabilidad del modelado."
    )
    lineas.append("=" * 70)

    RUTA_RESUMEN.write_text("\n".join(lineas), encoding="utf-8")


# =========================================================
# FUNCION PRINCIPAL
# =========================================================
def main() -> None:
    bronze, silver = cargar_datasets()

    print("Generando tablas del estado del dataset...")
    tabla_estado = construir_tabla_resumen_estado(bronze, silver)
    tabla_nulos_bronze = construir_tabla_nulos(bronze, "bronze_1m")
    tabla_nulos_silver = construir_tabla_nulos(silver, "silver_1m")
    tabla_desc_bronze = construir_tabla_descriptivos(bronze, "bronze_1m")
    tabla_desc_silver = construir_tabla_descriptivos(silver, "silver_1m")

    print("Generando métricas de ruido y outliers...")
    tabla_metricas_ruido = construir_tabla_metricas_ruido(bronze, silver)
    tabla_outliers = construir_tabla_outliers(bronze, silver)
    tabla_ruido_mensual = construir_tabla_ruido_mensual(bronze)
    top_ruidosos, top_estables = construir_top_meses_ruidosos_y_estables(
        tabla_ruido_mensual
    )

    print("Generando gráficas...")
    grafico_close_1m_muestreado(bronze)
    grafico_hist_close(bronze)
    grafico_hist_volume(bronze)
    grafico_hist_log_return(bronze)
    grafico_boxplot_log_return(bronze)
    grafico_ruido_mensual(tabla_ruido_mensual)
    grafico_top_ruidosos(top_ruidosos)
    grafico_top_estables(top_estables)
    grafico_boxplot_rango_hl_pct(bronze)
    grafico_dispersion_abs_return_vs_volume(bronze)
    grafico_volatilidad_rolling(bronze)

    guardar_resumen_narrativo(
        tabla_estado=tabla_estado,
        tabla_metricas_ruido=tabla_metricas_ruido,
        tabla_outliers=tabla_outliers,
        top_ruidosos=top_ruidosos,
        top_estables=top_estables,
    )

    _ = tabla_nulos_bronze
    _ = tabla_nulos_silver
    _ = tabla_desc_bronze
    _ = tabla_desc_silver

    print("=" * 70)
    print("ANALISIS DE RUIDO TERMINADO")
    print(f"Tablas guardadas en:  {CARPETA_TABLAS}")
    print(f"Figuras guardadas en: {CARPETA_FIGURAS}")
    print(f"Resumen guardado en:  {RUTA_RESUMEN}")
    print("=" * 70)


if __name__ == "__main__":
    main()
