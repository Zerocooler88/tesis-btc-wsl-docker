from __future__ import (
    annotations,
)  # Permite usar anotaciones de tipos modernas de forma más flexible

import csv  # Sirve para escribir el archivo de log en formato CSV
import hashlib  # Sirve para calcular hashes SHA256 y verificar integridad
import time  # Sirve para pausar reintentos entre descargas
import zipfile  # Sirve para abrir y extraer archivos ZIP
from pathlib import (
    Path,
)  # Sirve para manejar rutas de archivos y carpetas de forma ordenada
from datetime import datetime  # Sirve para registrar fecha y hora actual en el log
from urllib.request import (
    Request,
    urlopen,
)  # Sirve para hacer solicitudes HTTP y descargar archivos
from urllib.error import (
    HTTPError,
    URLError,
)  # Sirve para capturar errores HTTP y de conexión

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
SYMBOL = "BTCUSDT"  # Define el símbolo del mercado que vas a descargar
INTERVAL = "1m"  # Define el intervalo de velas: 1 minuto

START_YEAR = 2019  # Año inicial de descarga
START_MONTH = 1  # Mes inicial de descarga
END_YEAR = 2026  # Año final de descarga
END_MONTH = 2  # Mes final de descarga

BASE_URL = f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}"  # URL base del repositorio mensual público de Binance

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)  # Obtiene la carpeta raíz del proyecto a partir de este script
ZIP_DIR = (
    PROJECT_ROOT / "data" / "raw_zips" / "spot" / SYMBOL / INTERVAL
)  # Carpeta donde se guardarán los ZIP descargados
CSV_DIR = (
    PROJECT_ROOT / "data" / "raw_csv" / "spot" / SYMBOL / INTERVAL
)  # Carpeta donde se extraerán los CSV
LOG_DIR = PROJECT_ROOT / "logs"  # Carpeta donde se guardará el log del proceso

ZIP_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de ZIP si no existe
CSV_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de CSV si no existe
LOG_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de logs si no existe

LOG_FILE = (
    LOG_DIR / "download_btcusdt_1m_monthly_2019_2026.csv"
)  # Ruta completa del archivo CSV de log

RETRIES = 3  # Número máximo de intentos de descarga por archivo
SLEEP_SECONDS = 2  # Segundos de espera entre reintentos
TIMEOUT = 60  # Tiempo máximo de espera de la conexión HTTP en segundos


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def month_range(
    start_year: int, start_month: int, end_year: int, end_month: int
):  # Genera todos los meses entre una fecha inicial y una final
    year, month = start_year, start_month  # Inicializa el año y mes actuales
    while (year, month) <= (
        end_year,
        end_month,
    ):  # Repite mientras no se pase del mes final
        yield year, month  # Devuelve el año y mes actual uno por uno
        month += 1  # Avanza al siguiente mes
        if month > 12:  # Si el mes pasa de diciembre
            month = 1  # Reinicia el mes a enero
            year += 1  # Avanza al siguiente año


def total_months(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> int:  # Calcula cuántos meses hay en total en el rango
    return (end_year - start_year) * 12 + (end_month - start_month) + 1  # Devuelve el total de periodos mensuales


def build_zip_filename(
    year: int, month: int
) -> str:  # Construye el nombre del archivo ZIP mensual
    return f"{SYMBOL}-{INTERVAL}-{year}-{month:02d}.zip"  # Devuelve un nombre tipo BTCUSDT-1m-2019-01.zip


def build_checksum_filename(
    year: int, month: int
) -> str:  # Construye el nombre del archivo CHECKSUM
    return f"{build_zip_filename(year, month)}.CHECKSUM"  # Devuelve un nombre tipo BTCUSDT-1m-2019-01.zip.CHECKSUM


def build_zip_url(
    year: int, month: int
) -> str:  # Construye la URL completa del ZIP mensual
    return f"{BASE_URL}/{build_zip_filename(year, month)}"  # Une la URL base con el nombre del ZIP


def build_checksum_url(
    year: int, month: int
) -> str:  # Construye la URL completa del archivo CHECKSUM
    return f"{BASE_URL}/{build_checksum_filename(year, month)}"  # Une la URL base con el nombre del CHECKSUM


def sha256_file(file_path: Path) -> str:  # Calcula el hash SHA256 de un archivo
    h = hashlib.sha256()  # Crea un objeto de hash SHA256 vacío
    with open(file_path, "rb") as f:  # Abre el archivo en modo binario de lectura
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):  # Lee el archivo por bloques de 1 MB hasta terminar
            h.update(chunk)  # Va agregando cada bloque al cálculo del hash
    return h.hexdigest()  # Devuelve el hash final en formato hexadecimal


def http_get_bytes(url: str) -> bytes:  # Descarga el contenido de una URL como bytes
    request = Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )  # Crea la solicitud HTTP con un encabezado User-Agent
    with urlopen(
        request, timeout=TIMEOUT
    ) as response:  # Abre la conexión HTTP con tiempo máximo de espera
        return response.read()  # Devuelve el contenido descargado como bytes


def download_file(
    url: str, output_path: Path, retries: int = RETRIES
) -> bool:  # Descarga un archivo y lo guarda localmente
    if (
        output_path.exists() and output_path.stat().st_size > 0
    ):  # Si el archivo ya existe y no está vacío
        print(f"[YA EXISTE] {output_path.name}")  # Muestra que el archivo ya estaba descargado
        return True  # No lo vuelve a descargar y lo da por correcto

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )  # Crea una ruta temporal con extensión .part

    for attempt in range(
        1, retries + 1
    ):  # Recorre los intentos desde 1 hasta el máximo permitido
        try:  # Intenta descargar el archivo
            print(
                f"[DOWNLOAD {attempt}/{retries}] {url}"
            )  # Muestra en pantalla el intento actual
            data = http_get_bytes(url)  # Descarga el contenido del archivo desde la URL
            tmp_path.write_bytes(
                data
            )  # Guarda primero el contenido en el archivo temporal
            tmp_path.replace(
                output_path
            )  # Renombra el archivo temporal al nombre definitivo
            print(f"[OK] {output_path.name}")  # Muestra mensaje de descarga exitosa
            return True  # Indica que la descarga salió bien
        except HTTPError as e:  # Captura errores HTTP como 404 o 500
            print(
                f"[HTTP ERROR] {output_path.name} -> {e.code}"
            )  # Muestra el código del error HTTP
            if e.code == 404:  # Si el error es 404 archivo no encontrado
                return (
                    False  # Devuelve falso porque ese archivo no existe en el servidor
                )
        except URLError as e:  # Captura errores de red o conexión
            print(
                f"[URL ERROR] {output_path.name} -> {e}"
            )  # Muestra el error de conexión
        except Exception as e:  # Captura cualquier otro error inesperado
            print(f"[ERROR] {output_path.name} -> {e}")  # Muestra el error general

        time.sleep(SLEEP_SECONDS)  # Espera unos segundos antes de volver a intentar

    return False  # Si agotó todos los intentos y no pudo descargar, devuelve falso


def download_checksum_text(
    url: str,
) -> str | None:  # Descarga el texto del archivo CHECKSUM
    for attempt in range(
        1, RETRIES + 1
    ):  # Repite la descarga varias veces si hace falta
        try:  # Intenta descargar el checksum
            print(f"[CHECKSUM DOWNLOAD {attempt}/{RETRIES}] {url}")  # Muestra el intento de descarga del checksum
            data = http_get_bytes(url)  # Descarga el contenido del checksum en bytes
            return data.decode(
                "utf-8", errors="replace"
            )  # Convierte los bytes a texto UTF-8
        except HTTPError as e:  # Captura errores HTTP
            print(
                f"[HTTP ERROR CHECKSUM] {url} -> {e.code}"
            )  # Muestra el error HTTP del checksum
            if e.code == 404:  # Si el checksum no existe
                return None  # Devuelve None y continúa sin validación
        except Exception as e:  # Captura cualquier otro error inesperado
            print(
                f"[ERROR CHECKSUM] {url} -> {e}"
            )  # Muestra el error general del checksum
        time.sleep(SLEEP_SECONDS)  # Espera antes de intentar de nuevo
    return None  # Si no pudo descargar el checksum, devuelve None


def verify_checksum(
    zip_path: Path, checksum_text: str | None
) -> tuple[
    bool | None, str | None, str | None
]:  # Verifica si el ZIP coincide con el hash esperado
    """
    Retorna:
      checksum_ok, expected_hash, actual_hash

    checksum_ok:
      True  -> coincide
      False -> no coincide
      None  -> no hubo checksum disponible
    """
    if checksum_text is None:  # Si no hay texto de checksum
        return None, None, None  # No se puede validar nada, devuelve vacíos

    parts = checksum_text.strip().split()  # Limpia el texto y lo separa por espacios
    if not parts:  # Si el archivo checksum vino vacío o mal formado
        return None, None, None  # No se puede validar nada, devuelve vacíos

    expected_hash = (
        parts[0].strip().lower()
    )  # Toma el hash esperado desde el checksum y lo pasa a minúsculas
    actual_hash = sha256_file(
        zip_path
    ).lower()  # Calcula el hash real del ZIP descargado

    return (
        actual_hash == expected_hash,
        expected_hash,
        actual_hash,
    )  # Devuelve si coinciden y ambos hashes


def extract_zip(
    zip_path: Path, extract_to: Path
) -> bool:  # Extrae un archivo ZIP a una carpeta destino
    try:  # Intenta abrir y extraer el ZIP
        with zipfile.ZipFile(
            zip_path, "r"
        ) as zf:  # Abre el archivo ZIP en modo lectura
            zf.extractall(
                extract_to
            )  # Extrae todo el contenido dentro de la carpeta destino
        print(f"[EXTRACT OK] {zip_path.name}")  # Muestra mensaje de extracción exitosa
        return True  # Indica que la extracción salió bien
    except zipfile.BadZipFile:  # Captura el caso en que el ZIP está dañado
        print(f"[BAD ZIP] {zip_path.name}")  # Muestra que el ZIP es inválido o corrupto
        return False  # Indica fallo en extracción
    except Exception as e:  # Captura cualquier otro error inesperado
        print(
            f"[EXTRACT ERROR] {zip_path.name} -> {e}"
        )  # Muestra el error general de extracción
        return False  # Indica fallo en extracción


def append_log(rows: list[dict]) -> None:  # Agrega filas al archivo CSV de log
    file_exists = LOG_FILE.exists()  # Verifica si el archivo de log ya existe
    with open(
        LOG_FILE, "a", newline="", encoding="utf-8"
    ) as f:  # Abre el log en modo agregar texto UTF-8
        writer = csv.DictWriter(  # Crea un escritor CSV que recibirá diccionarios
            f,  # Usa el archivo abierto como destino
            fieldnames=[  # Define el orden de las columnas del CSV
                "timestamp",  # Fecha y hora del registro
                "period",  # Periodo descargado en formato YYYY-MM
                "zip_filename",  # Nombre del archivo ZIP
                "zip_url",  # URL del ZIP
                "checksum_url",  # URL del archivo CHECKSUM
                "download_ok",  # Indica si la descarga fue exitosa
                "checksum_ok",  # Indica si la validación del checksum fue exitosa
                "extract_ok",  # Indica si la extracción fue exitosa
                "zip_size_bytes",  # Tamaño del ZIP en bytes
                "expected_hash",  # Hash esperado según el CHECKSUM
                "actual_hash",  # Hash real calculado del ZIP descargado
            ],
        )
        if not file_exists:  # Si el log todavía no existía
            writer.writeheader()  # Escribe la cabecera del CSV una sola vez
        writer.writerows(rows)  # Escribe todas las filas acumuladas en el log


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main():  # Define la función principal que controla todo el proceso
    print("=" * 70)  # Imprime una línea separadora
    print("DESCARGA MASIVA BTCUSDT SPOT 1m MONTHLY")  # Imprime el título del proceso
    print(
        f"Rango: {START_YEAR}-{START_MONTH:02d} -> {END_YEAR}-{END_MONTH:02d}"
    )  # Muestra el rango de meses a descargar
    print(f"ZIP_DIR: {ZIP_DIR}")  # Muestra la carpeta donde se guardarán los ZIP
    print(f"CSV_DIR: {CSV_DIR}")  # Muestra la carpeta donde se extraerán los CSV
    total_periods = total_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)  # Calcula cuántos periodos mensuales se procesarán
    print(f"TOTAL DE PERIODOS A PROCESAR: {total_periods}")  # Muestra el total de meses a descargar
    print("=" * 70)  # Imprime otra línea separadora

    rows_to_log = []  # Crea una lista vacía para acumular registros del log

    for idx, (year, month) in enumerate(
        month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH), start=1
    ):  # Recorre todos los meses del rango configurado con índice
        period = f"{year}-{month:02d}"  # Crea una etiqueta de periodo tipo 2019-01
        percent = (idx / total_periods) * 100  # Calcula el porcentaje de avance
        print("\n" + "=" * 70)  # Imprime una línea separadora antes de cada periodo
        print(f"[PROGRESO] Periodo {idx}/{total_periods} -> {period} ({percent:.2f}%)")  # Muestra el progreso del proceso
        print("=" * 70)  # Imprime una línea separadora después del progreso

        zip_filename = build_zip_filename(
            year, month
        )  # Construye el nombre del ZIP de ese mes
        checksum_filename = build_checksum_filename(
            year, month
        )  # Construye el nombre del CHECKSUM de ese mes

        zip_url = build_zip_url(year, month)  # Construye la URL del ZIP
        checksum_url = build_checksum_url(year, month)  # Construye la URL del CHECKSUM

        print(f"[ARCHIVO] ZIP: {zip_filename}")  # Muestra el nombre del ZIP actual
        print(f"[ARCHIVO] CHECKSUM: {checksum_filename}")  # Muestra el nombre del checksum actual
        print(f"[URL ZIP] {zip_url}")  # Muestra la URL de descarga del ZIP
        print(f"[URL CHECKSUM] {checksum_url}")  # Muestra la URL de descarga del checksum

        zip_path = (
            ZIP_DIR / zip_filename
        )  # Construye la ruta local donde se guardará el ZIP

        download_ok = download_file(
            zip_url, zip_path
        )  # Descarga el ZIP del mes y devuelve si salió bien
        checksum_ok = None  # Inicializa el estado de validación del checksum
        extract_ok = False  # Inicializa el estado de extracción del ZIP
        expected_hash = None  # Inicializa el hash esperado
        actual_hash = None  # Inicializa el hash real

        if (
            download_ok and zip_path.exists()
        ):  # Solo continúa si la descarga salió bien y el ZIP existe
            checksum_text = download_checksum_text(
                checksum_url
            )  # Descarga el contenido textual del checksum
            checksum_ok, expected_hash, actual_hash = verify_checksum(
                zip_path, checksum_text
            )  # Verifica la integridad del ZIP

            if checksum_ok is True:  # Si el checksum coincide correctamente
                print(f"[CHECKSUM OK] {zip_filename}")  # Muestra que el checksum fue correcto
            elif checksum_ok is False:  # Si el checksum no coincide
                print(f"[CHECKSUM FAIL] {zip_filename}")  # Muestra que el checksum falló
            else:  # Si no hubo checksum disponible
                print(f"[CHECKSUM NO DISPONIBLE] {zip_filename}")  # Muestra que no se pudo validar checksum

            if (
                checksum_ok is False
            ):  # Si el checksum indica que el ZIP está corrupto o no coincide
                print(
                    f"[REINTENTO] El archivo {zip_filename} no pasó checksum. Se intentará descargar nuevamente."
                )  # Muestra que se hará un reintento de descarga
                try:  # Intenta borrar el archivo ZIP defectuoso
                    zip_path.unlink(missing_ok=True)  # Elimina el ZIP dañado si existe
                except Exception:  # Captura cualquier error al borrar el archivo
                    pass  # Lo ignora y sigue para no romper el flujo

                download_ok = download_file(
                    zip_url, zip_path, retries=1
                )  # Re-descarga el ZIP solo una vez
                if (
                    download_ok and zip_path.exists()
                ):  # Si se volvió a descargar correctamente
                    checksum_ok, expected_hash, actual_hash = verify_checksum(
                        zip_path, checksum_text
                    )  # Verifica otra vez el checksum

                    if checksum_ok is True:  # Si el checksum fue correcto tras el reintento
                        print(f"[CHECKSUM OK TRAS REINTENTO] {zip_filename}")  # Muestra checksum correcto después del reintento
                    elif checksum_ok is False:  # Si el checksum sigue fallando tras el reintento
                        print(f"[CHECKSUM FAIL TRAS REINTENTO] {zip_filename}")  # Muestra checksum fallido después del reintento
                    else:  # Si tampoco hubo checksum disponible tras el reintento
                        print(f"[CHECKSUM NO DISPONIBLE TRAS REINTENTO] {zip_filename}")  # Muestra que no hubo checksum tras el reintento

            if download_ok and (
                checksum_ok is True or checksum_ok is None
            ):  # Si el ZIP está bien o no había checksum disponible
                extract_ok = extract_zip(
                    zip_path, CSV_DIR
                )  # Extrae el contenido del ZIP a la carpeta CSV

                if extract_ok:  # Si la extracción salió bien
                    print(f"[EXTRACCION OK] {zip_filename}")  # Muestra extracción correcta
                else:  # Si la extracción falló
                    print(f"[EXTRACCION FAIL] {zip_filename}")  # Muestra extracción fallida

        zip_size = (
            zip_path.stat().st_size if zip_path.exists() else 0
        )  # Obtiene el tamaño del ZIP en bytes si existe
        print(f"[TAMANO ZIP] {zip_size} bytes")  # Muestra el tamaño del archivo ZIP

        rows_to_log.append(  # Agrega una nueva fila de información a la lista del log
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),  # Fecha y hora actual del registro
                "period": period,  # Periodo procesado
                "zip_filename": zip_filename,  # Nombre del ZIP
                "zip_url": zip_url,  # URL del ZIP
                "checksum_url": checksum_url,  # URL del checksum
                "download_ok": download_ok,  # Resultado de la descarga
                "checksum_ok": checksum_ok,  # Resultado de la verificación de integridad
                "extract_ok": extract_ok,  # Resultado de la extracción
                "zip_size_bytes": zip_size,  # Tamaño del ZIP descargado
                "expected_hash": expected_hash,  # Hash esperado según el checksum
                "actual_hash": actual_hash,  # Hash real calculado del archivo descargado
            }
        )

        print(
            f"[RESUMEN PERIODO] {period} | download_ok={download_ok} | checksum_ok={checksum_ok} | extract_ok={extract_ok}"
        )  # Muestra el resumen final del periodo procesado

    append_log(
        rows_to_log
    )  # Escribe todas las filas acumuladas en el archivo CSV de log

    print("\n" + "=" * 70)  # Imprime una línea final con salto de línea antes
    print("PROCESO TERMINADO")  # Muestra mensaje de fin del proceso
    print(f"Log: {LOG_FILE}")  # Muestra la ruta del archivo de log generado
    print("=" * 70)  # Imprime la línea final de cierre


if __name__ == "__main__":  # Verifica que este archivo se esté ejecutando directamente
    main()  # Llama a la función principal para iniciar todo el proceso