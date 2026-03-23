# Tesis BTC WSL Docker

Repositorio del pipeline de procesamiento en **WSL Ubuntu + Docker + Airflow** para el proyecto de tesis de predicción del precio de Bitcoin.

## Descripción

Este repositorio contiene la parte del proyecto orientada a la ejecución en entorno Linux mediante **WSL**, uso de **contenedores Docker**, componentes de **Airflow** para orquestación y scripts auxiliares de procesamiento.

Forma parte de una arquitectura de tesis dividida en tres componentes principales:

1. **Pipeline local**
2. **Pipeline WSL/Docker**
3. **Entrenamiento en Windows con CUDA**

Este repositorio corresponde al componente **WSL/Docker**.

---

## Objetivo

Organizar y ejecutar procesos del pipeline de datos en un entorno reproducible basado en Linux, apoyado en contenedores y herramientas de orquestación, como parte de la solución técnica de la tesis.

---

## Tecnologías utilizadas

- **WSL Ubuntu**
- **Python**
- **Docker**
- **Apache Airflow**
- **Spark jobs**
- **Git y GitHub**

---

## Estructura del proyecto

```text
tesis_btc/
├── airflow/
├── docker/
├── spark_jobs/
├── src/
├── requirements_wsl_docker.txt
├── .gitignore
└── README.md
Descripción de carpetas
airflow/: archivos y componentes relacionados con orquestación de tareas.
docker/: configuración de contenedores y servicios necesarios para el pipeline.
spark_jobs/: procesos orientados al procesamiento distribuido.
src/: scripts auxiliares y lógica complementaria del proyecto.
requirements_wsl_docker.txt: dependencias del entorno Python para este repositorio.
Requisitos previos

Antes de usar este repositorio, se recomienda contar con:

WSL Ubuntu instalado
Python 3 disponible en WSL
Docker instalado y funcionando
Git configurado
Entorno virtual de Python
Instalación del entorno
1. Crear entorno virtual
python3 -m venv .venv
2. Activar entorno virtual
source .venv/bin/activate
3. Instalar dependencias
pip install -r requirements_wsl_docker.txt
Ejecución

La ejecución depende del componente específico que se quiera usar.

Ejecutar un script Python
python src/nombre_del_script.py
Ejecución de componentes Docker

Dependiendo de la configuración del proyecto, los servicios pueden levantarse con comandos Docker o Docker Compose definidos en este repositorio.

Ejecución de Airflow

Los archivos dentro de airflow/ corresponden al flujo de orquestación definido para el pipeline.

Alcance del repositorio

Este repositorio puede incluir, entre otros:

preparación de entorno Linux/WSL para el pipeline
integración con Docker
soporte de orquestación con Airflow
jobs de procesamiento con Spark
scripts de apoyo para automatización del flujo de datos
Archivos excluidos del repositorio

Por control de tamaño y orden del proyecto, no se incluyen en GitHub:

datos pesados
archivos temporales
backups comprimidos
notebooks temporales
configuración local de VS Code
entornos virtuales

Esto se controla mediante el archivo .gitignore.

Relación con la tesis

Este repositorio forma parte del proyecto de tesis orientado al procesamiento y preparación de datos para el análisis y modelado predictivo del precio de Bitcoin.

El componente de entrenamiento de modelos y experimentación GPU se mantiene en un repositorio separado, al igual que el pipeline local de preparación y análisis.

Autor

Juan Andres Logacho Torres

Maestría en Ciencia de Datos y Big Data

Notas

Este repositorio documenta únicamente el componente WSL/Docker del proyecto general.
Para la solución completa, revisar también los repositorios complementarios del pipeline local y del entrenamiento en Windows con CUDA.
