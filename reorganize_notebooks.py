#!/usr/bin/env python3
"""
Script para reorganizar notebooks con estructura correcta de ejecución.
Ordena las celdas: Markdown -> pip install -> Importaciones -> Cargar datos -> Análisis
"""

import json
import sys
from pathlib import Path

def create_ordered_cells(archivo_datos):
    """Crea celdas en orden correcto de ejecución"""
    return [
        # 0. Título
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# Análisis Exploratorio de Datos - {archivo_datos}\n\nEste notebook realiza un análisis exploratorio completo del archivo de datos, utilizando todas las funciones disponibles en `data_explorer.py` para una revisión exhaustiva de los datos."]
        },
        # 1. Instalación
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 0. Instalación de Dependencias"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "\n",
                "# Instalar dependencias requeridas\n",
                "subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-r\", \"requirements.txt\", \"-q\"])"
            ]
        },
        # 2. Importaciones
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Importar Librerías"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from data_explorer import DataExplorer\n",
                "import warnings\n",
                "\n",
                "# Configuración de visualización\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "sns.set_palette(\"husl\")\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)\n",
                "warnings.filterwarnings('ignore')"
            ]
        },
        # 3. Cargar datos
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Cargar y Explorar Datos"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"archivo = \"{archivo_datos}\"\n",
                "\n",
                "# Intentar cargar con diferentes delimitadores\n",
                "try:\n",
                "    df = pd.read_csv(archivo, sep=',')\n",
                "    print(\"✓ Archivo cargado con delimitador COMA\")\n",
                "except:\n",
                "    try:\n",
                "        df = pd.read_csv(archivo, sep=';')\n",
                "        print(\"✓ Archivo cargado con delimitador PUNTO Y COMA\")\n",
                "    except:\n",
                "        try:\n",
                "            df = pd.read_csv(archivo, sep='\\\\t')\n",
                "            print(\"✓ Archivo cargado con delimitador TAB\")\n",
                "        except:\n",
                "            try:\n",
                "                df = pd.read_csv(archivo, sep='\\\\s+')\n",
                "                print(\"✓ Archivo cargado con delimitador WHITESPACE\")\n",
                "            except:\n",
                "                df = pd.read_csv(archivo)\n",
                "                print(\"✓ Archivo cargado con delimitador por defecto\")\n",
                "\n",
                "print(f\"\\\\n✓ Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas\")\n",
                "print(f\"  Tamaño en memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\")"
            ]
        },
        # 4. Vista previa
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Vista Previa de Datos"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\\\n=== PRIMERAS FILAS ===\")\n",
                "print(df.head(10))\n",
                "\n",
                "print(\"\\\\n=== ÚLTIMAS FILAS ===\")\n",
                "print(df.tail())\n",
                "\n",
                "print(\"\\\\n=== ESTADÍSTICAS DESCRIPTIVAS ===\")\n",
                "print(df.describe())\n",
                "\n",
                "print(\"\\\\n=== TIPOS DE DATOS ===\")\n",
                "print(df.dtypes)\n",
                "\n",
                "print(\"\\\\n=== INFORMACIÓN GENERAL ===\")\n",
                "print(df.info())"
            ]
        },
        # 5. DataExplorer
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Crear Instancia de DataExplorer"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["explorer = DataExplorer(df, verbose=True)"]
        },
        # 6. Resumen Estructural
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Resumen Estructural de Datos"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "resumen_estructural = explorer.get_structural_summary()\n",
                "print(\"Resumen Estructural:\")\n",
                "print(resumen_estructural)"
            ]
        },
        # 7. Análisis de nulos
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Análisis de Valores Faltantes"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "analisis_nulos = explorer.analyze_nulls()\n",
                "print(\"Análisis de Valores Faltantes:\")\n",
                "print(analisis_nulos)"
            ]
        },
        # 8. Análisis de duplicados
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Análisis de Duplicados"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "analisis_duplicados = explorer.analyze_duplicates()\n",
                "print(\"Análisis de Duplicados:\")\n",
                "for key, value in analisis_duplicados.items():\n",
                "    print(f\"  {key}: {value}\")"
            ]
        },
        # 9. Baja varianza
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Detección de Variables con Baja Varianza"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "bajo_varianza = explorer.detect_low_variance()\n",
                "print(f\"Variables con baja varianza detectadas: {len(bajo_varianza)}\")\n",
                "if bajo_varianza:\n",
                "    print(\"Columnas:\", bajo_varianza)"
            ]
        },
        # 10. Outliers
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Detección de Outliers"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "outliers_df = explorer.detect_outliers()\n",
                "print(f\"Total de outliers detectados: {len(outliers_df)}\\\\n\")\n",
                "if len(outliers_df) > 0:\n",
                "    print(outliers_df.head(10))"
            ]
        },
        # 11. Normalidad
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Análisis de Normalidad - Gráficos Q-Q"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = explorer.plot_normality()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # 12. Correlación
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Matriz de Correlación"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = explorer.plot_correlation_heatmap()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # 13. Scatter
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 12. Gráficos Scatter de Relaciones Bivariadas"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    fig = explorer.plot_scatter()\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "except Exception as e:\n",
                "    print(f\"Nota: No se pudieron generar scatter plots. Razón: {str(e)[:100]}\")"
            ]
        },
        # 14. Alertas
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 13. Resumen de Alertas y Problemas Detectados"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "alerts_df = explorer.generate_alert_summary()\n",
                "print(f\"Total de alertas: {len(alerts_df)}\\\\n\")\n",
                "if len(alerts_df) > 0:\n",
                "    print(alerts_df)"
            ]
        },
        # 15. Pipeline
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 14. Pipeline Summary para Downstream Processing"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline_summary = explorer.get_pipeline_summary()\n",
                "print(\"Pipeline Summary para Downstream Processing:\")\n",
                "print(f\"  Número de variables numéricas: {len(pipeline_summary['numeric_cols'])}\")\n",
                "print(f\"  Número de variables categóricas: {len(pipeline_summary['categorical_cols'])}\")\n",
                "print(f\"  Columnas con valores faltantes: {len(pipeline_summary['null_columns'])}\")\n",
                "print(f\"  Columnas de baja varianza: {len(pipeline_summary['low_variance_cols'])}\")\n",
                "print(f\"\\\\nVARIABLES NUMÉRICAS:\")\n",
                "print(pipeline_summary['numeric_cols'][:10])\n",
                "print(f\"\\\\nVARIABLES CATEGÓRICAS:\")\n",
                "print(pipeline_summary['categorical_cols'][:10])"
            ]
        },
        # 16. EDA completo
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 15. Análisis EDA Completo"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "full_eda = explorer.run_full_eda()\n",
                "print(\"✓ Análisis EDA completo ejecutado exitosamente\")\n",
                "print(f\"  Secciones generadas: {len(full_eda)} componentes\")"
            ]
        }
    ]

def reorganize_notebook(notebook_path, archivo_datos):
    """Reorganiza un notebook con estructura correcta"""
    # Crear nueva estructura
    new_notebook = {
        "cells": create_ordered_cells(archivo_datos),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Escribir el nuevo notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(new_notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Reorganizado: {notebook_path}")

def main():
    notebooks = [
        ("analisis_guerras_tidy_eda.ipynb", "Datos_guerras_tidy.csv"),
        ("analisis_astronomia_eda.ipynb", "Datos de astronomía.txt"),
        ("analisis_vehiculos_eda.ipynb", "Datos de consumo en vehículos.txt"),
    ]
    
    base_path = Path("/workspaces/Analisis-de-datos-II")
    
    for notebook_name, archivo_datos in notebooks:
        notebook_path = base_path / notebook_name
        if notebook_path.exists():
            reorganize_notebook(notebook_path, archivo_datos)
        else:
            print(f"✗ No encontrado: {notebook_path}")

if __name__ == "__main__":
    main()
    print("\n✓ Reorganización completada exitosamente")
