# 🛠️ Configuración del Entorno - Análisis de Datos II

## 📋 Dependencias Instaladas

Este proyecto utiliza un pipeline modular de análisis de datos con los siguientes componentes:

### Módulos principales:
- **`data_loader.py`** - Carga universal de CSV, Excel y TXT
- **`data_explorer.py`** - Análisis exploratorio completo (EDA)
- **`data_cleaner.py`** - Limpieza y recomendaciones de datos
- **`data_reggresion.py`** - Comparación y selección de modelos de regresión

### Librerías requeridas:

| Librería | Versión | Propósito |
|----------|---------|-----------|
| pandas | ≥1.5.0 | Manipulación de datos |
| numpy | ≥1.23.0 | Operaciones numéricas |
| matplotlib | ≥3.5.0 | Visualización estática |
| seaborn | ≥0.12.0 | Visualización estadística |
| scipy | ≥1.9.0 | Análisis estadístico |
| scikit-learn | ≥1.2.0 | Machine learning |
| openpyxl | ≥3.9.0 | Lectura de Excel |

### Librerías opcionales (descomentar si necesitas):
- `xgboost` - Gradient boosting
- `lightgbm` - Light gradient boosting
- `catboost` - Categorical boosting
- `py-earth` - Multivariate adaptive regression splines
- `rulefit` - Interpretable model extraction

---

## 🚀 Inicio Rápido

### Opción 1: Usar requirements.txt (RECOMENDADO)

```bash
# Ejecutar una sola vez al abrir un nuevo Codespace
pip install -r requirements.txt
```

### Opción 2: Con .devcontainer (Automático en nuevos Codespaces)

Si está configurado, las dependencias se instalan automáticamente:
1. Crear nuevo Codespace
2. Esperar a que termine la instalación
3. Abrir `analisis_guerras_eda.ipynb`
4. ¡Listo! Todo funciona

### Opción 3: En el notebook (con instalación silenciosa)

El notebook ejecuta automáticamente:
```python
pip install -r requirements.txt -q
```

---

## 📊 Estructura del Proyecto

```
Analisis-de-datos-II/
├── data_loader.py              # Descarga universal de archivos
├── data_explorer.py            # EDA completo
├── data_cleaner.py             # Limpieza de datos
├── data_reggresion.py          # Modelos de regresión
├── analisis_guerras_eda.ipynb  # Notebook principal
├── requirements.txt            # Dependencias Python
├── .devcontainer/
│   └── devcontainer.json      # Configuración GitHub Codespaces
└── Datos_Guerras.csv          # Dataset de ejemplo
```

---

## ✅ Verificar Instalación

Para verificar que todo está correctamente instalado:

```python
# En Python o Jupyter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import datasets
from data_loader import load_data
from data_explorer import DataExplorer
from data_cleaner import DataCleaner

print("✓ Todas las librerías están correctamente instaladas")
```

---

## 🔄 GitHub Codespaces

### Comportamiento de persistencia:

| Escenario | Librerías | Kernel |
|-----------|-----------|--------|
| Cerrar y reabrir mismo Codespace | ✅ Persisten | Reinicia |
| Crear nuevo Codespace | ❌ Desaparecen | Nuevo |
| Ejecutar notebook | ✅ Usa cached | Activo |
| Instalar ejecutable | ✅ Persiste | - |

### Flujo recomendado:

1. **Primer acceso:** 
   ```bash
   pip install -r requirements.txt
   ```

2. **Sesiones siguientes:**
   - Mismo Codespace → No requiere nada (librerías persisten)
   - Nuevo Codespace → Ejecutar paso 1

---

## 🐛 Solucionar Problemas

### Error: "ModuleNotFoundError"

```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "No module named 'openpyxl'"

```bash
# Instalar solo openpyxl
pip install openpyxl
```

### El notebook no encuentra módulos locales

Asegúrate de ejecutar el notebook desde la raíz del proyecto:
- ✅ Correcto: `/workspaces/Analisis-de-datos-II/`
- ❌ Incorrecto: `/workspaces/Analisis-de-datos-II/notebook/`

---

## 📝 Notas

- Las librerías se especifican con versiones mínimas (`≥`) para permitir actualizaciones
- El archivo `.devcontainer.json` es opcional pero recomendado para Codespaces nuevos
- El notebook incluye instalación automática como medida de seguridad

---

**Última actualización:** Abril 2026
