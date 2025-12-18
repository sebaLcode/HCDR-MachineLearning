# Home Credit Default Risk - Modelo de Scoring Crediticio
- Autor: Sebastián Loncón

En este proyecto se implementa una solución de Machine Learning para predecir la probabilidad de incumplimiento de pago (default) de clientes bancarios.

El sistema sigue la metodología **CRISP-DM**, se integran múltiples fuentes de datos relacionales, aplicando la ingeniería de características avanzada y desplegando el modelo final mediante una **API REST** con FastAPI.

## Estructura del Proyecto

El repositorio sigue una arquitectura de microservicios.
├── 01_data_understanding/ # Notebooks para EDA

├── 02_data_preparation/ # Scripts de limpieza, ingeniería de features e integración 

├── 03_modeling/ # Entrenamiento del modelo (Random Forest) y validación 

├── 04_evaluation/ # Generación de métricas y gráficos de desempeño 

├── 05_deployment/ # API REST (FastAPI) para predicción en tiempo real 

├── artifacts/ # Modelos serializados (.joblib) y metadatos 

├── data/ # Fuentes de datos entregadas y  aquellas procesadas 

├── requirements.txt # Dependencias del proyecto 

└── README.md # Documentación

## Instalación y Configuración del Proyecto

1. Clonar el repositorio ```git clone https://github.com/sebaLcode/HCDR-MachineLearning.git```
2. Instalar dependencias ```pip install -r requirements.txt```
3. Con esto ya tenemos configurado e instaladas las dependencias necesarias.

## Ejecución del Proyecto.
Se recomienda ejecutar el archivo 01_data_understanding/data_understanding.ipynb, para entender los datos que son tratados, mediante run all.

Por lo tanto, los archivos .py deben ser ejecutados en el orden que se dará a continuación, además se explicará brevemente lo que se realizó en cada carpeta.

1. En 02_data_preparation, se integran las tablas, limpiando las anomalías y generando ratios financieros.
2. 03_modeling, se entrena un Random Forest con manejo de desbalance y validación cruzada 5-fold CV
3. 04_evaluation, generamos reporte de clasificación y gráficos en la carpeta /plots.
4. 05_deployment, se genera un despliegue a traves de una API construida con FastAPI.

## Despliegue de la API
A continuación se detallará como ejecutar la API.

1. Para levantar el servidor ejecutar: ```uvicorn 05_deployment.main:app --reload```
2. Para acceder a la documentación, abrir navegador y pegar: ```http://127.0.0.1:8000/docs```
3. Finalmente se puede probar el Endpoint /evaluate_risk, se puede utilizar el siguiente ejemplo:
```json
{
  "features": {
    "NAME_CONTRACT_TYPE": 0,
    "CODE_GENDER": 0,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1,
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 200000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 25000.0,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -2000,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.6
  }
}
```

## Decisiones claves
Se eligió Random Forest dada su capacidad de capturar relaciones no lineales y la robustez frente a outliers, sin necesitdad de escalado profundo.

Además se manejo un desbalance para penalizar los errores en la clase minoritaria Default, priorizando la sensibilidad.

Se calcularon ratios manuales, demostrando la importancia predictiva de esas variables.

Finalmente se optó por el Label Encoding, para evitar la explosión de dimensionalidad dado el gran número de categorías.
