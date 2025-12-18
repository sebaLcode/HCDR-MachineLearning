import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

#Configuración
app = FastAPI(
    title="API de Scoring de Crédito (HCDR)",
    description="Microservicio para evaluación de riesgo crediticio en tiempo real.",
    version="1.0.0"
)

# Rutas a los artefactos
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "imputer.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_list.joblib")

# Variables globales para cargar los artefactos en memoria
model = None
imputer = None
features_list = None

"""
    Carga de artefactos.
"""
@app.on_event("startup")
def load_artifacts():
    global model, imputer, features_list
    print("Iniciando API y cargando artefactos...")
    
    # Verificación de seguridad
    if not os.path.exists(MODEL_PATH):
        print(f"Error Crítico, no se encontró el modelo en {MODEL_PATH}")
        return

    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        features_list = joblib.load(FEATURES_PATH)
        print("Modelo, Imputer y Features cargados correctamente, el sistema está listo.")
    except Exception as e:
        print(f"Error al cargar artefactos: {e}")

#Esquema para los datos en el Input
class ClientData(BaseModel):
    # Usamos un diccionario flexible para permitir enviar solo los datos disponibles y no obligar a enviar las 300 columnas si no se tienen.
    features: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "NAME_CONTRACT_TYPE": 0,
                    "CODE_GENDER": 0,
                    "AMT_INCOME_TOTAL": 200000.0,
                    "AMT_CREDIT": 500000.0,
                    "AMT_ANNUITY": 25000.0,
                    "DAYS_BIRTH": -15000,
                    "DAYS_EMPLOYED": -2000,
                    "EXT_SOURCE_2": 0.5,
                    "EXT_SOURCE_3": 0.6
                }
            }
        }

# Endpoint para la predicción
"""
    Recibe datos de un cliente, calcula variables derivadas, imputa nulos y 
    devuelve la probabilidad de incumplimiento.
"""
@app.post("/evaluate_risk")
def predict_risk(data: ClientData):
    if not model:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")
    try:
        # JSON a DataFrame
        input_data = data.features
        df = pd.DataFrame([input_data])
        
        # Calculamos datos para que coincida con los del entrenamiento
        if 'RATIO_CREDIT_INCOME' not in df.columns and 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['RATIO_CREDIT_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
            
        if 'RATIO_ANNUITY_INCOME' not in df.columns and 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['RATIO_ANNUITY_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
            
        if 'RATIO_CREDIT_TERM' not in df.columns and 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
            df['RATIO_CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
            
        if 'RATIO_DAYS_EMPLOYED_PERCENT' not in df.columns and 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
            df['RATIO_DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        # Alineamos las columnas, si no se entrega se rellena con NaN
        df = df.reindex(columns=features_list, fill_value=np.nan)
        
        # Limpieza de infinitos y nulos
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Usamos el Imputer cargado
        X_imputed = imputer.transform(df)

        # Predicción, devuelve [[prob_clase_0, prob_clase_1]]
        prob_default = float(model.predict_proba(X_imputed)[:, 1])
        
        decision = "REVISIÓN MANUAL"
        color_alerta = "AMARILLO"
        mensaje = "El cliente requiere análisis adicional."
        
        if prob_default < 0.30:
            decision = "APROBAR"
            color_alerta = "VERDE"
            mensaje = "Riesgo bajo. Crédito pre-aprobado."
        elif prob_default > 0.60:
            decision = "RECHAZAR"
            color_alerta = "ROJO"
            mensaje = "Riesgo alto de incumplimiento."

        # Respuesta JSON
        return {
            "status": "success",
            "resultado": {
                "probabilidad_default": round(prob_default, 4),
                "score_riesgo": int(prob_default * 1000),
                "decision": decision,
                "alerta": color_alerta,
                "mensaje": mensaje
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento: {str(e)}")

@app.get("/")
def home():
    return {"message": "API de Scoring Crediticio HCDR activa. Visite /docs para documentación."}