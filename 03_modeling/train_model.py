import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import ks_2samp

# Configuración
DATA_PATH = 'data/master_table.parquet'
ARTIFACTS_DIR = 'artifacts'

# Crear carpeta artifacts
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

"""
    Calcula el estadístico KS.
"""
def calculate_ks(y_true, y_prob):
    df_ks = pd.DataFrame({'target': y_true, 'prob': y_prob})
    class0 = df_ks[df_ks['target'] == 0]['prob']
    class1 = df_ks[df_ks['target'] == 1]['prob']
    ks_stat, p_value = ks_2samp(class0, class1)
    return ks_stat

def main():
    print("Iniciando el entrenamiento...\n")
    
    # Cargamos los datos de master_table.parquet
    if not os.path.exists(DATA_PATH):
        print(f"Error: No existe el archivo {DATA_PATH}. Ejecuta el paso 02 primero (02_data_preparation/data_preparation.py).")
        return

    print(f"   -> Cargando {DATA_PATH}...")
    
    # Leemos el archivo
    df = pd.read_parquet(DATA_PATH)
    
    # Separamos Features y Target, excluimos id para no memorizar y target dado que es lo que se espera predecir.
    features = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = df[features]
    y = df['TARGET']
    
    print(f"   -> Dataset cargado. X: {X.shape}, y: {y.shape}")

    # Split del train y test
    print("   -> Dividiendo datos (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputación, fit solo en train, para evitar data leakage
    print("\nAprendiendo imputación (Mediana) sobre Train...")
    
    # Manejo de infinitos
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Guardamos el imputer para la API
    joblib.dump(imputer, os.path.join(ARTIFACTS_DIR, 'imputer.joblib'))
    print("Imputer guardado...")

    # Definiendo el modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10, 
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Validación Cruzada
    print("\n   -> Ejecutando Cross-Validation (5-Fold Stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_imputed, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"ROC-AUC Promedio (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Entrenamiento final
    print("\n   -> Entrenando modelo final en todo el set de Train...")
    model.fit(X_train_imputed, y_train)
    
    # Guardar modelo y lista de features
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'model.joblib'))
    joblib.dump(features, os.path.join(ARTIFACTS_DIR, 'features_list.joblib'))
    print("\nModelo y features guardados...")

    # Evaluación del test
    print("\nRESULTADOS FINALES EN TEST SET:")
    y_prob = model.predict_proba(X_test_imputed)[:, 1]
    y_pred = model.predict(X_test_imputed)
    
    auc = roc_auc_score(y_test, y_prob)
    ks = calculate_ks(y_test, y_prob)
    
    print(f"   -> ROC-AUC: {auc:.4f}")
    print(f"   -> KS Statistic: {ks*100:.2f}% (Objetivo > 20%)")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()