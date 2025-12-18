import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve

# Configuración
DATA_PATH = 'data/master_table.parquet'
ARTIFACTS_DIR = 'artifacts'

# Carpeta para guardar los gráficos
PLOTS_DIR = '04_evaluation/plots'

# Crear carpeta de plots si no existe
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


"""
    Genera y guarda la matriz de confusión normalizada y en conteo absoluto.
"""
def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusión'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicción (0: Paga, 1: Default)')
    plt.ylabel('Realidad')
    
    path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico Matriz Confusión guardado: {path}")


"""
    Genera y guarda la Curva ROC.
"""
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC - Capacidad de Discriminación')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico Curva ROC guardado: {path}")


"""
    Genera Curva Precision-Recall. 
 """
def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall (Impacto del Desbalance)')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    
    path = os.path.join(PLOTS_DIR, 'precision_recall_curve.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico Curva Precision-Recall guardado: {path}")


"""
    Genera y guarda el top 20 de variables más importantes.
"""
def plot_feature_importance(model, features_list):
    importances = model.feature_importances_
    
    # Crear DataFrame
    feature_imp = pd.DataFrame({'Feature': features_list, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_imp, palette='viridis', legend=False)
    plt.title('Top 20 Variables Determinantes del Riesgo')
    plt.xlabel('Importancia (Gini Impurity)')
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico Top 20 variables más importantes guardado: {path}")

def main():
    print("Iniciando la evaluación del modelo...\n")
    
    # Validamos los archivos creados
    required_files = [
        os.path.join(ARTIFACTS_DIR, 'model.joblib'),
        os.path.join(ARTIFACTS_DIR, 'features_list.joblib'),
        os.path.join(ARTIFACTS_DIR, 'imputer.joblib')
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Falta el artefacto {f}. Ejecuta el paso 03 primero (03_modeling/train_model.py).")
            return

    # Cargamos los artefactos
    print("   -> Cargando modelo y preprocesadores...")
    model = joblib.load(required_files[0])
    features_list = joblib.load(required_files[1])
    imputer = joblib.load(required_files[2])
    
    # Cargamos y preparamos los datos
    print("   -> Cargando datos de prueba...")
    df = pd.read_parquet(DATA_PATH)
    
    X = df[features_list]
    y = df['TARGET']
    
    # Split idéntico al entrenamiento (random_state=42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Imputación
    print("   -> Aplicando imputación al set de prueba...")
    
    # Limpieza de infinitos antes de imputar
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Transformamos usando el imputer APRENDIDO en entrenamiento
    X_test_imputed = imputer.transform(X_test)
    
    # Predicciones
    print("   -> Generando predicciones...")
    y_pred = model.predict(X_test_imputed)
    y_prob = model.predict_proba(X_test_imputed)[:, 1]
    
    # Reporte Númerico
    print("\n" + "="*50)
    print("REPORTE FINAL DE DESEMPEÑO (TEST SET)")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # Generación de Gráficos
    print("\nGenerando evidencias visuales en '/04_evaluation/plots'...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_feature_importance(model, features_list)
    
    print("\nEvaluación completada. Revisa los gráficos en la carpeta de Plots")

if __name__ == "__main__":
    main()