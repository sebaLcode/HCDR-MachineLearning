import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder

# Configuraciones 
DATA_DIR = "data" 
OUTPUT_FILE = "master_table.parquet"
OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_FILE)

"""
    Carga un archivo parquet de la carpeta de datos.
"""
def load_data(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {file_name}")
        return None
    print(f"   -> Cargando {file_name}...")
    return pd.read_parquet(path)

"""
    Agrupa tablas satélite y calcula estadísticas.
"""
def aggregate_table(df, group_col, prefix):
    # Seleccionar columnas numéricas
    num_cols = [c for c in df.columns if df[c].dtype != 'object' and 'SK_ID' not in c]
    
    if not num_cols:
        return pd.DataFrame()

    print(f"Agrupando {prefix} (cols: {len(num_cols)})...")
    
    # Calcular estadísticas agregadas
    agg = df.groupby(group_col)[num_cols].agg(['min', 'max', 'mean', 'sum'])
    agg.columns = [f'{prefix}_{c}_{stat.upper()}' for c, stat in agg.columns]
    
    # Añadir conteo de registros por grupo
    agg[f'{prefix}_COUNT'] = df.groupby(group_col).size()
    return agg

def main():
    print("Inicio de la preparación de los datos...\n")

    # Cargar tabla principal
    df_app = load_data("application_.parquet")
    if df_app is None: return
    print(f"   Dimensiones Iniciales: {df_app.shape}")

    #Limpiamos outliers
    print("\n[1/4] Limpiando anomalías...")
    # 365243 en DAYS_EMPLOYED es un error/valor default. Lo reemplazamos por NaN.
    df_app['DAYS_EMPLOYED'] = df_app['DAYS_EMPLOYED'].replace(365243, np.nan)


    print("\n[2/4] Creando Ratios Financieros...")

    # Porcentaje del crédito respecto al ingreso, ¿Está sobreendeudado?
    df_app['RATIO_CREDIT_INCOME'] = df_app['AMT_CREDIT'] / df_app['AMT_INCOME_TOTAL']

    # Carga financiera (Anualidad / Ingreso)
    df_app['RATIO_ANNUITY_INCOME'] = df_app['AMT_ANNUITY'] / df_app['AMT_INCOME_TOTAL']
    
    # Plazo del crédito aproximado (Crédito / Anualidad)
    df_app['RATIO_CREDIT_TERM'] = df_app['AMT_CREDIT'] / df_app['AMT_ANNUITY']
    
    # Porcentaje de días trabajados respecto a la edad (Los 2 son negativos)
    df_app['RATIO_DAYS_EMPLOYED_PERCENT'] = df_app['DAYS_EMPLOYED'] / df_app['DAYS_BIRTH']

    # Integración de las tablas
    print("\n[3/4] Integrando Tablas")

    # Boreau y Bureau Balance
    df_bb = load_data("bureau_balance.parquet")
    df_bureau = load_data("bureau.parquet")
    
    if df_bureau is not None and df_bb is not None:
        bb_agg = aggregate_table(df_bb, 'SK_ID_BUREAU', 'BB')
        df_bureau = df_bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
        
        bureau_agg = aggregate_table(df_bureau, 'SK_ID_CURR', 'BUREAU')
        df_app = df_app.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        # Limpieza de memoria
        del df_bb, df_bureau, bb_agg, bureau_agg
        gc.collect()

    # Previous Application
    df_prev = load_data("previous_application.parquet")
    if df_prev is not None:
        prev_agg = aggregate_table(df_prev, 'SK_ID_CURR', 'PREV')
        df_app = df_app.merge(prev_agg, on='SK_ID_CURR', how='left')
        del df_prev, prev_agg; gc.collect()

    # POS CASH Balance
    df_pos = load_data("POS_CASH_balance.parquet")
    if df_pos is not None:
        pos_agg = aggregate_table(df_pos, 'SK_ID_CURR', 'POS')
        df_app = df_app.merge(pos_agg, on='SK_ID_CURR', how='left')
        del df_pos, pos_agg; gc.collect()

    # Installments Payments
    df_ins = load_data("installments_payments.parquet")
    if df_ins is not None:
        ins_agg = aggregate_table(df_ins, 'SK_ID_CURR', 'INSTAL')
        df_app = df_app.merge(ins_agg, on='SK_ID_CURR', how='left')
        del df_ins, ins_agg; gc.collect()

    # Credit Card Balance
    df_cc = load_data("credit_card_balance.parquet")
    if df_cc is not None:
        cc_agg = aggregate_table(df_cc, 'SK_ID_CURR', 'CC')
        df_app = df_app.merge(cc_agg, on='SK_ID_CURR', how='left')
        del df_cc, cc_agg; gc.collect()

    print("\n[4/4] Finalizando Preprocesamiento...")    
    # Label Encoding para variables categóricas (necesario para Random Forest). Convertimos strings a números. Mantenemos los nulos como nulos (o un valor especial)
    # para que el Imputer de la etapa 03 los maneje correctamente.
    obj_cols = df_app.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df_app[col] = df_app[col].astype(str)
        le = LabelEncoder()
        df_app[col] = le.fit_transform(df_app[col])
        
    print(f"Dimensiones Finales Master Table (filas, columnas): {df_app.shape}")
    
    # Guardar
    print(f"Guardando en {OUTPUT_PATH}...")
    df_app.to_parquet(OUTPUT_PATH, index=False)
    print("\nPROCESO FINALIZADO EXITOSAMENTE.")

if __name__ == "__main__":
    main()