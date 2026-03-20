"""
train.py - Pipeline de Entrenamiento de Inteligencia Artificial
---------------------------------------------------------------
Este script realiza el proceso completo de Machine Learning:
1. Ingeniería de Atributos: Transforma los datos relacionales (sesiones, logs, catálogo)
   en un vector de características por sesión, calculando fatiga acumulada por zona.
2. Preprocesamiento: Escala variables numéricas y codifica variables categóricas.
3. Entrenamiento: Entrena un modelo XGBoost optimizado para predecir probabilidades.
4. Evaluación: Calcula métricas de precisión (ROC-AUC y Brier Score).
5. Exportación: Guarda el modelo, el preprocesador y el orden de los atributos.

Uso: python models/train.py
"""
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss

def engineer_features():
    """
    Carga los datos de la base de datos relacional y realiza 
    el feature engineering necesario para el modelo.
    """
    print(">>> 1. Cargando las tablas relacionales...")
    users = pd.read_csv('data/users.csv')
    sessions = pd.read_csv('data/workout_sessions.csv')
    logs = pd.read_csv('data/workout_exercises_log.csv')
    catalog = pd.read_csv('data/exercises_catalog.csv')
    
    # FEATURE ENGINEERING: Agregar datos por sesión
    print(">>> 2. Construyendo variables predictoras por sesión (Feature Engineering)...")
    # Unimos logs con catalog para obtener el perfil anatómico del ejercicio
    logs_with_cat = logs.merge(catalog[['id', 'zonas']], left_on='exercise_id', right_on='id', how='left')
    
    # Calculamos el conteo de impactos por zona anatómica usando dummies
    zonas_dummies = logs_with_cat['zonas'].str.get_dummies(sep=',')
    logs_with_cat = pd.concat([logs_with_cat, zonas_dummies], axis=1)
    
    # Diccionario de agregaciones por sesión
    agg_dict = {'exercise_id': 'count'}
    for zona in zonas_dummies.columns:
        agg_dict[zona] = 'sum'
        
    # Agrupamos los logs a nivel "Sesión"
    session_agg = logs_with_cat.groupby('session_id').agg(agg_dict).reset_index()
    session_agg.rename(columns={'exercise_id': 'num_exercises'}, inplace=True)
    
    # Renombrar columnas de zonas para el modelo
    zone_cols = {z: f"zone_{z}" for z in zonas_dummies.columns}
    session_agg.rename(columns=zone_cols, inplace=True)

    # MERGE FINAL: Unir perfil de usuario con métricas de sesión
    master_df = sessions.merge(users, on='user_id', how='left')
    master_df = master_df.merge(session_agg, on='session_id', how='left')
    
    master_df.fillna(0, inplace=True)
    return master_df

def train_model():
    """
    Ejecuta el pipeline de entrenamiento ML y registra métricas en MLflow.
    """
    df = engineer_features()
    
    # Selección de variables y Target
    cols_to_drop = ['session_id', 'user_id', 'date', 'injury_event']
    X = df.drop(columns=cols_to_drop)
    y = df['injury_event']
    
    categorical_cols = ['experience_level', 'previous_condition']
    continuous_cols = [c for c in X.columns if c not in categorical_cols]
    
    print(">>> 3. Particionando datos (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline de transformación de datos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print(">>> 4. Entrenando XGBoost enfocado en Probabilidades...")
    
    # MLflow Tracking
    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("SmartTrainer_XGB_Relational")
    
    with mlflow.start_run(run_name="Inferencia_Probabilistica"):
        # Manejo de desbalanceo de clases
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, 
            learning_rate=0.03, 
            max_depth=4, 
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        xgb_model.fit(X_train_processed, y_train)
        
        # Cálculo de métricas de confianza
        probs_xgb = xgb_model.predict_proba(X_test_processed)[:, 1]
        roc_auc_xgb = roc_auc_score(y_test, probs_xgb)
        brier_score = brier_score_loss(y_test, probs_xgb)
        
        mlflow.log_metric("roc_auc", roc_auc_xgb)
        mlflow.log_metric("brier_score", brier_score)
        
        print(f"--- Métricas del Modelo ---")
        print(f"ROC-AUC: {roc_auc_xgb:.4f}")
        print(f"Brier Score: {brier_score:.4f}\n")
        
        joblib.dump(xgb_model, 'models/xgb_model.pkl')
        
        # Guardar metadatos de los atributos
        try:
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            all_features = continuous_cols + list(cat_features)
            joblib.dump(all_features, 'models/features_order.pkl')
        except Exception:
            pass

    print(">>> 5. Pipeline completado con éxito.")

if __name__ == "__main__":
    train_model()
