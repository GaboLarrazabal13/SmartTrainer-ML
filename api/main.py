"""
main.py - Servidor Principal FastAPI de SmartTrainer ML
-------------------------------------------------------
Este script orquesta el backend del proyecto. Se encarga de:
1. Cargar los modelos entrenados (XGBoost) y el preprocesador al inicio.
2. Exponer endpoints para consultar el catálogo de ejercicios.
3. Procesar las solicitudes de predicción, calculando fatiga SNC y Periférica
   basándose en el volumen y la intensidad real de la sesión.
4. Ejecutar la inferencia del modelo y devolver recomendaciones accionables.

Ejecución: uvicorn api.main:app --reload
"""
import pandas as pd
import numpy as np
import joblib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionRequest, PredictionResponse
from api.rules_engine import apply_rules

# ─────────────────────────────────────────────
#  Carga de modelos al arrancar el servidor
# ─────────────────────────────────────────────
_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el catálogo de ejercicios y los modelos ML una sola vez al arrancar."""
    try:
        _models["catalog"]      = pd.read_csv("data/exercises_catalog.csv")
        _models["preprocessor"] = joblib.load("models/preprocessor.pkl")
        _models["xgb_model"]    = joblib.load("models/xgb_model.pkl")
        print("✅ SmartTrainer: Modelos cargados correctamente.")
    except FileNotFoundError as e:
        print(f"❌ Error cargando modelos. Ejecuta primero train.py: {e}")
    yield
    _models.clear()


# ─────────────────────────────────────────────
#  Instancia de la App
# ─────────────────────────────────────────────
app = FastAPI(
    title="SmartTrainer ML - API de Predicción de Riesgo",
    description=(
        "Predice la probabilidad de lesión de una sesión de entrenamiento "
        "basándose en el perfil del atleta, el volumen, la intensidad y las "
        "zonas anatómicas involucradas. Proporciona recomendaciones de recuperación "
        "por zona corporal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción limitar al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Helper: calcula métricas y zonas de la sesión
# ─────────────────────────────────────────────
def _compute_session_metrics(request: PredictionRequest) -> tuple[dict, dict]:
    """
    Calcula las métricas agregadas de la sesión (volumen, reps, sets) 
    y la carga de fatiga fisiológica estimada.
    
    Args:
        request: Objeto con los datos del atleta y los ejercicios realizados.
        
    Returns:
        tuple (metrics, zone_counts):
            - metrics: Diccionario con totales de reps, sets, volumen y fatiga.
            - zone_counts: Diccionario con el conteo de impactos por zona anatómica.
    """
    catalog: pd.DataFrame = _models["catalog"]
    weight = request.weight_kg

    effort_mult_map = {"Bajo": 0.8, "Moderado": 1.0, "Alto": 1.2, "Fallo": 1.5}

    total_sets = 0
    total_reps = 0
    total_volume_kg = 0.0
    total_cns = 0.0
    total_periph = 0.0
    zone_counts: dict[str, int] = {}

    for ex_input in request.exercises:
        row_candidates = catalog[catalog["id"] == ex_input.exercise_id]
        if row_candidates.empty:
            raise HTTPException(
                status_code=422,
                detail=f"El exercise_id {ex_input.exercise_id} no existe en el catálogo."
            )
        ex = row_candidates.iloc[0]

        reps = ex_input.reps_per_set
        loads = ex_input.load_kg_per_set
        s = ex_input.sets

        if len(reps) != s or len(loads) != s:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Para exercise_id {ex_input.exercise_id} se indicaron "
                    f"{s} sets pero reps_per_set ({len(reps)}) o "
                    f"load_kg_per_set ({len(loads)}) no coinciden."
                )
            )

        avg_reps = sum(reps) / s
        avg_load = sum(loads) / s
        effort_mult = effort_mult_map.get(ex_input.effort_sensation, 1.0)

        # Volumen bruto del ejercicio
        ex_volume = sum(r * w for r, w in zip(reps, loads))

        # Cálculo de fatiga (fórmula biomecánica simplificada):
        # Fatiga = %Basal_Ejercicio * Multiplicador_Volumen * Multiplicador_Esfuerzo
        volume_fac = (avg_reps * avg_load * s) / weight
        if avg_load == 0:
            volume_fac = avg_reps * s * 0.5
        net_fatigue = (ex["fatigue_pct"] / 100.0) * volume_fac * effort_mult
        
        # Acumulación de métricas según tipo de impacto (SNC o muscular)
        total_sets += s
        total_reps += sum(reps)
        total_volume_kg += ex_volume

        if ex["type"] == "SNC":
            total_cns += net_fatigue
        else:
            total_periph += net_fatigue

        # Mapeo de zonas anatómicas afectadas para el motor de reglas
        for zona in str(ex["zonas"]).split(","):
            z = zona.strip()
            zone_counts[z] = zone_counts.get(z, 0) + 1

    metrics = {
        "total_exercises": len(request.exercises),
        "total_sets": total_sets,
        "total_reps": total_reps,
        "total_volume_kg": total_volume_kg,
        "total_cns": total_cns,
        "total_periph": total_periph,
        "rest_hours": request.rest_hours_since_last,
    }

    return metrics, zone_counts


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "SmartTrainer ML API está activa. Visita /docs para la documentación interactiva."}


@app.get("/health", tags=["Health"])
def health():
    models_loaded = "xgb_model" in _models and "preprocessor" in _models
    return {"status": "healthy" if models_loaded else "degraded", "models_loaded": models_loaded}


@app.get("/catalog", tags=["Ejercicios"])
def get_catalog(body_part: str | None = None):
    """
    Devuelve el catálogo completo de ejercicios.
    Filtra opcionalmente por: Superior, Inferior, Core.
    """
    catalog: pd.DataFrame = _models.get("catalog", pd.DataFrame())
    if catalog.empty:
        raise HTTPException(status_code=503, detail="Catálogo no disponible. Reinicia el servidor.")
    
    if body_part:
        catalog = catalog[catalog["body_part"].str.lower() == body_part.lower()]
        if catalog.empty:
            raise HTTPException(status_code=404, detail=f"Zona '{body_part}' no encontrada. Usa: Superior, Inferior, Core.")
    
    return catalog.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint principal. Recibe el perfil del atleta y los ejercicios de su sesión
    y devuelve la probabilidad de lesión junto a recomendaciones de recuperación por zona.
    """
    if "xgb_model" not in _models:
        raise HTTPException(status_code=503, detail="Modelos ML no disponibles. Reinicia el servidor.")

    # 1. Calcular métricas de sesión
    metrics, zone_counts = _compute_session_metrics(request)

    # 2. Construir el vector de features para el modelo
    row_data = {
        "age": request.age,
        "weight_kg": request.weight_kg,
        "experience_level": request.experience_level,
        "previous_condition": request.previous_condition,
        "rest_hours_since_last": request.rest_hours_since_last,
        "total_cns_fatigue": metrics["total_cns"],
        "total_periph_fatigue": metrics["total_periph"],
        "num_exercises": metrics["total_exercises"],
    }
    # Añadir conteos de zonas como features (zone_lumbar, zone_rodillas, etc.)
    for z, count in zone_counts.items():
        row_data[f"zone_{z}"] = count

    df_inference = pd.DataFrame([row_data])

    # Alinear con las columnas esperadas por el preprocesador
    required_cols = list(_models["preprocessor"].feature_names_in_)
    for col in required_cols:
        if col not in df_inference.columns:
            df_inference[col] = 0
    df_inference = df_inference[required_cols]

    # 3. Predicción XGBoost
    X_processed = _models["preprocessor"].transform(df_inference)
    risk_prob = float(_models["xgb_model"].predict_proba(X_processed)[0][1])

    # 4. Motor de Reglas → Response final
    return apply_rules(risk_prob, zone_counts, metrics)
