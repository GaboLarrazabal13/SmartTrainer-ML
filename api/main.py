"""
main.py - Servidor Principal FastAPI de SmartTrainer ML
-------------------------------------------------------
"""
import pandas as pd
import numpy as np
import joblib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime

from api.schemas import (
    PredictionRequest, PredictionResponse,
    UserCreate, UserLogin, WorkoutSessionCreate,
    ExerciseCreate, InjuryConditionCreate
)
from api.rules_engine import apply_rules
from api.database import get_db
from api.models import Exercise, InjuryCondition, User, WorkoutSession

# ─────────────────────────────────────────────
#  Carga de modelos al arrancar el servidor
# ─────────────────────────────────────────────
_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos ML una sola vez al arrancar."""
    try:
        _models["preprocessor"] = joblib.load("models/preprocessor.pkl")
        _models["xgb_model"]    = joblib.load("models/xgb_model.pkl")
        print("✅ SmartTrainer: Modelos cargados correctamente.")
    except FileNotFoundError as e:
        print(f"❌ Error cargando modelos. Ejecuta primero train.py: {e}")
    yield
    _models.clear()


app = FastAPI(
    title="SmartTrainer ML - API de Predicción de Riesgo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Helper: calcula métricas y zonas de la sesión
# ─────────────────────────────────────────────
def _compute_session_metrics(request: PredictionRequest, db: Session) -> tuple[dict, dict]:
    weight = request.weight_kg
    effort_mult_map = {"Bajo": 0.8, "Moderado": 1.0, "Alto": 1.2, "Fallo": 1.5}

    total_sets = 0
    total_reps = 0
    total_volume_kg = 0.0
    total_cns = 0.0
    total_periph = 0.0
    zone_counts: dict[str, int] = {}

    for ex_input in request.exercises:
        ex = db.query(Exercise).filter(Exercise.id == ex_input.exercise_id).first()
        if not ex:
            raise HTTPException(status_code=422, detail=f"El exercise_id {ex_input.exercise_id} no existe en la DB.")

        reps = ex_input.reps_per_set
        loads = ex_input.load_kg_per_set
        s = ex_input.sets

        if len(reps) != s or len(loads) != s:
            raise HTTPException(status_code=422, detail="Mismatched sets/reps/loads")

        avg_reps = sum(reps) / s
        avg_load = sum(loads) / s
        effort_mult = effort_mult_map.get(ex_input.effort_sensation, 1.0)
        ex_volume = sum(r * w for r, w in zip(reps, loads))

        volume_fac = (avg_reps * avg_load * s) / weight
        if avg_load == 0:
            volume_fac = avg_reps * s * 0.5
        
        # Determine fatigue base from DB (e.g. using CNS and Periph factors as base)
        # Fallback to older mechanism if fatigue_pct is strictly required (we infer from factors here)
        cns_f = ex.cns_impact_factor or 0.0
        periph_f = ex.periph_impact_factor or 0.0
        fatigue_pct = (cns_f + periph_f) * 100 / 2.0  # Approx

        net_fatigue = (fatigue_pct / 100.0) * volume_fac * effort_mult
        
        total_sets += s
        total_reps += sum(reps)
        total_volume_kg += ex_volume

        if cns_f >= periph_f:
            total_cns += net_fatigue
        else:
            total_periph += net_fatigue

        for zona in str(ex.zonas).split(","):
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
    return {"status": "ok"}

@app.get("/catalog", tags=["Ejercicios"])
def get_catalog(body_part: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Exercise)
    if body_part:
        query = query.filter(Exercise.body_part.ilike(f"%{body_part}%"))
    results = query.all()
    # Serialize to list of dicts for front-end compatibility
    return [{"id": r.id, "name": r.name, "body_part": r.body_part, "zonas": r.zonas, "cns_impact_factor": r.cns_impact_factor, "periph_impact_factor": r.periph_impact_factor} for r in results]

@app.get("/injuries", tags=["Lesiones"])
def get_injuries(db: Session = Depends(get_db)):
    results = db.query(InjuryCondition).all()
    return [{"id": r.id, "zona": r.zona_articulacion, "lesion": r.lesion_comun, "ejercicio": r.ejercicio_riesgo, "nivel": r.nivel_esfuerzo_rpe, "fatiga": r.fatiga_estimada, "tipo": r.tipo_fatiga} for r in results]

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    if "xgb_model" not in _models:
        raise HTTPException(status_code=503, detail="Modelos ML no disponibles.")

    metrics, zone_counts = _compute_session_metrics(request, db)

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
    for z, count in zone_counts.items():
        row_data[f"zone_{z}"] = count

    df_inference = pd.DataFrame([row_data])

    required_cols = list(_models["preprocessor"].feature_names_in_)
    for col in required_cols:
        if col not in df_inference.columns:
            df_inference[col] = 0
    df_inference = df_inference[required_cols]

    X_processed = _models["preprocessor"].transform(df_inference)
    risk_prob = float(_models["xgb_model"].predict_proba(X_processed)[0][1])

    # Buscar lesión en BD para generar alerta dinámica
    injury_alert = ""
    if request.previous_condition and request.previous_condition != "Ninguna":
        injury_rec = db.query(InjuryCondition).filter(InjuryCondition.lesion_comun == request.previous_condition).first()
        if injury_rec:
            injury_alert = f"🚩 ALERTA CLÍNICA: Historial de {request.previous_condition}. Zona vulnerable: {injury_rec.zona_articulacion}. Evitar: {injury_rec.ejercicio_riesgo}."

    metrics["previous_condition"] = request.previous_condition
    metrics["injury_alert"] = injury_alert
    return apply_rules(risk_prob, zone_counts, metrics)

# ─────────────────────────────────────────────
#  Módulo de Usuarios
# ─────────────────────────────────────────────

@app.post("/register", tags=["Usuarios"])
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="El email ya está registrado.")
    
    new_user = User(
        email=user.email,
        age=user.age,
        weight=user.weight,
        height=user.height,
        experience_level=user.experience_level,
        injury_history_id=user.injury_history_id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "Usuario registrado exitosamente", "email": new_user.email}

@app.post("/login", tags=["Usuarios"])
def login_user(login: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    return {
        "email": user.email,
        "age": user.age,
        "weight": user.weight,
        "height": user.height,
        "experience_level": user.experience_level,
        "injury_history_id": user.injury_history_id
    }

# ─────────────────────────────────────────────
#  Módulo MLOps y Registro de Sesiones
# ─────────────────────────────────────────────

def _trigger_mlops_retraining(db: Session):
    """ Función que simula o dispara el pipeline de MLOps en backgrouund """
    untrained_sessions = db.query(WorkoutSession).filter(WorkoutSession.is_trained == 0).all()
    if len(untrained_sessions) >= 1000:
        print("🚀 [MLOps] 1000 nuevas sesiones detectadas. Iniciando Reentrenamiento Automático de XGBoost...")
        # Aquí iría un subprocess o celery task que corra models/train.py usando la DB actual
        # ...
        # Marcamos como procesadas
        for session in untrained_sessions:
            session.is_trained = 1
        db.commit()
        print("✅ [MLOps] Pipeline completado con éxito. Sesiones marcadas.")

@app.post("/workouts/log", tags=["MLOps"])
def log_workout_session(session_data: WorkoutSessionCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # 1. Validar usuario
    user = db.query(User).filter(User.email == session_data.user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Debe estar logueado para guardar sesiones.")

    # 2. Guardar sesión
    new_session = WorkoutSession(
        user_email=session_data.user_email,
        date=datetime.now().isoformat(),
        exercise_ids=session_data.exercise_ids,
        total_cns_fatigue=session_data.total_cns_fatigue,
        total_periph_fatigue=session_data.total_periph_fatigue,
        risk_probability=session_data.risk_probability,
        is_trained=0
    )
    db.add(new_session)
    db.commit()
    
    # 3. Lanzar verificación asíncrona de MLOps
    background_tasks.add_task(_trigger_mlops_retraining, db)
    
    return {"message": "Sesión registrada con éxito para MLOps."}

# ─────────────────────────────────────────────
#  Módulo Administrador (Catálogo)
# ─────────────────────────────────────────────

@app.post("/admin/exercises", tags=["Administrador"])
def add_exercise(ex: ExerciseCreate, db: Session = Depends(get_db)):
    existing = db.query(Exercise).filter(Exercise.name == ex.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="El ejercicio ya existe.")
    
    new_ex = Exercise(
        name=ex.name,
        body_part=ex.body_part,
        zonas=ex.zonas,
        cns_impact_factor=ex.cns_impact_factor,
        periph_impact_factor=ex.periph_impact_factor
    )
    db.add(new_ex)
    db.commit()
    return {"message": "Ejercicio añadido exitosamente"}

@app.post("/admin/injuries", tags=["Administrador"])
def add_injury(inj: InjuryConditionCreate, db: Session = Depends(get_db)):
    new_inj = InjuryCondition(
        zona_articulacion=inj.zona_articulacion,
        lesion_comun=inj.lesion_comun,
        ejercicio_riesgo=inj.ejercicio_riesgo,
        nivel_esfuerzo_rpe=inj.nivel_esfuerzo_rpe,
        fatiga_estimada=inj.fatiga_estimada,
        tipo_fatiga=inj.tipo_fatiga
    )
    db.add(new_inj)
    db.commit()
    return {"message": "Condición clínica añadida catalogada"}

