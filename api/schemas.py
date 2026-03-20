"""
schemas.py - Modelos de Datos (Pydantic)
---------------------------------------
Define la estructura de entrada (Request) y salida (Response) de la API.
Garantiza que los datos recibidos (edad, peso, ejercicios) cumplan con 
los tipos y rangos necesarios para que el modelo ML procese la información.
"""
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# --- MODELOS DE REQUEST ---

class ExerciseSetInput(BaseModel):
    """Un bloque de sets para un ejercicio concreto."""
    exercise_id: int = Field(..., description="ID numérico del ejercicio del catálogo.")
    sets: int = Field(..., ge=1, le=10, description="Número total de sets realizados.")
    reps_per_set: List[int] = Field(..., description="Lista de repeticiones por cada set.")
    load_kg_per_set: List[float] = Field(..., description="Lista de pesos (kg) por cada set.")
    effort_sensation: Literal["Bajo", "Moderado", "Alto", "Fallo"] = Field(
        "Moderado", description="Sensación subjetiva de esfuerzo en la serie."
    )


class PredictionRequest(BaseModel):
    """Cuerpo completo de la solicitud de predicción de riesgo."""
    # Perfil del atleta
    age: int = Field(..., ge=14, le=80, description="Edad en años.")
    weight_kg: float = Field(..., ge=40.0, le=200.0, description="Peso corporal en kg.")
    experience_level: Literal["Principiante", "Intermedio", "Avanzado"]
    previous_condition: Literal[
        "Ninguna", "Desgarro LCA", "Hernia Lumbar", "Tendinitis Hombro"
    ] = "Ninguna"
    rest_hours_since_last: int = Field(
        ..., ge=0, le=336, description="Horas de descanso desde la última sesión intensa."
    )
    # Ejercicios de la sesión
    exercises: List[ExerciseSetInput] = Field(
        ..., min_length=1, description="Lista de ejercicios con sus sets y métricas."
    )


# --- MODELOS DE RESPONSE ---

class ZoneAlert(BaseModel):
    """Alerta para una zona corporal específica sobrecargada."""
    zone: str
    exercise_count: int
    recommendation: str
    rest_hours_suggested: int


class PredictionResponse(BaseModel):
    """Respuesta completa con probabilidad, métricas y recomendaciones."""
    # Probabilidad del modelo
    injury_risk_probability: float = Field(..., description="Probabilidad de lesión/sobrecarga (0.0 a 1.0).")
    risk_level: Literal["BAJO", "MODERADO", "ALTO", "CRÍTICO"]

    # Métricas de rendimiento de la sesión
    total_exercises: int
    total_sets: int
    total_reps: int
    total_volume_kg: float = Field(..., description="Tonelaje total levantado en la sesión (series × reps × peso).")
    estimated_cns_load: float = Field(..., description="Carga acumulada en el Sistema Nervioso Central.")
    estimated_peripheral_load: float = Field(..., description="Carga acumulada periférica-muscular.")

    # Recomendaciones del motor de reglas
    alert_zones: List[ZoneAlert]
    general_recommendation: str
