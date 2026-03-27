"""
rules_engine.py - Motor de Recomendaciones y Reglas Fisiológicas
----------------------------------------------------------------
Este módulo actúa como el 'intérprete' del modelo de Machine Learning.
Toma la probabilidad cruda calculada por el XGBoost y la cruza con:
1. Umbrales de riesgo clínico (Bajo, Moderado, Alto, Crítico).
2. Conteo de ejercicios por zona anatómica (para detectar sobrecarga local).
3. Historial de descanso previo.

Traduce la probabilidad matemática en consejos de salud y tiempos de descanso 
específicos para cada zona anatómica involucrada.
"""
from api.schemas import ZoneAlert, PredictionResponse

# Umbrales de riesgo
THRESHOLDS = {
    "BAJO":     (0.00, 0.30),
    "MODERADO": (0.30, 0.50),
    "ALTO":     (0.50, 0.70),
    "CRÍTICO":  (0.70, 1.01),
}

# Horas de descanso recomendadas por zona según impacto (NotebookLM 2026)
ZONE_REST_MAP = {
    "lumbar":        (72, "⛔ RIESGO SNC ALTO: Evitar Peso Muerto y Sentadilla. Peligro de hernia L4-L5."),
    "columna":       (72, "⛔ RIESGO SISTÉMICO: Evitar cargas axiales. Priorizar estabilidad de core."),
    "rodillas":      (48, "⚠️ RIESGO LCA/Menisco: Evitar Sentadilla y Prensa. Vigilar valgo dinámico."),
    "hombros":       (48, "⚠️ RIESGO MANGUITO: Evitar Press Militar y Press Banca a 90°. Usar 'thumbs up' en laterales."),
    "cadera":        (48, "Evitar Hip Thrust y Peso Muerto Sumo. Vigilar pinzamiento acetabular."),
    "codos":         (24, "Riesgo de Epicondilitis: Reducir curls y remos pesados. Revisar técnica de agarre."),
    "muñecas":       (24, "Riesgo de Tendinitis: Evitar hiperextensión en Press Banca y Curls pesados."),
    "cervicales":    (48, "Evitar Press tras nuca y tirones de cuello en Crunches. Peligro de distensión."),
    "isquios":       (48, "Riesgo de Desgarro: Evitar P. Muerto Rumano y Leg Curl explosivo."),
    "pecho":         (48, "Riesgo de Desgarro Pectoral: Evitar Press de Banca pesado con rebote."),
    "dorsales":      (48, "Evitar Dominadas con rotación interna excesiva si hay inestabilidad."),
    "tobillos":      (24, "Riesgo de Aquiles: Evitar saltos al cajón y aceleraciones súbitas."),
    "aductores":     (24, "Evitar Sentadilla Sumo. Riesgo de distensión en aductores."),
    "core":          (24, "Mantener 'Abdominal Bracing'. Evitar arqueo lumbar en levantamientos."),
    "abdomen":       (24, "Priorizar 'Dead Bug' o 'Bird-Dog' si hay antecedentes de dolor lumbar.")
}

DEFAULT_REST = (24, "Seguir protocolo de recuperación periférica estándar: 24-48h.")

def classify_risk(probability: float) -> str:
    for level, (lo, hi) in THRESHOLDS.items():
        if lo <= probability < hi:
            return level
    return "CRÍTICO"


def build_zone_alerts(zone_counts: dict[str, int]) -> list[ZoneAlert]:
    """Genera alertas para todas las zonas que superan el umbral de 2 ejercicios."""
    alerts = []
    for zone, count in zone_counts.items():
        if count >= 2:
            rest_h, advice = ZONE_REST_MAP.get(zone, DEFAULT_REST)
            alerts.append(ZoneAlert(
                zone=zone.upper(),
                exercise_count=count,
                recommendation=advice,
                rest_hours_suggested=rest_h,
            ))
    return alerts


def build_general_recommendation(risk_level: str, alert_zones: list[ZoneAlert], rest_hours: int, condition: str = "Ninguna") -> str:
    if risk_level == "CRÍTICO":
        if alert_zones:
            zones = ", ".join(a.zone for a in alert_zones)
            return (
                f"⛔ RIESGO CRÍTICO detεctado. Las zonas [{zones}] están al límite. "
                f"Se recomienda descanso activo de al menos 72h y revisión con un profesional "
                f"si hay dolor articular persistente."
            )
        return (
            "⛔ RIESGO CRÍTICO sistémico. El volumen e intensidad combinados superan ampliamente "
            f"tu ventana de recuperación de {rest_hours}h. Descansa 48-72h."
        )
    elif risk_level == "ALTO":
        return (
            "⚠️ Zona de riesgo ALTO. Tu carga de hoy es elevada. "
            "Asegúrate de dormir ≥ 8h y mantener hidratación y nutrición adecuadas antes de tu próxima sesión."
        )
    elif risk_level == "MODERADO":
        return (
            "🟡 Riesgo MODERADO. Sesión intensa pero dentro de márgenes. "
            "Considera una sesión de recuperación activa (movilidad, cardio suave) antes de volver a cargar con alta intensidad."
        )
    else:
        base_rec = (
            "🟢 Excelente. Sesión en zona ÓPTIMA de entrenamiento. "
            "Tu carga, volumen y descanso están bien balanceados. Puedes progresar en tu próxima sesión."
        )
    
    # Añadir alerta específica por historial
    history_alert = condition  # Condition here actually holds the string of metric["injury_alert"] now (we will rename parameter in the caller or here)
    if history_alert and history_alert != "Ninguna":
        return f"{history_alert}\n\n{base_rec}"
    return base_rec


def apply_rules(
    risk_prob: float,
    zone_counts: dict[str, int],
    metrics: dict,
) -> PredictionResponse:
    """Punto de entrada principal del motor de reglas."""
    risk_level = classify_risk(risk_prob)
    alert_zones = build_zone_alerts(zone_counts)
    
    # La condición en sí ahora pasa el mensaje compilado por la BD (si existe)
    injury_alert = metrics.get("injury_alert", "")
    general_rec = build_general_recommendation(risk_level, alert_zones, metrics["rest_hours"], injury_alert)

    return PredictionResponse(
        injury_risk_probability=round(risk_prob, 4),
        risk_level=risk_level,
        total_exercises=metrics["total_exercises"],
        total_sets=metrics["total_sets"],
        total_reps=metrics["total_reps"],
        total_volume_kg=round(metrics["total_volume_kg"], 1),
        estimated_cns_load=round(metrics["total_cns"], 2),
        estimated_peripheral_load=round(metrics["total_periph"], 2),
        alert_zones=alert_zones,
        general_recommendation=general_rec,
    )
