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

# Horas de descanso recomendadas por zona según impacto (>=2 ejercicios)
ZONE_REST_MAP = {
    "lumbar":        (72, "Evitar ejercicios de cadena posterior: Peso Muerto, Remo Inclinado, Sentadillas."),
    "columna":       (72, "Evitar cualquier ejercicio axial con carga sobre la columna vertebral."),
    "rodillas":      (48, "Evitar sentadillas, prensa y cualquier flexión de rodilla con carga."),
    "hombros":       (48, "Evitar press militar, elevaciones laterales y remos por encima de la cabeza."),
    "cadera":        (48, "Evitar hip thrust, peso muerto sumo y ejercicios unilaterales de cadera."),
    "codos":         (24, "Evitar curls y extensiones de tríceps con carga media-alta."),
    "muñecas":       (24, "Reducir peso en ejercicios de agarre y evitar pronación forzada."),
    "cervicales":    (48, "Evitar press militar, dominadas detrás de la nuca y jalones verticales."),
    "isquios":       (48, "Evitar peso muerto rumano, leg curl y zancadas profundas."),
    "pecho":         (48, "Evitar press de banca y aperturas hasta que la tensión remita."),
    "dorsales":      (48, "Evitar jalones y remos hasta recuperación completa."),
    "tobillos":      (24, "Reducir ejercicios con saltos o cambios de dirección bruscos."),
    "aductores":     (24, "Evitar sentadilla sumo y ejercicios de aducción con carga."),
    "core":          (24, "Reducir plancha y trabajo de estabilización si hay hiperlordosis."),
    "abdomen":       (24, "Reducir crunches y leg raises si hay dolor lumbar asociado.")
}

DEFAULT_REST = (24, "Evitar ejercicios directos sobre esta zona hasta la recuperación.")


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


def build_general_recommendation(risk_level: str, alert_zones: list[ZoneAlert], rest_hours: int) -> str:
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
        return (
            "🟢 Excelente. Sesión en zona ÓPTIMA de entrenamiento. "
            "Tu carga, volumen y descanso están bien balanceados. Puedes progresar en tu próxima sesión."
        )


def apply_rules(
    risk_prob: float,
    zone_counts: dict[str, int],
    metrics: dict,
) -> PredictionResponse:
    """Punto de entrada principal del motor de reglas."""
    risk_level = classify_risk(risk_prob)
    alert_zones = build_zone_alerts(zone_counts)
    general_rec = build_general_recommendation(risk_level, alert_zones, metrics["rest_hours"])

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
