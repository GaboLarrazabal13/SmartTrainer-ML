from sqlalchemy import Column, Integer, String, Float
from api.database import Base

class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    body_part = Column(String)  # Inferior, Superior, Core
    zonas = Column(String)      # Cuadriceps, Isquios, Pecho, etc.
    cns_impact_factor = Column(Float)
    periph_impact_factor = Column(Float)


class InjuryCondition(Base):
    __tablename__ = "injury_conditions"

    id = Column(Integer, primary_key=True, index=True)
    zona_articulacion = Column(String, index=True)
    lesion_comun = Column(String)
    ejercicio_riesgo = Column(String)
    nivel_esfuerzo_rpe = Column(String)
    fatiga_estimada = Column(String)
    tipo_fatiga = Column(String)

class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True, index=True)
    age = Column(Integer)
    weight = Column(Float)
    height = Column(Float)
    experience_level = Column(String)
    injury_history_id = Column(Integer) # Opcional foreign key manual a injury_conditions.id

class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    date = Column(String)  # ISO format string or timestamp
    exercise_ids = Column(String) # JSON string
    total_cns_fatigue = Column(Float)
    total_periph_fatigue = Column(Float)
    risk_probability = Column(Float)
    is_trained = Column(Integer, default=0) # 0 False, 1 True flag for MLOps
