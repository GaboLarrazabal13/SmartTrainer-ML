import pandas as pd
import json
import os
import sys

# Ajustar PYTHONPATH para permitir importaciones desde la raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.database import engine, Base, SessionLocal
from api.models import Exercise, InjuryCondition

def migrate_data():
    print("Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    # 1. Migrar Ejercicios desde CSV
    print("Migrando ejercicios desde data/exercises_catalog.csv...")
    csv_path = "data/exercises_catalog.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Revisar si ya existe
            existing = db.query(Exercise).filter_by(id=row["id"]).first()
            if not existing:
                ex = Exercise(
                    id=row["id"],
                    name=row["name"],
                    body_part=row["body_part"],
                    zonas=row["zonas"],
                    cns_impact_factor=row["cns_impact_factor"],
                    periph_impact_factor=row["periph_impact_factor"]
                )
                db.add(ex)
        db.commit()
    else:
        print(f"Advertencia: No se encontró {csv_path}")

    # 2. Seed de Lesiones desde NotebookLM
    print("Sembrando base de datos de lesiones obtenidas desde NotebookLM...")
    lesiones_notebooklm = [
        {"zona": "Zona Lumbar (L4-L5)", "lesion": "Hernia discal, distensión", "ee": "Peso Muerto, Sentadilla [1, 2]", "rpe": "9–10 (Máximo)", "fe": "95–100% [2]", "tf": "SNC / Sistémica [3]"},
        {"zona": "Manguito Rotador", "lesion": "Pinzamiento, desgarro", "ee": "Press Militar, Press Banca [4, 5]", "rpe": "8–10 (Alto)", "fe": "80–85% [2]", "tf": "SNC y Periférica [6, 7]"},
        {"zona": "Rodilla (Menisco/LCA)", "lesion": "Desgarro, esguince", "ee": "Sentadilla, Prensa [8-10]", "rpe": "8–10 (Alto)", "fe": "90–95% [2]", "tf": "SNC / Sistémica [6]"},
        {"zona": "Codo (Lateral/Medial)", "lesion": "Epicondilitis (Tenista/Golfer)", "ee": "Curls, Remos pesados [11]", "rpe": "7–9 (Vigoro)", "fe": "30–50% [2]", "tf": "Periférica [3]"},
        {"zona": "Muñeca", "lesion": "Tendinitis, esguince", "ee": "Press Banca, Curl Barra [11, 12]", "rpe": "7–9 (Vigoro)", "fe": "30–80% [2]", "tf": "Periférica [3]"},
        {"zona": "Bíceps (Largo)", "lesion": "Tendinitis, desgarro", "ee": "Peso Muerto (agarre mixto) [13, 14]", "rpe": "9–10 (Máximo)", "fe": "95–100% [2]", "tf": "SNC / Sistémica [6]"},
        {"zona": "Columna Cervical", "lesion": "Distensión, dolor agudo", "ee": "Crunches, Press tras nuca [1, 14]", "rpe": "6–8 (Moderado)", "fe": "20–25% [2]", "tf": "Periférica [3]"},
        {"zona": "Isquiotibiales", "lesion": "Desgarro o distensión", "ee": "Peso Muerto Rumano [15]", "rpe": "8–10 (Alto)", "fe": "85–90% [2]", "tf": "SNC y Periférica [6]"},
        {"zona": "Pectoral Mayor", "lesion": "Desgarro muscular", "ee": "Press de Banca pesado [13, 16]", "rpe": "9–10 (Máximo)", "fe": "80–85% [2]", "tf": "SNC y Periférica [6]"},
        {"zona": "Tobillo / Aquiles", "lesion": "Ruptura, esguince", "ee": "Saltos al cajón, Cleans [17, 18]", "rpe": "8–10 (Alto)", "fe": "90–95% [2]", "tf": "SNC / Sistémica [6]"},
    ]

    for item in lesiones_notebooklm:
         existing = db.query(InjuryCondition).filter_by(zona_articulacion=item["zona"]).first()
         if not existing:
             ic = InjuryCondition(
                 zona_articulacion=item["zona"],
                 lesion_comun=item["lesion"],
                 ejercicio_riesgo=item["ee"],
                 nivel_esfuerzo_rpe=item["rpe"],
                 fatiga_estimada=item["fe"],
                 tipo_fatiga=item["tf"]
             )
             db.add(ic)

    db.commit()
    db.close()
    print("¡Migración completada con éxito!")

if __name__ == "__main__":
    migrate_data()
