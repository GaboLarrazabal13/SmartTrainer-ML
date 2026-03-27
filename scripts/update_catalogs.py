import os
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.models import Base, Exercise, InjuryCondition
from dotenv import load_dotenv

def classify_body_part(raw_part, name):
    raw = raw_part.lower()
    n = name.lower()
    
    # Inferior
    if any(k in raw for k in ["pierna", "cuád", "isquio", "glúteo", "pantorrilla", "aductor", "cadena post"]):
        return "Inferior"
    if any(k in n for k in ["peso muerto", "sentadilla", "squat", "clean", "swing", "zancada", "lunge"]):
        return "Inferior"
        
    # Core
    if any(k in raw for k in ["core", "abdomen", "oblicuo"]):
        if "hombro" in raw or "brazo" in raw or "espalda" in raw:
            return "Superior"
        return "Core"
        
    # Superior
    return "Superior"

# Cargar variables de entorno
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL no encontrada en .env")
    exit(1)

# Configurar SQLAlchemy
# Usamos sslmode=require para compatibilidad con Supabase desde fuera
if "sslmode" not in DATABASE_URL:
    if "?" in DATABASE_URL:
        DATABASE_URL += "&sslmode=require"
    else:
        DATABASE_URL += "?sslmode=require"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def parse_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # --- PARSE INJURIES ---
    injuries = []
    injury_section = re.search(r"Anatomía del Riesgo: Las 20 Lesiones y Fatiga en el Entrenamiento Pesado(.*?)(?=--------------------------------------------------------------------------------|$)", content, re.S)
    if injury_section:
        lines = injury_section.group(1).strip().split("\n")
        # Identificar el inicio de los datos (buscando "1. ")
        start_index = 0
        for idx, line in enumerate(lines):
            if "1. " in line:
                start_index = idx
                break
        
        data_lines = [l.strip() for l in lines[start_index:] if l.strip()]
        
        for i in range(0, len(data_lines), 6):
            if i + 5 < len(data_lines):
                zona = data_lines[i].split(". ", 1)[-1] if ". " in data_lines[i] else data_lines[i]
                injuries.append({
                    "zona_articulacion": zona,
                    "lesion_comun": data_lines[i+1],
                    "ejercicio_riesgo": data_lines[i+2],
                    "nivel_esfuerzo_rpe": data_lines[i+3],
                    "fatiga_estimada": data_lines[i+4],
                    "tipo_fatiga": data_lines[i+5]
                })

    # --- PARSE EXERCISES ---
    exercises = []
    exercise_section = re.search(r"Guía de Fatiga y Desgaste \(100 Ejercicios\)(.*?)(?=Consideraciones sobre la Copa|$)", content, re.S)
    if not exercise_section:
         exercise_section = re.search(r"Guía Maestra de Fatiga y Desgaste en 100 Ejercicios(.*?)(?=Consideraciones sobre la Copa|$)", content, re.S)

    if exercise_section:
        lines = exercise_section.group(1).strip().split("\n")
        
        # Identificar el inicio de los datos (buscando "1" solo en una línea)
        start_index = 0
        for idx, line in enumerate(lines):
            if line.strip() == "1":
                start_index = idx
                break
        
        data_lines = [l.strip() for l in lines[start_index:] if l.strip()]
        
        for i in range(0, len(data_lines), 6):
            if i + 5 < len(data_lines):
                # Validar que data_lines[i] sea un número (el ID)
                if not data_lines[i].isdigit():
                    continue

                name = data_lines[i+1]
                original_body_part = data_lines[i+2]
                effort = data_lines[i+3]
                fatigue_str = data_lines[i+4].replace("%", "").strip()
                try:
                    fatigue_val = float(fatigue_str)
                except:
                    fatigue_val = 50.0
                
                zonas_afectadas = data_lines[i+5]
                # Preservamos la información granular añadiéndola a la zona
                zonas_afectadas = f"[{original_body_part}] {zonas_afectadas}"
                
                # Clasificamos a las grandes 3 familias para que el frontend funcione
                body_part = classify_body_part(original_body_part, name)
                
                total_factor = fatigue_val / 50.0 
                
                if any(kw in effort for kw in ["Muy Alto", "Alto"]):
                    cns = total_factor * 0.6
                    periph = total_factor * 0.4
                else:
                    cns = total_factor * 0.4
                    periph = total_factor * 0.6

                exercises.append({
                    "name": name,
                    "body_part": body_part,
                    "zonas": zonas_afectadas,
                    "cns_impact_factor": round(cns, 2),
                    "periph_impact_factor": round(periph, 2)
                })

    return injuries, exercises

def sync_to_db(injuries, exercises):
    db = SessionLocal()
    try:
        # 1. Update Injuries (Clear and reload)
        print(f"🔄 Sincronizando {len(injuries)} lesiones...")
        db.query(InjuryCondition).delete()
        for inj_data in injuries:
            db.add(InjuryCondition(**inj_data))
        
        # 2. Update Exercises (Clear and reload to avoid ID conflicts and ensure exact 100 items)
        print(f"🔄 Sincronizando {len(exercises)} ejercicios...")
        db.query(Exercise).delete()
        for ex_data in exercises:
            db.add(Exercise(**ex_data))
        
        db.commit()
        print("✅ Sincronización completada con éxito.")
    except Exception as e:
        db.rollback()
        print(f"❌ ERROR durante la sincronización: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    inj_list, ex_list = parse_data("/tmp/notebooklm_data.txt")
    if not inj_list or not ex_list:
        print(f"⚠️ Alerta: Se parsearon {len(inj_list)} lesiones y {len(ex_list)} ejercicios. Revisa los regex.")
    
    sync_to_db(inj_list, ex_list)
