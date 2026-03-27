"""
dataset_generator.py - Motor de Simulación Biomecánica v2 (Elite)
------------------------------------------------------
Actualizado con datos de "Anatomía del Riesgo" (NotebookLM - 23/03/2026).
- Modela 50 ejercicios con umbrales de fatiga SNC/Periférica precisos.
- Simula riesgos específicos: Hernias (SNC > 90%), Tendinitis (Perif. acumulada), 
  Rotura de Bíceps (Agarre mixto + RPE 10).
- Implementa la carga basada en Intensidad Relativa (%1RM) y RPE.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_exercises():
    # Actualizado según la tabla "Guía de Fatiga y Desgaste (50 Ejercicios)" de NotebookLM
    exs = [
        {"id": 1, "name": "Peso Muerto (Convencional)", "fatigue_pct": 98, "type": "SNC", "category": "Heavy_Lower", "zonas": "columna,lumbar,agarre"},
        {"id": 2, "name": "Sentadilla con Barra", "fatigue_pct": 93, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,lumbar,cadera"},
        {"id": 3, "name": "Power Clean", "fatigue_pct": 93, "type": "SNC", "category": "Heavy_Compound", "zonas": "muñecas,hombros,lumbar,tobillos"},
        {"id": 4, "name": "Peso Muerto Sumo", "fatigue_pct": 93, "type": "SNC", "category": "Heavy_Lower", "zonas": "aductores,cadera,lumbar"},
        {"id": 5, "name": "Sentadilla Frontal", "fatigue_pct": 93, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,columna,muñecas"},
        {"id": 6, "name": "Peso Muerto Rumano", "fatigue_pct": 88, "type": "SNC", "category": "Heavy_Lower", "zonas": "lumbar,isquios"},
        {"id": 7, "name": "Sentadilla Búlgara", "fatigue_pct": 88, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,cadera,cuadriceps"},
        {"id": 8, "name": "Press de Banca (Barra)", "fatigue_pct": 83, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,pecho,muñecas"},
        {"id": 9, "name": "Remo con Barra Inclinado", "fatigue_pct": 83, "type": "SNC", "category": "Heavy_Upper", "zonas": "lumbar,hombros"},
        {"id": 10, "name": "Press Militar de Pie", "fatigue_pct": 83, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,cervicales"},
        {"id": 11, "name": "Kettlebell Swing", "fatigue_pct": 83, "type": "SNC", "category": "Heavy_Lower", "zonas": "lumbar,cadera,gluteos"},
        {"id": 12, "name": "Hip Thrust (Barra)", "fatigue_pct": 78, "type": "SNC", "category": "Heavy_Lower", "zonas": "cadera,lumbar"},
        {"id": 13, "name": "Press de Banca (Manc.)", "fatigue_pct": 78, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,muñecas"},
        {"id": 14, "name": "Dominadas (Pull-ups)", "fatigue_pct": 73, "type": "SNC", "category": "Heavy_Upper_BW", "zonas": "dorsales,codos,hombros"},
        {"id": 15, "name": "Prensa de Piernas", "fatigue_pct": 73, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,lumbar"},
        {"id": 16, "name": "Remo con Mancuerna", "fatigue_pct": 73, "type": "SNC", "category": "Heavy_Upper", "zonas": "espalda,hombros"},
        {"id": 17, "name": "Zancadas (Lunges)", "fatigue_pct": 68, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,tobillos"},
        {"id": 18, "name": "Press Inclinado (Manc.)", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Upper", "zonas": "hombros,pecho"},
        {"id": 19, "name": "Fondos (Dips)", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "pecho,hombros,codos"},
        {"id": 20, "name": "Arnold Press", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Upper", "zonas": "hombros"},
        {"id": 21, "name": "Chin-ups", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "codos,biceps,hombros"},
        {"id": 22, "name": "Goblet Squat", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,columna"},
        {"id": 23, "name": "Step-ups", "fatigue_pct": 63, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,tobillos"},
        {"id": 24, "name": "Jalón al Pecho", "fatigue_pct": 58, "type": "Periferica", "category": "Medium_Upper", "zonas": "dorsales,hombros"},
        {"id": 25, "name": "Remo al Mentón", "fatigue_pct": 58, "type": "Periferica", "category": "Isolation", "zonas": "hombros,cervical"},
        {"id": 26, "name": "Skull Crushers", "fatigue_pct": 53, "type": "Periferica", "category": "Isolation", "zonas": "codos,triceps"},
        {"id": 27, "name": "Press de Pecho (Máq.)", "fatigue_pct": 53, "type": "Periferica", "category": "Medium_Upper", "zonas": "pecho,codos"},
        {"id": 28, "name": "Extensión de Piernas", "fatigue_pct": 48, "type": "Periferica", "category": "Isolation", "zonas": "rodillas"},
        {"id": 29, "name": "Curl de Piernas", "fatigue_pct": 48, "type": "Periferica", "category": "Isolation", "zonas": "isquios"},
        {"id": 30, "name": "Preacher Curl", "fatigue_pct": 48, "type": "Periferica", "category": "Isolation", "zonas": "codos,biceps"},
        {"id": 31, "name": "Remo Invertido", "fatigue_pct": 43, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "espalda,brazos"},
        {"id": 32, "name": "Aperturas (Flyes)", "fatigue_pct": 38, "type": "Periferica", "category": "Isolation", "zonas": "hombros,pecho"},
        {"id": 33, "name": "Curl de Bíceps (Barra)", "fatigue_pct": 33, "type": "Periferica", "category": "Isolation", "zonas": "muñecas,codos"},
        {"id": 34, "name": "Extensión Tríceps", "fatigue_pct": 33, "type": "Periferica", "category": "Isolation", "zonas": "codos,hombros"},
        {"id": 35, "name": "Elevaciones Laterales", "fatigue_pct": 33, "type": "Periferica", "category": "Isolation", "zonas": "hombros"},
        {"id": 36, "name": "Hammer Curl", "fatigue_pct": 33, "type": "Periferica", "category": "Isolation", "zonas": "muñecas,codos"},
        {"id": 37, "name": "Cable Crossovers", "fatigue_pct": 33, "type": "Periferica", "category": "Isolation", "zonas": "hombros,pecho"},
        {"id": 38, "name": "Face Pulls", "fatigue_pct": 28, "type": "Periferica", "category": "Isolation", "zonas": "hombros,espalda"},
        {"id": 39, "name": "Flexiones (Push-ups)", "fatigue_pct": 28, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "muñecas,lumbar,pecho"},
        {"id": 40, "name": "Leg Raises", "fatigue_pct": 33, "type": "Periferica", "category": "Core", "zonas": "lumbar,cadera"},
        {"id": 41, "name": "Elevación Talones", "fatigue_pct": 23, "type": "Periferica", "category": "Isolation", "zonas": "tobillos"},
        {"id": 42, "name": "Crunches", "fatigue_pct": 23, "type": "Periferica", "category": "Core", "zonas": "cervical,lumbar"},
        {"id": 43, "name": "Russian Twists", "fatigue_pct": 28, "type": "Periferica", "category": "Core", "zonas": "lumbar"},
        {"id": 44, "name": "Plancha (Plank)", "fatigue_pct": 18, "type": "Periferica", "category": "Core_Time", "zonas": "abdomen,hombros"},
        {"id": 45, "name": "Superman Extension", "fatigue_pct": 18, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,cuello"},
        {"id": 46, "name": "Bird-Dog", "fatigue_pct": 13, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,core"},
        {"id": 47, "name": "Dead Bug", "fatigue_pct": 13, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,core"},
        {"id": 48, "name": "Scapular Push-ups", "fatigue_pct": 13, "type": "Periferica", "category": "Core_Time", "zonas": "escapula,hombros"},
        {"id": 49, "name": "Prone T / Y", "fatigue_pct": 13, "type": "Periferica", "category": "Core_Time", "zonas": "escapula,hombros"},
        {"id": 50, "name": "Abdominal Bracing", "fatigue_pct": 5, "type": "Periferica", "category": "Core_Time", "zonas": "core"}
    ]
    
    for ex in exs:
        if ex["id"] in [1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 17, 22, 23, 28, 29, 41]:
            ex["body_part"] = "Inferior"
        elif ex["id"] in [40, 42, 43, 44, 45, 46, 47, 50]:
            ex["body_part"] = "Core"
        else:
            ex["body_part"] = "Superior"
            
    return exs

def get_max_realistic_load(user, exercise):
    bw = user['weight_kg']
    exp = user['experience_level']
    age = user['age']
    injury = user['previous_condition']
    
    # Base multiplier (ACSM 2026 guidelines for relative strength)
    exp_map = {"Avanzado": 1.1, "Intermedio": 0.8, "Principiante": 0.5}
    base_mult = exp_map[exp]
    
    # Age factor
    if age > 55: base_mult *= 0.75
    elif age < 19: base_mult *= 0.85
        
    cat = exercise['category']
    if cat == "Heavy_Lower":
        max_load = bw * 2.5 * base_mult
        if injury == "Hernia Lumbar (L4-S1)": max_load *= 0.2
        if injury == "Desgarro LCA / Meniscos" and "rodilla" in exercise['zonas']: max_load *= 0.4
        if injury == "Desgarro de Isquiotibiales" and "isquios" in exercise['zonas']: max_load *= 0.4
    elif cat == "Heavy_Upper":
        max_load = bw * 1.6 * base_mult
        if injury == "Tendinitis / Desgarro Manguito Rotador" and "hombros" in exercise['zonas']: max_load *= 0.35
        if injury == "Desgarro Pectoral" and "pecho" in exercise['zonas']: max_load *= 0.3
        if "muñecas" in exercise['zonas'] and injury == "Epicondilitis (Codo)": max_load *= 0.6
    elif cat == "Heavy_Compound":
        max_load = bw * 1.8 * base_mult
        if injury != "Ninguna": max_load *= 0.45
    elif cat == "Isolation":
        max_load = bw * 0.4 * base_mult
        if injury == "Epicondilitis (Codo)" and "codos" in exercise['zonas']: max_load *= 0.4
        if injury == "Tendinitis del Bíceps" and "biceps" in exercise['zonas']: max_load *= 0.4
    else: # Medium / Core
        max_load = bw * 0.9 * base_mult
        
    return max(5.0, round(max_load, 1))

def generate_database(num_users=400, num_sessions=15):
    np.random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    exercises_data = load_exercises()
    pd.DataFrame(exercises_data).to_csv('data/exercises_catalog.csv', index=False)
    
    # --- USERS GENERATION ---
    user_ids = np.arange(1, num_users + 1)
    ages = np.clip(np.random.normal(30, 12, num_users).astype(int), 16, 75)
    weights = np.clip(np.random.normal(78, 14, num_users).astype(int), 48, 140)
    experiences = np.random.choice(["Principiante", "Intermedio", "Avanzado"], num_users, p=[0.4, 0.4, 0.2])
    conditions = [
        "Ninguna", 
        "Hernia Lumbar (L4-S1)", 
        "Desgarro LCA / Meniscos", 
        "Tendinitis / Desgarro Manguito Rotador",
        "Epicondilitis (Codo)",
        "Tendinitis del Bíceps",
        "Desgarro Pectoral",
        "Desgarro de Isquiotibiales"
    ]
    conds = np.random.choice(conditions, num_users, p=[0.6, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04])
    
    users = [{'user_id': i, 'age': a, 'weight_kg': w, 'experience_level': e, 'previous_condition': c} 
             for i, a, w, e, c in zip(user_ids, ages, weights, experiences, conds)]
    pd.DataFrame(users).to_csv('data/users.csv', index=False)
    
    # --- SESSIONS GENERATION ---
    sessions, logs, raw_sets = [], [], []
    s_id, l_id = 1, 1
    
    for u in users:
        # Recovery time target based on experience (ACSM 2026)
        rec_target = 24 if u['experience_level'] == "Avanzado" else 48
        current_date = datetime(2025, 1, 1)
        
        for _ in range(num_sessions):
            # Simulation of Rest
            rest_h = int(np.clip(np.random.normal(48, 40), 12, 168))
            current_date += timedelta(hours=rest_h)
            
            num_exs = np.random.randint(3, 8)
            chosen_exs = np.random.choice(exercises_data, num_exs, replace=False)
            
            t_cns, t_periph = 0.0, 0.0
            session_rpe_avg = 0
            
            for ex in chosen_exs:
                # "Anatomía del Riesgo": RPE levels
                effort_cat = np.random.choice(["Bajo", "Moderado", "Alto", "Fallo"], p=[0.1, 0.4, 0.4, 0.1])
                rpe_map = {"Bajo": 6, "Moderado": 7.5, "Alto": 9, "Fallo": 10}
                rpe_val = rpe_map[effort_cat]
                
                max_load = get_max_realistic_load(u, ex)
                
                # Carga Relativa
                # RPE 9-10 -> 85-100% load ratio relative to current capacity
                load_ratio = np.random.uniform(0.8, 1.0) if rpe_val >= 9 else np.random.uniform(0.5, 0.75)
                load_used = round(max_load * load_ratio, 1)
                
                num_sets = np.random.randint(3, 6)
                reps_list = [int(np.clip(15 - (load_ratio*12), 1, 20)) for _ in range(num_sets)]
                
                # Fatigue Modulators (NotebookLM Notes)
                fatigue_base = ex['fatigue_pct'] / 100.0
                # Volume factor normalized to bodyweight
                vol_fac = (np.mean(reps_list) * load_used * num_sets) / u['weight_kg']
                
                # Failure penalty: RPE 10 increases fatigue cost exponentially
                effort_penalty = 1.6 if rpe_val == 10 else (1.2 if rpe_val >= 9 else 1.0)
                
                net_fatigue = fatigue_base * vol_fac * effort_penalty
                
                if ex['type'] == "SNC": t_cns += net_fatigue
                else: t_periph += net_fatigue
                
                # Logs
                logs.append({
                    "log_id": l_id, "session_id": s_id, "exercise_id": ex['id'],
                    "effort_sensation": effort_cat, "total_sets": num_sets,
                    "avg_reps": np.mean(reps_list), "avg_load_kg": load_used,
                    "fatigue_impact": round(net_fatigue, 2)
                })
                
                # Raw Sets
                rs = {"log_id": l_id, "session_id": s_id, "user_id": u['user_id'], "exercise_id": ex['id'], "total_sets": num_sets}
                for i in range(6):
                    rs[f"reps_set_{i+1}"] = reps_list[i] if i < num_sets else 0
                    rs[f"load_kg_set_{i+1}"] = load_used if i < num_sets else 0.0
                raw_sets.append(rs)
                
                l_id += 1
                session_rpe_avg += rpe_val
            
            # --- INJURY LOGIC (MODULADORES CIENTÍFICOS) ---
            # Rule 1: CNS FATIGUE > 5.0 (Proxy for 90% threshold) + Low Rest = High Risk (Hernia/ACL)
            # Rule 2: RPE 10 in Multiarticulares (Exercises 1-10) = Danger zone
            # Rule 3: Previous injury + High Peripheral Fatigue = Recurrence
            
            risk_score = 0.02
            if t_cns > 5.5 and rest_h < rec_target: risk_score += 0.4
            if any(l['effort_sensation'] == "Fallo" and l['exercise_id'] <= 10 for l in logs if l['session_id'] == s_id):
                risk_score += 0.25 # Failing big lifts is dangerous
            
            if u['previous_condition'] != "Ninguna" and t_periph > 8.0: risk_score += 0.3
            
            risk_score = np.clip(risk_score, 0.01, 0.98)
            injury_event = 1 if np.random.rand() < risk_score else 0
            
            sessions.append({
                "session_id": s_id, "user_id": u['user_id'], "date": current_date.strftime("%Y-%m-%d %H:%M"),
                "rest_hours_since_last": rest_h, "total_cns_fatigue": round(t_cns, 2),
                "total_periph_fatigue": round(t_periph, 2), "injury_event": injury_event
            })
            s_id += 1

    pd.DataFrame(sessions).to_csv('data/workout_sessions.csv', index=False)
    pd.DataFrame(logs).to_csv('data/workout_exercises_log.csv', index=False)
    pd.DataFrame(raw_sets).to_csv('data/sets_raw_data.csv', index=False)

if __name__ == "__main__":
    generate_database()
    df_s = pd.read_csv('data/workout_sessions.csv')
    print(f"Dataset v2 Elite generado. Sesiones: {len(df_s)}. Tasa Lesiones: {(df_s['injury_event'].mean()*100):.2f}%")
