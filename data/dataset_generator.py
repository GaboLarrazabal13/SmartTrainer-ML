"""
dataset_generator.py - Motor de Simulación Biomecánica
------------------------------------------------------
Este script genera un dataset sintético ultra-realista basado en principios 
de fisiología del ejercicio.
- Modela 50 ejercicios con sus perfiles de fatiga SNC y Periférica.
- Simula 1000 usuarios con diferentes edades, pesos y niveles de experiencia.
- Genera sesiones de entrenamiento con cargas (kg) y repeticiones realistas,
  penalizando el rendimiento según lesiones previas.
- Etiqueta eventos de lesión basados en la acumulación de fatiga y falta de descanso.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_exercises():
    exs = [
        {"id": 1, "name": "Peso Muerto", "fatigue_pct": 95, "type": "SNC", "category": "Heavy_Lower", "zonas": "columna,lumbar,agarre"},
        {"id": 2, "name": "Sentadilla", "fatigue_pct": 90, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,lumbar,cadera"},
        {"id": 3, "name": "Power Clean", "fatigue_pct": 90, "type": "SNC", "category": "Heavy_Compound", "zonas": "muñecas,hombros,lumbar,tobillos"},
        {"id": 4, "name": "Peso Muerto Sumo", "fatigue_pct": 90, "type": "SNC", "category": "Heavy_Lower", "zonas": "aductores,cadera,lumbar"},
        {"id": 5, "name": "Sentadilla Frontal", "fatigue_pct": 90, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,columna,muñecas"},
        {"id": 6, "name": "Peso Muerto Rumano", "fatigue_pct": 85, "type": "SNC", "category": "Heavy_Lower", "zonas": "lumbar,isquios"},
        {"id": 7, "name": "Sentadilla Bulgara", "fatigue_pct": 85, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,cadera,cuadriceps"},
        {"id": 8, "name": "Press Banca", "fatigue_pct": 80, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,pecho,muñecas"},
        {"id": 9, "name": "Remo con Barra Inclinado", "fatigue_pct": 80, "type": "SNC", "category": "Heavy_Upper", "zonas": "lumbar,hombros"},
        {"id": 10, "name": "Press Militar", "fatigue_pct": 80, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,cervicales"},
        {"id": 11, "name": "Kettlebell Swing", "fatigue_pct": 80, "type": "SNC", "category": "Heavy_Lower", "zonas": "lumbar,cadera,gluteos"},
        {"id": 12, "name": "Hip Thrust", "fatigue_pct": 75, "type": "SNC", "category": "Heavy_Lower", "zonas": "cadera,lumbar"},
        {"id": 13, "name": "Press Banca Mancuernas", "fatigue_pct": 75, "type": "SNC", "category": "Heavy_Upper", "zonas": "hombros,muñecas"},
        {"id": 14, "name": "Dominadas", "fatigue_pct": 70, "type": "SNC", "category": "Heavy_Upper_BW", "zonas": "dorsales,codos,hombros"},
        {"id": 15, "name": "Prensa Piernas", "fatigue_pct": 70, "type": "SNC", "category": "Heavy_Lower", "zonas": "rodillas,lumbar"},
        {"id": 16, "name": "Remo con Mancuerna", "fatigue_pct": 70, "type": "SNC", "category": "Heavy_Upper", "zonas": "espalda,hombros"},
        {"id": 17, "name": "Zancadas", "fatigue_pct": 65, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,tobillos"},
        {"id": 18, "name": "Press Inclinado Manc.", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Upper", "zonas": "hombros,pecho"},
        {"id": 19, "name": "Fondos (Dips)", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "pecho,hombros,codos"},
        {"id": 20, "name": "Arnold Press", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Upper", "zonas": "hombros"},
        {"id": 21, "name": "Chin-ups", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "codos,biceps,hombros"},
        {"id": 22, "name": "Goblet Squat", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,columna"},
        {"id": 23, "name": "Step-ups", "fatigue_pct": 60, "type": "Periferica", "category": "Medium_Lower", "zonas": "rodillas,tobillos"},
        {"id": 24, "name": "Jalon al Pecho", "fatigue_pct": 55, "type": "Periferica", "category": "Medium_Upper", "zonas": "dorsales,hombros"},
        {"id": 25, "name": "Remo al Menton", "fatigue_pct": 55, "type": "Periferica", "category": "Isolation", "zonas": "hombros"},
        {"id": 26, "name": "Skull Crushers", "fatigue_pct": 50, "type": "Periferica", "category": "Isolation", "zonas": "codos,triceps"},
        {"id": 27, "name": "Press Pecho Maquina", "fatigue_pct": 50, "type": "Periferica", "category": "Medium_Upper", "zonas": "pecho,codos"},
        {"id": 28, "name": "Extension Piernas", "fatigue_pct": 45, "type": "Periferica", "category": "Isolation", "zonas": "rodillas"},
        {"id": 29, "name": "Curl Piernas", "fatigue_pct": 45, "type": "Periferica", "category": "Isolation", "zonas": "isquios"},
        {"id": 30, "name": "Preacher Curl", "fatigue_pct": 45, "type": "Periferica", "category": "Isolation", "zonas": "codos,biceps"},
        {"id": 31, "name": "Remo Invertido", "fatigue_pct": 40, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "espalda,brazos"},
        {"id": 32, "name": "Aperturas Flyes", "fatigue_pct": 35, "type": "Periferica", "category": "Isolation", "zonas": "hombros,pecho"},
        {"id": 33, "name": "Curl Biceps Barra", "fatigue_pct": 30, "type": "Periferica", "category": "Isolation", "zonas": "muñecas,codos"},
        {"id": 34, "name": "Extension Triceps", "fatigue_pct": 30, "type": "Periferica", "category": "Isolation", "zonas": "codos,hombros"},
        {"id": 35, "name": "Elevaciones Laterales", "fatigue_pct": 30, "type": "Periferica", "category": "Isolation", "zonas": "hombros"},
        {"id": 36, "name": "Hammer Curl", "fatigue_pct": 30, "type": "Periferica", "category": "Isolation", "zonas": "muñecas,codos"},
        {"id": 37, "name": "Cable Crossovers", "fatigue_pct": 30, "type": "Periferica", "category": "Isolation", "zonas": "hombros,pecho"},
        {"id": 38, "name": "Face Pulls", "fatigue_pct": 25, "type": "Periferica", "category": "Isolation", "zonas": "hombros,espalda"},
        {"id": 39, "name": "Flexiones", "fatigue_pct": 25, "type": "Periferica", "category": "Medium_Upper_BW", "zonas": "muñecas,lumbar,pecho"},
        {"id": 40, "name": "Leg Raises", "fatigue_pct": 30, "type": "Periferica", "category": "Core", "zonas": "lumbar,cadera"},
        {"id": 41, "name": "Elevacion Talones", "fatigue_pct": 20, "type": "Periferica", "category": "Isolation", "zonas": "tobillos"},
        {"id": 42, "name": "Crunches", "fatigue_pct": 20, "type": "Periferica", "category": "Core", "zonas": "cervical,lumbar"},
        {"id": 43, "name": "Russian Twists", "fatigue_pct": 25, "type": "Periferica", "category": "Core", "zonas": "lumbar"},
        {"id": 44, "name": "Plancha", "fatigue_pct": 15, "type": "Periferica", "category": "Core_Time", "zonas": "abdomen,hombros"},
        {"id": 45, "name": "Superman Extension", "fatigue_pct": 15, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,cuello"},
        {"id": 46, "name": "Bird-Dog", "fatigue_pct": 10, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,core"},
        {"id": 47, "name": "Dead Bug", "fatigue_pct": 10, "type": "Periferica", "category": "Core_Time", "zonas": "lumbar,core"},
        {"id": 48, "name": "Scapular Push-ups", "fatigue_pct": 10, "type": "Periferica", "category": "Core_Time", "zonas": "escapula,hombros"},
        {"id": 49, "name": "Prone T / Y", "fatigue_pct": 10, "type": "Periferica", "category": "Core_Time", "zonas": "escapula,hombros"},
        {"id": 50, "name": "Abdominal Bracing", "fatigue_pct": 5, "type": "Periferica", "category": "Core_Time", "zonas": "core"}
    ]
    
    # Mapeo a zonas amigables (Upper, Lower, Core) para UX
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
    
    # Base multiplier depending on experience
    if exp == "Avanzado":
        base_mult = 1.0
    elif exp == "Intermedio":
        base_mult = 0.7
    else: # Principiante
        base_mult = 0.4
        
    # Age penalty
    if age > 50:
        base_mult *= 0.8
    elif age < 18:
        base_mult *= 0.85
        
    cat = exercise['category']
    # Specific multipliers for exercises relative to Body Weight
    if cat == "Heavy_Lower":
        max_load = bw * 2.2 * base_mult
        if injury == "Hernia Lumbar": max_load *= 0.3 # Huge penalty to protect back
        if injury == "Desgarro LCA" and "rodilla" in exercise['zonas']: max_load *= 0.5
    elif cat == "Heavy_Upper":
        max_load = bw * 1.5 * base_mult
        if injury == "Tendinitis Hombro": max_load *= 0.4
    elif cat == "Heavy_Compound":
        max_load = bw * 1.5 * base_mult
        if injury != "Ninguna": max_load *= 0.5
    elif cat == "Medium_Lower":
        max_load = bw * 1.0 * base_mult
    elif cat == "Medium_Upper":
        max_load = bw * 0.8 * base_mult
        if injury == "Tendinitis Hombro": max_load *= 0.4
    elif "BW" in cat: # Bodyweight (Pushups, Pullups) + Additional Weight occasionally
        max_load = bw * 0.4 * base_mult # represents added plates
    elif cat == "Isolation":
        max_load = bw * 0.3 * base_mult
        if injury == "Tendinitis Hombro" and "hombro" in exercise['zonas']: max_load *= 0.3
    else: # Core / Time based
        max_load = 15 # fixed max added weight
        
    # Floor limit for empty bar / dumbbells
    max_load = max(2.5, max_load)
    return round(max_load, 1)

def generate_database(num_users=2000, num_sessions=20):
    np.random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    exercises_data = load_exercises()
    df_exercises = pd.DataFrame(exercises_data)
    df_exercises.to_csv('data/exercises_catalog.csv', index=False)
    
    # USERS
    user_ids = np.arange(1, num_users + 1)
    ages = np.clip(np.random.normal(32, 10, num_users).astype(int), 16, 70)
    weights = np.clip(np.random.normal(80, 15, num_users).astype(int), 50, 130)
    experiences = np.random.choice(["Principiante", "Intermedio", "Avanzado"], size=num_users, p=[0.5, 0.3, 0.2])
    conditions = ["Ninguna", "Desgarro LCA", "Hernia Lumbar", "Tendinitis Hombro"]
    conds = np.random.choice(conditions, size=num_users, p=[0.75, 0.05, 0.10, 0.10])
    
    users_list = []
    for i in range(num_users):
        users_list.append({
            'user_id': user_ids[i], 'age': ages[i], 'weight_kg': weights[i],
            'experience_level': experiences[i], 'previous_condition': conds[i]
        })
    df_users = pd.DataFrame(users_list)
    df_users.to_csv('data/users.csv', index=False)
    
    # SESSIONS, LOGS, RAW SETS
    sessions_list = []
    logs_list = []
    raw_sets_list = []
    
    session_id = 1
    log_id = 1
    
    for u in users_list:
        base_recovery_needed = 24 if u['experience_level'] == "Avanzado" else 48
        current_date = datetime(2025, 1, 1)
        
        for s in range(num_sessions):
            # Advance time
            rest_h = int(np.clip(np.random.normal(48, 36), 12, 168))
            current_date += timedelta(hours=rest_h)
            
            # Select 3-6 exercises per session
            num_exs = np.random.randint(3, 7)
            chosen_exs = np.random.choice(exercises_data, size=num_exs, replace=False)
            
            total_cns = 0.0
            total_periph = 0.0
            
            for ex in chosen_exs:
                # Effort selection
                effort = np.random.choice(["Bajo", "Moderado", "Alto", "Fallo"], p=[0.1, 0.5, 0.3, 0.1])
                effort_mult = {"Bajo": 0.8, "Moderado": 1.0, "Alto": 1.2, "Fallo": 1.5}[effort]
                
                # Determine max safe load for this user on this exercise
                true_user_max = get_max_realistic_load(u, ex)
                
                num_sets = np.random.randint(2, 7)
                set_reps_list = []
                set_weights_list = []
                
                # Generate each set
                for st in range(num_sets):
                    # Higher effort means pushing closer to their max capacity
                    load_ratio = np.random.uniform(0.6, 0.95) if effort in ["Alto", "Fallo"] else np.random.uniform(0.4, 0.7)
                    if ex['category'] in ["Core", "Core_Time", "Heavy_Upper_BW"]:
                        if np.random.rand() > 0.3: load_ratio = 0 # bodyweight only
                    
                    load_used = round(true_user_max * load_ratio, 1)
                    
                    # Inverse relationship: heavier load -> fewer reps
                    target_reps = int(np.clip(np.random.normal(12 - (load_ratio*10), 3), 1, 20))
                    if ex['category'] == "Core_Time":
                        target_reps = int(np.random.uniform(30, 90)) # represents seconds instead of reps
                    
                    set_reps_list.append(target_reps)
                    set_weights_list.append(load_used)
                
                avg_r = round(np.mean(set_reps_list), 1)
                avg_w = round(np.mean(set_weights_list), 1)
                
                # Impact calculation (Fatigue Net)
                fatigue_base = ex['fatigue_pct']/100.0
                volume_fac = (avg_r * avg_w * num_sets) / u['weight_kg'] # Normalized to bw
                if avg_w == 0: volume_fac = (avg_r * num_sets * 0.5) # small proxy for BW volume
                
                net_fatigue = fatigue_base * volume_fac * effort_mult
                
                # Accumulate
                if ex['type'] == "SNC": total_cns += net_fatigue
                else: total_periph += net_fatigue
                
                # Save raw sets
                raw_set_row = {
                    "log_id": log_id, "session_id": session_id,
                    "user_id": u['user_id'], "exercise_id": ex['id'], 
                    "total_sets": num_sets
                }
                for i in range(6):
                    if i < num_sets:
                        raw_set_row[f"reps_set_{i+1}"] = set_reps_list[i]
                        raw_set_row[f"load_kg_set_{i+1}"] = set_weights_list[i]
                    else:
                        raw_set_row[f"reps_set_{i+1}"] = 0
                        raw_set_row[f"load_kg_set_{i+1}"] = 0.0
                
                raw_sets_list.append(raw_set_row)
                
                # Save Log
                logs_list.append({
                    "log_id": log_id, "session_id": session_id, "exercise_id": ex['id'],
                    "effort_sensation": effort, "total_sets": num_sets,
                    "avg_reps": avg_r, "avg_load_kg": avg_w, 
                    "fatigue_impact": round(net_fatigue, 2)
                })
                log_id += 1
            
            # Formulate Injury Probability 
            # High CNS + Low Rest = High Risk
            injury_prob = 0.01
            if total_cns > 5.0 and rest_h < base_recovery_needed:
                injury_prob += 0.3
            if u['previous_condition'] != "Ninguna" and total_periph > 10.0:
                injury_prob += 0.2
            
            injury_prob = np.clip(injury_prob, 0.01, 0.95)
            injury_event = 1 if np.random.rand() < injury_prob else 0
            
            sessions_list.append({
                "session_id": session_id, "user_id": u['user_id'], 
                "date": current_date.strftime("%Y-%m-%d %H:%M"),
                "rest_hours_since_last": rest_h,
                "total_cns_fatigue": round(total_cns, 2),
                "total_periph_fatigue": round(total_periph, 2),
                "injury_event": injury_event
            })
            session_id += 1

    df_sessions = pd.DataFrame(sessions_list)
    df_logs = pd.DataFrame(logs_list)
    df_raw = pd.DataFrame(raw_sets_list)
    
    df_sessions.to_csv('data/workout_sessions.csv', index=False)
    df_logs.to_csv('data/workout_exercises_log.csv', index=False)
    df_raw.to_csv('data/sets_raw_data.csv', index=False)

if __name__ == "__main__":
    print(">>> Iniciando simulación de datos ultra-realista...")
    generate_database(num_users=1000, num_sessions=20)
    print("\nArchivos creados:")
    print("1. data/users.csv")
    print("2. data/exercises_catalog.csv")
    print("3. data/workout_sessions.csv")
    print("4. data/workout_exercises_log.csv")
    print("5. data/sets_raw_data.csv")
    
    df_sn = pd.read_csv('data/workout_sessions.csv')
    print(f"\nGeneradas {len(df_sn)} sesiones. Tasa de lesiones simulada: {(df_sn['injury_event'].mean()*100):.2f}%")
