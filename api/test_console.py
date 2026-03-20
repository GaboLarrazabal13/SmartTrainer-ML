"""
test_console.py - Simulador Interactivo de Entrenamiento (CLI)
--------------------------------------------------------------
Este script permite probar la lógica de SmartTrainer desde la consola.
Flujo:
1. Captura datos del perfil (edad, peso, experiencia, lesiones).
2. Permite construir una sesión eligiendo zonas (Superior, Inferior, Core).
3. Calcula en tiempo real la fatiga SNC y Periférica acumulada.
4. Ejecuta la inferencia con el modelo XGBoost para predecir riesgo.
5. Muestra métricas de rendimiento (tonelaje, sets, reps).

Uso: python api/test_console.py
"""
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

def run_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==================================================")
    print(" S M A R T   T R A I N E R   -   S I M U L A D O R")
    print("==================================================\n")
    
    # 1. Cargar Modelos y Catálogo
    try:
        catalog = pd.read_csv('data/exercises_catalog.csv')
        preprocessor = joblib.load('models/preprocessor.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
    except Exception as e:
        print("Error: Modelos o catálogo no encontrados. Ejecuta train.py primero.")
        return

    # 2. Perfil de Usuario
    print("--- 1. PERFIL DEL USUARIO ---")
    age_str = input("Edad (Ej: 30): ").strip()
    age = int(age_str) if age_str else 30
    
    we_str = input("Peso corporal en kg (Ej: 80): ").strip()
    weight = float(we_str) if we_str else 80.0
    
    print("Nivel de experiencia: 1. Principiante | 2. Intermedio | 3. Avanzado")
    exp_opt = input("Opción [2]: ")
    exp_map = {"1": "Principiante", "2": "Intermedio", "3": "Avanzado"}
    experience = exp_map.get(exp_opt, "Intermedio")
    
    print("Condición previa: 1. Ninguna | 2. Desgarro LCA | 3. Hernia Lumbar | 4. Tendinitis Hombro")
    cond_opt = input("Opción [1]: ")
    cond_map = {"1": "Ninguna", "2": "Desgarro LCA", "3": "Hernia Lumbar", "4": "Tendinitis Hombro"}
    condition = cond_map.get(cond_opt, "Ninguna")

    rh_str = input("Horas de descanso desde la última sesión intensa (ej. 48): ").strip()
    rest_h = int(rh_str) if rh_str else 48
    
    # 3. Construccion de la Sesion
    print("\n--- 2. CONSTRUCCIÓN DE LA SESIÓN ---")
    session_exercises = []
    total_cns = 0.0
    total_periph = 0.0
    zone_counts = {}
    
    # Nuevas métricas solicitadas por el usuario
    total_sets_session = 0
    total_reps_session = 0
    total_volume_session = 0.0
    
    while True:
        print("\n===============================")
        print("¿Qué zona deseas entrenar hoy?")
        print(" 1. Zona Superior")
        print(" 2. Zona Inferior")
        print(" 3. Zona Core")
        print(" 4. >>> TERMINAR SESIÓN Y CALCULAR RIESGO <<<")
        print("===============================")
        zone_opt = input("Elige una opción: ").strip()
        
        if zone_opt == "4":
            if len(session_exercises) == 0:
                print("⚠️ No puedes terminar sin añadir ejercicios.")
                continue
            break
            
        zona_target = "Superior" if zone_opt == "1" else "Inferior" if zone_opt == "2" else "Core" if zone_opt == "3" else None
        if not zona_target: 
            print("Opción inválida.")
            continue
        
        print(f"\n--- Ejercicios disponibles [{zona_target.upper()}] ---")
        subset = catalog[catalog['body_part'] == zona_target]
        for idx, row in subset.iterrows():
            print(f" [{row['id']}] {row['name']} (Fatiga Base: {row['fatigue_pct']}%)")
            
        ex_id_str = input("\nIngresa el ID numérico del ejercicio que realizarás: ").strip()
        try:
            ex_id = int(ex_id_str)
            ex_row = catalog[catalog['id'] == ex_id].iloc[0]
        except (ValueError, IndexError):
            print("⚠️ ID inválido o no existe en esa categoría.")
            continue
            
        sets_str = input(f"¿Cuántos sets de {ex_row['name']} vas a realizar? (1-10): ").strip()
        sets = int(sets_str) if sets_str else 3
        
        reps_list = []
        load_list = []
        for s in range(sets):
            r_str = input(f"  └─ Set {s+1} - Repeticiones: ").strip()
            w_str = input(f"  └─ Set {s+1} - Peso (kg): ").strip()
            reps_list.append(int(r_str) if r_str else 10)
            load_list.append(float(w_str) if w_str else 0.0)
            
        avg_reps = sum(reps_list) / sets
        avg_load = sum(load_list) / sets
        
        print("\nSensación de esfuerzo (1. Bajo | 2. Moderado | 3. Alto | 4. Fallo)")
        eff_opt = input("Opción [2]: ").strip()
        eff_map = {"1": "Bajo", "2": "Moderado", "3": "Alto", "4": "Fallo"}
        effort = eff_map.get(eff_opt, "Moderado")
        effort_mult = {"Bajo": 0.8, "Moderado": 1.0, "Alto": 1.2, "Fallo": 1.5}[effort]
        
        # Calcular Fatiga al igual que dataset_generator.py
        exercise_volume = sum([r * w for r, w in zip(reps_list, load_list)])
        volume_fac = (avg_reps * avg_load * sets) / weight
        if avg_load == 0: volume_fac = (avg_reps * sets * 0.5)
        
        net_fatigue = (ex_row['fatigue_pct'] / 100.0) * volume_fac * effort_mult
        
        # Acumular métricas generales
        total_sets_session += sets
        total_reps_session += sum(reps_list)
        total_volume_session += exercise_volume

        if ex_row['type'] == "SNC": 
            total_cns += net_fatigue
        else: 
            total_periph += net_fatigue    
        # Contar Zonas anatómicas estresadas
        zonas_raw = str(ex_row['zonas']).split(',')
        for z in zonas_raw:
            z = z.strip()
            feature_name = f"zone_{z}"
            zone_counts[feature_name] = zone_counts.get(feature_name, 0) + 1
            
        session_exercises.append(ex_row['name'])
        print(f"\n✅ {ex_row['name']} añadido con éxito. (Impacto {ex_row['type']} acumulado: +{net_fatigue:.1f})\n")

    # 4. Inferencia del Modelo ML
    print("\n==================================================")
    print(" 🤖 CALCULANDO PREDICCIÓN CON XGBOOST...")
    print("==================================================\n")
    
    row_data = {
        'age': age,
        'weight_kg': weight,
        'experience_level': experience,
        'previous_condition': condition,
        'rest_hours_since_last': rest_h,
        'total_cns_fatigue': total_cns,
        'total_periph_fatigue': total_periph,
        'num_exercises': len(session_exercises)
    }
    row_data.update(zone_counts)
    
    df_inference = pd.DataFrame([row_data])
    
    # Alinear columnas con las requeridas por el modelo
    required_cols = list(preprocessor.feature_names_in_)
    for col in required_cols:
        if col not in df_inference.columns:
            df_inference[col] = 0
            
    df_inference = df_inference[required_cols]
    
    try:
        X_processed = preprocessor.transform(df_inference)
        risk_prob = xgb_model.predict_proba(X_processed)[0][1]
    except Exception as e:
        print("❌ Error transformando datos. Detalles:", e)
        return
        
    print(f"📋 RESUMEN DE RENDIMIENTO:")
    print(f"- Ejercicios realizados: {len(session_exercises)}")
    print(f"- Sets totales: {total_sets_session}")
    print(f"- Repeticiones totales: {total_reps_session}")
    print(f"- Volumen de peso total: {total_volume_session:,.1f} kg (Tonelaje)")
    
    print(f"\n🧠 CARGA FISIOLÓGICA:")
    print(f"- Impacto estimado SNC (Cerebral): {total_cns:.1f}")
    print(f"- Impacto estimado Periférico (Músculo): {total_periph:.1f}")
    
    print(f"\n⚡ PROBABILIDAD PREDICHA DE LESIÓN/SOBRECARGA: {risk_prob*100:.1f}%\n")
    
    zonas_afectadas = {z.replace('zone_', ''): count for z, count in zone_counts.items() if count >= 2}
    
    if risk_prob > 0.50 and len(zonas_afectadas) > 0:
        zonas_str = ", ".join(zonas_afectadas.keys()).upper()
        print(f"⛔ ALERTA ROJA ⛔")
        print(f"=> Sobrecarga masiva detectada en: {zonas_str}.")
        print(f"=> RECOMENDACIÓN CLÍNICA: Descansar mínimo de 72h para estas articulaciones. Si entrenas mañana, que sea activo y alejado de estas áreas.")
    elif risk_prob > 0.50:
        print(f"⚠️ ALERTA NARANJA ⚠️")
        print(f"=> Fatiga sistémica crítica. Tu carga de hoy supera tu capacidad de recuperación ({rest_h}h).")
    else:
        if len(zonas_afectadas) > 0:
             zonas_str = ", ".join(zonas_afectadas.keys()).upper()
             print(f"🟢 ENTRENAMIENTO ÓPTIMO 🟢")
             print(f"=> Las zonas [{zonas_str}] recibirán estimulo denso pero dentro de los límites de riesgo. Buen trabajo.")
        else:
             print("🟢 ENTRENAMIENTO ÓPTIMO 🟢")
             print("=> Volumen totalmente seguro para tus articulaciones. Puedes subir intensidades en tu siguiente sesión si tu cuerpo lo permite.")

if __name__ == '__main__':
    run_console()
