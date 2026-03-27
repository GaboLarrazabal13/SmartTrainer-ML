"""
SmartTrainer ML - SaaS Edition
------------------------------------
Versión Escalable. MLOps Integrado, Autenticación y Panel de Administración.
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import json

# ==========================================
# CONFIGURACIÓN DEL ENTORNO DE PRODUCCIÓN
# ==========================================
# El endpoint Fastapi alojado en onRender.
# Primero intentamos leer de st.secrets (Streamlit Cloud) o fallback local.
API_BASE = st.secrets.get("API_URL", "https://smarttrainer-ml.onrender.com")

# ==========================================
# CONFIG DE PÁGINA Y CSS
# ==========================================
st.set_page_config(page_title="SmartTrainer ML | SaaS", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #05070A; background-image: radial-gradient(at 0% 0%, #111827 0, transparent 50%), radial-gradient(at 50% 0%, #001E3C 0, transparent 50%); }
    [data-testid="stSidebar"] { background-color: #0B0E14 !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    div.stButton > button { background: linear-gradient(135deg, #00D4FF 0%, #0052D4 100%) !important; color: white !important; border-radius: 8px !important; width: 100%; font-weight: 600; }
    .glass-card { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; }
    .elite-alert { padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; border-left: 6px solid; }
    .alert-warning { background: rgba(251, 191, 36, 0.08); border-color: #fbbf24; color: #fbbf24; }
    .alert-danger { background: rgba(239, 68, 68, 0.08); border-color: #ef4444; color: #ef4444; }
    .alert-success { background: rgba(33, 197, 93, 0.08); border-color: #21c55d; color: #21c55d; }
    .rec-box { background: linear-gradient(90deg, rgba(0,212,255,0.1) 0%, rgba(0,0,0,0) 100%); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(0,212,255,0.2); }
</style>
""", unsafe_allow_html=True)


# ==========================================
# GESTIÓN DE ESTADOS (SESSION STATE)
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_data' not in st.session_state: st.session_state.user_data = None
if 'prediction_data' not in st.session_state: st.session_state.prediction_data = None
if 'last_exercises_input' not in st.session_state: st.session_state.last_exercises_input = []
if 'routing' not in st.session_state: st.session_state.routing = "Login"

# Funciones de consulta a la API
@st.cache_data(ttl=60)
def fetch_catalog():
    try: return pd.DataFrame(requests.get(f"{API_BASE}/catalog", timeout=10).json())
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_injuries():
    try: return requests.get(f"{API_BASE}/injuries", timeout=10).json()
    except: return []

catalog_df = fetch_catalog()
injuries_data = fetch_injuries()
injuries_dict = {i["lesion"]: i["id"] for i in injuries_data} if injuries_data else {}
inj_options = ["Ninguna"] + list(injuries_dict.keys())

# ==========================================
# RUTAS PÚBLICAS (Login y Registro)
# ==========================================
if not st.session_state.logged_in:
    
    st.markdown("<center><h1 style='font-size:3rem; margin-top:5vh;'>SmartTrainer <span style='color:#00D4FF'>SaaS</span></h1></center>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Iniciar Sesión", "📝 Registrarse"])
        
        with tab1:
            with st.form("login_form"):
                log_email = st.text_input("Correo Electrónico")
                submit_login = st.form_submit_button("Ingresar")
                if submit_login:
                    res = requests.post(f"{API_BASE}/login", json={"email": log_email})
                    if res.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_data = res.json()
                        st.rerun()
                    else: st.error("Usuario no encontrado.")
                    
        with tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Correo Electrónico (ID Único)")
                c_a, c_b = st.columns(2)
                reg_age = c_a.number_input("Edad", 14, 80, 25)
                reg_weight = c_b.number_input("Peso (kg)", 40.0, 150.0, 75.0)
                reg_height = c_a.number_input("Altura (m)", 1.0, 2.50, 1.75)
                reg_exp = c_b.selectbox("Nivel Experiencia", ["Principiante", "Intermedio", "Avanzado"])
                reg_inj = st.selectbox("Historial Lesión Previa", inj_options)
                
                submit_reg = st.form_submit_button("Crear Cuenta")
                if submit_reg:
                    inj_id = injuries_dict.get(reg_inj, None)
                    payload = {
                        "email": reg_email, "age": reg_age, "weight": reg_weight, "height": reg_height,
                        "experience_level": reg_exp, "injury_history_id": inj_id
                    }
                    res = requests.post(f"{API_BASE}/register", json=payload)
                    if res.status_code == 200:
                        st.success("Cuenta Creada. Inicie sesión en la otra pestaña.")
                    else: st.error(f"Error: {res.text}")

# ==========================================
# RUTAS PRIVADAS (Logged In)
# ==========================================
else:
    # --- MENÚ LATERAL SECRETO ---
    with st.sidebar:
        st.markdown(f"### 👋 Hola, \n**{st.session_state.user_data['email']}**")
        st.divider()
        menu = st.radio("Módulos de Sistema", ["📋 Nueva Sesión", "📊 Dashboard Predictivo", "⚙️ Panel de Administración"])
        if st.button("Cerrar Sesión"):
            st.session_state.logged_in = False
            st.rerun()
            
    # --- PÁGINA 1: NUEVA SESIÓN (Cálculo Predicción) ---
    if menu == "📋 Nueva Sesión":
        st.markdown("# CONSTRUYE TU <span style='color:#00D4FF'>SESIÓN</span>", unsafe_allow_html=True)
        if catalog_df.empty: st.error("❌ API de Catálogo no conectada."); st.stop()
        
        # Opciones de Entrenamiento
        rest_h = st.slider("Horas Descanso Múscular Global", 0, 168, 48)
        selected_zones = st.multiselect("Zonas Objetivo", ["Superior", "Inferior", "Core"], default=["Inferior"])
        
        exercises_in_zones = catalog_df[catalog_df["body_part"].isin(selected_zones)].copy()
        ex_options = {row['name']: row['id'] for _, row in exercises_in_zones.iterrows()}
        sel_labels = st.multiselect("Selecciona Ejercicios", options=list(ex_options.keys()))
        
        exs_input = []
        if sel_labels:
            for label in sel_labels:
                eid = ex_options[label]
                with st.expander(f"⚙️ Intensidad: {label}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        sets = st.number_input("Sets", 1, 8, 3, key=f"s_{eid}")
                        effort = st.select_slider("RPE", ["Bajo", "Moderado", "Alto", "Fallo"], "Moderado", key=f"e_{eid}")
                    with c2:
                        reps, loads = [], []
                        for i in range(sets):
                            r = st.number_input(f"Reps S{i+1}", 1, 50, 10, key=f"r_{eid}_{i}")
                            l = st.number_input(f"Kg S{i+1}", 0.0, 500.0, 60.0, key=f"l_{eid}_{i}")
                            reps.append(int(r)); loads.append(float(l))
                    exs_input.append({
                        "exercise_id": int(eid), "sets": int(sets), 
                        "reps_per_set": reps, "load_kg_per_set": loads, "effort_sensation": effort
                    })
                    
            if st.button("🚀 ANALIZAR MI SESIÓN"):
                u_data = st.session_state.user_data
                # Buscar nombre de lesión previa para el prompt a la API
                inj_name = "Ninguna"
                for i in injuries_data:
                    if i["id"] == u_data["injury_history_id"]: inj_name = i["lesion"]
                
                payload = {
                    "age": u_data["age"], "weight_kg": u_data["weight"], 
                    "experience_level": u_data["experience_level"], "previous_condition": inj_name, 
                    "rest_hours_since_last": rest_h, "exercises": exs_input
                }
                
                try:
                    r = requests.post(f"{API_BASE}/predict", json=payload)
                    if r.status_code == 200:
                        st.session_state.prediction_data = r.json()
                        st.session_state.last_exercises_input = exs_input
                        st.success("🎯 Análisis listo. Cambia al DASHBOARD PREDICTIVO para revisarlo o guardarlo.")
                    else: st.error(f"Error API: {r.text}")
                except Exception as e: st.error(f"Error conexión: {e}")

    # --- PÁGINA 2: DASHBOARD (Revisión y Loggeo MLOps) ---
    elif menu == "📊 Dashboard Predictivo":
        st.markdown("# ANÁLISIS Y <span style='color:#00D4FF'>PIPELINE</span>", unsafe_allow_html=True)
        if not st.session_state.prediction_data: st.warning("Configura y analiza tu sesión primero en la otra pestaña.")
        else:
            d = st.session_state.prediction_data
            prob = d.get("injury_risk_probability", 0) * 100
            lvl = d.get("risk_level", "DESCONOCIDO")
            
            # Guardado MLOps
            with st.container(border=True):
                st.markdown("### 💾 Acciones de Plataforma")
                if st.button("✅ CONFIRMAR Y GUARDAR SESIÓN (ALIMENTA EL MODELO)"):
                    ex_ids = [ex["exercise_id"] for ex in st.session_state.last_exercises_input]
                    log_payload = {
                        "user_email": st.session_state.user_data["email"],
                        "exercise_ids": json.dumps(ex_ids),
                        "total_cns_fatigue": d.get("estimated_cns_load", 0),
                        "total_periph_fatigue": d.get("estimated_peripheral_load", 0),
                        "risk_probability": d.get("injury_risk_probability", 0)
                    }
                    log_res = requests.post(f"{API_BASE}/workouts/log", json=log_payload)
                    if log_res.status_code == 200: st.success("¡Sesión guardada en Supabase! Entrará al circuito auto-entrenamiento.")
                    else: st.error("Falló el guardado.")

            # UI Análisis Original (Resumido para espacio)
            c1, c2 = st.columns([1, 1.2])
            with c1:
                clr = {"BAJO": "#21c55d", "MODERADO": "#fbbf24", "ALTO": "#f97316", "CRÍTICO": "#ef4444"}.get(lvl, "#00D4FF")
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob, number={'suffix': "%", 'font': {'color': clr}}, title={'text': f"ESTADO: {lvl}"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': clr}}))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFF"}, height=350)
                st.plotly_chart(fig, use_column_width=True)
            with c2:
                st.markdown("### Métricas Generadas en Vivo")
                m1, m2 = st.columns(2)
                m1.metric("Tonelaje", f"{d.get('total_volume_kg', 0):,.0f}kg")
                m2.metric("Impacto SNC", f"{d.get('estimated_cns_load', 0):.1f}%")

            if d.get("alert_zones"):
                st.markdown("### 📍 Alertas Anatómicas")
                for a in d["alert_zones"]:
                    st.markdown(f'<div class="elite-alert alert-warning"><b>{a["zone"]}</b> | {a["recommendation"]}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="rec-box">💡 <b>RECOMENDACIÓN DEL ENGINE:</b><br>{d.get("general_recommendation", "")}</div>', unsafe_allow_html=True)

    # --- PÁGINA 3: PANEL ADMINISTRADOR ---
    elif menu == "⚙️ Panel de Administración":
        st.markdown("# GESTIÓN DE <span style='color:#00D4FF'>CATÁLOGO</span>", unsafe_allow_html=True)
        # Verificación simple (Podría mejorarse con JWT o boolean 'is_admin')
        if not st.session_state.user_data["email"].startswith("admin"):
            st.warning("⚠️ Debes iniciar sesión con un correo que empiece por 'admin' para usar este panel.")
        else:
            t1, t2 = st.tabs(["Añadir Ejercicio", "Añadir Categoría Lesión"])
            
            with t1:
                with st.form("admin_ex_form"):
                    ex_name = st.text_input("Nombre del Ejercicio")
                    ex_part = st.selectbox("Parte del Cuerpo Principal", ["Superior", "Inferior", "Core"])
                    ex_zonas = st.text_input("Zonas (Ej: hombros,pecho,muñecas)")
                    ex_cns = st.number_input("Factor Impacto SNC", 0.0, 100.0, 10.0)
                    ex_perf = st.number_input("Factor Impacto Periférico", 0.0, 100.0, 50.0)
                    
                    if st.form_submit_button("Guardar en Supabase"):
                        payload = {"name": ex_name, "body_part": ex_part, "zonas": ex_zonas, "cns_impact_factor": ex_cns, "periph_impact_factor": ex_perf}
                        res = requests.post(f"{API_BASE}/admin/exercises", json=payload)
                        if res.status_code == 200: st.success("Agregado exitosamente.")
                        else: st.error(res.text)
                        
            with t2:
                with st.form("admin_inj_form"):
                    i_zone = st.text_input("Zona / Articulación (ej. Manguito Rotador)")
                    i_les = st.text_input("Nombre de Lesión Común")
                    i_ej = st.text_input("Ejercicios de Riesgo")
                    i_rpe = st.text_input("Nivel Esfuerzo Peligroso (ej. 8-10)")
                    i_fatig = st.text_input("Fatiga Tolerada Max (ej. 80%)")
                    i_tipo = st.selectbox("Tipo de Agotamiento Sensible", ["SNC", "Periférica", "Ambos"])
                    
                    if st.form_submit_button("Guardar Tipo de Lesión"):
                        payload = {"zona_articulacion": i_zone, "lesion_comun": i_les, "ejercicio_riesgo": i_ej, "nivel_esfuerzo_rpe": i_rpe, "fatiga_estimada": i_fatig, "tipo_fatiga": i_tipo}
                        res = requests.post(f"{API_BASE}/admin/injuries", json=payload)
                        if res.status_code == 200: st.success("Lesión Clínica agregada exitosamente.")
                        else: st.error(res.text)

