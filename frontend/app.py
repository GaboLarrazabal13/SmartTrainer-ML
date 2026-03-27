"""
SmartTrainer Pro - Frontend
----------------------------------
Versión Escalable con Dashboard plenamente funcional y gestión de sesiones por fecha.
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import json
import re
from datetime import datetime

# ==========================================
# CONFIG DEL ENTORNO
# ==========================================
API_BASE = st.secrets.get("API_URL", "https://smarttrainer-ml.onrender.com")

st.set_page_config(page_title="SmartTrainer Pro", page_icon="⚡", layout="wide")

def clean_zone_label(label):
    """Limpia etiquetas como '[Superior] Codos [5' a 'CODOS'"""
    # Remueve contenido entre corchetes [...]
    label = re.sub(r'\[.*?\]', '', label)
    # Remueve números y caracteres especiales
    label = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ ]', '', label)
    return label.strip().upper()

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  * { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #05070A; background-image: radial-gradient(at 0% 0%, #0f172a 0, transparent 50%), radial-gradient(at 100% 0%, #001E3C 0, transparent 50%); }
  
  /* ==================== SIDEBAR ==================== */
  [data-testid="stSidebar"] { background-color: #0B0E14 !important; border-right: 1px solid rgba(255,255,255,0.06); }
  [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
  
  /* Botones generales */
  div.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0052D4 100%) !important;
    color: white !important; border-radius: 8px !important; width: 100%;
    font-weight: 600; border: none !important; padding: 0.5rem 1rem;
    transition: opacity 0.2s;
  }
  div.stButton > button:hover { opacity: 0.85; }
  
  .glass-card {
    background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 20px;
  }
  .elite-alert { padding: 1.2rem; border-radius: 12px; margin-bottom: 0.7rem; border-left: 6px solid; font-size: 0.9rem; }
  .alert-warning { background: rgba(251,191,36,0.08); border-color: #fbbf24; color: #fbbf24; }
  .alert-danger  { background: rgba(239,68,68,0.08); border-color: #ef4444; color: #ef4444; }
  .alert-success { background: rgba(33,197,93,0.08); border-color: #21c55d; color: #21c55d; }
  .rec-box { background: linear-gradient(90deg,rgba(0,212,255,0.1) 0%,rgba(0,0,0,0) 100%); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(0,212,255,0.2); }
  
  /* Nav pills en sidebar */
  .nav-pill {
    display: flex; align-items: center; gap: 10px;
    padding: 0.7rem 1rem; border-radius: 10px;
    margin-bottom: 0.25rem; cursor: pointer;
    color: rgba(255,255,255,0.55); font-size: 0.9rem; font-weight: 500;
    transition: all 0.2s;
  }
  .logout-btn > div > button {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    color: #ef4444 !important;
  }
</style>
""", unsafe_allow_html=True)


# ==========================================
# ESTADO
# ==========================================
if 'logged_in'          not in st.session_state: st.session_state.logged_in = False
if 'user_data'          not in st.session_state: st.session_state.user_data = None
if 'prediction_data'    not in st.session_state: st.session_state.prediction_data = None
if 'last_exercises_input' not in st.session_state: st.session_state.last_exercises_input = []
if 'page'               not in st.session_state: st.session_state.page = "nueva_sesion"
if 'session_date'       not in st.session_state: st.session_state.session_date = datetime.now().date()


# ==========================================
# DATOS DEL CATÁLOGO (CACHEADOS)
# ==========================================
@st.cache_data(ttl=60)
def fetch_catalog():
    try: return pd.DataFrame(requests.get(f"{API_BASE}/catalog", timeout=10).json())
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_injuries():
    try: return requests.get(f"{API_BASE}/injuries", timeout=10).json()
    except: return []

catalog_df    = fetch_catalog()
injuries_data = fetch_injuries()
injuries_dict = {f"{i['zona']} + {i['lesion']}": i["id"] for i in injuries_data} if injuries_data else {}
inj_options   = ["Ninguna"] + list(injuries_dict.keys())
inj_id_to_name = {v: k for k, v in injuries_dict.items()}


# ==========================================
# RUTAS PÚBLICAS (Login / Registro)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<center><h1 style='font-size:3rem; margin-top:6vh;'>SmartTrainer <span style='color:#00D4FF'>Pro</span></h1><p style='color:rgba(255,255,255,0.4); margin-top:-0.5rem;'>Plataforma Inteligente de Entrenamiento</p></center>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Iniciar Sesión", "📝 Registrarse"])

        with tab1:
            with st.form("login_form"):
                log_email = st.text_input("Correo Electrónico")
                submit_login = st.form_submit_button("Ingresar", use_container_width=True)
                if submit_login:
                    res = requests.post(f"{API_BASE}/login", json={"email": log_email})
                    if res.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_data = res.json()
                        st.rerun()
                    else:
                        st.error("Usuario no encontrado.")

        with tab2:
            with st.form("register_form"):
                reg_email  = st.text_input("Correo Electrónico (tu ID único)")
                c_a, c_b   = st.columns(2)
                reg_age    = c_a.number_input("Edad", 14, 80, 25)
                reg_weight = c_b.number_input("Peso (kg)", 40.0, 150.0, 75.0)
                reg_height = c_a.number_input("Altura (m)", 1.0, 2.50, 1.75)
                reg_exp    = c_b.selectbox("Nivel Experiencia", ["Principiante", "Intermedio", "Avanzado"])
                reg_inj    = st.selectbox("Historial Lesión Previa", inj_options)

                if st.form_submit_button("Crear Cuenta", use_container_width=True):
                    inj_id  = injuries_dict.get(reg_inj, None)
                    payload = {"email": reg_email, "age": reg_age, "weight": reg_weight,
                               "height": reg_height, "experience_level": reg_exp, "injury_history_id": inj_id}
                    res = requests.post(f"{API_BASE}/register", json=payload)
                    if res.status_code == 200:
                        st.success("✅ Cuenta creada. Inicia sesión en la pestaña de arriba.")
                    else:
                        st.error(f"Error: {res.text}")

# ==========================================
# RUTAS PRIVADAS
# ==========================================
else:
    u = st.session_state.user_data
    is_admin = u["email"].lower() == "admin@admin.com"

    # --- SIDEBAR NAVEGACIÓN PREMIUM ---
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:0 0.5rem 0.5rem;'>
          <div style='font-size:1.4rem; font-weight:800; color:#00D4FF;'>⚡ SmartTrainer<span style='color:#fff'> Pro</span></div>
          <div style='font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:2px;'>Plataforma de Rendimiento</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr class='sidebar-sep'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='padding:0.6rem 0.8rem; background:rgba(255,255,255,0.04); border-radius:10px; margin-bottom:0.5rem;'>
          <div style='font-size:0.78rem; color:rgba(255,255,255,0.4);'>Hola 👋</div>
          <div style='font-size:0.9rem; font-weight:600; color:#fff; word-break:break-all;'>{u["email"]}</div>
          <div style='font-size:0.75rem; color:#00D4FF; margin-top:2px;'>{u.get("experience_level","")}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr class='sidebar-sep'>", unsafe_allow_html=True)

        pages = [
            ("nueva_sesion",  "🏋️", "Nueva Sesión"),
            ("dashboard",     "📊", "Dashboard Predictivo"),
            ("editar_perfil", "✏️", "Editar Perfil"),
        ]
        if is_admin: pages.append(("admin", "⚙️", "Panel de Administración"))

        for page_id, icon, label in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
                st.session_state.page = page_id
                st.rerun()

        st.markdown("<hr class='sidebar-sep'>", unsafe_allow_html=True)
        if st.button("⏏  Cerrar Sesión", use_container_width=True, key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.prediction_data = None
            st.rerun()

    # --- PÁGINA 1: NUEVA SESIÓN ---
    if st.session_state.page == "nueva_sesion":
        st.markdown("# CONSTRUYE TU <span style='color:#00D4FF'>SESIÓN</span>", unsafe_allow_html=True)
        
        c_top1, c_top2 = st.columns(2)
        with c_top1:
            st.session_state.session_date = st.date_input("Fecha de Entrenamiento", st.session_state.session_date)
        with c_top2:
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
                        sets   = st.number_input("Sets", 1, 8, 3, key=f"s_{eid}")
                        effort = st.select_slider("RPE", ["Bajo", "Moderado", "Alto", "Fallo"], "Moderado", key=f"e_{eid}")
                    with c2:
                        reps, loads = [], []
                        for i in range(sets):
                            r = st.number_input(f"Reps S{i+1}", 1, 50, 10, key=f"r_{eid}_{i}")
                            l = st.number_input(f"Kg S{i+1}",  0.0, 500.0, 60.0, key=f"l_{eid}_{i}")
                            reps.append(int(r)); loads.append(float(l))
                    exs_input.append({"exercise_id": int(eid), "sets": int(sets),
                                      "reps_per_set": reps, "load_kg_per_set": loads, "effort_sensation": effort})

            if st.button("🚀 ANALIZAR MI SESIÓN"):
                inj_name = "Ninguna"
                for i in injuries_data:
                    if i["id"] == u.get("injury_history_id"):
                        inj_name = i["lesion"]
                payload = {"age": u["age"], "weight_kg": u["weight"],
                           "experience_level": u["experience_level"], "previous_condition": inj_name,
                           "rest_hours_since_last": rest_h, "exercises": exs_input}
                try:
                    r = requests.post(f"{API_BASE}/predict", json=payload)
                    if r.status_code == 200:
                        st.session_state.prediction_data       = r.json()
                        st.session_state.last_exercises_input  = exs_input
                        st.success("🎯 Análisis listo. Revisa tu Dashboard.")
                        st.session_state.page = "dashboard"
                        st.rerun()
                    else:
                        st.error(f"Error API: {r.text}")
                except Exception as e:
                    st.error(f"Error conexión: {e}")


    # --- PÁGINA 2: DASHBOARD PREDICTIVO ---
    elif st.session_state.page == "dashboard":
        st.markdown("# ANÁLISIS DE <span style='color:#00D4FF'>ENTRENAMIENTO</span>", unsafe_allow_html=True)

        if not st.session_state.prediction_data:
            st.warning("Configura y analiza tu sesión primero.")
        else:
            d    = st.session_state.prediction_data
            prob = d.get("injury_risk_probability", 0) * 100
            lvl  = d.get("risk_level", "DESCONOCIDO")
            clr  = {"BAJO": "#21c55d", "MODERADO": "#fbbf24", "ALTO": "#f97316", "CRÍTICO": "#ef4444"}.get(lvl, "#00D4FF")

            with st.container(border=True):
                st.markdown(f"### 📅 Sesión del {st.session_state.session_date}")
                c_save, c_redo = st.columns(2)

                with c_save:
                    # Verificar si existe antes de guardar
                    try:
                        check_res = requests.get(f"{API_BASE}/workouts/check", 
                                               params={"email": u["email"], "date": str(st.session_state.session_date)}).json()
                        if check_res.get("exists"):
                            st.warning("⚠️ Ya existe una sesión guardada para esta fecha.")
                    except: pass
                    
                    if st.button("✅ CONFIRMAR Y GUARDAR"):
                        ex_ids      = [ex["exercise_id"] for ex in st.session_state.last_exercises_input]
                        log_payload = {
                            "user_email":        u["email"],
                            "session_date":      str(st.session_state.session_date),
                            "exercise_ids":      json.dumps(ex_ids),
                            "total_cns_fatigue":     d.get("estimated_cns_load", 0),
                            "total_periph_fatigue":  d.get("estimated_peripheral_load", 0),
                            "risk_probability":      d.get("injury_risk_probability", 0)
                        }
                        log_res = requests.post(f"{API_BASE}/workouts/log", json=log_payload)
                        if log_res.status_code == 200:
                            st.success("¡Sesión guardada/actualizada con éxito!")
                        else:
                            st.error(f"Falló el guardado: {log_res.text}")

                with c_redo:
                    if st.button("🔄 REHACER SESIÓN"):
                        st.session_state.prediction_data      = None
                        st.session_state.last_exercises_input = []
                        st.session_state.page = "nueva_sesion"
                        st.rerun()

            c1, c2 = st.columns([1, 1.2])
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    number={'suffix': "%", 'font': {'color': clr, 'size': 48}},
                    title={'text': f"ESTADO: {lvl}", 'font': {'color': '#fff', 'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#555'},
                        'bar': {'color': clr},
                        'steps': [
                            {'range': [0, 35],  'color': 'rgba(33,197,93,0.12)'},
                            {'range': [35, 65], 'color': 'rgba(251,191,36,0.12)'},
                            {'range': [65, 85], 'color': 'rgba(249,115,22,0.12)'},
                            {'range': [85, 100],'color': 'rgba(239,68,68,0.12)'},
                        ],
                        'threshold': {'line': {'color': clr, 'width': 3}, 'thickness': 0.75, 'value': prob}
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FFF"}, height=330, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### 📐 Métricas de la Sesión")
                m1, m2 = st.columns(2)
                m1.metric("Tonelaje", f"{d.get('total_volume_kg', 0):,.0f} kg")
                m2.metric("Ejercicios", d.get('total_exercises', 0))
                m1.metric("Sets Totales", d.get('total_sets', 0))
                m2.metric("Reps Totales", d.get('total_reps', 0))

                st.markdown("#### ⚡ Carga Neuro-Muscular")
                cns_val   = min(d.get("estimated_cns_load", 0), 100)
                periph_val= min(d.get("estimated_peripheral_load", 0), 100)

                fig_bars = go.Figure()
                fig_bars.add_trace(go.Bar(
                    x=[cns_val], y=["SNC"], orientation='h', marker_color='#ef4444', name='SNC',
                    text=[f"{cns_val:.1f}%"], textposition='inside'
                ))
                fig_bars.add_trace(go.Bar(
                    x=[periph_val], y=["Periférico"], orientation='h', marker_color='#3b82f6', name='Periférico',
                    text=[f"{periph_val:.1f}%"], textposition='inside'
                ))
                fig_bars.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': '#fff'}, barmode='group',
                    xaxis=dict(range=[0, 100], showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False, height=140, margin=dict(t=5, b=5, l=5, r=5)
                )
                st.plotly_chart(fig_bars, use_container_width=True)

            st.divider()

            c3, c4 = st.columns([1, 1])
            with c3:
                alert_zones = d.get("alert_zones", [])
                if alert_zones:
                    # Limpiamos los nombres de zona para el radar
                    zone_names  = [clean_zone_label(a["zone"]) for a in alert_zones]
                    zone_counts = [a["exercise_count"] for a in alert_zones]
                    
                    # Para que el radar se vea como un polígono, necesitamos al menos 3 puntos.
                    # Si hay menos, completamos con ceros.
                    if len(zone_names) < 3:
                        zone_names += ["ESTADÍSTICA", "TÉCNICA", "RECUPERACIÓN"][:3-len(zone_names)]
                        zone_counts += [0] * (3 - len(zone_counts))

                    # Cerrar el radar
                    plot_names = zone_names + [zone_names[0]]
                    plot_counts = zone_counts + [zone_counts[0]]

                    fig_radar = go.Figure(go.Scatterpolar(
                        r=plot_counts, theta=plot_names, fill='toself',
                        line_color='#00D4FF', fillcolor='rgba(0,212,255,0.25)',
                        name='Carga Articular'
                    ))
                    fig_radar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        polar=dict(
                            bgcolor="rgba(0,10,20,0.5)",
                            radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.4)', range=[0, max(zone_counts)+1]),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.7)', font=dict(size=11))
                        ),
                        showlegend=False, height=360, margin=dict(t=40, b=40)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Sin zonas de alerta detectadas.")

            with c4:
                st.markdown("### 📍 Alertas Anatómicas")
                if alert_zones:
                    for a in alert_zones:
                        z_clean = clean_zone_label(a["zone"])
                        st.markdown(
                            f'<div class="elite-alert alert-warning"><b>{z_clean}</b> &nbsp;|&nbsp; {a["recommendation"]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown('<div class="elite-alert alert-success">✅ Sin zonas en riesgo detectadas.</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="rec-box" style="margin-top:1rem;">💡 <b>RECOMENDACIÓN DEL ENGINE:</b><br>{d.get("general_recommendation", "")}</div>', unsafe_allow_html=True)


    # --- PÁGINA 3: EDITAR PERFIL ---
    elif st.session_state.page == "editar_perfil":
        st.markdown("# EDITAR <span style='color:#00D4FF'>PERFIL</span>", unsafe_allow_html=True)
        st.caption("Actualiza tus datos físicos y nivel de experiencia.")

        with st.form("edit_profile_form"):
            c_a, c_b = st.columns(2)
            new_age    = c_a.number_input("Edad",     14, 80,  value=u.get("age", 25))
            new_weight = c_b.number_input("Peso (kg)",40.0, 200.0, value=float(u.get("weight", 75)))
            new_height = c_a.number_input("Altura (m)",1.0, 2.5, value=float(u.get("height", 1.75)))
            new_exp    = c_b.selectbox("Nivel Experiencia", ["Principiante", "Intermedio", "Avanzado"],
                index=["Principiante", "Intermedio", "Avanzado"].index(u.get("experience_level", "Principiante")))

            current_inj_id   = u.get("injury_history_id")
            current_inj_name = inj_id_to_name.get(current_inj_id, "Ninguna")
            try: inj_current_idx = inj_options.index(current_inj_name)
            except: inj_current_idx = 0
            new_inj = st.selectbox("Historial Lesión Previa", inj_options, index=inj_current_idx)

            if st.form_submit_button("💾 Guardar Cambios"):
                new_inj_id = injuries_dict.get(new_inj, None)
                payload    = {"age": new_age, "weight": new_weight, "height": new_height,
                              "experience_level": new_exp, "injury_history_id": new_inj_id}
                res = requests.patch(f"{API_BASE}/users/{u['email']}", json=payload)
                if res.status_code == 200:
                    st.session_state.user_data.update(res.json())
                    st.success("✅ Perfil actualizado correctamente.")
                else:
                    st.error(f"Error al actualizar: {res.text}")


    # --- PÁGINA 4: PANEL ADMINISTRADOR ---
    elif st.session_state.page == "admin" and is_admin:
        st.markdown("# GESTIÓN DE <span style='color:#00D4FF'>CATÁLOGO</span>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["Añadir Ejercicio", "Añadir Categoría Lesión"])

        with t1:
            with st.form("admin_ex_form"):
                ex_name  = st.text_input("Nombre del Ejercicio")
                ex_part  = st.selectbox("Familia Principal", ["Superior", "Inferior", "Core"])
                ex_zonas = st.text_input("Zonas afectadas (ej: hombros, pecho, muñecas)")
                ex_cns   = st.number_input("Factor Impacto SNC",        0.0, 5.0, 0.5)
                ex_perf  = st.number_input("Factor Impacto Periférico", 0.0, 5.0, 0.8)
                if st.form_submit_button("Guardar en Catálogo"):
                    payload = {"name": ex_name, "body_part": ex_part, "zonas": ex_zonas,
                               "cns_impact_factor": ex_cns, "periph_impact_factor": ex_perf}
                    res = requests.post(f"{API_BASE}/admin/exercises", json=payload)
                    if res.status_code == 200: st.success("Añadido correctamente.")
                    else: st.error(res.text)

        with t2:
            with st.form("admin_inj_form"):
                i_zone = st.text_input("Zona / Articulación (ej. Manguito Rotador)")
                i_les  = st.text_input("Nombre de la Lesión Común")
                i_ej   = st.text_input("Ejercicios de Riesgo")
                i_rpe  = st.text_input("Nivel Esfuerzo Peligroso")
                i_fatig= st.text_input("Fatiga Tolerada Max")
                i_tipo = st.selectbox("Tipo de Agotamiento", ["SNC", "Periférica", "Ambos"])
                if st.form_submit_button("Guardar Tipo de Lesión"):
                    payload = {"zona_articulacion": i_zone, "lesion_comun": i_les,
                               "ejercicio_riesgo": i_ej, "nivel_esfuerzo_rpe": i_rpe,
                               "fatiga_estimada": i_fatig, "tipo_fatiga": i_tipo}
                    res = requests.post(f"{API_BASE}/admin/injuries", json=payload)
                    if res.status_code == 200: st.success("Lesión catalogada exitosamente.")
                    else: st.error(res.text)
