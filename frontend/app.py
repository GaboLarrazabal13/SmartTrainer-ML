"""
SmartTrainer ML - Frontend Streamlit
------------------------------------
Este módulo implementa la interfaz de usuario interactiva utilizando Streamlit.
Permite a los usuarios:
1. Configurar su perfil físico y nivel de experiencia.
2. Seleccionar ejercicios de un catálogo dinámico (organizado por Superior, Inferior, Core).
3. Ingresar métricas reales de su entrenamiento (sets, reps, peso, esfuerzo).
4. Visualizar predicciones de riesgo de lesión mediante el modelo XGBoost y reglas fisiológicas.

Uso: streamlit run frontend/app.py (requiere la API FastAPI activa).
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# ─────────────────────────────────────────────
#  Configuración de la API (Docker vs Local)
# ─────────────────────────────────────────────
API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000")

# ─────────────────────────────────────────────
#  Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartTrainer ML",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Estilos personalizados
# ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {background: #0f1117;}
  [data-testid="stSidebar"] {background: #1a1d27;}
  .metric-card {
      background: #1e2130; border-radius: 12px;
      padding: 18px 22px; margin-bottom: 12px;
      border-left: 4px solid #4f8ef7;
  }
  .zone-alert-red   { background:#2a1a1a; border-left:4px solid #ff4b4b; border-radius:10px; padding:14px 18px; margin:8px 0;}
  .zone-alert-orange{ background:#2a2210; border-left:4px solid #ffa500; border-radius:10px; padding:14px 18px; margin:8px 0;}
  .zone-alert-green { background:#142214; border-left:4px solid #21c55d; border-radius:10px; padding:14px 18px; margin:8px 0;}
  h1,h2,h3 {color:#e8eaf6;}
  .bignum {font-size:2.4rem; font-weight:700; color:#4f8ef7;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar – Perfil del Atleta
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/weightlifting.png", width=60)
    st.title("SmartTrainer ML")
    st.caption("Predictor de Riesgo de Lesión")
    st.divider()

    st.header("👤 Tu Perfil")
    age     = st.number_input("Edad", min_value=14, max_value=80, value=28, step=1)
    weight  = st.number_input("Peso corporal (kg)", min_value=40.0, max_value=200.0, value=80.0, step=0.5)
    exp     = st.selectbox("Nivel de Experiencia", ["Principiante", "Intermedio", "Avanzado"], index=1)
    cond    = st.selectbox("Condición Previa", ["Ninguna", "Desgarro LCA", "Hernia Lumbar", "Tendinitis Hombro"])
    rest_h  = st.slider("Horas de descanso desde la última sesión intensa", 0, 168, 48, step=1)

    st.divider()
    st.caption("⚡ API: " + API_BASE)


# ─────────────────────────────────────────────
#  Obtener catálogo de ejercicios
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_catalog():
    try:
        r = requests.get(f"{API_BASE}/catalog", timeout=5)
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()

catalog_df = fetch_catalog()

if catalog_df.empty:
    st.error("❌ No se pudo conectar con la API. Asegúrate de que esté corriendo con: `uvicorn api.main:app --reload`")
    st.stop()


# ─────────────────────────────────────────────
#  Cuerpo principal
# ─────────────────────────────────────────────
st.title("🏋️ SmartTrainer — Simulador de Riesgo")
st.markdown("Diseña tu sesión de entrenamiento y obtén en segundos la **probabilidad de lesión** y las **recomendaciones de recuperación** por zona anatómica.")

st.divider()

# ─── Selector de ejercicios ────────────────────
st.header("📋 Construye tu Sesión")

BODY_PARTS = ["Superior", "Inferior", "Core"]
selected_zones = st.multiselect("1️⃣  Selecciona las zonas que entrenas hoy", BODY_PARTS, default=["Inferior"])

if not selected_zones:
    st.info("Selecciona al menos una zona para ver los ejercicios disponibles.")
    st.stop()

exercises_in_zones = catalog_df[catalog_df["body_part"].isin(selected_zones)].copy()
exercise_options   = {f"[{row['id']}] {row['name']} ({row['fatigue_pct']}% fatiga)": row['id']
                      for _, row in exercises_in_zones.iterrows()}

selected_ex_labels = st.multiselect(
    "2️⃣  Elige los ejercicios de la sesión",
    options=list(exercise_options.keys()),
)

if not selected_ex_labels:
    st.info("Selecciona al menos un ejercicio para configurar los sets.")
    st.stop()

st.divider()
st.header("⚙️ Configura los Sets")

exercises_input = []
for label in selected_ex_labels:
    ex_id   = exercise_options[label]
    ex_row  = exercises_in_zones[exercises_in_zones["id"] == ex_id].iloc[0]
    fatiga  = ex_row["fatigue_pct"]

    with st.expander(f"🔸 **{ex_row['name']}** — Fatiga base: {fatiga}%", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            num_sets = st.number_input("Número de sets", 1, 10, 3, key=f"sets_{ex_id}")
        with col2:
            effort   = st.select_slider(
                "Sensación de esfuerzo",
                options=["Bajo", "Moderado", "Alto", "Fallo"],
                value="Moderado",
                key=f"effort_{ex_id}"
            )

        reps_list  = []
        loads_list = []
        cols = st.columns(num_sets)
        for i, c in enumerate(cols):
            with c:
                st.markdown(f"**Set {i+1}**")
                reps  = st.number_input("Reps", 1, 50, 10,  key=f"reps_{ex_id}_{i}")
                load  = st.number_input("Kg",   0.0, 500.0, 60.0, step=2.5, key=f"load_{ex_id}_{i}")
                reps_list.append(int(reps))
                loads_list.append(float(load))

        exercises_input.append({
            "exercise_id":     int(ex_id),
            "sets":            int(num_sets),
            "reps_per_set":    reps_list,
            "load_kg_per_set": loads_list,
            "effort_sensation": effort,
        })


# ─── Botón de predicción ───────────────────────
# Se utiliza 'use_container_width' para que el botón ocupe todo el ancho disponible.
st.divider()
predict_btn = st.button("🚀 CALCULAR RIESGO DE LESIÓN", type="primary", use_container_width=True)

if predict_btn:
    payload = {
        "age":                    int(age),
        "weight_kg":              float(weight),
        "experience_level":       exp,
        "previous_condition":     cond,
        "rest_hours_since_last":  int(rest_h),
        "exercises":              exercises_input,
    }

    with st.spinner("Analizando sesión con el modelo XGBoost..."):
        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ API no disponible. Ejecuta `uvicorn api.main:app --reload` en otra terminal.")
            st.stop()
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.stop()

    # ─────────────────────────────────────────────
    #  RESULTADOS
    # ─────────────────────────────────────────────
    st.divider()
    st.header("📊 Resultados de tu Sesión")

    risk_prob  = data["injury_risk_probability"]
    risk_level = data["risk_level"]

    # Colores según nivel
    color_map  = {"BAJO": "#21c55d", "MODERADO": "#fbbf24", "ALTO": "#f97316", "CRÍTICO": "#ef4444"}
    gauge_color = color_map.get(risk_level, "#4f8ef7")

    col_gauge, col_metrics = st.columns([1, 1.4])

    with col_gauge:
        st.subheader("🎯 Probabilidad de Lesión")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": gauge_color, "thickness": 0.35},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0,  30],  "color": "#142214"},
                    {"range": [30, 50],  "color": "#222010"},
                    {"range": [50, 70],  "color": "#2a2010"},
                    {"range": [70, 100], "color": "#2a1414"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 4}, "thickness": 0.9, "value": risk_prob * 100},
            },
            title={"text": f"Nivel: <b>{risk_level}</b>", "font": {"size": 20, "color": gauge_color}},
        ))
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e8eaf6"},
            margin=dict(l=20, r=20, t=40, b=10),
        )
        # Se renderiza el gráfico de Plotly ocupando el ancho del contenedor.
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.subheader("💪 Métricas de Rendimiento")
        m1, m2 = st.columns(2)
        m1.metric("🏋️ Ejercicios", data["total_exercises"])
        m2.metric("📦 Sets Totales", data["total_sets"])

        m3, m4 = st.columns(2)
        m3.metric("🔁 Repeticiones", data["total_reps"])
        m4.metric("⚖️ Tonelaje (kg)", f"{data['total_volume_kg']:,.0f}")

        st.divider()
        st.subheader("🧠 Carga Fisiológica")
        max_load    = max(data["estimated_cns_load"] + data["estimated_peripheral_load"], 1)
        cns_pct     = data["estimated_cns_load"] / max_load
        periph_pct  = data["estimated_peripheral_load"] / max_load

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="SNC (Central)", x=["Carga"], y=[round(data["estimated_cns_load"], 1)],
                              marker_color="#ef4444"))
        fig2.add_trace(go.Bar(name="Periférica (Muscular)", x=["Carga"], y=[round(data["estimated_peripheral_load"], 1)],
                              marker_color="#4f8ef7"))
        fig2.update_layout(
            barmode="stack", height=160, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e8eaf6"}, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.3),
        )
        # Visualización de la carga SNC vs Periférica acumulada.
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ─── Recomendación general
    rec_html_class = "zone-alert-green"
    if risk_level in ("ALTO", "CRÍTICO"):
        rec_html_class = "zone-alert-red"
    elif risk_level == "MODERADO":
        rec_html_class = "zone-alert-orange"

    st.markdown(
        f'<div class="{rec_html_class}"><b>💡 Recomendación General</b><br><br>{data["general_recommendation"]}</div>',
        unsafe_allow_html=True,
    )

    # ─── Alertas por zona
    if data["alert_zones"]:
        st.subheader("⚠️ Zonas Anatómicas en Riesgo")

        for alert in data["alert_zones"]:
            alert_class = "zone-alert-red" if risk_level in ("ALTO", "CRÍTICO") else "zone-alert-orange"
            st.markdown(
                f"""<div class="{alert_class}">
                    <b>🦴 Zona: {alert['zone']}</b> &nbsp;|&nbsp; 
                    {alert['exercise_count']} ejercicios que la impactan &nbsp;|&nbsp;
                    ⏱️ Descanso sugerido: <b>{alert['rest_hours_suggested']}h</b><br><br>
                    {alert['recommendation']}
                </div>""",
                unsafe_allow_html=True,
            )

        # Mini gráfico de zonas afectadas
        st.subheader("📍 Distribución de Estrés por Zona")
        zone_data = {a["zone"]: a["exercise_count"] for a in data["alert_zones"]}
        all_zones = {
            z.replace("zone_", "").upper(): v
            for entry in exercises_input
            for z, v in {}.items()  # placeholder
        }
        # Construimos del payload directo
        zone_counts_local: dict[str, int] = {}
        for ex_inp in exercises_input:
            ex_row2 = catalog_df[catalog_df["id"] == ex_inp["exercise_id"]]
            if not ex_row2.empty:
                for z in str(ex_row2.iloc[0]["zonas"]).split(","):
                    z = z.strip().upper()
                    zone_counts_local[z] = zone_counts_local.get(z, 0) + 1

        if zone_counts_local:
            df_zones = pd.DataFrame(
                {"zona": list(zone_counts_local.keys()), "impactos": list(zone_counts_local.values())}
            ).sort_values("impactos", ascending=True)

            fig3 = px.bar(df_zones, x="impactos", y="zona", orientation="h",
                          color="impactos", color_continuous_scale="Reds",
                          labels={"impactos": "Ejercicios que la tocan", "zona": "Zona Anatómica"})
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e8eaf6"}, height=max(250, len(zone_counts_local) * 32),
                showlegend=False, coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            # Gráfico de barras horizontal para identificar las zonas con más impactos.
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown(
            '<div class="zone-alert-green">✅ Ninguna zona anatómica presenta señales de sobrecarga en esta sesión.</div>',
            unsafe_allow_html=True,
        )
