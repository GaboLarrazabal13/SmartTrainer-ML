# 🏋️ SmartTrainer ML: Predictor de Riesgo y Optimización Biofísica

![Status](https://img.shields.io/badge/Status-Functional_MVP-success)
![Tech Stack](https://img.shields.io/badge/Stack-Python_|_FastAPI_|_XGBoost_|_Streamlit-blue)

**SmartTrainer ML** es un sistema inteligente diseñado para atletas y entrenadores que buscan cuantificar el riesgo de lesión y la fatiga acumulada durante una sesión de entrenamiento de fuerza. Utilizando un modelo de gradiente aumentado (**XGBoost**) y un motor de reglas basado en fisiología aplicada, el sistema predice la probabilidad de sobrecarga y ofrece recomendaciones de recuperación precisas.

---

## 🚀 Características Principales

- **Simulación Biomecánica**: Generador de datos sintéticos que modela 50 ejercicios con perfiles de fatiga específicos (Sistema Nervioso Central vs. Periférica).
- **Inferencia Probalística**: No solo detecta si hay riesgo, sino que calcula un **% de probabilidad** exacto (XGBoost).
- **Motor de Reglas Dinámico**: Identifica sobrecarga local en zonas anatómicas (Lumbar, Rodillas, Hombros, etc.) basándose en el volumen real de la sesión.
- **Dashboard Interactivo**: Interfaz moderna en **Streamlit** (Dark Mode) con indicadores de riesgo, métricas de volumen y gráficos de impacto anatómico.
- **Backend Robusto**: API desarrollada con **FastAPI** y validación de tipos vía Pydantic.

---

## 📂 Estructura del Proyecto

- `api/`: Lógica del servidor, esquemas de datos y motor de reglas.
- `data/`: Generador de bases de datos relacionales y catálogos de ejercicios.
- `models/`: Scripts de entrenamiento, preprocesamiento y modelos serializados (.pkl).
- `frontend/`: Aplicación visual en Streamlit.
- `mlruns/`: Trazabilidad de experimentos mediante MLflow.

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio** e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generar la Base de Datos** (opcional):
   ```bash
   python data/dataset_generator.py
   ```

3. **Entrenar el Modelo**:
   ```bash
   python models/train.py
   ```

---

## 🎮 Cómo Ejecutar

Para disfrutar de la experiencia completa, inicia ambos servicios:

### 1. Iniciar la API (Servidor)
```bash
uvicorn api.main:app --reload
```

Acceso a la documentación interactiva: `http://localhost:8000/docs`

### 2. Iniciar el Dashboard (Interfaz)
```bash
streamlit run frontend/app.py
```

---

## 🐳 Despliegue con Docker

Puedes ejecutar el proyecto de dos formas:

### A. Usando Docker Hub (Sin descargar código)

Ideal para usuarios finales. Solo necesitas Docker instalado:

```bash
# Descargar imágenes
docker pull gabolarrazabal13/smarttrainerml-api:1.0
docker pull gabolarrazabal13/smarttrainerml-frontend:1.0

# Ejecutar manualmente o vía compose
docker compose up
```

### B. Ejecución Local con Docker Compose

Si has clonado este repositorio:

```bash
docker-compose up --build
```

- **API**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`

---

## 🧠 Metodología de Cálculo de Fatiga

El sistema utiliza la siguiente fórmula para estimar el estrés fisiológico de cada ejercicio:

$$Fatiga_{Neta} = (\%\ Fatiga_{Basal}) \times \left( \frac{Sets \times Reps \times Carga}{Peso\ Corporal} \right) \times Multiplicador_{Esfuerzo}$$

- **SNC (Sistema Nervioso Central)**: Generada por ejercicios multiarticulares pesados (ej. Peso Muerto).
- **Periférica**: Agotamiento muscular localizado (ej. Curl de Bíceps).

---

## 🛡️ Límites de Seguridad e Inferencia

El modelo tiene en cuenta:

- **Edad y Peso**: Proporcionalidad de las cargas.
- **Nivel de Experiencia**: Capacidad de recuperación adaptativa.
- **Regla del Doble Impacto**: Si una zona anatómica recibe 2 o más ejercicios intensos, se dispara una alerta de recuperación forzada (48h-72h).

---

## 🔬 Origen e Investigación (NotebookLM)

Este proyecto no es solo una herramienta de código, sino el resultado de una investigación exhaustiva basada en **Ciencia del Deporte y Fisiología del Ejercicio**.

El proceso de desarrollo siguió esta metodología:
1.  **Ingesta de Evidencia**: Se utilizó **NotebookLM** para procesar y sintetizar múltiples fuentes científicas sobre fatiga del Sistema Nervioso Central (SNC), sobrecarga progresiva y prevención de lesiones biomecánicas.
2.  **Extracción de Reglas**: De esta investigación se derivaron las constantes y multiplicadores utilizados en el simulador (`dataset_generator.py`), como la "Regla del Doble Impacto" por zona anatómica.
3.  **Simulación de Datos**: Creamos un "Gemelo Digital" que recrea la respuesta fisiológica humana ante diferentes intensidades y descansos.
4.  **Entrenamiento de IA**: El modelo XGBoost fue entrenado para reconocer patrones en este rastro de datos científicos, permitiendo una inferencia proactiva.

> [!NOTE]
> Puedes consultar el cuaderno de investigación de NotebookLM utilizado para este proyecto aquí: 
> **[Enlace al Cuaderno de SmartTrainer](https://notebooklm.google.com/notebook/d8b1ca55-fe6c-4f6e-ac1e-cc82e7e664a3)** *(Nota: Reemplaza con el enlace real si es compartido públicamente).*

---

Desarrollado con ❤️ para la comunidad de entrenamiento de fuerza basado en evidencia.
