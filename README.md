# 🤖 SmartTrainer Pro - MLOps Edition

¡Bienvenido a la versión más avanzada de **SmartTrainer Pro**! Esta plataforma combina Machine Learning (XGBoost) con un Motor de Reglas Clínicas para prevenir lesiones y optimizar el rendimiento de atletas de elite.

## 🚀 Características Principales

- **Predicción de Riesgo en Tiempo Real**: Un modelo entrenado con miles de sesiones detecta patrones de sobrecarga antes de que se conviertan en lesiones.
- **Historial Completo de Sesiones**: Nueva página dedicada para revisar tu progreso, cargas, repeticiones y fatiga acumulada.
- **Motor de Reglas Biomecánicas**: Interpretación quirúrgica de los datos para darte consejos específicos sobre cada articulación.
- **Sincronización con Supabase**: Persistencia robusta y escalable para tus datos de entrenamiento.
- **Interfaz Premium**: Diseño oscuro, estético y dinámico con integración de marca.

## 📁 Estructura del Proyecto

```text
├── api/                # Backend (FastAPI + SQLAlchemy)
│   ├── main.py         # Punto de entrada y endpoints
│   ├── models.py       # Modelos de base de datos (Supabase)
│   ├── database.py     # Configuración de conexión y ORM
│   ├── schemas.py      # Validación de datos (Pydantic)
│   └── rules_engine.py # Lógica de recomendaciones clínicas
├── frontend/           # Interfaz de Usuario (Streamlit)
│   ├── app.py          # Aplicación principal
│   └── assets/         # Recursos visuales (logo.png)
├── data/               # Ingeniería de Datos
│   └── dataset_generator.py # Generador de datos sintéticos biomecánicos
├── models/             # Modelos de Machine Learning (XGBoost)
└── requirements.txt    # Dependencias del proyecto
```

## 🛠️ Instalación y Uso Local

### 1. Requisitos Previos
- Python 3.10+
- Una cuenta en Supabase (o cualquier PostgreSQL).

### 2. Configuración
Crea un archivo `.env` en la raíz con tu URL de base de datos:
```env
DATABASE_URL=tu_url_de_postgresql
```

### 3. Instalación
```bash
pip install -r requirements.txt
```

### 4. Ejecución
**Servidor API:**
```bash
uvicorn api.main:app --reload
```

**Frontend:**
```bash
streamlit run frontend/app.py
```

## 🧠 El Motor de Inteligencia

El sistema utiliza un procesador de variables que transforma tu edad, experiencia y fatiga acumulada en una probabilidad de riesgo. Si esta probabilidad supera el **70%**, el sistema activará alertas críticas y bloqueará ejercicios de alta carga axial para proteger tu salud.

---
*Desarrollado con ❤️ para atletas que buscan la excelencia sin comprometer su longevidad.*
