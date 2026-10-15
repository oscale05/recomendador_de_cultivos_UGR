# ==============================================================================
# APLICACIÓN STREAMLIT - RECOMENDADOR DE CULTIVOS
#
# Ejecutar localmente:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Desplegar en Streamlit Community Cloud:
#   1. Subir este repo a GitHub (app.py + modelo_recomendacion_cultivos.pkl + requirements.txt).
#   2. https://share.streamlit.io → New app → Main file path: app.py → Deploy.
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(
    page_title="Recomendador de Cultivos Inteligente",
    page_icon="🌱",
    layout="centered"
)

# 2. Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    # Busca el archivo .pkl en el directorio local
    return joblib.load('modelo_recomendacion_cultivos.pkl')

try:
    modelo = cargar_modelo()
    modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    modelo_cargado = False

# 3. Interfaz de usuario
st.title("🌱 Sistema de Recomendación de Cultivos")
st.write("""
Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado con parámetros edafoclimáticos
para recomendar el cultivo más óptimo para tu lote de producción.
""")

if modelo_cargado:
    st.subheader("Parámetros del Suelo y Clima")

    # Dos columnas para organizar los controles deslizantes
    col1, col2 = st.columns(2)

    with col1:
        n = st.slider("Nitrógeno (N) en mg/kg", 0, 150, 80)
        p = st.slider("Fósforo (P) en mg/kg", 5, 150, 50)
        k = st.slider("Potasio (K) en mg/kg", 5, 220, 40)
        ph = st.slider("pH del Suelo", 3.5, 10.0, 6.5, step=0.1)

    with col2:
        temp = st.slider("Temperatura Ambiente (°C)", 5.0, 45.0, 25.0, step=0.5)
        hum = st.slider("Humedad Relativa (%)", 10, 100, 70)
        rain = st.slider("Precipitación Promedio (mm)", 20, 300, 100)

    # Crear DataFrame de entrada con la misma estructura que en el entrenamiento
    input_data = pd.DataFrame([{
        'N': n,
        'P': p,
        'K': k,
        'temperature': temp,
        'humidity': hum,
        'ph': ph,
        'rainfall': rain
    }])

    st.write("---")

    # Botón de ejecución para predicción
    if st.button("🤖 Obtener Recomendación", use_container_width=True):
        # Realizar predicción
        prediccion = modelo.predict(input_data)[0]

        # Obtener confianza usando las probabilidades del clasificador
        probabilidades = modelo.predict_proba(input_data)
        confianza = np.max(probabilidades) * 100

        # Mostrar resultados de forma atractiva
        st.balloons()
        st.success(f"### ¡Cultivo Recomendado: **{prediccion.upper()}**!")
        st.metric(label="Nivel de Confianza del Modelo", value=f"{confianza:.2f}%")