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
#      IMPORTANTE: elegir Python 3.12 en "Advanced settings" (scikit-learn 1.6.1
#      solo tiene wheels precompilados hasta Python 3.13).
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Configuración de la página
st.set_page_config(
    page_title="Recomendador de Cultivos Inteligente",
    page_icon="🌱",
    layout="centered"
)

# 2. Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    # Busca el archivo .pkl en el directorio local / repositorio
    return joblib.load('modelo_recomendacion_cultivos.pkl')


# 3. Cargar el dataset de referencia: rangos para los sliders y umbral de
#    "fuera de distribución" (reproduce el mismo split del notebook:
#    test_size=0.20, random_state=42, estratificado por cultivo)
@st.cache_resource
def cargar_referencia():
    df = pd.read_csv('Crop_recommendation.csv')
    X = df.drop(columns=['label'])
    mu = X.mean()
    sd = X.std().replace(0.0, 1.0)
    Xtr, Xte, _, _ = train_test_split(
        X, df['label'], test_size=0.20, random_state=42, stratify=df['label']
    )
    ref = ((Xtr - mu) / sd).to_numpy()
    d_test = np.array([
        np.linalg.norm(ref - z, axis=1).min()
        for z in ((Xte - mu) / sd).to_numpy()
    ])
    return {
        'min': X.min(),
        'max': X.max(),
        'p05': X.quantile(0.05),
        'p95': X.quantile(0.95),
        'mu': mu.to_numpy(),
        'sd': sd.to_numpy(),
        'ref': ref,
        'umbral': float(np.percentile(d_test, 95)),
    }


def limites(col, step, ref):
    """Límites de slider alineados al step e incluyendo todo el rango del dataset."""
    lo = np.floor(ref['min'][col] / step) * step
    hi = np.ceil(ref['max'][col] / step) * step
    if step >= 1:
        return int(lo), int(hi)
    return round(float(lo), 1), round(float(hi), 1)


def fmt(col, v):
    return f"{v:.1f}" if col in ('temperature', 'humidity', 'ph') else f"{v:.0f}"


def distancia_entrada(entrada, ref):
    """Distancia (z-score) al punto de train más cercano."""
    z = (np.asarray(entrada, dtype=float) - ref['mu']) / ref['sd']
    return float(np.linalg.norm(ref['ref'] - z, axis=1).min())


try:
    modelo = cargar_modelo()
    ref = cargar_referencia()
    modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar el modelo o los datos de referencia: {e}")
    modelo_cargado = False

# 4. Interfaz de usuario
st.title("🌱 Sistema de Recomendación de Cultivos")
st.write("""
Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado con parámetros edafoclimáticos
para recomendar el cultivo más óptimo para tu lote de producción. Los rangos de cada parámetro indican los
valores presentes en los datos de entrenamiento; a mayor cercanía a esa zona, mayor la confianza del modelo.
""")

if modelo_cargado:
    st.subheader("Parámetros del Suelo y Clima")

    # Dos columnas para organizar los controles deslizantes
    col1, col2 = st.columns(2)

    with col1:
        n_min, n_max = limites('N', 1, ref)
        n = st.slider("Nitrógeno (N) en mg/kg", n_min, n_max, 80)
        st.caption(f"📊 Rango en los datos: {fmt('N', ref['min']['N'])}–{fmt('N', ref['max']['N'])}"
                   f" · mayoría de muestras (5–95%): {fmt('N', ref['p05']['N'])}–{fmt('N', ref['p95']['N'])}")

        p_min, p_max = limites('P', 1, ref)
        p = st.slider("Fósforo (P) en mg/kg", p_min, p_max, 50)
        st.caption(f"📊 Rango en los datos: {fmt('P', ref['min']['P'])}–{fmt('P', ref['max']['P'])}"
                   f" · mayoría de muestras (5–95%): {fmt('P', ref['p05']['P'])}–{fmt('P', ref['p95']['P'])}")

        k_min, k_max = limites('K', 1, ref)
        k = st.slider("Potasio (K) en mg/kg", k_min, k_max, 40)
        st.caption(f"📊 Rango en los datos: {fmt('K', ref['min']['K'])}–{fmt('K', ref['max']['K'])}"
                   f" · mayoría de muestras (5–95%): {fmt('K', ref['p05']['K'])}–{fmt('K', ref['p95']['K'])}")

        ph_min, ph_max = limites('ph', 0.1, ref)
        ph = st.slider("pH del Suelo", ph_min, ph_max, 6.5, step=0.1)
        st.caption(f"📊 Rango en los datos: {fmt('ph', ref['min']['ph'])}–{fmt('ph', ref['max']['ph'])}"
                   f" · mayoría de muestras (5–95%): {fmt('ph', ref['p05']['ph'])}–{fmt('ph', ref['p95']['ph'])}")

    with col2:
        t_min, t_max = limites('temperature', 0.5, ref)
        temp = st.slider("Temperatura Ambiente (°C)", t_min, t_max, 25.0, step=0.5)
        st.caption(f"📊 Rango en los datos: {fmt('temperature', ref['min']['temperature'])}–{fmt('temperature', ref['max']['temperature'])}"
                   f" · mayoría de muestras (5–95%): {fmt('temperature', ref['p05']['temperature'])}–{fmt('temperature', ref['p95']['temperature'])}")

        h_min, h_max = limites('humidity', 1, ref)
        hum = st.slider("Humedad Relativa (%)", h_min, h_max, 70)
        st.caption(f"📊 Rango en los datos: {fmt('humidity', ref['min']['humidity'])}–{fmt('humidity', ref['max']['humidity'])}"
                   f" · mayoría de muestras (5–95%): {fmt('humidity', ref['p05']['humidity'])}–{fmt('humidity', ref['p95']['humidity'])}")

        r_min, r_max = limites('rainfall', 1, ref)
        rain = st.slider("Precipitación Promedio (mm)", r_min, r_max, 100)
        st.caption(f"📊 Rango en los datos: {fmt('rainfall', ref['min']['rainfall'])}–{fmt('rainfall', ref['max']['rainfall'])}"
                   f" · mayoría de muestras (5–95%): {fmt('rainfall', ref['p05']['rainfall'])}–{fmt('rainfall', ref['p95']['rainfall'])}")

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
        # Realizar predicción y obtener probabilidades por cultivo
        probas = modelo.predict_proba(input_data)[0]
        prediccion = modelo.predict(input_data)[0]
        confianza = float(np.max(probas)) * 100

        # Distancia de la entrada a los datos de entrenamiento
        d = distancia_entrada(input_data.iloc[0], ref)
        fuera_de_distribucion = d > ref['umbral']

        # Aviso 1: combinación poco representada en los datos de entrenamiento
        if fuera_de_distribucion:
            st.warning(
                "⚠️ **Combinación fuera de la zona estudiada.** Los parámetros ingresados están "
                "lejos de las muestras usadas para entrenar el modelo, por lo que la recomendación "
                "no tiene respaldo empírico. Ajustá los valores hacia la 'zona común' indicada en cada parámetro."
            )

        # Aviso 2 / resultado principal según la confianza individual
        if confianza < 50:
            st.warning(
                f"🤔 **Confianza baja ({confianza:.1f}%).** Varias condiciones de cultivo son viables "
                "para esta combinación de parámetros. Revisá el ranking de cultivos posibles y acercá "
                "los valores a los rangos más frecuentes de los datos."
            )
        else:
            st.balloons()
            st.success(f"### ¡Cultivo Recomendado: **{prediccion.upper()}**!")

        st.metric(label="Nivel de Confianza del Modelo", value=f"{confianza:.2f}%")

        # Ranking top-3 de cultivos posibles (transparencia hacia el usuario)
        top3 = np.argsort(probas)[::-1][:3]
        st.markdown("#### 🏆 Ranking de cultivos posibles (top-3)")
        c1, c2, c3 = st.columns(3)
        for columna, idx in zip((c1, c2, c3), top3):
            columna.metric(label=modelo.classes_[idx].upper(), value=f"{probas[idx] * 100:.1f}%")

        # Detalles técnicos para trazabilidad
        with st.expander("ℹ️ Detalles técnicos de esta recomendación"):
            st.write(f"- **Distancia a los datos de entrenamiento:** {d:.2f} "
                     f"(umbral del 95% del conjunto de prueba: {ref['umbral']:.2f})")
            st.write(f"- **Fuera de distribución:** {'Sí' if fuera_de_distribucion else 'No'}")
            st.write("- **Interpretación de la confianza:** probabilidad asignada por el Random Forest "
                     "al cultivo ganador (promedio de los votos de sus 100 árboles).")
