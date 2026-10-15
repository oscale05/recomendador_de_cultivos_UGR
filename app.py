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
import altair as alt
from sklearn.model_selection import train_test_split

# 1. Configuración de la página
st.set_page_config(
    page_title="Recomendador de Cultivos Inteligente",
    page_icon="🌱",
    layout="centered"
)

# 2. Estilos personalizados (HTML/CSS propio; degrada con elegancia si cambia Streamlit)
CSS = """
<style>
/* ---------- Hero ---------- */
.crop-hero {
  background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #43a047 100%);
  border-radius: 1rem;
  padding: 1.6rem 1.8rem;
  color: #fff;
  margin: .2rem 0 1.1rem 0;
  box-shadow: 0 8px 24px rgba(46, 125, 50, .25);
  animation: cropFade .6s ease both;
}
.crop-hero-title { font-size: 1.7rem; font-weight: 800; letter-spacing: .2px; }
.crop-hero-sub { opacity: .93; margin-top: .4rem; font-size: .95rem; line-height: 1.45; }
.crop-chips { margin-top: .85rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.crop-chips span {
  background: rgba(255, 255, 255, .16);
  border: 1px solid rgba(255, 255, 255, .28);
  padding: .22rem .7rem;
  border-radius: 999px;
  font-size: .78rem;
}
/* ---------- Mini-barra de rango por parámetro ---------- */
.crop-bar { margin: -.35rem 0 .75rem 0; }
.crop-bar-track {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: linear-gradient(90deg, #e8f0e7, #dfe9dd);
  animation: cropFade .5s ease both;
}
.crop-bar-band {
  position: absolute;
  top: 0;
  height: 100%;
  border-radius: 999px;
  background: rgba(46, 125, 50, .30);
}
.crop-bar-dot {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  margin-left: -6px;
  margin-top: -6px;
  background: #2e7d32;
  border: 2px solid #fff;
  border-radius: 50%;
  box-shadow: 0 1px 4px rgba(0, 0, 0, .25);
  animation: cropPop .45s cubic-bezier(.2, .9, .3, 1.2) both;
}
.crop-bar-labels {
  display: flex;
  justify-content: space-between;
  font-size: .68rem;
  color: #6a7b6a;
  margin-top: .18rem;
}
/* ---------- Card de resultado ---------- */
.crop-card {
  border-radius: 1rem;
  padding: 1.15rem 1.35rem;
  margin: .35rem 0 1rem 0;
  background: #ffffff;
  border: 1px solid rgba(46, 125, 50, .15);
  border-left: 6px solid #2e7d32;
  box-shadow: 0 6px 20px rgba(27, 42, 27, .08);
  animation: cropSlide .55s ease both;
}
.crop-card-label {
  font-size: .75rem;
  text-transform: uppercase;
  letter-spacing: .09em;
  color: #5f7260;
}
.crop-card-crop { font-size: 2rem; font-weight: 800; color: #1b2a1b; margin: .15rem 0 .65rem 0; }
.crop-card--ok { border-left-color: #2e7d32; }
.crop-card--mid { border-left-color: #ff8f00; }
.crop-card--low { border-left-color: #c62828; }
.crop-conf { height: 10px; background: #eef4ea; border-radius: 999px; overflow: hidden; }
.crop-conf-fill { height: 100%; border-radius: 999px; animation: cropGrow .9s ease both; }
.crop-conf-caption { font-size: .78rem; color: #5f7260; margin-top: .3rem; }
.crop-rank { margin-top: .95rem; border-top: 1px dashed rgba(46, 125, 50, .25); padding-top: .7rem; }
.crop-rank-title { font-size: .75rem; text-transform: uppercase; letter-spacing: .09em; color: #5f7260; margin-bottom: .45rem; }
.crop-rank-row { display: flex; align-items: center; gap: .6rem; margin: .38rem 0; }
.crop-rank-name { min-width: 7.5rem; font-size: .85rem; font-weight: 600; color: #2c3e2c; }
.crop-rank-track { flex: 1; height: 10px; background: #eef4ea; border-radius: 999px; overflow: hidden; }
.crop-rank-fill {
  height: 100%;
  background: linear-gradient(90deg, #66bb6a, #2e7d32);
  border-radius: 999px;
  animation: cropGrow .9s ease both;
}
.crop-rank-row--first .crop-rank-fill { background: linear-gradient(90deg, #43a047, #1b5e20); }
.crop-rank-pct {
  min-width: 3.4rem;
  text-align: right;
  font-size: .8rem;
  color: #4a5c4b;
  font-variant-numeric: tabular-nums;
}
/* ---------- Botón principal ---------- */
.stButton > button {
  background: linear-gradient(135deg, #2e7d32, #43a047) !important;
  color: #fff !important;
  border: none !important;
  border-radius: .75rem !important;
  padding: .6rem 1rem !important;
  font-weight: 700 !important;
  letter-spacing: .2px;
  box-shadow: 0 4px 14px rgba(46, 125, 50, .30);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(46, 125, 50, .38);
  filter: brightness(1.05);
}
.stButton > button:active { transform: translateY(0); }
/* ---------- Sidebar / footer ---------- */
.crop-brand { line-height: 1.35; margin-bottom: .2rem; }
.crop-brand b { font-size: 1.02rem; color: #1b5e20; }
.crop-brand span { font-size: .78rem; color: #6a7b6a; }
.crop-footer { text-align: center; color: #7a8b7a; font-size: .78rem; margin-top: 1.6rem; }
.crop-footer a { color: #2e7d32; text-decoration: none; }
/* ---------- Animaciones ---------- */
@keyframes cropFade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@keyframes cropSlide { from { opacity: 0; transform: translateY(10px) scale(.99); } to { opacity: 1; transform: none; } }
@keyframes cropPop { from { opacity: 0; transform: scale(.3); } to { opacity: 1; transform: scale(1); } }
@keyframes cropGrow { from { width: 0 !important; } }
</style>
"""

CROP_EMOJI = {
    'apple': '🍎', 'banana': '🍌', 'blackgram': '🫘', 'chickpea': '🥜',
    'coconut': '🥥', 'coffee': '☕', 'cotton': '🌸', 'grapes': '🍇',
    'jute': '🧵', 'kidneybeans': '🫘', 'lentil': '🌱', 'maize': '🌽',
    'mango': '🥭', 'mothbeans': '🫘', 'mungbean': '🫘', 'muskmelon': '🍈',
    'orange': '🍊', 'papaya': '🧡', 'pomegranate': '🍎', 'pigeonpeas': '🫘',
    'rice': '🍚', 'watermelon': '🍉',
}


# 3. Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    # Busca el archivo .pkl en el directorio local / repositorio
    return joblib.load('modelo_recomendacion_cultivos.pkl')


# 4. Cargar el dataset de referencia: rangos para los sliders y umbral de
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
        'X': X,
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


def barra_rango(col, valor, ref):
    """Mini-barra HTML: posición del valor dentro de min–max con banda p5–p95."""
    lo, hi = float(ref['min'][col]), float(ref['max'][col])
    span = hi - lo if hi > lo else 1.0
    pct = min(max((valor - lo) / span * 100.0, 0.0), 100.0)
    b0 = (float(ref['p05'][col]) - lo) / span * 100.0
    b1 = (float(ref['p95'][col]) - lo) / span * 100.0
    return f"""
<div class="crop-bar" title="Tu valor: {fmt(col, valor)} | Zona común (5–95%): {fmt(col, ref['p05'][col])}–{fmt(col, ref['p95'][col])}">
  <div class="crop-bar-track">
    <div class="crop-bar-band" style="left:{b0:.1f}%; width:{max(b1 - b0, 1.0):.1f}%"></div>
    <div class="crop-bar-dot" style="left:{pct:.1f}%"></div>
  </div>
  <div class="crop-bar-labels">
    <span>{fmt(col, lo)}</span><span>zona común 5–95%</span><span>{fmt(col, hi)}</span>
  </div>
</div>"""


def card_resultado(crop, confianza, probas, clases):
    """Tarjeta HTML con cultivo sugerido, barra de confianza y ranking top-3."""
    if confianza >= 70:
        nivel, color = 'crop-card--ok', '#2e7d32'
    elif confianza >= 50:
        nivel, color = 'crop-card--mid', '#ff8f00'
    else:
        nivel, color = 'crop-card--low', '#c62828'
    emoji = CROP_EMOJI.get(crop, '🌱')
    top3 = np.argsort(probas)[::-1][:3]
    filas = ''
    for i, idx in enumerate(top3):
        pct = float(probas[idx]) * 100
        nombre = f"{CROP_EMOJI.get(clases[idx], '🌱')} {clases[idx].capitalize()}"
        primera = ' crop-rank-row--first' if i == 0 else ''
        filas += f"""
  <div class="crop-rank-row{primera}">
    <div class="crop-rank-name">{nombre}</div>
    <div class="crop-rank-track"><div class="crop-rank-fill" style="width:{pct:.1f}%"></div></div>
    <div class="crop-rank-pct">{pct:.1f}%</div>
  </div>"""
    return f"""
<div class="crop-card {nivel}">
  <div class="crop-card-label">Cultivo recomendado</div>
  <div class="crop-card-crop">{emoji} {crop.upper()}</div>
  <div class="crop-conf"><div class="crop-conf-fill" style="width:{confianza:.1f}%; background:{color}"></div></div>
  <div class="crop-conf-caption">Confianza del modelo: <b>{confianza:.1f}%</b></div>
  <div class="crop-rank">
    <div class="crop-rank-title">Ranking de cultivos posibles (top-3)</div>
    {filas}
  </div>
</div>"""


def grafico_distribucion(valores):
    """Histogramas de cada parámetro (datos de entrenamiento) con banda 5–95% y marcador del valor actual."""
    charts = []
    for col, valor in valores.items():
        datos = ref['X'][col].to_numpy()
        cuenta, bordes = np.histogram(datos, bins=30)
        hist = pd.DataFrame({'x0': bordes[:-1], 'x1': bordes[1:], 'cuenta': cuenta})
        barras = alt.Chart(hist).mark_bar(color='#8BC34A', opacity=0.55, cornerRadius=2).encode(
            x=alt.X('x0:Q', title=None, scale=alt.Scale(domain=[float(ref['min'][col]), float(ref['max'][col])])),
            x2='x1:Q',
            y=alt.Y('cuenta:Q', title=None, axis=None),
        )
        banda = alt.Chart(pd.DataFrame({'x0': [float(ref['p05'][col])], 'x1': [float(ref['p95'][col])]})).mark_rect(
            color='#2E7D32', opacity=0.07
        ).encode(x='x0:Q', x2='x1:Q')
        marca = alt.Chart(pd.DataFrame({'v': [float(valor)]})).mark_rule(
            color='#C62828', strokeWidth=2
        ).encode(x='v:Q')
        charts.append(
            alt.layer(banda, barras, marca).properties(
                height=55, width=280,
                title=alt.TitleParams(text=f'{col} = {fmt(col, valor)}', fontSize=11),
            )
        )
    return alt.vconcat(*charts).properties(
        title=alt.TitleParams(
            text='Distribución de tus valores dentro de los datos de entrenamiento (banda verde = zona común 5–95%)',
            fontSize=12,
        )
    )


try:
    modelo = cargar_modelo()
    ref = cargar_referencia()
    modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar el modelo o los datos de referencia: {e}")
    modelo_cargado = False

st.markdown(CSS, unsafe_allow_html=True)

# 5. Sidebar informativa
with st.sidebar:
    st.markdown(
        '<div class="crop-brand">🌱 <b>Recomendador de Cultivos</b><br>'
        '<span>UGR · Ingeniería de Software · TP</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Modelo**")
    st.write("Random Forest · 22 cultivos · 7 variables edafoclimáticas")
    st.write("Entrenado con 2.200 muestras (scikit-learn 1.6.1)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy test", "99.55%")
    c2.metric("CV 5-fold", "99.45%")
    st.divider()
    st.markdown("[📁 Repositorio en GitHub](https://github.com/oscale05/recomendador_de_cultivos_UGR)")
    st.caption(
        "⚠️ Herramienta orientativa basada en un dataset público. "
        "Validá siempre la recomendación con un asesor agronómico."
    )

# 6. Interfaz principal
st.markdown(
    """
<div class="crop-hero">
  <div class="crop-hero-title">🌱 Sistema de Recomendación de Cultivos</div>
  <div class="crop-hero-sub">
    Ingresá los parámetros edafoclimáticos de tu lote: el modelo Random Forest recomienda el cultivo
    más óptimo, con su nivel de confianza y avisos cuando la combinación queda fuera de los datos de entrenamiento.
  </div>
  <div class="crop-chips">
    <span>🍃 22 cultivos</span>
    <span>🎯 Accuracy 99.55%</span>
    <span>🔁 Validación cruzada 5-fold: 99.45%</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if modelo_cargado:
    st.subheader("Parámetros del Suelo y Clima")

    # Dos columnas para organizar los controles deslizantes
    col1, col2 = st.columns(2)

    with col1:
        n_min, n_max = limites('N', 1, ref)
        n = st.slider("Nitrógeno (N) en mg/kg", n_min, n_max, 80)
        st.caption(f"📊 Rango en los datos: {fmt('N', ref['min']['N'])}–{fmt('N', ref['max']['N'])}"
                   f" · mayoría de muestras (5–95%): {fmt('N', ref['p05']['N'])}–{fmt('N', ref['p95']['N'])}")
        st.markdown(barra_rango('N', n, ref), unsafe_allow_html=True)

        p_min, p_max = limites('P', 1, ref)
        p = st.slider("Fósforo (P) en mg/kg", p_min, p_max, 50)
        st.caption(f"📊 Rango en los datos: {fmt('P', ref['min']['P'])}–{fmt('P', ref['max']['P'])}"
                   f" · mayoría de muestras (5–95%): {fmt('P', ref['p05']['P'])}–{fmt('P', ref['p95']['P'])}")
        st.markdown(barra_rango('P', p, ref), unsafe_allow_html=True)

        k_min, k_max = limites('K', 1, ref)
        k = st.slider("Potasio (K) en mg/kg", k_min, k_max, 40)
        st.caption(f"📊 Rango en los datos: {fmt('K', ref['min']['K'])}–{fmt('K', ref['max']['K'])}"
                   f" · mayoría de muestras (5–95%): {fmt('K', ref['p05']['K'])}–{fmt('K', ref['p95']['K'])}")
        st.markdown(barra_rango('K', k, ref), unsafe_allow_html=True)

        ph_min, ph_max = limites('ph', 0.1, ref)
        ph = st.slider("pH del Suelo", ph_min, ph_max, 6.5, step=0.1)
        st.caption(f"📊 Rango en los datos: {fmt('ph', ref['min']['ph'])}–{fmt('ph', ref['max']['ph'])}"
                   f" · mayoría de muestras (5–95%): {fmt('ph', ref['p05']['ph'])}–{fmt('ph', ref['p95']['ph'])}")
        st.markdown(barra_rango('ph', ph, ref), unsafe_allow_html=True)

    with col2:
        t_min, t_max = limites('temperature', 0.5, ref)
        temp = st.slider("Temperatura Ambiente (°C)", t_min, t_max, 25.0, step=0.5)
        st.caption(f"📊 Rango en los datos: {fmt('temperature', ref['min']['temperature'])}–{fmt('temperature', ref['max']['temperature'])}"
                   f" · mayoría de muestras (5–95%): {fmt('temperature', ref['p05']['temperature'])}–{fmt('temperature', ref['p95']['temperature'])}")
        st.markdown(barra_rango('temperature', temp, ref), unsafe_allow_html=True)

        h_min, h_max = limites('humidity', 1, ref)
        hum = st.slider("Humedad Relativa (%)", h_min, h_max, 70)
        st.caption(f"📊 Rango en los datos: {fmt('humidity', ref['min']['humidity'])}–{fmt('humidity', ref['max']['humidity'])}"
                   f" · mayoría de muestras (5–95%): {fmt('humidity', ref['p05']['humidity'])}–{fmt('humidity', ref['p95']['humidity'])}")
        st.markdown(barra_rango('humidity', hum, ref), unsafe_allow_html=True)

        r_min, r_max = limites('rainfall', 1, ref)
        rain = st.slider("Precipitación Promedio (mm)", r_min, r_max, 100)
        st.caption(f"📊 Rango en los datos: {fmt('rainfall', ref['min']['rainfall'])}–{fmt('rainfall', ref['max']['rainfall'])}"
                   f" · mayoría de muestras (5–95%): {fmt('rainfall', ref['p05']['rainfall'])}–{fmt('rainfall', ref['p95']['rainfall'])}")
        st.markdown(barra_rango('rainfall', rain, ref), unsafe_allow_html=True)

    # Distribución de cada valor dentro de los datos de entrenamiento
    with st.expander("📊 ¿Dónde caen tus valores en los datos de entrenamiento?"):
        st.altair_chart(
            grafico_distribucion({
                'N': n, 'P': p, 'K': k, 'ph': ph,
                'temperature': temp, 'humidity': hum, 'rainfall': rain,
            }),
            width='stretch',
        )

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
    if st.button("🤖 Obtener Recomendación", width='stretch'):
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

        # Tarjeta visual: cultivo + barra de confianza + ranking top-3
        st.markdown(
            card_resultado(prediccion, confianza, probas, modelo.classes_),
            unsafe_allow_html=True,
        )

        # Detalles técnicos para trazabilidad
        with st.expander("ℹ️ Detalles técnicos de esta recomendación"):
            st.write(f"- **Distancia a los datos de entrenamiento:** {d:.2f} "
                     f"(umbral del 95% del conjunto de prueba: {ref['umbral']:.2f})")
            st.write(f"- **Fuera de distribución:** {'Sí' if fuera_de_distribucion else 'No'}")
            st.write("- **Interpretación de la confianza:** probabilidad asignada por el Random Forest "
                     "al cultivo ganador (promedio de los votos de sus 100 árboles).")

# 7. Footer
st.markdown(
    '<div class="crop-footer">🌱 Recomendador de Cultivos · UGR — Ingeniería de Software · '
    '<a href="https://github.com/oscale05/recomendador_de_cultivos_UGR">Repositorio</a></div>',
    unsafe_allow_html=True,
)
