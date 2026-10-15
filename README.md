# 🌱 Recomendador de Cultivos (UGR - Ingeniería de Software)

Aplicación web de Machine Learning que recomienda el cultivo más óptimo para un lote de
producción a partir de parámetros edafoclimáticos (N, P, K, pH, temperatura, humedad y
precipitación), usando un modelo **RandomForestClassifier** entrenado con el dataset
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/arpitjain007/crop-recommendation-dataset)
(22 cultivos).

**Resultados:** accuracy 99.55% en test (2/440 errores) y **99.45% ± 0.34%** con
validación cruzada de 5 pliegues. La confianza individual (máxima probabilidad) tiene
mediana 0.99 sobre datos similares a los del entrenamiento y **cae en combinaciones no
representadas en el dataset** — la app lo detecta y lo informa al usuario.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `UGR_IdS_TP (2).ipynb` | Notebook con EDA, entrenamiento, evaluación (matriz de confusión, CV, confianza) y exportación del modelo |
| `modelo_recomendacion_cultivos.pkl` | Modelo entrenado serializado con joblib (scikit-learn 1.6.1) |
| `app.py` | Aplicación Streamlit: predicción con rangos visibles, avisos de confianza/fuera de distribución y top-3 |
| `Crop_recommendation.csv` | Dataset utilizado para el entrenamiento (también sirve de referencia de rangos en la app) |
| `requirements.txt` | Dependencias para ejecutar/deplegar la app |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`. Cada slider muestra el rango real de los datos y
la "zona común" (percentiles 5–95). Presioná **🤖 Obtener Recomendación**: verás el cultivo
sugerido, su confianza, el ranking top-3 y —si corresponde— avisos de confianza baja o de
combinación fuera de la zona estudiada.

## Desplegar en Streamlit Community Cloud

1. Subí este repositorio a GitHub (ya está configurado: https://github.com/oscale05/recomendador_de_cultivos_UGR).
2. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. **New app → Deploy from a repo**, seleccioná `oscale05/recomendador_de_cultivos_UGR` y como Main file path: `app.py`.
4. **Antes de desplegar:** hacé clic en **Advanced settings → Python version** y elegí **Python 3.12** (imprescindible: sin esta selección, Streamlit Cloud usa Python 3.14 y `scikit-learn==1.6.1` no tiene wheel precompilado → falla la instalación).
5. **Deploy**. La app quedará publicada en `https://<tu-usuario>.streamlit.app/`.

> Nota: `requirements.txt` fija `scikit-learn==1.6.1`, la versión con la que se entrenó
> y exportó el `.pkl`. No la cambies sin re-entrenar el modelo, o podrías romper la
> deserialización.

## Reentrenar el modelo

Abrí el notebook en Google Colab y ejecutá todas las celdas en orden: el dataset
`Crop_recommendation.csv` se descarga **automáticamente** desde el raw de este repositorio
(no hace falta subirlo a mano) y la última celda exporta y **descarga** el
`modelo_recomendacion_cultivos.pkl`. Reemplazá ese archivo en la raíz del repo y hacé push
(Streamlit Cloud redesplegará solo al detectar el cambio).
