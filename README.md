# 🌱 Recomendador de Cultivos (UGR - Ingeniería de Software)

Aplicación web de Machine Learning que recomienda el cultivo más óptimo para un lote de
producción a partir de parámetros edafoclimáticos (N, P, K, pH, temperatura, humedad y
precipitación), usando un modelo **RandomForestClassifier** entrenado con el dataset
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/arpitjain007/crop-recommendation-dataset)
(22 cultivos, accuracy ≈ 0.9955).

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `UGR_IdS_TP (2).ipynb` | Notebook con EDA, entrenamiento, evaluación y exportación del modelo |
| `modelo_recomendacion_cultivos.pkl` | Modelo entrenado serializado con joblib (scikit-learn 1.4.2) |
| `app.py` | Aplicación Streamlit que carga el modelo y sirve predicciones |
| `Crop_recommendation.csv` | Dataset utilizado para el entrenamiento |
| `requirements.txt` | Dependencias para ejecutar/deplegar la app |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`. Ajustá los sliders y presioná
**🤖 Obtener Recomendación**.

## Desplegar en Streamlit Community Cloud

1. Subí este repositorio a GitHub (ya está configurado: https://github.com/oscale05/recomendador_de_cultivos_UGR).
2. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. **New app → Deploy from a repo**, seleccioná `oscale05/recomendador_de_cultivos_UGR` y como Main file path: `app.py`.
4. **Deploy**. La app quedará publicada en `https://<tu-usuario>.streamlit.app/`.

> Nota: `requirements.txt` fija `scikit-learn==1.4.2`, la versión con la que se entrenó
> el `.pkl`. No la cambies sin re-entrenar el modelo, o podrías romper la deserialización.

## Reentrenar el modelo

Abrí el notebook en Google Colab (usa `google.colab.files.upload()` para subir el CSV),
ejecutá todas las celdas en orden y descargá el `.pkl` generado. Reemplazá
`modelo_recomendacion_cultivos.pkl` en la raíz del repo y hacé push.
