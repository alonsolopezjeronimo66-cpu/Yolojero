import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Rostros en Tiempo Real",
    page_icon="😊",
    layout="wide"
)

# -------------------------------
# FUNCIONES
# -------------------------------

@st.cache_resource
def load_yolov5_face_model():
    """Carga el modelo YOLOv5-Face desde Torch Hub o archivo local"""
    try:
        st.info("🔵 Cargando modelo YOLOv5-Face (detección de rostros)...")
        # Intentar cargar el modelo YOLOv5-Face
        model = torch.hub.load(
            'WongKinYiu/yolov5', 
            'custom', 
            path='yolov5n-face.pt',  # Puedes reemplazar por yolov5s-face.pt si lo tienes
            source='github'
        )
        return model
    except Exception as e:
        st.error(f"❌ No se pudo cargar YOLOv5-Face: {e}")
        st.info("""
        Solución:
        1. Descarga el modelo desde: https://github.com/deepcam-cn/yolov5-face
        2. Colócalo en la carpeta del proyecto con el nombre 'yolov5n-face.pt'
        """)
        return None


# -------------------------------
# INTERFAZ PRINCIPAL
# -------------------------------

st.title("😊 Detección de Rostros en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5-Face** para detectar rostros humanos en imágenes
capturadas con tu cámara o subidas desde tu dispositivo.
""")

# Cargar modelo YOLOv5-Face
with st.spinner("Cargando modelo YOLOv5-Face..."):
    model = load_yolov5_face_model()

if model:
    # Barra lateral
    st.sidebar.title("Configuración")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Opciones de entrada")
    opcion = st.sidebar.radio("Selecciona una fuente:", ["📸 Cámara", "🖼️ Subir imagen"])

    # Contenedor principal
    main_container = st.container()

    with main_container:
        if opcion == "📸 Cámara":
            picture = st.camera_input("Captura una imagen con tu cámara", key="camera")
        else:
            picture = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("Detectando rostros..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            # Parsear resultados
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                label_names = model.names

                # Filtrar solo categorías con "face" o "person"
                filtered_indices = [
                    i for i, c in enumerate(categories)
                    if 'face' in label_names[int(c)] or 'person' in label_names[int(c)]
                ]
                if len(filtered_indices) > 0:
                    boxes = boxes[filtered_indices]
                    scores = scores[filtered_indices]
                    categories = categories[filtered_indices]
                else:
                    boxes = []
                    scores = []
                    categories = []

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🧠 Rostros detectados")
                    results.render()
                    st.image(results.ims[0], channels='BGR', use_container_width=True)

                with col2:
                    st.subheader("📋 Información de detección")
                    if len(categories) > 0:
                        df = pd.DataFrame({
                            "Rostro #": range(1, len(categories) + 1),
                            "Confianza": [f"{s.item():.2f}" for s in scores]
                        })
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index("Rostro #")["Confianza"])
                    else:
                        st.info("No se detectaron rostros. Prueba con otra imagen o baja el umbral de confianza.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
else:
    st.error("No se pudo cargar el modelo YOLOv5-Face.")
    st.stop()

# -------------------------------
# PIE DE PÁGINA
# -------------------------------
st.markdown("---")
st.caption("""
**Aplicación de Detección Facial** desarrollada con Streamlit, PyTorch y YOLOv5-Face.  
Creada por Jerónimo Alonso ⚽  
""")

