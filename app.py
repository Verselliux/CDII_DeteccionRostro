import os
import pickle
import tempfile

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Facial — Identidad & Emoción",
    page_icon="🧠",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# Estilos CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo y fuente general */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Título principal */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Tarjeta de resultado */
    .result-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #3a3f6b;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-name {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .result-emotion {
        font-size: 1.4rem;
        margin-top: 0.3rem;
        color: #34d399;
    }
    .result-conf {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }

    /* Badge persona desconocida */
    .unknown { color: #f87171 !important; }

    /* Separador */
    hr { border-color: #2d3062; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1a1d2e;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }

    /* Barra de probabilidad */
    .prob-bar-container {
        background: #1e2035;
        border-radius: 8px;
        margin: 3px 0;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 22px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
EMOCIONES_ES = {
    "angry":   ("Enojo",     "😠"),
    "disgust": ("Asco",      "🤢"),
    "fear":    ("Miedo",     "😨"),
    "happy":   ("Felicidad", "😊"),
    "sad":     ("Tristeza",  "😢"),
    "surprise":("Sorpresa",  "😲"),
    "neutral": ("Neutral",   "😐"),
}

COLORES_EMO = {
    "Felicidad": "#2ecc71", "Neutral": "#95a5a6", "Tristeza": "#3498db",
    "Enojo": "#e74c3c",     "Sorpresa": "#f39c12", "Miedo": "#9b59b6",
    "Asco": "#1abc9c",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
UMBRAL_CONFIANZA = 0.50


# ─────────────────────────────────────────────────────────────
# Carga de modelos (cacheados)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Cargando modelo SVM...")
def cargar_svm():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["svm"], data["label_encoder"]


@st.cache_resource(show_spinner="⏳ Cargando InsightFace...")
def cargar_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# ─────────────────────────────────────────────────────────────
# Funciones de análisis
# ─────────────────────────────────────────────────────────────
def analizar_emocion(img_bgr_crop):
    """Devuelve (nombre_es, emoji, scores_dict) de la emoción."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr_crop)
        tmp_path = tmp.name
    try:
        result = DeepFace.analyze(
            img_path=tmp_path,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        emotions_raw = result[0]["emotion"]
        dom_raw = result[0]["dominant_emotion"]
        nombre_es, emoji = EMOCIONES_ES.get(dom_raw, (dom_raw, "❓"))
        scores = {
            EMOCIONES_ES.get(k, (k, ""))[0]: round(v, 1)
            for k, v in emotions_raw.items()
        }
        return nombre_es, emoji, scores
    except Exception:
        return "Error", "❓", {}
    finally:
        os.unlink(tmp_path)


def predecir_identidad(face_embedding, svm, le):
    """Devuelve (persona, confianza) o ('Desconocido', 0.0)."""
    if face_embedding is None:
        return "Desconocido", 0.0
    emb = normalize(face_embedding.reshape(1, -1), norm="l2")
    probs = svm.predict_proba(emb)[0]
    idx = np.argmax(probs)
    if probs[idx] >= UMBRAL_CONFIANZA:
        return str(le.classes_[idx]), float(probs[idx])
    return "Desconocido", float(probs[idx])


def procesar_imagen(img_bgr, face_app, svm, le):
    """Pipeline completo: detecta, identifica y analiza emoción en cada rostro."""
    faces = face_app.get(img_bgr)
    if not faces:
        return img_bgr, []

    img_out = img_bgr.copy()
    resultados = []
    H, W = img_bgr.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # Identidad
        persona, conf = predecir_identidad(face.embedding, svm, le)

        # Crop con padding para emoción
        pad = int(0.10 * min(x2 - x1, y2 - y1))
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
        crop = img_bgr[y1p:y2p, x1p:x2p]
        emocion, emoji_emo, scores = analizar_emocion(crop)

        # Anotar imagen
        color_bgr = (80, 200, 80) if persona != "Desconocido" else (80, 80, 200)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color_bgr, 2)

        label1 = f"{persona} ({conf*100:.0f}%)" if persona != "Desconocido" else "Desconocido"
        label2 = f"{emoji_emo} {emocion}"

        cv2.rectangle(img_out, (x1, y1 - 46), (x2, y1), color_bgr, -1)
        cv2.putText(img_out, label1, (x1 + 4, y1 - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_out, emocion, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        resultados.append({
            "persona": persona,
            "confianza": conf,
            "emocion": emocion,
            "emoji_emo": emoji_emo,
            "scores": scores,
            "bbox": (x1, y1, x2, y2),
        })

    return img_out, resultados


def mostrar_resultados(resultados):
    """Renderiza las tarjetas de resultado en Streamlit."""
    if not resultados:
        st.warning("⚠️ No se detectó ningún rostro en la imagen.")
        return

    st.markdown(f"### 📋 {len(resultados)} rostro(s) detectado(s)")

    for i, r in enumerate(resultados, 1):
        es_conocido = r["persona"] != "Desconocido"
        nombre_class = "" if es_conocido else "unknown"
        conf_pct = r["confianza"] * 100

        st.markdown(f"""
        <div class="result-card">
            <div style="color:#9ca3af; font-size:0.8rem; margin-bottom:4px">ROSTRO {i}</div>
            <div class="result-name {nombre_class}">
                {"👤 " + r["persona"] if es_conocido else "❓ Desconocido"}
            </div>
            <div class="result-emotion">{r["emoji_emo"]} {r["emocion"]}</div>
            <div class="result-conf">Confianza identidad: {conf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Barras de probabilidad de emoción
        with st.expander(f"📊 Distribución de emociones — Rostro {i}"):
            scores_sorted = sorted(r["scores"].items(), key=lambda x: x[1], reverse=True)
            for emo, pct in scores_sorted:
                color = COLORES_EMO.get(emo, "#667eea")
                width = max(pct, 2)
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{width}%; background:{color};">
                        {emo}: {pct:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UI principal
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Análisis Facial</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identidad · Emoción · InsightFace + SVM + DeepFace</div>', unsafe_allow_html=True)

# Cargar modelos
with st.spinner("Cargando modelos..."):
    svm, le = cargar_svm()
    face_app = cargar_insightface()

clases = list(le.classes_)
st.markdown(
    f"<div style='text-align:center;color:#6b7280;font-size:0.85rem;margin-bottom:1.5rem'>"
    f"👥 Personas en el modelo: {', '.join(clases)}</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Tabs: Foto / Cámara
# ─────────────────────────────────────────────────────────────
tab_foto, tab_camara = st.tabs(["📁  Subir Foto", "📷  Usar Cámara"])


# ── TAB 1: Subir foto ────────────────────────────────────────
with tab_foto:
    st.markdown("#### Sube una imagen para analizar")
    uploaded = st.file_uploader(
        "Formatos soportados: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        # Decodificar imagen
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.markdown("**Imagen original**")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("🔍 Analizando rostros..."):
            img_anotada, resultados = procesar_imagen(img_bgr, face_app, svm, le)

        with col_img:
            st.markdown("**Resultado anotado**")
            st.image(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_res:
            mostrar_resultados(resultados)


# ── TAB 2: Cámara ────────────────────────────────────────────
with tab_camara:
    st.markdown("#### Captura desde la cámara")
    st.info("📸 Toma una foto con tu cámara y el sistema la analizará automáticamente.")

    foto = st.camera_input("Tomar foto", label_visibility="collapsed")

    if foto:
        file_bytes = np.frombuffer(foto.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analizando rostros..."):
            img_anotada, resultados = procesar_imagen(img_bgr, face_app, svm, le)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Resultado anotado**")
            st.image(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            mostrar_resultados(resultados)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:0.8rem">
    InsightFace <code>buffalo_l</code> · ArcFace 512-dim · SVM RBF · DeepFace FER-2013<br>
    Visión por Computadora — PC2
</div>
""", unsafe_allow_html=True)
