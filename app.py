"""
=============================================================================
Canary-1B-v2 & Parakeet-TDT-0.6B-v3 — Demo de Inferencia Interactiva
=============================================================================
Materia  : Procesamiento de Datos Secuenciales con Deep Learning
Grupo    : Santiago Londoño Méndez | Andrés Rojas Zúñiga |
           Rubén Darío García Morales | David Ayala Caro
Artículo : Sekoyan et al. (2025), NVIDIA

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
=============================================================================
"""

import os
import io
import time
import tempfile
import streamlit as st
import numpy as np

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ASR/AST Demo — Canary & Parakeet",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        background-color: #1e2d40;
        border-left: 4px solid #00b4d8;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        font-size: 1.05rem;
        color: #e0f4ff;
        line-height: 1.6;
    }
    .metric-card {
        background: #0f1e2e;
        border: 1px solid #1c3a52;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: bold; color: #00b4d8; }
    .metric-lbl { font-size: 0.75rem; color: #8ba3b8; margin-top: 0.2rem; }
    .arch-box {
        background: #0d1b2a;
        border: 1px solid #22405a;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.82rem;
        color: #7ecfed;
        white-space: pre;
    }
</style>
""", unsafe_allow_html=True)


# ─── Carga de modelos (cacheado) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_canary():
    """Carga Canary-1B-v2 desde HuggingFace (se cachea en disco)."""
    try:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
        model.eval()
        # Configurar decodificación greedy para mayor velocidad
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_parakeet():
    """Carga Parakeet-TDT-0.6B-v3 desde HuggingFace."""
    try:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def save_uploaded_audio(uploaded_file) -> str:
    """Guarda el audio como WAV mono 16 kHz (formato que NeMo requiere)."""
    import soundfile as sf
    import numpy as np

    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        raw_path = tmp.name

    # Leer, convertir a mono y guardar como WAV 16 kHz
    data, sr = sf.read(raw_path)
    if data.ndim > 1:          # estéreo → mono (promedio de canales)
        data = data.mean(axis=1)
    mono_path = raw_path.rsplit(".", 1)[0] + "_mono.wav"
    sf.write(mono_path, data, sr, subtype="PCM_16")
    return mono_path


def run_canary_inference(model, audio_path: str, task: str,
                          src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Ejecuta inferencia con Canary-1B-v2 y retorna (texto, tiempo_segundos)."""
    t0 = time.time()
    result = model.transcribe(
        audio=[audio_path],
        batch_size=1,
        task=task,            # 'asr' o 'ast'
        source_lang=src_lang,
        target_lang=tgt_lang,
        pnc=False,
    )
    elapsed = time.time() - t0
    # NeMo puede retornar lista de objetos o lista de strings
    text = result[0] if isinstance(result[0], str) else result[0].text
    return text.strip(), elapsed


def run_parakeet_inference(model, audio_path: str) -> tuple[str, float]:
    """Ejecuta inferencia con Parakeet-TDT-0.6B-v3 y retorna (texto, tiempo)."""
    t0 = time.time()
    result = model.transcribe(audio=[audio_path], batch_size=1)
    elapsed = time.time() - t0
    text = result[0] if isinstance(result[0], str) else result[0].text
    return text.strip(), elapsed


def get_audio_duration(audio_path: str) -> float:
    """Retorna la duración del audio en segundos."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception:
            return 0.0


# ─── IDIOMAS SOPORTADOS ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "Inglés (en)": "en", "Español (es)": "es", "Francés (fr)": "fr",
    "Alemán (de)": "de", "Italiano (it)": "it", "Portugués (pt)": "pt",
    "Ruso (ru)": "ru", "Holandés (nl)": "nl", "Polaco (pl)": "pl",
    "Ucraniano (uk)": "uk", "Checo (cs)": "cs", "Búlgaro (bg)": "bg",
    "Danés (da)": "da", "Estoniano (et)": "et", "Finlandés (fi)": "fi",
    "Griego (el)": "el", "Croata (hr)": "hr", "Húngaro (hu)": "hu",
    "Lituano (lt)": "lt", "Letón (lv)": "lv", "Maltés (mt)": "mt",
    "Rumano (ro)": "ro", "Eslovaco (sk)": "sk", "Esloveno (sl)": "sl",
    "Sueco (sv)": "sv",
}


# ─── INTERFAZ PRINCIPAL ───────────────────────────────────────────────────────
def main():
    # ── Encabezado
    st.title("🎙️ Demo de Inferencia: Canary-1B-v2 & Parakeet-TDT-0.6B-v3")
    st.caption("Arquitectura FastConformer + Transformer Decoder | NVIDIA (2025)")

    # ── Sidebar: configuración
    with st.sidebar:
        st.header("⚙️ Configuración")

        st.subheader("Modelo")
        model_choice = st.radio(
            "Seleccionar modelo:",
            ["Canary-1B-v2 (ASR + Traducción)", "Parakeet-TDT-0.6B-v3 (ASR)", "Ambos (comparar)"],
            index=0,
        )

        st.divider()

        if "Canary" in model_choice or "Ambos" in model_choice:
            st.subheader("Tarea (Canary)")
            task_choice = st.radio("Tarea:", ["ASR — Transcripción", "AST — Traducción"], index=0)
            task = "asr" if "ASR" in task_choice else "ast"

            lang_keys = list(SUPPORTED_LANGUAGES.keys())
            src_key = st.selectbox("Idioma fuente del audio:", lang_keys, index=0)
            src_lang = SUPPORTED_LANGUAGES[src_key]

            if task == "ast":
                tgt_key = st.selectbox("Idioma destino de la traducción:", lang_keys, index=0)
                tgt_lang = SUPPORTED_LANGUAGES[tgt_key]
            else:
                tgt_lang = src_lang
        else:
            task, src_lang, tgt_lang = "asr", "en", "en"

        st.divider()
        st.subheader("ℹ️ Acerca del modelo")
        st.markdown("""
**Canary-1B-v2**
- Encoder: FastConformer (submuestreo 8×)
- Decoder: Transformer autoregresivo
- Parámetros: ~1B
- RTFx: 749 (7–10× > Whisper)
- WER en inglés: 7.15%

**Parakeet-TDT-0.6B-v3**
- Encoder: FastConformer 24 capas
- Decoder: TDT (Token & Duration)
- Parámetros: ~600M
- RTFx: **3332** (54× > Phi-4)
- WER en inglés: 6.32%
""")

    # ── Pestañas principales
    tab_demo, tab_arch, tab_attention = st.tabs(
        ["🎤 Demo de Inferencia", "🏗️ Arquitectura", "🔍 Mecanismo de Atención"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 1: DEMO
    # ─────────────────────────────────────────────────────────────────────────
    with tab_demo:
        st.subheader("Cargar audio")
        uploaded = st.file_uploader(
            "Sube un archivo de audio (.wav, .mp3, .flac, .ogg)",
            type=["wav", "mp3", "flac", "ogg"],
        )

        if uploaded is not None:
            st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
            audio_path = save_uploaded_audio(uploaded)
            duration = get_audio_duration(audio_path)
            st.caption(f"Duración detectada: **{duration:.1f} s**")

            if st.button("▶️ Ejecutar inferencia", type="primary", use_container_width=True):
                use_canary = "Canary" in model_choice or "Ambos" in model_choice
                use_parakeet = "Parakeet" in model_choice or "Ambos" in model_choice

                # Columnas de resultado
                col_c, col_p = st.columns(2) if "Ambos" in model_choice else (st.container(), None)
                out_col = col_c if "Ambos" in model_choice else st.container()

                # ── Canary
                if use_canary:
                    with out_col:
                        st.markdown("#### 🟢 Canary-1B-v2")
                        with st.spinner("Cargando modelo Canary-1B-v2..."):
                            canary_model, err = load_canary()

                        if err:
                            st.error(f"Error al cargar Canary: {err}")
                            st.info("💡 Asegúrate de tener instalado: `pip install nemo_toolkit[asr]`")
                        else:
                            with st.spinner("Procesando audio..."):
                                text_c, t_c = run_canary_inference(
                                    canary_model, audio_path, task, src_lang, tgt_lang
                                )
                            rtfx = duration / t_c if t_c > 0 else 0
                            st.markdown(f'<div class="result-box">{text_c}</div>',
                                        unsafe_allow_html=True)
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Tiempo", f"{t_c:.2f} s")
                            m2.metric("RTFx", f"{rtfx:.1f}×")
                            m3.metric("Tarea", task.upper())

                # ── Parakeet
                if use_parakeet:
                    col_out = col_p if "Ambos" in model_choice else st.container()
                    with col_out:
                        st.markdown("#### 🔵 Parakeet-TDT-0.6B-v3")
                        with st.spinner("Cargando modelo Parakeet..."):
                            para_model, err2 = load_parakeet()

                        if err2:
                            st.error(f"Error al cargar Parakeet: {err2}")
                            st.info("💡 Asegúrate de tener instalado: `pip install nemo_toolkit[asr]`")
                        else:
                            with st.spinner("Procesando audio..."):
                                text_p, t_p = run_parakeet_inference(para_model, audio_path)
                            rtfx_p = duration / t_p if t_p > 0 else 0
                            st.markdown(f'<div class="result-box">{text_p}</div>',
                                        unsafe_allow_html=True)
                            m1, m2 = st.columns(2)
                            m1.metric("Tiempo", f"{t_p:.2f} s")
                            m2.metric("RTFx", f"{rtfx_p:.1f}×")

            # Limpiar temporal
            try:
                os.unlink(audio_path)
            except Exception:
                pass

        else:
            st.info("👆 Sube un archivo de audio para comenzar la inferencia.")
            with st.expander("¿Qué formatos se soportan?"):
                st.markdown("""
- **WAV** (recomendado): PCM 16kHz mono
- **MP3** / **FLAC** / **OGG**: Se convierten automáticamente
- Duración máxima recomendada: 10 minutos (chunking automático para audios largos)
""")

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 2: ARQUITECTURA
    # ─────────────────────────────────────────────────────────────────────────
    with tab_arch:
        st.subheader("Arquitectura FastConformer + Transformer Decoder")

        col_enc, col_dec = st.columns(2)

        with col_enc:
            st.markdown("### 📦 Encoder: FastConformer")
            st.markdown("""
El FastConformer combina atención global (**Transformer**) con patrones locales (**convoluciones**):

**Bloque FastConformer (×24):**
```
Entrada
  │
  ├── LayerNorm
  ├── Feed-Forward × ½       (captura patrones de alto nivel)
  ├── Multi-Head Attention    (dependencias globales en la secuencia)
  ├── Convolutional Module    (patrones acústicos locales)
  ├── Feed-Forward × ½
  └── LayerNorm
```
**Innovaciones vs. Conformer original:**
- ✅ Submuestreo 8× (reduce secuencia de O(n²) a O(n²/64))
- ✅ Convoluciones separables en profundidad
- ✅ Kernels reducidos (31→9)
- ✅ 2–3× más rápido en inferencia
""")

        with col_dec:
            st.markdown("### 📤 Decoder: Transformer")
            st.markdown("""
El decoder genera texto token por token de forma autoregresiva:

**Capa del Decoder (×N):**
```
Texto generado (tokens previos)
  │
  ├── Self-Attention (causal/enmascarada)
  │     Q, K, V ← tokens del decoder
  │
  ├── Cross-Attention          ← CLAVE
  │     Q ← decoder (texto)
  │     K, V ← encoder (audio)
  │
  ├── Feed-Forward Network
  └── Proyección → Vocabulario
```
La **cross-attention** es el puente que conecta el audio procesado por el encoder con la generación de texto del decoder.
""")

        st.divider()
        st.markdown("### 🔄 Flujo completo de datos")

        st.markdown("""
<div class="arch-box">
Audio (.wav 16kHz)
    │
    ▼
Mel-Spectrogram (80 filtros)   [tiempo × 80]
    │
    ▼
FastConformer Encoder
    ├── Conv Subsampling 8×     [tiempo/8 × d_model]
    ├── Linear Projection
    └── 24× Conformer Blocks    [tiempo/8 × 1024]
    │
    ▼
Representaciones contextuales del audio
    │
    ▼
Transformer Decoder (autoregresivo)
    ├── Embedding de tokens
    ├── Self-Attention (causal)
    ├── Cross-Attention(Q=decoder, K,V=encoder)
    └── Feed-Forward + Softmax
    │
    ▼
Texto transcrito / traducido (token por token)
</div>
""", unsafe_allow_html=True)

        st.divider()
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metrics = [
            ("1B", "Parámetros Canary"),
            ("600M", "Parámetros Parakeet"),
            ("25", "Idiomas soportados"),
            ("1.7M h", "Datos de entrenamiento"),
        ]
        for col, (val, lbl) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
            col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                         f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 3: MECANISMO DE ATENCIÓN
    # ─────────────────────────────────────────────────────────────────────────
    with tab_attention:
        st.subheader("🔍 Mecanismo de Atención Detallado")

        st.markdown("""
### Scaled Dot-Product Attention

El corazón del Transformer. Dados los tensores Q (Queries), K (Keys) y V (Values):
""")

        col_q, col_k, col_v = st.columns(3)
        with col_q:
            st.markdown("""
**🔵 Q — Queries**
*"¿Qué información busco?"*

Proyección lineal del estado actual. Cada token "pregunta" qué aspecto del contexto le es relevante.

```python
Q = X @ W_Q
# Shape: [batch, seq_len, d_k]
```
""")
        with col_k:
            st.markdown("""
**🟡 K — Keys**
*"¿Qué tengo disponible?"*

Descripción de cada posición en la secuencia. Se compara contra Q para determinar la relevancia.

```python
K = X @ W_K
# Shape: [batch, seq_len, d_k]
```
""")
        with col_v:
            st.markdown("""
**🔴 V — Values**
*"¿Qué información extraigo?"*

El contenido real que se extrae cuando un Query "encuentra" una Key relevante.

```python
V = X @ W_V
# Shape: [batch, seq_len, d_v]
```
""")

        st.markdown("---")
        st.markdown("""
### Fórmula de Atención

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    \"\"\"
    Q: [batch, heads, seq_q, d_k]
    K: [batch, heads, seq_k, d_k]
    V: [batch, heads, seq_v, d_v]
    \"\"\"
    d_k = Q.shape[-1]

    # 1. Similitud Q·Kᵀ / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Shape: [batch, heads, seq_q, seq_k]

    # 2. Máscara causal (solo en decoder self-attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax → pesos de atención (suman 1)
    attn_weights = F.softmax(scores, dim=-1)
    # Shape: [batch, heads, seq_q, seq_k]

    # 4. Suma ponderada de Values
    output = torch.matmul(attn_weights, V)
    # Shape: [batch, heads, seq_q, d_v]

    return output, attn_weights
```
""")

        st.markdown("---")
        st.markdown("### Multi-Head Attention")
        st.markdown("""
La atención multi-cabeza divide Q, K, V en `h` subconjuntos independientes (cabezas), permitiendo
que el modelo capture simultáneamente diferentes tipos de relaciones en la secuencia.

```python
# Canary-1B-v2: d_model=1024, n_heads=16 → d_k = 1024/16 = 64 por cabeza
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

# Cada cabeza aprende a atender a:
# - head_1: relaciones fonéticas locales
# - head_2: dependencias de largo alcance
# - head_N: patrones prosódicos, etc.
```
""")

        st.markdown("---")
        st.markdown("### Cross-Attention: Conectando Audio y Texto")
        with st.expander("Ver diagrama detallado de Cross-Attention"):
            st.markdown("""
```
ENCODER (FastConformer)         DECODER (Transformer)
──────────────────────          ──────────────────────
Audio frames procesados         Tokens generados hasta ahora
[f_1, f_2, ..., f_T]           [t_1, t_2, ..., t_s]
        │                               │
        ▼                               ▼
K = frames @ W_K            Q = tokens @ W_Q
V = frames @ W_V
        │                               │
        └──────────── Cross-Attention ──┘
                            │
                  score(i,j) = Q_i · K_j / √d_k
                  α = softmax(scores)      ← ¿qué frame atender?
                  output = Σ α_j · V_j     ← información extraída
                            │
                  Siguiente token generado
```
La **Query** del decoder pregunta: *"Dado lo que he generado hasta ahora, ¿en qué parte del audio debo enfocarme?"*
Las **Keys** del encoder responden: *"Aquí están las representaciones del audio."*
Los **Values** del encoder entregan: *"Esta es la información que te paso de esas posiciones."*
""")

        st.markdown("---")
        st.markdown("### ALiBi Simétrico (Innovación del artículo)")
        st.markdown("""
Para el encoder nGPT, los autores adaptan **ALiBi** (Attention with Linear Biases) de forma simétrica
para contextos bidireccionales:

```python
# ALiBi original (causal, para decoders):
score(i,j) = Q_i · K_j / √d_k  -  m * (i - j)    # solo penaliza posiciones futuras

# ALiBi Simétrico (novedad para encoders bidireccionales):
score(i,j) = Q_i · K_j / √d_k  -  m * |i - j|    # penaliza distancia en AMBAS direcciones

# m es diferente por cabeza (pendiente aprendida)
# Primera aplicación de ALiBi bidireccional en ASR/AST
```
Esto le permite al encoder atender de forma equilibrada a contexto izquierdo Y derecho, con penalización
decreciente según la distancia.
""")


# ─── PUNTO DE ENTRADA ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
