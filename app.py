import streamlit as st
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator, MyMemoryTranslator
from audio_recorder_streamlit import audio_recorder
import subprocess
import tempfile
import time
import os

st.set_page_config(page_title="AI Tourist Communication Assistant", page_icon="🗣️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main { background-color: #faf9f6; }
.hero {
    background: linear-gradient(135deg, #0f766e, #14b8a6);
    padding: 2.2rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.8rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p { margin: 0.6rem 0 0; opacity: 0.92; font-size: 1rem; }
.card {
    background: white;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 14px rgba(15, 118, 110, 0.08);
    margin-bottom: 1.2rem;
    border: 1px solid #eef2f1;
}
.section-title {
    font-weight: 600;
    font-size: 1.05rem;
    color: #0f766e;
    margin-bottom: 0.8rem;
}
.intent-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}
.intent-emergency { background: #fee2e2; color: #b91c1c; }
.intent-normal { background: #e0f2fe; color: #0369a1; }
.result-box {
    background: #f0fdfa;
    border-left: 4px solid #0f766e;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 1.15rem;
    font-weight: 500;
    color: #134e4a;
}
div.stButton > button {
    background-color: #0f766e;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1.6rem;
    font-weight: 600;
}
div.stButton > button:hover { background-color: #115e59; color: white; }
</style>
""", unsafe_allow_html=True)

LANGUAGES = {
    "Tamil": "ta",
    "English": "en",
    "Japanese": "ja",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Chinese": "zh",
    "Korean": "ko",
    "Arabic": "ar",
}

TRANSLATE_CODE_OVERRIDES = {"zh": "zh-CN"}
TTS_CODE_OVERRIDES = {"zh": "cmn"}
MYMEMORY_CODE_OVERRIDES = {
    "ta": "ta-IN",
    "en": "en-GB",
    "ja": "ja-JP",
    "fr": "fr-FR",
    "es": "es-ES",
    "de": "de-DE",
    "hi": "hi-IN",
    "zh": "zh-CN",
    "ko": "ko-KR",
    "ar": "ar-SA",
}

INTENTS = {
    "Emergency": ["emergency", "help", "danger", "fire", "police", "ambulance", "accident", "lost", "steal", "stolen", "robbed"],
    "Medical": ["hospital", "doctor", "medicine", "pain", "sick", "clinic", "pharmacy", "injured"],
    "Directions": ["where", "nearest", "how to reach", "direction", "way to", "far", "near"],
    "Transport": ["bus", "train", "taxi", "auto", "airport", "station", "ticket"],
    "Food": ["food", "restaurant", "eat", "hungry", "water", "drink", "menu"],
    "Accommodation": ["hotel", "room", "stay", "booking", "check in", "check out"],
    "Shopping": ["price", "cost", "buy", "shop", "how much", "discount"],
}

@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

def detect_intent(text):
    text_lower = text.lower()
    for intent, keywords in INTENTS.items():
        for kw in keywords:
            if kw in text_lower:
                return intent
    return "General"

def speech_to_text(audio_bytes, lang_code):
    model = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    segments, _ = model.transcribe(path, language=lang_code)
    os.remove(path)
    return " ".join(seg.text for seg in segments).strip()

@st.cache_data(show_spinner=False)
def translate_text(text, source, target):
    src = TRANSLATE_CODE_OVERRIDES.get(source, source)
    tgt = TRANSLATE_CODE_OVERRIDES.get(target, target)
    for attempt in range(3):
        try:
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    mm_src = MYMEMORY_CODE_OVERRIDES.get(source, source)
    mm_tgt = MYMEMORY_CODE_OVERRIDES.get(target, target)
    return MyMemoryTranslator(source=mm_src, target=mm_tgt).translate(text)

def text_to_speech(text, lang_code):
    voice = TTS_CODE_OVERRIDES.get(lang_code, lang_code)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    subprocess.run(["espeak-ng", "-v", voice, "-w", path, text], check=True)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data

st.markdown("""
<div class="hero">
    <h1>🗣️ AI Tourist Communication Assistant</h1>
    <p>Speak in your language, get instantly translated speech back — built for real tourist situations like hospitals, directions and emergencies.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Choose languages</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("Your language", list(LANGUAGES.keys()), index=0)
with col2:
    target_lang_name = st.selectbox("Translate to", list(LANGUAGES.keys()), index=1)
st.markdown('</div>', unsafe_allow_html=True)

source_lang = LANGUAGES[source_lang_name]
target_lang = LANGUAGES[target_lang_name]

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">1. Say something</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎤 Record", "⌨️ Type"])

input_text = None

with tab1:
    audio_bytes = audio_recorder(text="Click to record", recording_color="#e63946", neutral_color="#0f766e")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Transcribing..."):
            try:
                input_text = speech_to_text(audio_bytes, source_lang)
                st.success(f"Heard: {input_text}")
            except Exception as e:
                st.error(f"Could not recognize speech: {e}")

with tab2:
    typed = st.text_input("Type your sentence")
    if typed:
        input_text = typed

st.markdown('</div>', unsafe_allow_html=True)

if input_text:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Translation</div>', unsafe_allow_html=True)

    intent = detect_intent(input_text)
    badge_class = "intent-emergency" if intent == "Emergency" else "intent-normal"
    badge_text = f"🚨 {intent}" if intent == "Emergency" else intent
    st.markdown(f'<span class="intent-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    with st.spinner("Translating..."):
        try:
            translated = translate_text(input_text, source_lang, target_lang)
            st.markdown(f'<div class="result-box">{translated}</div>', unsafe_allow_html=True)
            audio_data = text_to_speech(translated, target_lang)
            st.audio(audio_data, format="audio/wav")
        except Exception as e:
            st.error(f"Translation failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">3. Foreigner\'s reply</div>', unsafe_allow_html=True)
    reply_audio = audio_recorder(text="Record reply", key="reply_recorder", recording_color="#f59e0b", neutral_color="#0f766e")
    if reply_audio:
        st.audio(reply_audio, format="audio/wav")
        with st.spinner("Processing reply..."):
            try:
                reply_text = speech_to_text(reply_audio, target_lang)
                st.success(f"Heard: {reply_text}")
                back_translated = translate_text(reply_text, target_lang, source_lang)
                st.markdown(f'<div class="result-box">{back_translated}</div>', unsafe_allow_html=True)
                back_audio = text_to_speech(back_translated, source_lang)
                st.audio(back_audio, format="audio/wav")
            except Exception as e:
                st.error(f"Could not process reply: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
