import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
import io
import tempfile

st.set_page_config(page_title="AI Tourist Communication Assistant", page_icon="🗣️", layout="centered")

LANGUAGES = {
    "Tamil": "ta",
    "English": "en",
    "Japanese": "ja",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh-CN",
    "Korean": "ko",
    "Arabic": "ar",
}

SPEECH_LANG_CODES = {
    "ta": "ta-IN",
    "en": "en-IN",
    "ja": "ja-JP",
    "fr": "fr-FR",
    "es": "es-ES",
    "de": "de-DE",
    "hi": "hi-IN",
    "zh-CN": "zh-CN",
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

QUICK_PHRASES = [
    "Nearest hospital enga iruku?",
    "Police station enga iruku?",
    "Naan varazhi thondrudhu",
    "Evlo per",
    "Enakku udhavi venum",
]

def detect_intent(text):
    text_lower = text.lower()
    for intent, keywords in INTENTS.items():
        for kw in keywords:
            if kw in text_lower:
                return intent
    return "General"

def speech_to_text(audio_bytes, lang_code):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with sr.AudioFile(tmp.name) as source:
            audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=SPEECH_LANG_CODES.get(lang_code, "en-IN"))

def translate_text(text, source, target):
    return GoogleTranslator(source=source, target=target).translate(text)

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

st.title("🗣️ AI Tourist Communication Assistant")
st.caption("Speak in your language, get instantly translated speech back — built for real tourist situations like hospitals, directions and emergencies.")

col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("Your language", list(LANGUAGES.keys()), index=0)
with col2:
    target_lang_name = st.selectbox("Translate to", list(LANGUAGES.keys()), index=1)

source_lang = LANGUAGES[source_lang_name]
target_lang = LANGUAGES[target_lang_name]

st.divider()
st.subheader("1. Say something")

tab1, tab2, tab3 = st.tabs(["🎤 Record", "⌨️ Type", "⚡ Quick Phrases"])

input_text = None

with tab1:
    audio_bytes = audio_recorder(text="Click to record", recording_color="#e63946", neutral_color="#457b9d")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        try:
            input_text = speech_to_text(audio_bytes, source_lang)
            st.success(f"Heard: {input_text}")
        except Exception as e:
            st.error(f"Could not recognize speech: {e}")

with tab2:
    typed = st.text_input("Type your sentence")
    if typed:
        input_text = typed

with tab3:
    picked = st.selectbox("Common phrases", QUICK_PHRASES)
    if st.button("Use this phrase"):
        input_text = picked

if input_text:
    st.divider()
    st.subheader("2. Translation")

    intent = detect_intent(input_text)
    if intent == "Emergency":
        st.error(f"🚨 Intent detected: {intent}")
    else:
        st.info(f"Intent detected: {intent}")

    try:
        translated = translate_text(input_text, source_lang, target_lang)
        st.markdown(f"**{target_lang_name}:** {translated}")

        audio_buf = text_to_speech(translated, target_lang)
        st.audio(audio_buf, format="audio/mp3")
    except Exception as e:
        st.error(f"Translation failed: {e}")

    st.divider()
    st.subheader("3. Foreigner's reply")
    reply_audio = audio_recorder(text="Record reply", key="reply_recorder", recording_color="#2a9d8f", neutral_color="#457b9d")
    if reply_audio:
        st.audio(reply_audio, format="audio/wav")
        try:
            reply_text = speech_to_text(reply_audio, target_lang)
            st.success(f"Heard: {reply_text}")
            back_translated = translate_text(reply_text, target_lang, source_lang)
            st.markdown(f"**{source_lang_name}:** {back_translated}")
            back_audio = text_to_speech(back_translated, source_lang)
            st.audio(back_audio, format="audio/mp3")
        except Exception as e:
            st.error(f"Could not process reply: {e}")
