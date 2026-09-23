import streamlit as st
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from audio_recorder_streamlit import audio_recorder
import subprocess
import tempfile
import os

st.set_page_config(page_title="AI Tourist Communication Assistant", page_icon="🗣️", layout="centered")

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

@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

@st.cache_resource
def load_translator():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

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

def translate_text(text, source, target):
    tokenizer, model = load_translator()
    tokenizer.src_lang = source
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target))
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def text_to_speech(text, lang_code):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    subprocess.run(["espeak-ng", "-v", lang_code, "-w", path, text], check=True)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data

st.title("🗣️ AI Tourist Communication Assistant")
st.caption("Fully offline — speech recognition, translation and voice output all run locally, no external APIs.")

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

    with st.spinner("Translating..."):
        try:
            translated = translate_text(input_text, source_lang, target_lang)
            st.markdown(f"**{target_lang_name}:** {translated}")
            audio_data = text_to_speech(translated, target_lang)
            st.audio(audio_data, format="audio/wav")
        except Exception as e:
            st.error(f"Translation failed: {e}")

    st.divider()
    st.subheader("3. Foreigner's reply")
    reply_audio = audio_recorder(text="Record reply", key="reply_recorder", recording_color="#2a9d8f", neutral_color="#457b9d")
    if reply_audio:
        st.audio(reply_audio, format="audio/wav")
        with st.spinner("Processing reply..."):
            try:
                reply_text = speech_to_text(reply_audio, target_lang)
                st.success(f"Heard: {reply_text}")
                back_translated = translate_text(reply_text, target_lang, source_lang)
                st.markdown(f"**{source_lang_name}:** {back_translated}")
                back_audio = text_to_speech(back_translated, source_lang)
                st.audio(back_audio, format="audio/wav")
            except Exception as e:
                st.error(f"Could not process reply: {e}")
