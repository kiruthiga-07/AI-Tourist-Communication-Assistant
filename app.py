import streamlit as st
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator, MyMemoryTranslator
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import io
import subprocess
import tempfile
import time
import os

try:
    import speech_recognition as sr
    GOOGLE_STT_AVAILABLE = True
except ImportError:
    GOOGLE_STT_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

st.set_page_config(page_title="AI Tourist Communication Assistant", page_icon="🗣️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0f766e, #14b8a6);
    padding: 2.2rem 2rem; border-radius: 20px; color: white; margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p { margin: 0.6rem 0 0; opacity: 0.92; font-size: 1rem; }
.intent-badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
    font-weight: 600; font-size: 0.78rem; margin-top: 0.3rem;
}
.intent-emergency { background: #fee2e2; color: #b91c1c; }
.intent-normal { background: #e0f2fe; color: #0369a1; }
div.stButton > button {
    background-color: #0f766e; color: white; border-radius: 10px;
    border: none; padding: 0.45rem 1.2rem; font-weight: 600;
}
div.stButton > button:hover { background-color: #115e59; color: white; }
</style>
""", unsafe_allow_html=True)

LANGUAGES = {
    "Auto-detect (speech only)": None,
    "Tamil": "ta", "English": "en", "Japanese": "ja", "French": "fr",
    "Spanish": "es", "German": "de", "Hindi": "hi", "Chinese": "zh",
    "Korean": "ko", "Arabic": "ar",
}
CODE_TO_NAME = {v: k for k, v in LANGUAGES.items() if v is not None}
TRANSLATE_CODE_OVERRIDES = {"zh": "zh-CN"}
TTS_CODE_OVERRIDES = {"zh": "cmn"}
GTTS_LANG_OVERRIDES = {"zh": "zh-CN"}
MYMEMORY_CODE_OVERRIDES = {
    "ta": "ta-IN", "en": "en-GB", "ja": "ja-JP", "fr": "fr-FR", "es": "es-ES",
    "de": "de-DE", "hi": "hi-IN", "zh": "zh-CN", "ko": "ko-KR", "ar": "ar-SA",
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
TOURIST_PROMPT = (
    "Tourist phrases about hospital, doctor, police, emergency, hotel, bus, "
    "train, taxi, restaurant, directions, nearest, help, lost, enga iruku, "
    "epdi poganum, evlo, venum, romba nandri."
)
QUICK_PHRASES = [
    "Where is the nearest hospital?", "I need a doctor.", "Please call the police.",
    "I am lost.", "I need water.", "How much does this cost?",
]


@st.cache_resource
def load_whisper(model_size):
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def detect_intent(text):
    text_lower = text.lower()
    for intent, keywords in INTENTS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "General"


def preprocess_audio(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_frame_rate(16000).set_channels(1)
    audio = audio.apply_gain(-20.0 - audio.dBFS)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def speech_to_text(audio_bytes, lang_code, model_size):
    """Returns (text, detected_lang_code). lang_code=None lets Whisper auto-detect."""
    if not audio_bytes or len(audio_bytes) < 8000:
        return "", lang_code
    model = load_whisper(model_size)
    audio_bytes = preprocess_audio(audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        segments, info = model.transcribe(
            path, language=lang_code, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            beam_size=5, best_of=5, condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            initial_prompt=TOURIST_PROMPT if lang_code in ("ta", "en", None) else None,
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text, (info.language if lang_code is None else lang_code)
    finally:
        os.remove(path)


def speech_to_text_google(audio_bytes, lang_code):
    if not GOOGLE_STT_AVAILABLE or lang_code is None:
        return ""
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(preprocess_audio(audio_bytes))
        path = tmp.name
    try:
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language=MYMEMORY_CODE_OVERRIDES.get(lang_code, lang_code))
    except (sr.UnknownValueError, sr.RequestError):
        return ""
    finally:
        os.remove(path)


def transcribe(audio_bytes, lang_code, model_size, use_google_stt):
    if use_google_stt and lang_code is not None:
        text = speech_to_text_google(audio_bytes, lang_code)
        if text:
            return text, lang_code
    return speech_to_text(audio_bytes, lang_code, model_size)


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


def text_to_speech(text, lang_code, natural=False):
    if natural and GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text, lang=GTTS_LANG_OVERRIDES.get(lang_code, lang_code))
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()
        except Exception:
            pass
    voice = TTS_CODE_OVERRIDES.get(lang_code, lang_code)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        subprocess.run(["espeak-ng", "-v", voice, "-w", path, text], check=True)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


# ---------------- Session state ----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "turn" not in st.session_state:
    st.session_state.turn = 0

st.markdown("""
<div class="hero">
<h1>🗣️ AI Tourist Communication Assistant</h1>
<p>A live, turn-by-turn translated conversation between a traveler and a local — speak or type, either side, either turn.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar settings ----------------
with st.sidebar:
    st.markdown("### Languages")
    source_lang_name = st.selectbox("🧳 Traveler speaks", list(LANGUAGES.keys()), index=1)
    target_lang_name = st.selectbox("🧑‍🤝‍🧑 Local person speaks", list(LANGUAGES.keys()), index=2)
    source_lang, target_lang = LANGUAGES[source_lang_name], LANGUAGES[target_lang_name]

    st.markdown("### Settings")
    model_size = st.selectbox("Speech recognition model", ["tiny", "base", "small", "medium"], index=2,
                               help="Bigger = more accurate but slower and more RAM. 'medium' may not fit on Streamlit Cloud's free tier.")
    use_google_stt = st.checkbox("Higher-accuracy online STT (needs internet)", value=False, disabled=not GOOGLE_STT_AVAILABLE)
    use_natural_voice = st.checkbox("Natural-sounding voice (needs internet)", value=GTTS_AVAILABLE, disabled=not GTTS_AVAILABLE)

    st.markdown("---")
    if st.button("🔄 Reset conversation"):
        st.session_state.conversation = []
        st.session_state.turn = 0
        st.rerun()

    if st.session_state.conversation:
        transcript = "\n\n".join(
            f"{'Traveler' if t['speaker']=='traveler' else 'Local'} ({t['lang_name']}): {t['original']}\n"
            f"→ ({t['target_lang_name']}): {t['translated']}"
            for t in st.session_state.conversation
        )
        st.download_button("⬇️ Download transcript", transcript, file_name="conversation.txt")

# ---------------- Quick emergency phrases ----------------
with st.expander("🚨 Quick emergency phrases"):
    if target_lang is None:
        st.info("Pick a specific language for the local person (not Auto-detect) to use quick phrases.")
    else:
        for phrase in QUICK_PHRASES:
            if st.button(phrase, key=f"qp_{phrase}"):
                translated = translate_text(phrase, "en", target_lang)
                st.success(translated)
                st.audio(text_to_speech(translated, target_lang, natural=use_natural_voice), format="audio/mp3" if use_natural_voice else "audio/wav")

# ---------------- Conversation history ----------------
for t in st.session_state.conversation:
    avatar = "🧳" if t["speaker"] == "traveler" else "🧑‍🤝‍🧑"
    with st.chat_message(t["speaker"], avatar=avatar):
        badge_class = "intent-emergency" if t["intent"] == "Emergency" else "intent-normal"
        st.markdown(f'<span class="intent-badge {badge_class}">{t["intent"]}</span>', unsafe_allow_html=True)
        st.markdown(f"**{t['lang_name']}:** {t['original']}")
        st.markdown(f"**→ {t['target_lang_name']}:** {t['translated']}")
        if t.get("audio"):
            st.audio(t["audio"], format="audio/mp3" if use_natural_voice else "audio/wav")

# ---------------- Next turn input ----------------
speaker_is_traveler = st.session_state.turn % 2 == 0
current_lang = source_lang if speaker_is_traveler else target_lang
other_lang = target_lang if speaker_is_traveler else source_lang
current_label = source_lang_name if speaker_is_traveler else target_lang_name
avatar = "🧳" if speaker_is_traveler else "🧑‍🤝‍🧑"


def add_turn(original_text, detected_lang_code, target_code):
    if target_code is None:
        st.warning("The other person's language is set to Auto-detect — please pick a specific language in the sidebar to translate into.")
        return
    translated = translate_text(original_text, detected_lang_code, target_code)
    audio = text_to_speech(translated, target_code, natural=use_natural_voice)
    st.session_state.conversation.append({
        "speaker": "traveler" if speaker_is_traveler else "local",
        "lang_name": CODE_TO_NAME.get(detected_lang_code, detected_lang_code or "unknown"),
        "original": original_text,
        "translated": translated,
        "target_lang_name": CODE_TO_NAME.get(target_code, target_code),
        "intent": detect_intent(original_text),
        "audio": audio,
    })
    st.session_state.turn += 1
    st.rerun()


st.markdown(f"#### Turn {st.session_state.turn + 1}: {avatar} {current_label}'s turn")
mode = st.radio("Respond with:", ["🎤 Speak", "⌨️ Type"], horizontal=True, key=f"mode_{st.session_state.turn}")

if mode == "🎤 Speak":
    audio_bytes = audio_recorder(key=f"rec_{st.session_state.turn}", text="Click to record",
                                  recording_color="#e63946", neutral_color="#0f766e")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Transcribing..."):
            text, detected = transcribe(audio_bytes, current_lang, model_size, use_google_stt)
        if text:
            add_turn(text, detected, other_lang)
        else:
            st.warning("No clear speech detected — try recording again, closer to the mic.")
else:
    with st.form(key=f"form_{st.session_state.turn}", clear_on_submit=True):
        typed = st.text_input("Type here")
        submitted = st.form_submit_button("Send")
    if submitted and typed:
        add_turn(typed, current_lang if current_lang else source_lang, other_lang)
