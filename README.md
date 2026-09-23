# AI Tourist Communication Assistant (Offline)

A real-time speech-to-speech translation tool for tourists, built around actual travel scenarios — asking for a hospital, getting lost, calling for help — instead of generic phrase translation. Runs entirely offline: no Google/Cloud APIs, no internet calls at inference time.

## Pipeline

Speech (Tamil) → Local STT (Whisper) → Intent Detection → Local Translation (M2M100) → Local TTS (espeak-ng) → Foreigner hears it
Foreigner replies → Local STT → Translation back to Tamil → Local TTS → Tourist hears it

## Why it's meaningful

Most translator demos just swap words. This one adds **intent detection**: if the sentence matches emergency or medical keywords, it's flagged in red before translation, so urgency isn't lost even if the translation is imperfect. Quick-phrase buttons cover common tourist emergencies without needing a working mic (useful for demos/judging).

## Stack

- **STT:** `faster-whisper` (small model) — runs locally on CPU, no API key, no network calls
- **Translation:** `facebook/m2m100_418M` via HuggingFace `transformers` — local neural machine translation covering Tamil and 100 languages
- **TTS:** `espeak-ng` — local offline speech synthesizer (robotic but real, no external dependency)
- **Intent detection:** rule-based keyword classifier

## Features

- Bidirectional voice translation (10 languages)
- Live mic recording in-browser
- Typed input fallback
- Pre-built emergency/direction quick phrases
- Fully offline after first model download — works with no internet once set up

## Local setup

```bash
sudo apt install espeak-ng ffmpeg   # Linux; on Mac: brew install espeak ffmpeg
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the Whisper and M2M100 model weights (~2GB total) — after that it works fully offline.

## Deployment note

This stack is CPU/RAM-heavy: Whisper-small (~500MB) + M2M100-418M (~1.5GB) both load into memory. **Streamlit Community Cloud's free tier (1GB RAM) will likely fail to run this** — it's built for local use, a university lab machine, or a paid/self-hosted server with at least 4GB RAM. If you need it on Streamlit Cloud specifically, swap in smaller models: `WhisperModel("tiny")` and consider `facebook/m2m100_418M` is already the smallest M2M100 variant, so you may need a different host instead.

## Notes

- Whisper and M2M100 both officially support Tamil, so accuracy is reasonable, though not as strong as the big cloud APIs.
- espeak-ng's Tamil voice is functional but robotic-sounding — this is the tradeoff for zero external dependency.
- For a production version with better voice quality, a self-hosted TTS model (e.g. Coqui TTS) could replace espeak-ng while staying offline.
