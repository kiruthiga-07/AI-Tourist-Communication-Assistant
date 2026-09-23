# AI Tourist Communication Assistant

A real-time speech-to-speech translation tool for tourists, built around actual travel scenarios — asking for a hospital, getting lost, calling for help — instead of generic phrase translation.

## Pipeline

Speech (Tamil) → Local STT (Whisper) → Intent Detection → Translation (Google, no key needed) → Local TTS (espeak-ng) → Foreigner hears it
Foreigner replies → Local STT → Translation back to Tamil → Local TTS → Tourist hears it

## Why it's meaningful

Most translator demos just swap words. This one adds **intent detection**: if the sentence matches emergency or medical keywords, it's flagged in red before translation, so urgency isn't lost even if the translation is imperfect.

## Stack

- **STT:** `faster-whisper` (small model) — runs locally on CPU, no API key, no network calls
- **Translation:** `deep-translator`'s `GoogleTranslator` — hits Google Translate's public web backend, no API key or account needed, much more accurate than fully offline MT models for Tamil
- **TTS:** `espeak-ng` — local offline speech synthesizer (robotic but real, no external dependency)
- **Intent detection:** rule-based keyword classifier

## Features

- Bidirectional voice translation (10 languages)
- Live mic recording in-browser
- Typed input fallback
- Styled UI with intent badges and result cards

## Local setup

```bash
sudo apt install espeak-ng ffmpeg   # Linux; on Mac: brew install espeak ffmpeg
pip install -r requirements.txt
streamlit run app.py
```

## Deployment note

STT (Whisper-small, ~500MB) still needs real RAM/CPU. This will run on Streamlit Community Cloud's free tier more comfortably than the earlier M2M100 version, but if you hit memory limits, switch `WhisperModel("small", ...)` to `WhisperModel("tiny", ...)` in `app.py` — it's faster and lighter, at a small accuracy cost.

## Translation API notes

`deep-translator`'s Google backend needs no key and has no hard published limit, but it's an unofficial wrapper around Google's public translate endpoint, so heavy sustained use can get temporarily rate-limited (Google allows 5 requests/second, 200k/day per IP). The app now caches translations so Streamlit's automatic reruns don't resend the same request, and automatically falls back to MyMemory if Google throttles it. If you still hit limits under heavy use:

- **LibreTranslate** — open source; self-host it for genuinely unlimited use, or use the public instance with a free API key (rate-limited)
- **MyMemory** — free, no key for light use; free email registration raises the limit to 50,000 words/day

## Notes

- Whisper officially supports Tamil, so STT accuracy is solid.
- espeak-ng's Tamil voice is functional but robotic-sounding — that's the tradeoff for zero external TTS dependency. For better voice quality later, a self-hosted model like Coqui TTS could replace it while staying offline.
