# AI Tourist Communication Assistant

A real-time speech-to-speech translation tool for tourists, built around actual travel scenarios — asking for a hospital, getting lost, calling for help — instead of generic phrase translation.

## Pipeline

Speech (Tamil) → Speech-to-Text → Intent Detection → Translation → Text-to-Speech → Foreigner hears it
Foreigner replies → Speech-to-Text → Translation back to Tamil → Text-to-Speech → Tourist hears it

## Why it's meaningful

Most translator demos just swap words. This one adds **intent detection**: if the sentence matches emergency or medical keywords, it's flagged in red before translation, so urgency isn't lost even if the translation is imperfect. Quick-phrase buttons cover common tourist emergencies without needing a working mic (useful for demos/judging).

## Features

- Bidirectional voice translation (10 languages)
- Live mic recording in-browser (no local audio setup needed)
- Typed input fallback
- Pre-built emergency/direction quick phrases
- Rule-based intent classifier (Emergency, Medical, Directions, Transport, Food, Accommodation, Shopping, General)

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py`, `requirements.txt`, `packages.txt`, `README.md`)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub, select the repo
3. Set main file to `app.py` and deploy
4. `packages.txt` ensures `ffmpeg` is installed on the cloud machine for audio processing

## Notes

- STT and translation use free Google-backed APIs (`SpeechRecognition`'s Google Web Speech, `deep-translator`'s `GoogleTranslator`) — no API keys required, but they need internet access and have soft rate limits, fine for a demo/project.
- For a production version, swap in a paid STT/translation API (Google Cloud Speech-to-Text, Azure Translator) for reliability at scale.
