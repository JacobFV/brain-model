"""TRIBE v2 model loading and text-to-brain prediction utilities."""

import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np

CACHE_FOLDER = Path("./cache")


def load_model(cache_folder: Path = CACHE_FOLDER):
    """Load TRIBE v2 from HuggingFace with forced local extraction mode."""
    from tribev2 import TribeModel

    print("Loading TRIBE v2 model...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)

    # Force local computation mode for all feature extractors
    # (the default exca caching can hang on first run)
    for attr in ["text_feature", "audio_feature", "video_feature"]:
        feat = getattr(model.data, attr, None)
        if feat is not None and hasattr(feat, "infra"):
            feat.infra.mode = "force"

    print("Model loaded.")
    return model


def build_events_for_text(text: str, cache_folder: Path = CACHE_FOLDER):
    """
    Build a TRIBE v2 events dataframe from raw text, bypassing the
    whisperx subprocess. Uses gTTS to generate audio and constructs
    word-level events with evenly-distributed timing.
    """
    import pandas as pd
    import soundfile as sf
    from gtts import gTTS
    from langdetect import detect
    from neuralset.events.transforms import (
        AddContextToWords,
        AddSentenceToWords,
        AddText,
        ChunkEvents,
        RemoveMissing,
        standardize_events,
    )

    # Generate audio via gTTS
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    audio_dir = cache_folder / f"brain_radio_{text_hash}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio.mp3"

    if not audio_path.exists():
        lang = detect(text)
        tts = gTTS(text, lang=lang)
        tts.save(str(audio_path))

    # Get audio duration
    info = sf.info(str(audio_path))
    duration = info.duration

    # Build Audio event
    audio_event = {
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0.0,
        "duration": duration,
        "timeline": "default",
        "subject": "default",
    }

    # Build Word events with evenly-distributed timing
    words = text.split()
    n_words = len(words)
    word_duration = duration / max(n_words, 1)

    word_events = []
    for i, word in enumerate(words):
        word_events.append({
            "type": "Word",
            "text": word.strip(".,;:!?\"'()-"),
            "start": i * word_duration,
            "duration": word_duration * 0.8,
            "timeline": "default",
            "subject": "default",
            "language": "english",
            "sequence_id": 0,
        })

    df = pd.DataFrame([audio_event] + word_events)

    transforms = [
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.99),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    df = standardize_events(df)
    for transform in transforms:
        df = transform(df)
    return standardize_events(df, auto_fill=False)


def text_to_predictions(model, text: str, label: str = ""):
    """Run a text string through TRIBE v2 and return brain predictions."""
    tag = f" [{label}]" if label else ""
    print(f"Predicting brain response{tag}...")

    df = build_events_for_text(text, CACHE_FOLDER)
    preds, segments = model.predict(events=df)
    print(f"  -> {preds.shape[0]} timesteps x {preds.shape[1]} vertices")
    return preds, segments


def batch_predict(model, texts: dict[str, str]) -> dict[str, np.ndarray]:
    """Run multiple texts and return {label: preds} dict."""
    results = {}
    for label, text in texts.items():
        preds, _ = text_to_predictions(model, text, label=label)
        results[label] = preds
    return results
