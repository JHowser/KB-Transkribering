#!/usr/bin/env python3
"""Talardiarisering ("vem talar när") med pyannote.audio.

Diarisering är språkoberoende — den arbetar på akustiska röstembeddingar, inte på
ord. Därför finns ingen svensk-specifik diariseringsmodell; KB-Whisper sköter orden,
den här modulen sköter talarna. Körs lokalt på CPU. Inget ljud lämnar datorn; token
används bara för att ladda ner modellvikterna första gången.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
_pipeline = None
_pipeline_error = None


class DiarizationUnavailable(RuntimeError):
    """Diarisering går inte att köra (saknad token, ej godkända villkor, saknat
    bibliotek). Appen faller då tillbaka till en vanlig transkription."""


def read_token():
    """Läs Hugging Face-token från HF_TOKEN (eller hf_token.txt bredvid app.py)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok and tok.strip():
        return tok.strip()
    path = os.path.join(HERE, "hf_token.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                val = fh.read().strip()
                if val:
                    return val
        except OSError:
            pass
    return None


def get_diarizer():
    """Ladda pyannote-pipelinen en gång och cacha den i en modulglobal."""
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline
    if _pipeline_error is not None:
        raise _pipeline_error

    token = read_token()
    if not token:
        _pipeline_error = DiarizationUnavailable(
            "Ingen Hugging Face-token hittades (HF_TOKEN eller hf_token.txt)."
        )
        raise _pipeline_error

    try:
        import torch
        from pyannote.audio import Pipeline

        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        if pipe is None:
            # from_pretrained returnerar None när villkoren inte är godkända eller
            # token saknar behörighet.
            raise DiarizationUnavailable(
                "Kunde inte ladda pyannote/speaker-diarization-3.1. Kontrollera att "
                "din token är giltig och att du godkänt villkoren på både "
                "pyannote/speaker-diarization-3.1 och pyannote/segmentation-3.0."
            )
        # MPS-stödet är ojämnt på Apple Silicon — CPU är korrekt och snabbt nog.
        pipe.to(torch.device("cpu"))
    except DiarizationUnavailable as exc:
        _pipeline_error = exc
        raise
    except Exception as exc:  # noqa: BLE001
        _pipeline_error = DiarizationUnavailable(
            "Kunde inte ladda talarmodellen: " + str(exc)
        )
        raise _pipeline_error

    _pipeline = pipe
    return pipe


def diarize(waveform, sample_rate):
    """Diariserar den redan avkodade vågformen (16 kHz mono float32).

    Tar emot np.ndarray direkt från appens ``_decode`` så filen inte avkodas två
    gånger. Returnerar en tidssorterad lista ``[(start, end, speaker), ...]``.
    """
    import numpy as np
    import torch

    pipe = get_diarizer()
    wav = np.asarray(waveform, dtype="float32").reshape(1, -1)
    tensor = torch.from_numpy(wav)
    annotation = pipe({"waveform": tensor, "sample_rate": int(sample_rate)})

    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(segment.start), float(segment.end), str(speaker)))
    turns.sort(key=lambda t: t[0])
    return turns
