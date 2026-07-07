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
        import warnings
        # Vi matar in ljudet som en färdig vågform (se diarize() nedan), så pyannote
        # avkodar aldrig filer själv. Dämpa därför torchcodec/FFmpeg-varningen som
        # annars skräms i terminalen på pyannote.audio 4.x — den påverkar inte oss.
        warnings.filterwarnings("ignore", message=".*torchcodec.*")

        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline

        # pyannote.audio 4.x döpte om standardpipelinen till "community-1"; 3.x har
        # bara "3.1". Prova den nyare först på 4.x och fall tillbaka till 3.1.
        ver = getattr(pyannote.audio, "__version__", "") or ""
        major = int(ver.split(".")[0]) if ver[:1].isdigit() else 0
        if major >= 4:
            model_ids = ["pyannote/speaker-diarization-community-1",
                         "pyannote/speaker-diarization-3.1"]
        else:
            model_ids = ["pyannote/speaker-diarization-3.1"]

        pipe, load_error = None, None
        for model_id in model_ids:
            try:
                try:
                    # pyannote.audio >= 4 (och nyare huggingface_hub) använder token=
                    pipe = Pipeline.from_pretrained(model_id, token=token)
                except TypeError:
                    # pyannote.audio 3.x använder use_auth_token=
                    pipe = Pipeline.from_pretrained(model_id, use_auth_token=token)
            except Exception as exc:  # noqa: BLE001 — prova nästa modell i listan
                load_error = exc
                pipe = None
            if pipe is not None:
                break

        if pipe is None:
            # from_pretrained returnerar None (eller kastar) när villkoren inte är
            # godkända eller token saknar behörighet.
            hint = (" (" + str(load_error) + ")") if load_error else ""
            raise DiarizationUnavailable(
                "Kunde inte ladda talarpipelinen. Kontrollera att din token är giltig "
                "och att du godkänt villkoren på modellsidan: pyannote.audio 4.x kräver "
                "pyannote/speaker-diarization-community-1, medan 3.x kräver både "
                "pyannote/speaker-diarization-3.1 och pyannote/segmentation-3.0." + hint
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
