#!/usr/bin/env python3
"""Röstprover (enrollment): bygger en röstsignatur per namngiven deltagare med
ECAPA-TDNN (SpeechBrain) och matchar diariserade kluster mot deltagare via
cosinuslikhet.

Samma modell används för både uppladdade prover och klustren ur inspelningen så
att vektorerna lever i samma rum. Körs lokalt på CPU. Inget ljud lämnar datorn.
"""

import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Vikterna cachas under model/ (som redan är gitignorerat).
_CACHE = os.path.join(HERE, "model", "ecapa")
_classifier = None
_classifier_error = None


def get_embedder():
    """Ladda ECAPA-modellen en gång och cacha den i en modulglobal."""
    global _classifier, _classifier_error
    if _classifier is not None:
        return _classifier
    if _classifier_error is not None:
        raise _classifier_error

    try:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:  # noqa: BLE001 — äldre SpeechBrain
            from speechbrain.pretrained import EncoderClassifier

        clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_CACHE,
            run_opts={"device": "cpu"},
        )
    except Exception as exc:  # noqa: BLE001
        _classifier_error = RuntimeError("Kunde inte ladda röstmodellen (ECAPA): " + str(exc))
        raise _classifier_error

    _classifier = clf
    return clf


def _l2(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def embed(waveform, sample_rate=16000):
    """En 192-dim röstembedding för en vågform (16 kHz mono float32)."""
    import torch

    clf = get_embedder()
    wav = np.asarray(waveform, dtype="float32").reshape(-1)
    if wav.size == 0:
        raise ValueError("Tomt ljud för embedding.")
    tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
    with torch.no_grad():
        emb = clf.encode_batch(tensor).reshape(-1).cpu().numpy()
    return np.asarray(emb, dtype="float32")


def build_enrollment(samples, sample_rate=16000):
    """samples: [(name, waveform)]. Flera klipp per namn medelvärdesbildas till
    en centroid. Returnerar {name: normaliserad centroid}."""
    by_name = defaultdict(list)
    for name, wav in samples:
        try:
            by_name[name].append(_l2(embed(wav, sample_rate)))
        except Exception:  # noqa: BLE001 — hoppa över trasiga klipp
            continue
    out = {}
    for name, embs in by_name.items():
        if embs:
            out[name] = _l2(np.mean(embs, axis=0))
    return out


def cluster_centroids(waveform, turns, sample_rate=16000, per_cluster=4, min_dur=1.0):
    """Bygger en röstsignatur per diariserat kluster genom att embedda dess
    längsta talsegment ur inspelningen. Returnerar {speaker: centroid}."""
    wav = np.asarray(waveform, dtype="float32").reshape(-1)
    by_spk = defaultdict(list)
    for t0, t1, spk in turns:
        by_spk[spk].append((t1 - t0, t0, t1))

    out = {}
    for spk, segs in by_spk.items():
        segs.sort(reverse=True)  # längst först
        embs = []
        for dur, t0, t1 in segs[:per_cluster]:
            if dur < min_dur:
                continue
            a = max(0, int(t0 * sample_rate))
            b = min(wav.size, int(t1 * sample_rate))
            clip = wav[a:b]
            if clip.size < int(0.5 * sample_rate):
                continue
            try:
                embs.append(_l2(embed(clip, sample_rate)))
            except Exception:  # noqa: BLE001
                continue
        if embs:
            out[spk] = _l2(np.mean(embs, axis=0))
    return out


def match_clusters(cluster_centroids_map, enrollment, threshold=0.5):
    """Matchar kluster mot namngivna deltagare (Hungarian över cosinuslikhet).

    Returnerar {speaker: name} enbart för matchningar med likhet >= threshold.
    Kluster utan trygg matchning saknas i kartan och behåller generisk etikett.
    """
    if not cluster_centroids_map or not enrollment:
        return {}

    from scipy.optimize import linear_sum_assignment

    clusters = list(cluster_centroids_map.keys())
    names = list(enrollment.keys())
    sim = np.zeros((len(clusters), len(names)), dtype="float32")
    for i, c in enumerate(clusters):
        cv = _l2(cluster_centroids_map[c])
        for j, n in enumerate(names):
            sim[i, j] = float(np.dot(cv, _l2(enrollment[n])))

    rows, cols = linear_sum_assignment(-sim)  # maximera total likhet
    out = {}
    for i, j in zip(rows, cols):
        if sim[i, j] >= threshold:
            out[clusters[i]] = names[j]
    return out
