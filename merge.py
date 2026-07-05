#!/usr/bin/env python3
"""Slår ihop KB-Whisper-segment med pyannote-talarturer på tidsstämplar och
renderar en läsbar transkription märkt per talare.

Ordbelastningen ligger hos ASR (segment: [(start, end, text)]) och talarna hos
diariseringen (turer: [(start, end, speaker)]). Här kopplas de två ihop.
"""


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(s0, s1, turns):
    """Talaren vars tur överlappar segmentet mest i tid. Vid ingen överlapp
    (diariseringen missade en lucka) väljs närmaste tur i tid."""
    best_spk, best_ov = None, 0.0
    for t0, t1, spk in turns:
        ov = _overlap(s0, s1, t0, t1)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    if best_spk is not None:
        return best_spk

    mid = (s0 + s1) / 2.0
    nearest, dist = None, None
    for t0, t1, spk in turns:
        d = 0.0 if t0 <= mid <= t1 else min(abs(mid - t0), abs(mid - t1))
        if dist is None or d < dist:
            dist, nearest = d, spk
    return nearest


def assign_speakers(segments, turns):
    """segments: [(start, end, text)], turns: [(start, end, speaker)].

    Returnerar [(start, end, speaker, text)]. v1 delar inte segment vid en
    talarväxling — hela segmentet får den talare som överlappar mest.
    """
    labeled = []
    for s0, s1, text in segments:
        spk = _best_speaker(s0, s1, turns) if turns else None
        labeled.append((s0, s1, spk, text))
    return labeled


def group_turns(labeled):
    """Slår ihop intilliggande segment med samma talare till turer, så att
    utskriften blir 'en rad per replik' i stället för 'en rad per segment'."""
    groups = []
    for s0, s1, spk, text in labeled:
        text = (text or "").strip()
        if not text:
            continue
        if groups and groups[-1][2] == spk:
            g = groups[-1]
            groups[-1] = (g[0], s1, spk, (g[3] + " " + text).strip())
        else:
            groups.append((s0, s1, spk, text))
    return groups


def _auto_labels(groups):
    """Generiska etiketter 'Talare 1, Talare 2 …' i den ordning talarna dyker
    upp. Numreras stabilt oavsett om vissa senare får riktiga namn."""
    order = {}
    for _, _, spk, _ in groups:
        if spk is not None and spk not in order:
            order[spk] = "Talare " + str(len(order) + 1)
    return order


def _mmss(t):
    t = max(0, int(round(t)))
    return "{:02d}:{:02d}".format(t // 60, t % 60)


def render_transcript(groups, names=None):
    """Renderar 'Namn (mm:ss): text', en tur per rad.

    ``names`` är en valfri {speaker: riktigt namn}-karta från röstprover; talare
    utan trygg matchning behåller sin generiska etikett.
    """
    names = names or {}
    auto = _auto_labels(groups)
    lines = []
    for s0, _s1, spk, text in groups:
        label = names.get(spk) or auto.get(spk) or "Talare ?"
        lines.append("{} ({}): {}".format(label, _mmss(s0), text))
    return "\n\n".join(lines)
