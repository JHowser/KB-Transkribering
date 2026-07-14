#!/usr/bin/env python3
"""
KB Transkribering — helt lokalt. Ingen extern AI-tjänst, ingen API-nyckel.

Transkribering körs på GPU:n (Metal) via whisper.cpp. Inget ljud lämnar datorn.
Appen samlar tre lägen under varsin flik:

  1. Felsökning — spela in HELA skärmen + din röst medan du går igenom en webbplats
     och pekar ut problem. Rösten transkriberas lokalt (med tidsstämplar), skärmbilder
     klipps ut ur videon vid varje tidsstämpel och allt skrivs som en "handoff" i din
     projektmapp (context.md + frames/ + recording.*), redo för Claude att läsa och åtgärda.
  2. Transkribera — dra in en mötesinspelning och få tillbaka ren text.
  3. Talaridentifiering — som Transkribera men texten märks per talare ("Talare 1",
     "Talare 2" …) med diarisering (pyannote). Valfria röstprover ger riktiga namn (ECAPA).
     Flera mötesinspelningar kan laddas upp samtidigt — de läggs i kö och körs en i
     taget (praktiskt för långa nattkörningar).
"""

import os
import io
import sys
import json
import time
import shutil
import threading
import tempfile
import subprocess
import webbrowser
import urllib.request

from flask import Flask, request, jsonify, Response

import diarize  # lätt modul (endast os på toppnivå); tunga importer sker lazy inuti

# q5_0 = liten nedladdning (~1,1 GB), full noggrannhet, snabb på Metal.
MODEL_URL = "https://huggingface.co/KBLab/kb-whisper-large/resolve/main/ggml-model-q5_0.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MODEL_NAME = "kb-whisper-large-q5_0.bin"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CONFIG_PATH = os.path.join(HERE, ".felsokning_config.json")
SESSIONS_DIR = os.path.join(HERE, "sessions")
PORT = 8723
SR = 16000

# ── Plattform: appen körs på både macOS och Windows (och Linux i mån av mån) ──
# Skärminspelning och röstdiktering sker i webbläsaren och är plattformsoberoende;
# det är bara systemintegrationerna nedan (mappväljare, öppna mapp, öppna Terminal)
# som behöver skilja på operativsystem.
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = not IS_WINDOWS and not IS_MAC


def _pick_folder_dialog():
    """Öppna en native mappväljare för aktuellt OS. Returnerar vald sökväg eller ''.
    macOS -> osascript, Windows -> PowerShell, Linux -> zenity/kdialog. Faller
    tillbaka på en Tk-dialog (följer med Python) om det native verktyget saknas."""
    try:
        if IS_MAC:
            script = 'POSIX path of (choose folder with prompt "Välj din projektmapp")'
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
            path = (out.stdout or "").strip().rstrip("/")
            if path:
                return path
        elif IS_WINDOWS:
            # PowerShells mappväljare kräver inga extra beroenden.
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
                "$f.Description = 'Välj din projektmapp';"
                "$f.ShowNewFolderButton = $true;"
                "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
                " { [Console]::Out.Write($f.SelectedPath) }"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", ps],
                capture_output=True, text=True, timeout=300)
            path = (out.stdout or "").strip()
            if path:
                return path
        else:
            for cmd in (["zenity", "--file-selection", "--directory",
                         "--title=Välj din projektmapp"],
                        ["kdialog", "--getexistingdirectory",
                         os.path.expanduser("~")]):
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    path = (out.stdout or "").strip().rstrip("/")
                    if path:
                        return path
                except FileNotFoundError:
                    continue
    except Exception:  # noqa: BLE001 — faller vidare till Tk-reserven nedan.
        pass

    # Universell reserv: Tk följer med Python. Körs i en egen process så den inte
    # krockar med Flasks trådar (Tk måste ligga på huvudtråden på macOS).
    try:
        code = (
            "import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "r = tk.Tk(); r.withdraw()\n"
            "try:\n"
            "    r.attributes('-topmost', True)\n"
            "except Exception:\n"
            "    pass\n"
            "p = filedialog.askdirectory(title='Välj din projektmapp')\n"
            "import sys; sys.stdout.write(p or '')\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=300)
        return (out.stdout or "").strip().rstrip("/")
    except Exception:  # noqa: BLE001
        return ""


def _open_in_file_manager(path):
    """Öppna en mapp i systemets filhanterare (Finder / Utforskaren / filhanterare)."""
    if IS_MAC:
        subprocess.run(["open", path])
    elif IS_WINDOWS:
        os.startfile(path)  # noqa: SLF001 — Windows-specifik, finns bara där.
    else:
        subprocess.run(["xdg-open", path])


def _claude_command(project_dir, instruction):
    """Bygg ett färdigt CLI-kommando som startar Claude Code i projektmappen.
    På Windows behövs 'cd /d' för att kunna byta enhet (t.ex. C: -> D:)."""
    cd = 'cd /d "{}"'.format(project_dir) if IS_WINDOWS else 'cd "{}"'.format(project_dir)
    return '{} && claude "{}"'.format(cd, instruction)


def _open_terminal_with_command(command, cwd=None):
    """Öppna ett terminalfönster och kör ett kommando. Best-effort per OS."""
    if IS_MAC:
        esc = command.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(["osascript",
                        "-e", 'tell application "Terminal" to do script "{}"'.format(esc),
                        "-e", 'tell application "Terminal" to activate'])
        return True
    if IS_WINDOWS:
        # Skriv kommandot till en tillfällig .bat-fil så vi slipper trassla med
        # citattecken inuti 'start ... cmd /k'. Fönstret stannar öppet (cmd /k).
        fd, bat = tempfile.mkstemp(suffix=".bat", prefix="kb_claude_")
        os.close(fd)
        with open(bat, "w", encoding="utf-8") as fh:
            fh.write("@echo off\r\n" + command + "\r\n")
        subprocess.Popen(
            'start "KB Transkribering" cmd /k "{}"'.format(bat), shell=True)
        return True
    # Linux: prova vanliga terminaler i tur och ordning.
    for term in (
        ["gnome-terminal", "--", "bash", "-lc", command + "; exec bash"],
        ["konsole", "-e", "bash", "-lc", command + "; exec bash"],
        ["x-terminal-emulator", "-e", "bash", "-lc", command + "; exec bash"],
        ["xterm", "-e", "bash", "-lc", command + "; exec bash"],
    ):
        try:
            subprocess.Popen(term, cwd=cwd or None)
            return True
        except FileNotFoundError:
            continue
    return False

# Skärmbilder (felsökningsläget): minsta avstånd mellan utklipp (så varje replik får
# en egen bild men nära dubbletter undviks), maxantal, och nedskalning.
MIN_FRAME_GAP = 0.8
MAX_FRAMES = 80
FRAME_MAX_WIDTH = 1000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB (skärmvideo)

JOB = {
    "state": "idle",   # idle | downloading | loading_model | working | analyzing | diarizing | done | error
    "mode": "",        # transcribe | debug | diarize
    "stage": "",
    "progress": 0.0,
    "position": 0.0,
    "duration": 0.0,
    "text": "",           # ren/talarmärkt transkribering (transkribering + talaridentifiering)
    "transcript": "",     # läsbart transkript (felsökningsläge)
    "handoff_dir": "",
    "instruction": "",    # meningen att ge Claude
    "command": "",        # färdigt CLI-kommando (om projektmapp satt)
    "frames_count": 0,
    "project_dir": "",
    "filename": "",
    "error": "",
    "note": "",           # kort meddelande, t.ex. när talaridentifiering inte är tillgänglig
    # Batch-kö (talaridentifiering): flera möten kan laddas upp samtidigt och
    # bearbetas då ett i taget.
    "queue": [],          # filnamn som väntar i kön
    "batch_index": 0,     # 1-baserat index för filen som bearbetas just nu
    "batch_total": 0,     # totalt antal filer i batchen
    "results": [],        # färdiga resultat i körordning: {filename, text, note, error}
}
LOCK = threading.Lock()
_model = None


def _set(**kw):
    with LOCK:
        JOB.update(kw)


def _stage(text, **kw):
    with LOCK:
        JOB["stage"] = text
        JOB.update(kw)


# ── Konfiguration (projektmapp för felsökningsläget) sparas lokalt ──

def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return {"project_dir": data.get("project_dir", "")}
    except Exception:
        return {"project_dir": ""}


def _save_config(project_dir=None):
    cfg = _load_config()
    if project_dir is not None:
        cfg["project_dir"] = project_dir.strip()
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass
    return cfg


# ── Modell: hämta vid behov ──

def _ssl_contexts():
    """Försök i tur och ordning: certifi -> systemets -> overifierad (sista utväg).
    Anslutningen är krypterad i samtliga fall; sista läget hoppar bara över
    kedjekontrollen, vilket löser felkonfigurerade rotcertifikat på macOS."""
    import ssl
    ctxs = []
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    ctxs.append(ssl.create_default_context())
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    ctxs.append(unverified)
    return ctxs


def _download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = MODEL_PATH + ".part"
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "kb-transkribering"})
    last_err = None
    for ctx in _ssl_contexts():
        try:
            with urllib.request.urlopen(req, context=ctx) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    out.write(chunk)
                    read += len(chunk)
                    if total:
                        with LOCK:
                            JOB["progress"] = min(1.0, read / total)
            os.replace(tmp, MODEL_PATH)
            return MODEL_PATH
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise last_err


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            _download_model()
        from pywhispercpp.model import Model
        # whisper.cpp byggs med Metal som standard på Apple Silicon (GPU); på Windows
        # och Linux körs samma modell på CPU (eller CUDA om det byggts med stöd).
        _model = Model(MODEL_PATH, print_realtime=False, print_progress=False)
    return _model


def _decode_audio(path):
    """Avkoda valfri ljud-/videofil till 16 kHz mono float32 med PyAV (ingen ffmpeg krävs)."""
    import av
    import numpy as np

    container = av.open(path)
    try:
        if not container.streams.audio:
            raise RuntimeError("Filen innehåller inget ljudspår.")
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []

        def take(frames):
            if frames is None:
                return
            if not isinstance(frames, list):
                frames = [frames]
            for fr in frames:
                if fr is not None:
                    chunks.append(fr.to_ndarray().reshape(-1))

        for frame in container.decode(audio=0):
            take(resampler.resample(frame))
        try:
            take(resampler.resample(None))  # töm bufferten
        except Exception:
            pass
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def _start(s):
    return (getattr(s, "t0", 0) or 0) / 100.0  # whisper.cpp anger tid i centisekunder


def _end(s):
    return (getattr(s, "t1", 0) or 0) / 100.0


def _text(s):
    return (getattr(s, "text", "") or "").strip()


def _mmss(sec):
    sec = max(0, int(round(sec or 0)))
    return "{}:{:02d}".format(sec // 60, sec % 60)


def _paragraphs(segments):
    parts, prev = [], None
    for s in segments:
        t = _text(s)
        if not t:
            continue
        if prev is not None and (_start(s) - prev) > 2.0:
            parts.append("\n\n")
        elif parts:
            parts.append(" ")
        parts.append(t)
        prev = _end(s)
    return "".join(parts).strip()


def _segments(raw):
    """Strukturera whisper.cpp-segmenten till [(start, end, text)] för mergning."""
    out = []
    for s in raw:
        t = _text(s)
        if t:
            out.append((_start(s), _end(s), t))
    return out


# ── Talaridentifiering: diarisering + valfria röstprover ──

def _diarize_and_label(audio, segments, samples):
    """Diariserar ljudet, slår ihop med ASR-segmenten och (om röstprover finns)
    märker klustren med riktiga namn. Kastar vidare om diarisering inte går."""
    import merge

    turns = diarize.diarize(audio, SR)
    if not turns:
        raise RuntimeError("Diariseringen hittade inga talarturer.")

    labeled = merge.assign_speakers(segments, turns)
    groups = merge.group_turns(labeled)

    names = {}
    if samples:
        try:
            import enroll
            enrollment = enroll.build_enrollment(samples)
            centroids = enroll.cluster_centroids(audio, turns, SR)
            names = enroll.match_clusters(centroids, enrollment)
        except Exception:  # noqa: BLE001 — röstprover är en bonus; fall tillbaka
            names = {}                                     # till generiska etiketter

    return merge.render_transcript(groups, names)


# ── Bildextraktion ur skärmvideon (PyAV + Pillow) för felsökningsläget ──

def _frame_png(frame, max_w=FRAME_MAX_WIDTH):
    img = frame.to_image()  # kräver Pillow
    if img.width > max_w:
        r = max_w / float(img.width)
        img = img.resize((max_w, max(1, int(img.height * r))))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _extract_frames_multi(video_path, targets, max_w=FRAME_MAX_WIDTH):
    """Klipp ut en bild per tidsstämpel (sekunder) i ett enda genomlopp.
    Returnerar {index_i_targets: png_bytes}."""
    import av
    out = {}
    if not targets:
        return out
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    try:
        container = av.open(video_path)
    except Exception:
        return out
    try:
        if not container.streams.video:
            return out
        vstream = container.streams.video[0]
        try:
            vstream.thread_type = "AUTO"
        except Exception:
            pass
        tb = vstream.time_base
        oi = 0
        prev_frame = None
        prev_t = None
        for frame in container.decode(video=0):
            if frame.pts is not None and tb:
                ft = float(frame.pts * tb)
            else:
                ft = (prev_t + 0.1) if prev_t is not None else 0.0
            while oi < len(order) and ft >= targets[order[oi]]:
                ti = order[oi]
                tt = targets[ti]
                chosen = frame
                if prev_frame is not None and prev_t is not None and abs(prev_t - tt) < abs(ft - tt):
                    chosen = prev_frame
                try:
                    out[ti] = _frame_png(chosen, max_w)
                except Exception:
                    pass
                oi += 1
            prev_frame = frame
            prev_t = ft
            if oi >= len(order):
                break
        while oi < len(order) and prev_frame is not None:
            try:
                out[order[oi]] = _frame_png(prev_frame, max_w)
            except Exception:
                pass
            oi += 1
    finally:
        container.close()
    return out


def _handoff_targets(segments):
    """Välj tidsstämplar att klippa bilder vid: mitten av varje replik, glesat och kapat.
    Returnerar lista av (tidsstämpel_sekunder, segment_index)."""
    items = []
    last = -999.0
    for i, s in enumerate(segments):
        if not _text(s):
            continue
        mid = (_start(s) + _end(s)) / 2.0
        if mid - last >= MIN_FRAME_GAP:
            items.append((mid, i))
            last = mid
    if len(items) > MAX_FRAMES:
        step = len(items) / float(MAX_FRAMES)
        items = [items[int(k * step)] for k in range(MAX_FRAMES)]
    return items


# ── Handoff-skrivning (allt Claude behöver, i projektmappen) ──

INSTRUCTION = ("Read .felsokning/latest/context.md, view the screenshots in "
               ".felsokning/latest/frames, cross-reference the code in this project, "
               "and fix the issues I flagged while recording.")


def _write_handoff(target_root, segments, items, frames_flat, video_path, video_ext):
    root = os.path.join(target_root, ".felsokning")
    handoff = os.path.join(root, "latest")
    if os.path.isdir(handoff):
        shutil.rmtree(handoff, ignore_errors=True)
    frames_dir = os.path.join(handoff, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Håll inspelningar utanför git.
    try:
        with open(os.path.join(root, ".gitignore"), "w", encoding="utf-8") as fh:
            fh.write("*\n")
    except Exception:
        pass

    seg_to_file = {}
    for flat_i, (t, seg_i) in enumerate(items):
        png = frames_flat.get(flat_i)
        if not png:
            continue
        name = "t{:04d}.png".format(int(round(t)))
        with open(os.path.join(frames_dir, name), "wb") as fh:
            fh.write(png)
        seg_to_file[seg_i] = name

    vid_name = "recording" + video_ext
    try:
        shutil.copyfile(video_path, os.path.join(handoff, vid_name))
    except Exception:
        vid_name = ""

    lines = []
    lines.append("# Website review session — " + time.strftime("%Y-%m-%d %H:%M"))
    lines.append("")
    lines.append("I (the developer) recorded my screen and voice while walking through my "
                 "website and pointing out problems. Below is my narration, transcribed with "
                 "timestamps (Swedish). Many timestamps have a matching screenshot in ./frames.")
    lines.append("")
    lines.append("## Your task")
    lines.append("Read the narration, find where I describe a problem, bug or UX issue, open the "
                 "matching screenshot(s) in ./frames, cross-reference the code in this project, "
                 "and propose and implement fixes. Use a strong model (e.g. Opus). Ask me if "
                 "anything is ambiguous. When useful, group related remarks into one fix.")
    lines.append("")
    lines.append("## Narration (transcript with timestamps)")
    for i, s in enumerate(segments):
        t = _text(s)
        if not t:
            continue
        ref = " (frames/{})".format(seg_to_file[i]) if i in seg_to_file else ""
        lines.append("[{}] {}{}".format(_mmss(_start(s)), t, ref))
    lines.append("")
    lines.append("## Files")
    lines.append("- frames/ — screenshots captured from the recording, keyed by the timestamps above")
    if vid_name:
        lines.append("- {} — the full screen recording (extract more frames if you need them)".format(vid_name))
    with open(os.path.join(handoff, "context.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return handoff, len(seg_to_file)


def _save_upload(f):
    suffix = os.path.splitext(f.filename)[1] or ".m4a"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    f.save(path)
    return path


def _copy_temp(path):
    """Kopiera en tillfällig fil (röstprover delas av flera köade jobb; varje
    jobb får en egen kopia så att den kan raderas när jobbet är klart)."""
    fd, dst = tempfile.mkstemp(suffix=os.path.splitext(path)[1])
    os.close(fd)
    shutil.copyfile(path, dst)
    return dst


# ── Bakgrundsjobb ──

def _transcribe_only(path, filename):
    """Transkribera-fliken: ren text utan talaretiketter."""
    try:
        _set(state="downloading" if not os.path.exists(MODEL_PATH) else "loading_model",
             mode="transcribe", stage="", progress=0.0, position=0.0, duration=0.0,
             text="", transcript="", error="", note="", filename=filename)
        model = _load_model()
        audio = _decode_audio(path)
        duration = float(len(audio)) / SR if len(audio) else 0.0
        if duration <= 0:
            raise RuntimeError("Kunde inte läsa ljudet ur filen.")
        _set(state="working", duration=duration)

        def on_segment(seg):
            with LOCK:
                JOB["position"] = _end(seg)
                if duration > 0:
                    JOB["progress"] = max(0.0, min(1.0, _end(seg) / duration))

        segments = model.transcribe(audio, language="sv", new_segment_callback=on_segment)
        _set(state="done", progress=1.0, text=_paragraphs(segments))
    except Exception as exc:  # noqa: BLE001
        _set(state="error", error=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# Talaridentifiering körs som en kö: flera mötesinspelningar kan laddas upp
# samtidigt (eller läggas till medan en körning pågår) och bearbetas ett i taget
# av en enda arbetstråd. Resultaten samlas i körordning.
DIARIZE_QUEUE = []    # väntande jobb: {"path", "filename", "sample_paths"}
DIARIZE_RESULTS = []  # färdiga resultat: {"filename", "text", "note", "error"}
_diarize_worker_on = False


def _diarize_one(path, filename, sample_paths):
    """Bearbeta EN mötesinspelning (transkribering + diarisering, med valfria
    röstprover). Returnerar (text, note); fel kastas vidare till kö-arbetaren."""
    try:
        _set(state="downloading" if not os.path.exists(MODEL_PATH) else "loading_model",
             mode="diarize", stage="", progress=0.0, position=0.0, duration=0.0,
             text="", transcript="", error="", note="", filename=filename)
        model = _load_model()

        audio = _decode_audio(path)
        duration = float(len(audio)) / SR if len(audio) else 0.0
        if duration <= 0:
            raise RuntimeError("Kunde inte läsa ljudet ur filen.")
        _set(state="working", duration=duration)

        def on_segment(seg):
            with LOCK:
                JOB["position"] = _end(seg)
                if duration > 0:
                    JOB["progress"] = max(0.0, min(1.0, _end(seg) / duration))

        raw = model.transcribe(audio, language="sv", new_segment_callback=on_segment)
        segments = _segments(raw)

        # Avkoda ev. röstprover först nu (i arbetstråden), så uppladdningen går fort.
        samples = []
        for name, spath in sample_paths:
            try:
                wav = _decode_audio(spath)
                if len(wav):
                    samples.append((name, wav))
            except Exception:  # noqa: BLE001 — hoppa över trasiga prover
                pass

        # Talaridentifiering får aldrig fälla hela transkriberingen.
        text = _paragraphs(raw)
        note = ""
        try:
            _set(state="diarizing", progress=1.0, position=duration)
            text = _diarize_and_label(audio, segments, samples)
        except diarize.DiarizationUnavailable:
            note = "Talaridentifiering ej tillgänglig — visar text utan talare."
        except ImportError:
            note = "Talaridentifiering ej installerad — visar text utan talare."
        except Exception:  # noqa: BLE001
            note = "Talaridentifiering misslyckades — visar text utan talare."

        return text, note
    finally:
        for p in [path] + [sp for _n, sp in sample_paths]:
            try:
                os.remove(p)
            except OSError:
                pass


def _diarize_worker():
    """Kö-arbetare: kör jobben i DIARIZE_QUEUE ett i taget tills kön är tom.
    Ett fel i en fil stoppar inte resten av kön (viktigt vid nattkörningar)."""
    global _diarize_worker_on
    while True:
        with LOCK:
            job = DIARIZE_QUEUE.pop(0) if DIARIZE_QUEUE else None
            if job is None:
                _diarize_worker_on = False
                if len(DIARIZE_RESULTS) == 1 and DIARIZE_RESULTS[0]["error"]:
                    # En ensam fil som gick fel beter sig som tidigare (felvy).
                    JOB.update(state="error", error=DIARIZE_RESULTS[0]["error"])
                else:
                    JOB.update(state="done", progress=1.0)
                return
            JOB["batch_index"] += 1
            JOB["queue"] = [j["filename"] for j in DIARIZE_QUEUE]

        try:
            text, note = _diarize_one(job["path"], job["filename"], job["sample_paths"])
            result = {"filename": job["filename"], "text": text, "note": note, "error": ""}
        except Exception as exc:  # noqa: BLE001
            result = {"filename": job["filename"], "text": "", "note": "", "error": str(exc)}

        with LOCK:
            DIARIZE_RESULTS.append(result)
            JOB["results"] = list(DIARIZE_RESULTS)
            JOB["text"] = result["text"]
            JOB["note"] = result["note"]


def _prepare_handoff(video_path, session_dir, video_ext):
    """Felsökning-fliken: transkribera -> klipp ut bilder -> skriv handoff i projektmappen."""
    try:
        cfg = _load_config()
        project = cfg.get("project_dir") or ""
        use_project = bool(project) and os.path.isdir(project)
        target_root = project if use_project else session_dir

        _set(state="downloading" if not os.path.exists(MODEL_PATH) else "loading_model",
             mode="debug", stage="", progress=0.0, position=0.0, duration=0.0,
             text="", transcript="", handoff_dir="", instruction="", command="",
             frames_count=0, project_dir=(project if use_project else ""),
             error="", note="", filename="Felsökningssession")

        whisper = _load_model()

        audio = _decode_audio(video_path)
        duration = float(len(audio)) / SR if len(audio) else 0.0
        if duration <= 0:
            raise RuntimeError("Kunde inte läsa ljudet ur inspelningen.")
        _set(state="working", duration=duration, stage="Transkriberar din röst…")

        def on_segment(seg):
            with LOCK:
                JOB["position"] = _end(seg)
                if duration > 0:
                    JOB["progress"] = max(0.0, min(1.0, _end(seg) / duration))

        segments = whisper.transcribe(audio, language="sv", new_segment_callback=on_segment)
        readable = _paragraphs(segments)

        _set(state="analyzing", transcript=readable, progress=1.0)
        _stage("Klipper ut skärmbilder ur inspelningen…")
        items = _handoff_targets(segments)
        times = [t for t, _ in items]
        frames_flat = _extract_frames_multi(video_path, times, FRAME_MAX_WIDTH)

        _stage("Skriver handoff i projektmappen…")
        handoff, nframes = _write_handoff(target_root, segments, items, frames_flat, video_path, video_ext)

        if use_project:
            command = _claude_command(project, INSTRUCTION)
            instruction = INSTRUCTION
        else:
            instruction = ('Read the context.md in "{}", view the screenshots in its frames/ '
                           'folder, cross-reference the code, and fix the issues I flagged.'.format(handoff))
            command = ""

        _set(state="done", handoff_dir=handoff, instruction=instruction, command=command,
             frames_count=nframes, transcript=readable, stage="")
    except Exception as exc:  # noqa: BLE001
        _set(state="error", error=str(exc))
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


# ── Routes ──

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(ok=True)


@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        cfg = _save_config(project_dir=data.get("project_dir"))
        pd = cfg.get("project_dir", "")
        return jsonify(ok=True, project_dir=pd, valid=bool(pd and os.path.isdir(pd)))
    cfg = _load_config()
    pd = cfg.get("project_dir", "")
    return jsonify(project_dir=pd, valid=bool(pd and os.path.isdir(pd)))


@app.route("/pick-folder", methods=["POST"])
def pick_folder():
    """Öppna en native mappväljare (macOS/Windows/Linux) och spara valet."""
    try:
        path = _pick_folder_dialog()
        if path:
            _save_config(project_dir=path)
            return jsonify(ok=True, project_dir=path, valid=os.path.isdir(path))
        return jsonify(ok=False, error="Ingen mapp vald.")
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=False, error=str(exc))


@app.route("/open-folder", methods=["POST"])
def open_folder():
    data = request.get_json(silent=True) or {}
    path = data.get("path") or ""
    if path and os.path.isdir(path):
        try:
            _open_in_file_manager(path)
            return jsonify(ok=True)
        except Exception as exc:  # noqa: BLE001
            return jsonify(ok=False, error=str(exc))
    return jsonify(ok=False, error="Mappen finns inte."), 400


@app.route("/open-claude", methods=["POST"])
def open_claude():
    """Bäst-effort: öppna en terminal i projektmappen och kör Claude Code."""
    data = request.get_json(silent=True) or {}
    command = data.get("command") or ""
    if not command:
        return jsonify(ok=False, error="Inget kommando."), 400
    try:
        if _open_terminal_with_command(command):
            return jsonify(ok=True)
        return jsonify(ok=False, error="Hittade ingen terminal att öppna.")
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=False, error=str(exc))


def _busy():
    return JOB["state"] in ("downloading", "loading_model", "working", "analyzing", "diarizing")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    with LOCK:
        if _busy():
            return jsonify(ok=False, error="En körning pågår redan."), 409
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Ingen fil mottagen."), 400
    path = _save_upload(f)
    threading.Thread(target=_transcribe_only, args=(path, f.filename), daemon=True).start()
    return jsonify(ok=True)


@app.route("/diarize", methods=["POST"])
def diarize_route():
    """Talaridentifiering. Tar emot en eller flera mötesinspelningar (file[]);
    flera filer läggs i kö och körs en i taget. Fler filer kan även läggas till
    medan en kö-körning pågår — de hamnar då sist i kön."""
    global _diarize_worker_on
    files = [f for f in request.files.getlist("file") if f and f.filename]
    if not files:
        return jsonify(ok=False, error="Ingen fil mottagen."), 400
    with LOCK:
        if _busy() and not (JOB["mode"] == "diarize" and _diarize_worker_on):
            return jsonify(ok=False, error="En körning pågår redan."), 409

    # Valfria röstprover: parallella listor sample_name[] / sample_file[].
    # Proverna gäller alla filer i denna uppladdning.
    sample_paths = []
    sample_files = request.files.getlist("sample_file")
    sample_names = request.form.getlist("sample_name")
    for i, sf in enumerate(sample_files):
        if not sf or not sf.filename:
            continue
        name = (sample_names[i] if i < len(sample_names) else "").strip()
        if not name:
            continue
        sample_paths.append((name, _save_upload(sf)))

    jobs = []
    for i, f in enumerate(files):
        # Varje jobb äger (och raderar) sina egna proverfiler — kopiera för alla
        # utom det första jobbet.
        sp = sample_paths if i == 0 else [(n, _copy_temp(p)) for n, p in sample_paths]
        jobs.append({"path": _save_upload(f), "filename": f.filename, "sample_paths": sp})

    start = False
    with LOCK:
        if not _diarize_worker_on:
            if _busy():  # en annan flik hann starta medan filerna sparades
                for j in jobs:
                    for p in [j["path"]] + [sp for _n, sp in j["sample_paths"]]:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                return jsonify(ok=False, error="En körning pågår redan."), 409
            DIARIZE_QUEUE.clear()
            DIARIZE_RESULTS.clear()
            _diarize_worker_on = True
            start = True
            JOB.update(state="loading_model", mode="diarize", stage="", progress=0.0,
                       position=0.0, duration=0.0, text="", transcript="", error="",
                       note="", filename=jobs[0]["filename"],
                       queue=[], batch_index=0, batch_total=0, results=[])
        DIARIZE_QUEUE.extend(jobs)
        JOB["batch_total"] += len(jobs)
        JOB["queue"] = [j["filename"] for j in DIARIZE_QUEUE]
    if start:
        threading.Thread(target=_diarize_worker, daemon=True).start()
    return jsonify(ok=True, queued=len(jobs))


@app.route("/analyze", methods=["POST"])
def analyze():
    with LOCK:
        if _busy():
            return jsonify(ok=False, error="En körning pågår redan."), 409
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Ingen inspelning mottagen."), 400

    ext = os.path.splitext(f.filename)[1] or ".webm"
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    session_dir = os.path.join(SESSIONS_DIR, time.strftime("session-%Y%m%d-%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    f.save(path)

    threading.Thread(target=_prepare_handoff, args=(path, session_dir, ext), daemon=True).start()
    return jsonify(ok=True)


@app.route("/status")
def status():
    with LOCK:
        return jsonify(dict(JOB))


@app.route("/reset", methods=["POST"])
def reset():
    with LOCK:
        if _busy():
            return jsonify(ok=False), 409
        DIARIZE_QUEUE.clear()
        DIARIZE_RESULTS.clear()
        JOB.update(state="idle", mode="", stage="", progress=0.0, position=0.0, duration=0.0,
                   text="", transcript="", handoff_dir="", instruction="", command="",
                   frames_count=0, project_dir="", filename="", error="", note="",
                   queue=[], batch_index=0, batch_total=0, results=[])
    return jsonify(ok=True)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return jsonify(ok=True)


PAGE = r"""<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KB Transkribering</title>
<style>
  :root{
    --paper:#EAEEEF; --card:#FBFCFC; --ink:#16252B; --muted:#5E6E74;
    --line:#D3DADC; --accent:#1F5C73; --accent-soft:#D7E5EA; --danger:#9B3B2E;
    --ok:#2E7D5B; --rec:#C4453A;
    --display:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
    --ui:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;min-height:100%}
  body{background:var(--paper); color:var(--ink); font-family:var(--ui);
    display:flex; align-items:flex-start; justify-content:center; padding:32px; -webkit-font-smoothing:antialiased;}
  .card{width:100%; max-width:640px; background:var(--card); border:1px solid var(--line);
    border-radius:14px; padding:30px 34px 26px; box-shadow:0 1px 0 #fff inset, 0 18px 50px -28px rgba(22,37,43,.35);}
  .brand{display:flex; align-items:baseline; gap:10px; margin-bottom:16px}
  .brand h1{font-family:var(--display); font-weight:600; font-size:24px; letter-spacing:.2px; margin:0}
  .brand .tag{font-size:12px; color:var(--muted); letter-spacing:.04em; text-transform:uppercase}
  .tabs{display:flex; gap:6px; background:#EEF2F3; border:1px solid var(--line); border-radius:11px; padding:4px; margin-bottom:22px}
  .tabs button{flex:1; border:none; background:transparent; color:var(--muted); font-family:var(--ui);
    font-size:13.5px; font-weight:600; padding:9px 8px; border-radius:8px; cursor:pointer; transition:all .12s ease; white-space:nowrap}
  .tabs button.active{background:#fff; color:var(--ink); box-shadow:0 1px 3px rgba(22,37,43,.12)}
  .sub{color:var(--muted); font-size:13.5px; margin:0 0 22px; line-height:1.55}
  .drop{border:1.5px dashed var(--line); border-radius:12px; background:#fff; padding:46px 24px;
    text-align:center; cursor:pointer; transition:border-color .15s ease, background .15s ease;}
  .drop:hover{border-color:var(--accent); background:#FAFCFD}
  .drop.over{border-color:var(--accent); background:var(--accent-soft)}
  .drop .icon{width:42px; height:42px; margin:0 auto 14px; color:var(--accent)}
  .drop .big{font-size:15.5px; font-weight:550}
  .drop .small{font-size:12.5px; color:var(--muted); margin-top:6px}
  input[type=file]{display:none}
  .name{font-size:14px; font-weight:550; margin-bottom:16px; word-break:break-all}
  .track{height:8px; background:var(--accent-soft); border-radius:99px; overflow:hidden}
  .bar{height:100%; width:0%; background:var(--accent); border-radius:99px; transition:width .4s ease}
  .meta{display:flex; justify-content:space-between; margin-top:12px; font-size:13px; color:var(--muted)}
  .meta .state{color:var(--ink)}
  .spinner{display:inline-block; width:13px; height:13px; margin-right:7px; vertical-align:-2px;
    border:2px solid var(--accent-soft); border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  textarea{width:100%; height:220px; resize:vertical; border:1px solid var(--line); border-radius:10px;
    padding:14px 16px; font-family:var(--ui); font-size:14px; line-height:1.65; color:var(--ink); background:#fff;}
  textarea:focus{outline:none; border-color:var(--accent)}
  .actions{display:flex; gap:10px; margin-top:16px; flex-wrap:wrap}
  button.btn{font-family:var(--ui); font-size:14px; font-weight:550; cursor:pointer; border-radius:9px;
    padding:11px 16px; border:1px solid var(--line); background:#fff; color:var(--ink); transition:background .12s ease, border-color .12s ease;}
  button.btn:hover{background:#F2F5F6}
  button.primary{background:var(--accent); border-color:var(--accent); color:#fff}
  button.primary:hover{background:#1A4F62}
  button.rec{background:var(--rec); border-color:var(--rec); color:#fff}
  button.rec:hover{background:#A93a30}
  button:disabled{opacity:.5; cursor:not-allowed}
  .grow{flex:1}
  .err{color:var(--danger); font-size:13.5px; line-height:1.5; margin-top:4px; white-space:pre-wrap}
  .hidden{display:none}
  .foot{margin-top:22px; font-size:11.5px; color:var(--muted); text-align:center; letter-spacing:.02em}
  .foot a{color:var(--muted); cursor:pointer; text-decoration:underline}
  .settings{background:#F4F7F8; border:1px solid var(--line); border-radius:11px; padding:14px 16px; margin-bottom:20px}
  .settings label{font-size:12.5px; color:var(--muted); font-weight:600; display:block; margin-bottom:7px}
  .settings .row{display:flex; gap:10px; align-items:center; flex-wrap:wrap}
  .settings input{flex:1; min-width:200px; font-family:var(--ui); font-size:13px; border:1px solid var(--line);
    border-radius:8px; padding:9px 11px; background:#fff; color:var(--ink)}
  .projstate{font-size:12px; margin-top:9px; word-break:break-all}
  .projstate.ok{color:var(--ok)} .projstate.no{color:var(--muted)}
  .rec-dot{display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--rec); margin-right:7px;
    vertical-align:1px; animation:pulse 1.1s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .timer{font-family:var(--display); font-size:38px; font-weight:600; text-align:center; margin:10px 0 2px; letter-spacing:1px}
  .reclabel{text-align:center; font-size:13px; color:var(--muted); margin-bottom:16px}
  .hint{font-size:12.5px; color:var(--muted); line-height:1.55; margin:10px 0 0}
  .savedmsg{font-size:12px; color:var(--muted); margin-top:10px; word-break:break-all}
  .savedmsg.err-hint{color:var(--danger); font-weight:500; font-size:13px}
  .badge{display:inline-block; background:var(--accent-soft); color:var(--accent); font-size:11px; font-weight:600;
    border-radius:6px; padding:2px 7px; margin-left:6px}
  .instr{border:1px solid var(--line); border-radius:10px; background:#fff; padding:13px 15px; margin-top:4px;
    font-size:14px; line-height:1.5}
  .path{font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--muted); margin-top:8px; word-break:break-all}
  /* Röstprover (talaridentifiering) */
  .who{margin-top:16px}
  .who-head{font-size:12.5px; color:var(--muted); line-height:1.5}
  .enroll{margin-top:8px}
  .enroll summary{font-size:13px; font-weight:550; color:var(--accent); cursor:pointer; list-style:none;
    display:inline-block; padding:2px 0}
  .enroll summary::-webkit-details-marker{display:none}
  .enroll summary::before{content:"＋ "; font-weight:600}
  .enroll[open] summary::before{content:"－ "}
  .who-hint{font-size:12px; color:var(--muted); line-height:1.5; margin:8px 0 12px}
  .srow{display:flex; gap:8px; align-items:center; margin-bottom:8px}
  .srow input[type=text]{flex:1; min-width:0; border:1px solid var(--line); border-radius:8px;
    padding:8px 10px; font-family:var(--ui); font-size:13px; color:var(--ink); background:#fff}
  .srow input[type=text]:focus{outline:none; border-color:var(--accent)}
  .srow .pick{white-space:nowrap; max-width:150px; overflow:hidden; text-overflow:ellipsis; padding:8px 10px; font-size:12.5px}
  .srow .pick.has{border-color:var(--accent); color:var(--accent)}
  .srow .rm{padding:8px 11px; font-size:14px; line-height:1; color:var(--muted)}
  button.ghost{background:transparent; border:1px dashed var(--line); color:var(--accent); font-size:12.5px; padding:8px 12px}
  button.ghost:hover{background:var(--accent-soft); border-color:var(--accent)}
  .notice{background:var(--accent-soft); border:1px solid var(--line); border-radius:9px;
    padding:9px 12px; font-size:12.5px; color:var(--ink); margin-bottom:10px; line-height:1.5}
  /* Batch-kö (talaridentifiering) */
  .queueinfo{font-size:12.5px; color:var(--muted); margin-top:12px; line-height:1.55; word-break:break-word}
  .rlist{display:flex; flex-direction:column; gap:6px; margin-bottom:12px}
  .ritem{display:flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:9px;
    background:#fff; padding:8px 12px; font-size:13px}
  .ritem.sel{border-color:var(--accent); background:var(--accent-soft)}
  .ritem .fn{flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:pointer; font-weight:550}
  .ritem .fail{color:var(--danger); font-size:12px; white-space:nowrap}
  .ritem .mini{padding:5px 10px; font-size:12px; white-space:nowrap}
</style>
</head>
<body>
  <main class="card">
    <div class="brand"><h1>Transkribering</h1><span class="tag">KB-Whisper · lokalt</span></div>

    <div class="tabs">
      <button id="tabD" class="active" onclick="setTab('d')">Felsökning</button>
      <button id="tabT" onclick="setTab('t')">Transkribera</button>
      <button id="tabS" onclick="setTab('s')">Talaridentifiering</button>
    </div>

    <!-- ══════════ FELSÖKNING ══════════ -->
    <div id="paneD">
      <p class="sub">Spela in skärmen och din röst medan du går igenom din webbplats — peka ut problemen
        högt. Rösten transkriberas lokalt och skärmbilder klipps ut automatiskt. Allt skrivs som en
        handoff i din projektmapp, redo för Claude att läsa och åtgärda.</p>

      <div class="settings">
        <label>Projektmapp (dit handoff skrivs, så Claude ser den)</label>
        <div class="row">
          <input type="text" id="projPath" placeholder="Ingen mapp vald — klicka Välj mapp" readonly>
          <button class="btn primary" id="pickFolder">Välj mapp</button>
        </div>
        <div class="projstate no" id="projState">Ingen projektmapp vald. Handoff sparas då i appens egen mapp.</div>
      </div>

      <section id="d_idle">
        <div class="hint" style="margin-top:0">
          Klicka <b>Starta inspelning</b>, tillåt mikrofon, och välj <b>fönstret eller fliken med din
          webbplats</b> att dela. Klicka runt och beskriv felen högt. Klicka <b>Stoppa &amp; skapa handoff</b> när du är klar.
        </div>
        <div class="actions" style="margin-top:16px">
          <button class="btn rec grow" id="startRec">● Starta inspelning</button>
        </div>
        <div class="savedmsg" id="permHint"></div>
      </section>

      <section id="d_rec" class="hidden">
        <div class="timer"><span class="rec-dot"></span><span id="timer">0:00</span></div>
        <div class="reclabel">Spelar in skärm + röst. Prata på och peka ut problemen.</div>
        <div class="actions">
          <button class="btn primary grow" id="stopRec">Stoppa &amp; skapa handoff</button>
          <button class="btn" id="cancelRec">Avbryt</button>
        </div>
      </section>

      <section id="d_working" class="hidden">
        <div class="name">Bearbetar felsökningssession…</div>
        <div class="track"><div class="bar" id="d_bar"></div></div>
        <div class="meta"><span class="state" id="d_state"><span class="spinner"></span>Förbereder…</span><span id="d_time"></span></div>
      </section>

      <section id="d_done" class="hidden">
        <div class="name">Handoff klar <span class="badge" id="d_frames"></span></div>
        <div class="instr" id="d_instr"></div>
        <div class="path" id="d_path"></div>
        <div class="actions">
          <button class="btn primary grow" id="d_copy">📋 Kopiera instruktion till Claude</button>
          <button class="btn" id="d_openClaude">Öppna i Claude Code</button>
          <button class="btn" id="d_openFolder">Öppna mappen</button>
          <button class="btn" id="d_again">Ny session</button>
        </div>
        <div class="savedmsg" id="d_copyMsg"></div>
      </section>

      <section id="d_error" class="hidden">
        <p class="err" id="d_errMsg"></p>
        <div class="actions"><button class="btn grow" id="d_retry">Tillbaka</button></div>
      </section>
    </div>

    <!-- ══════════ TRANSKRIBERING ══════════ -->
    <div id="paneT" class="hidden">
      <p class="sub">Släpp in en mötesinspelning så får du tillbaka texten. Allt körs lokalt på din dator.</p>
      <section id="t_idle">
        <label class="drop" id="drop" for="file">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>
          </svg>
          <div class="big">Dra in din ljudfil</div>
          <div class="small">eller klicka för att välja · m4a, mp3, wav, mp4</div>
        </label>
        <input type="file" id="file" accept="audio/*,video/mp4,.m4a,.mp3,.wav,.mp4">
      </section>
      <section id="t_working" class="hidden">
        <div class="name" id="t_name"></div>
        <div class="track"><div class="bar" id="t_bar"></div></div>
        <div class="meta"><span class="state" id="t_state"><span class="spinner"></span>Förbereder…</span><span id="t_time"></span></div>
      </section>
      <section id="t_done" class="hidden">
        <div class="name" id="t_doneName"></div>
        <textarea id="t_out" spellcheck="false" style="height:300px"></textarea>
        <div class="actions">
          <button class="btn primary grow" id="t_download">Ladda ner .txt</button>
          <button class="btn" id="t_copy">Kopiera</button>
          <button class="btn" id="t_again">Ny fil</button>
        </div>
      </section>
      <section id="t_error" class="hidden">
        <p class="err" id="t_errMsg"></p>
        <div class="actions"><button class="btn grow" id="t_retry">Försök igen</button></div>
      </section>
    </div>

    <!-- ══════════ TALARIDENTIFIERING ══════════ -->
    <div id="paneS" class="hidden">
      <p class="sub">Släpp in en mötesinspelning så får du tillbaka texten <b>märkt per talare</b>
        (Talare 1, Talare 2 …) med tidsstämplar. Lägg gärna till röstprover för riktiga namn.
        Allt körs lokalt på din dator.</p>
      <section id="s_idle">
        <label class="drop" id="s_drop" for="s_file">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>
          </svg>
          <div class="big">Dra in en eller flera ljudfiler</div>
          <div class="small">eller klicka för att välja · flera filer köas och körs en i taget · m4a, mp3, wav, mp4</div>
        </label>
        <input type="file" id="s_file" accept="audio/*,video/mp4,.m4a,.mp3,.wav,.mp4" multiple>

        <div class="who">
          <div class="who-head">Repliker märks automatiskt per talare (Talare 1, Talare 2 …).</div>
          <details class="enroll">
            <summary>Lägg till röstprover (valfritt)</summary>
            <p class="who-hint">Ladda upp en kort inspelning (10–20 sek) där bara personen pratar,
              så märks repliker med namn i stället för "Talare 1". Lägg till personerna innan du
              släpper in mötesinspelningen.</p>
            <div id="s_samples"></div>
            <button type="button" id="s_addPerson" class="btn ghost">+ Lägg till person</button>
          </details>
        </div>
      </section>
      <section id="s_working" class="hidden">
        <div class="name" id="s_name"></div>
        <div class="track"><div class="bar" id="s_bar"></div></div>
        <div class="meta"><span class="state" id="s_state"><span class="spinner"></span>Förbereder…</span><span id="s_time"></span></div>
        <div class="queueinfo hidden" id="s_queue"></div>
        <div class="actions" style="margin-top:14px">
          <button class="btn" id="s_addMore">＋ Lägg till fler filer i kön</button>
        </div>
        <input type="file" id="s_moreFile" accept="audio/*,video/mp4,.m4a,.mp3,.wav,.mp4" multiple>
      </section>
      <section id="s_done" class="hidden">
        <div class="rlist hidden" id="s_results"></div>
        <div class="name" id="s_doneName"></div>
        <div class="notice hidden" id="s_note"></div>
        <textarea id="s_out" spellcheck="false" style="height:300px"></textarea>
        <div class="actions">
          <button class="btn primary grow" id="s_download">Ladda ner .txt</button>
          <button class="btn hidden" id="s_downloadAll">Ladda ner alla</button>
          <button class="btn" id="s_copy">Kopiera</button>
          <button class="btn" id="s_again">Ny fil</button>
        </div>
      </section>
      <section id="s_error" class="hidden">
        <p class="err" id="s_errMsg"></p>
        <div class="actions"><button class="btn grow" id="s_retry">Försök igen</button></div>
      </section>
    </div>

    <div class="foot">Allt lokalt · KB-Whisper large · GPU (Metal) · <a id="quit">Avsluta appen</a></div>
  </main>

<script>
const $ = id => document.getElementById(id);
function fmt(s){ s = Math.max(0, Math.round(s||0)); const m = Math.floor(s/60), x = s%60; return m+":"+String(x).padStart(2,"0"); }

function setTab(which){
  ["d","t","s"].forEach(k => {
    $("tab"+k.toUpperCase()).classList.toggle("active", which===k);
    $("pane"+k.toUpperCase()).classList.toggle("hidden", which!==k);
  });
}

/* ── Projektmapp (felsökning) ── */
async function loadConfig(){
  try{ const j = await (await fetch("/config")).json(); setProj(j.project_dir, j.valid); }catch(e){}
}
function setProj(path, valid){
  $("projPath").value = path || "";
  const el = $("projState");
  if (path && valid){ el.textContent = "✓ Handoff skrivs till "+path+"/.felsokning/latest/"; el.className="projstate ok"; }
  else if (path && !valid){ el.textContent = "Mappen hittades inte längre. Välj en ny."; el.className="projstate no"; }
  else { el.textContent = "Ingen projektmapp vald. Handoff sparas då i appens egen mapp."; el.className="projstate no"; }
}
$("pickFolder").addEventListener("click", async () => {
  $("pickFolder").textContent = "Väljer…"; $("pickFolder").disabled = true;
  try{ const j = await (await fetch("/pick-folder",{method:"POST"})).json(); if (j.ok) setProj(j.project_dir, j.valid); }
  catch(e){}
  $("pickFolder").textContent = "Välj mapp"; $("pickFolder").disabled = false;
});

/* ── Felsökning: inspelning ── */
const dShow = id => ["d_idle","d_rec","d_working","d_done","d_error"].forEach(s => $(s).classList.toggle("hidden", s!==id));

let micStream, screenStream, mediaRecorder, chunks = [];
let startTime = 0, timerInt = null, recMime = "video/webm", recExt = "webm";

function setPermHint(msg, isError){ const el=$("permHint"); el.textContent=msg; el.classList.toggle("err-hint", !!isError && !!msg); }
function resetStartBtn(){ const b=$("startRec"); b.disabled=false; b.textContent="● Starta inspelning"; }

$("startRec").addEventListener("click", startSession);
async function startSession(){
  const btn = $("startRec");
  setPermHint("");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia ||
      !navigator.mediaDevices.getDisplayMedia || typeof MediaRecorder === "undefined"){
    setPermHint("Den här webbläsaren stöder inte skärminspelning här. Öppna appen i Google Chrome (adress: http://127.0.0.1:8723).", true);
    return;
  }
  btn.disabled = true; btn.textContent = "Väljer fönster att dela…";
  try{
    screenStream = await navigator.mediaDevices.getDisplayMedia({video:{frameRate:10}, audio:false});
  }catch(e){
    resetStartBtn();
    setPermHint("Skärmdelningen avbröts. Klicka Starta igen och välj fönstret eller fliken med din webbplats. På Mac kan du behöva tillåta webbläsaren under Systeminställningar → Integritet & säkerhet → Skärminspelning. På Windows: tillåt skärmdelning i webbläsarens dialog.", true);
    return;
  }
  btn.textContent = "Väntar på mikrofon…";
  try{
    micStream = await navigator.mediaDevices.getUserMedia({audio:true});
  }catch(e){
    try{ screenStream.getTracks().forEach(t=>t.stop()); }catch(_){}
    resetStartBtn();
    setPermHint("Kunde inte komma åt mikrofonen. Tillåt mikrofon för webbläsaren och försök igen.", true);
    return;
  }
  const combined = new MediaStream([...screenStream.getVideoTracks(), ...micStream.getAudioTracks()]);
  const prefs = ["video/webm;codecs=vp9,opus","video/webm;codecs=vp8,opus","video/webm","video/mp4"];
  recMime = "";
  for (const m of prefs){ if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)){ recMime = m; break; } }
  recExt = (recMime.indexOf("mp4")>=0) ? "mp4" : "webm";
  chunks = [];
  try{ mediaRecorder = recMime ? new MediaRecorder(combined,{mimeType:recMime}) : new MediaRecorder(combined); }
  catch(e){ mediaRecorder = new MediaRecorder(combined); }
  mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  mediaRecorder.start(1000);
  startTime = Date.now(); $("timer").textContent = "0:00";
  clearInterval(timerInt);
  timerInt = setInterval(()=>{ $("timer").textContent = fmt((Date.now()-startTime)/1000); }, 500);
  resetStartBtn();
  dShow("d_rec");
}

$("cancelRec").addEventListener("click", () => { try{ mediaRecorder && mediaRecorder.state!=="inactive" && mediaRecorder.stop(); }catch(e){} teardown(); dShow("d_idle"); });
$("stopRec").addEventListener("click", async () => { const blob = await stopAndGetVideo(); teardown(); await analyze(blob); });

function stopAndGetVideo(){
  const type = (recMime || "video/webm").split(";")[0];
  return new Promise(resolve => {
    if (!mediaRecorder || mediaRecorder.state === "inactive"){ resolve(new Blob(chunks,{type})); return; }
    mediaRecorder.onstop = () => resolve(new Blob(chunks, {type}));
    mediaRecorder.stop();
  });
}
function teardown(){
  clearInterval(timerInt);
  try{ micStream && micStream.getTracks().forEach(t=>t.stop()); }catch(e){}
  try{ screenStream && screenStream.getTracks().forEach(t=>t.stop()); }catch(e){}
}

let doneData = {};
async function analyze(blob){
  $("d_bar").style.width = "0%"; $("d_state").innerHTML = '<span class="spinner"></span>Laddar upp inspelningen…'; $("d_time").textContent = "";
  dShow("d_working");
  const fd = new FormData();
  fd.append("file", blob, "inspelning." + recExt);
  const r = await fetch("/analyze", {method:"POST", body:fd});
  if (!r.ok){ const j = await r.json().catch(()=>({})); dFail(j.error || "Kunde inte starta."); return; }
  poll();
}
function dFail(msg){ $("d_errMsg").textContent = msg; dShow("d_error"); }
$("d_retry").addEventListener("click", dReset);
$("d_again").addEventListener("click", dReset);
async function dReset(){ await fetch("/reset",{method:"POST"}); dShow("d_idle"); loadConfig(); }

$("d_copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(doneData.instruction || "");
  $("d_copyMsg").textContent = "Kopierat ✓ — klistra in i Claude (i ditt projekt).";
  const b=$("d_copy"); b.textContent="Kopierat ✓"; setTimeout(()=>b.textContent="📋 Kopiera instruktion till Claude",1600);
});
$("d_openFolder").addEventListener("click", async () => {
  await fetch("/open-folder",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:doneData.handoff_dir})});
});
$("d_openClaude").addEventListener("click", async () => {
  if (!doneData.command){ $("d_copyMsg").textContent = "Välj en projektmapp först för att kunna starta Claude Code direkt."; return; }
  const j = await (await fetch("/open-claude",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({command:doneData.command})})).json();
  $("d_copyMsg").textContent = j.ok ? "Öppnar Claude Code i en terminal…" : ("Kunde inte öppna: " + (j.error||""));
});

/* ── Transkribering ── */
const tShow = id => ["t_idle","t_working","t_done","t_error"].forEach(s => $(s).classList.toggle("hidden", s!==id));
const drop = $("drop"), fileInput = $("file");
["dragenter","dragover"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.add("over");}));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.remove("over");}));
drop.addEventListener("drop", ev => { const f = ev.dataTransfer.files[0]; if (f) tUpload(f); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) tUpload(fileInput.files[0]); });
let tName = "transkribering";
async function tUpload(file){
  tName = file.name.replace(/\.[^.]+$/, "");
  $("t_name").textContent = file.name; $("t_bar").style.width = "0%";
  $("t_state").innerHTML = '<span class="spinner"></span>Förbereder…'; $("t_time").textContent = "";
  tShow("t_working");
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/transcribe", {method:"POST", body:fd});
  if (!r.ok){ const j = await r.json().catch(()=>({})); tFail(j.error || "Kunde inte starta."); return; }
  poll();
}
function tFail(msg){ $("t_errMsg").textContent = msg; tShow("t_error"); }
$("t_download").addEventListener("click", () => {
  const blob = new Blob([$("t_out").value], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = tName + ".txt"; a.click(); URL.revokeObjectURL(a.href);
});
$("t_copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("t_out").value);
  $("t_copy").textContent = "Kopierat ✓"; setTimeout(() => $("t_copy").textContent = "Kopiera", 1500);
});
async function tReset(){ await fetch("/reset", {method:"POST"}); fileInput.value=""; tShow("t_idle"); }
$("t_again").addEventListener("click", tReset);
$("t_retry").addEventListener("click", tReset);

/* ── Talaridentifiering (diarisering) ── */
const sShow = id => ["s_idle","s_working","s_done","s_error"].forEach(x => $(x).classList.toggle("hidden", x!==id));
const sDrop = $("s_drop"), sFile = $("s_file");
["dragenter","dragover"].forEach(e => sDrop.addEventListener(e, ev => {ev.preventDefault(); sDrop.classList.add("over");}));
["dragleave","drop"].forEach(e => sDrop.addEventListener(e, ev => {ev.preventDefault(); sDrop.classList.remove("over");}));
sDrop.addEventListener("drop", ev => { const fs = Array.from(ev.dataTransfer.files); if (fs.length) sUpload(fs); });
sFile.addEventListener("change", () => { const fs = Array.from(sFile.files); if (fs.length) sUpload(fs); });

function sAddPerson(){
  const row = document.createElement("div");
  row.className = "srow";
  const name = document.createElement("input");
  name.type = "text"; name.placeholder = "Namn"; name.className = "sname";
  const pick = document.createElement("button");
  pick.type = "button"; pick.className = "btn pick"; pick.textContent = "Välj ljudklipp";
  const fin = document.createElement("input");
  fin.type = "file"; fin.className = "sfile";
  fin.accept = "audio/*,video/mp4,.m4a,.mp3,.wav,.mp4"; fin.style.display = "none";
  pick.addEventListener("click", () => fin.click());
  fin.addEventListener("change", () => {
    if (fin.files[0]){ pick.textContent = fin.files[0].name; pick.classList.add("has"); }
  });
  const rm = document.createElement("button");
  rm.type = "button"; rm.className = "btn rm"; rm.textContent = "✕"; rm.title = "Ta bort";
  rm.addEventListener("click", () => row.remove());
  row.append(name, pick, fin, rm);
  $("s_samples").appendChild(row);
  name.focus();
}
$("s_addPerson").addEventListener("click", sAddPerson);
function sCollectSamples(){
  const out = [];
  document.querySelectorAll("#s_samples .srow").forEach(row => {
    const name = row.querySelector(".sname").value.trim();
    const file = row.querySelector(".sfile").files[0];
    if (name && file) out.push({name, file});
  });
  return out;
}
let sName = "transkribering";
function sMakeForm(files){
  const fd = new FormData();
  files.forEach(f => fd.append("file", f));
  sCollectSamples().forEach(s => { fd.append("sample_name", s.name); fd.append("sample_file", s.file); });
  return fd;
}
async function sUpload(files){
  const first = files[0];
  sName = first.name.replace(/\.[^.]+$/, "");
  $("s_name").textContent = first.name; $("s_bar").style.width = "0%";
  $("s_state").innerHTML = '<span class="spinner"></span>Förbereder…'; $("s_time").textContent = "";
  $("s_queue").textContent = ""; $("s_queue").classList.add("hidden");
  sShow("s_working");
  const r = await fetch("/diarize", {method:"POST", body:sMakeForm(files)});
  if (!r.ok){ const j = await r.json().catch(()=>({})); sFail(j.error || "Kunde inte starta."); return; }
  poll();
}
/* Lägg till fler filer i kön medan en körning pågår. */
$("s_addMore").addEventListener("click", () => $("s_moreFile").click());
$("s_moreFile").addEventListener("change", async () => {
  const fs = Array.from($("s_moreFile").files);
  $("s_moreFile").value = "";
  if (!fs.length) return;
  const r = await fetch("/diarize", {method:"POST", body:sMakeForm(fs)});
  if (!r.ok){
    const j = await r.json().catch(()=>({}));
    $("s_queue").textContent = "Kunde inte lägga till: " + (j.error || "okänt fel");
    $("s_queue").classList.remove("hidden");
  }
});
function sQueueInfo(j){
  const el = $("s_queue");
  if ((j.batch_total||0) > 1){
    let txt = "Fil " + (j.batch_index||0) + " av " + j.batch_total;
    if (j.queue && j.queue.length) txt += " · i kö: " + j.queue.join(", ");
    el.textContent = txt; el.classList.remove("hidden");
  } else { el.textContent = ""; el.classList.add("hidden"); }
}
function sFail(msg){ $("s_errMsg").textContent = msg; sShow("s_error"); }
/* Resultat för batch: lista med en rad per fil, klicka för att visa. */
let sResults = [];
function sSaveTxt(r){
  const blob = new Blob([r.text], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = r.filename.replace(/\.[^.]+$/, "") + ".txt"; a.click(); URL.revokeObjectURL(a.href);
}
function sSelectResult(i){
  const r = sResults[i]; if (!r) return;
  sName = r.filename.replace(/\.[^.]+$/, "");
  $("s_doneName").textContent = r.filename;
  $("s_out").value = r.error ? "" : r.text;
  const note = $("s_note");
  const msg = r.error ? ("Något gick fel med den här filen: " + r.error) : (r.note || "");
  if (msg){ note.textContent = msg; note.classList.remove("hidden"); }
  else { note.textContent = ""; note.classList.add("hidden"); }
  document.querySelectorAll("#s_results .ritem").forEach((el, k) => el.classList.toggle("sel", k===i));
}
function sRenderResults(){
  const list = $("s_results"); list.innerHTML = "";
  sResults.forEach((r, i) => {
    const row = document.createElement("div"); row.className = "ritem";
    const fn = document.createElement("span"); fn.className = "fn"; fn.textContent = r.filename;
    fn.addEventListener("click", () => sSelectResult(i));
    row.appendChild(fn);
    if (r.error){
      const w = document.createElement("span"); w.className = "fail"; w.textContent = "⚠ misslyckades";
      row.appendChild(w);
    } else {
      const dl = document.createElement("button"); dl.type = "button"; dl.className = "btn mini";
      dl.textContent = "Ladda ner"; dl.addEventListener("click", () => sSaveTxt(r));
      row.appendChild(dl);
    }
    list.appendChild(row);
  });
  list.classList.remove("hidden");
  $("s_downloadAll").classList.remove("hidden");
  let idx = sResults.findIndex(r => !r.error); if (idx < 0) idx = 0;
  sSelectResult(idx);
}
$("s_downloadAll").addEventListener("click", () => {
  sResults.filter(r => !r.error).forEach((r, k) => setTimeout(() => sSaveTxt(r), k*300));
});
$("s_download").addEventListener("click", () => {
  const blob = new Blob([$("s_out").value], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = sName + ".txt"; a.click(); URL.revokeObjectURL(a.href);
});
$("s_copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("s_out").value);
  $("s_copy").textContent = "Kopierat ✓"; setTimeout(() => $("s_copy").textContent = "Kopiera", 1500);
});
async function sReset(){
  await fetch("/reset", {method:"POST"});
  sFile.value=""; $("s_samples").innerHTML="";
  sResults = []; $("s_results").innerHTML=""; $("s_results").classList.add("hidden");
  $("s_downloadAll").classList.add("hidden");
  $("s_queue").textContent=""; $("s_queue").classList.add("hidden");
  sShow("s_idle");
}
$("s_again").addEventListener("click", sReset);
$("s_retry").addEventListener("click", sReset);

$("quit").addEventListener("click", async () => {
  await fetch("/shutdown",{method:"POST"}).catch(()=>{});
  document.body.innerHTML = '<div style="font-family:-apple-system,sans-serif;color:#5E6E74;padding:40px;text-align:center">Appen är avstängd. Du kan stänga fliken.</div>';
});

/* ── Delad polling ── */
let timer;
function poll(){
  clearTimeout(timer);
  timer = setTimeout(async () => {
    let j;
    try { j = await (await fetch("/status")).json(); } catch { return poll(); }
    const pct = Math.round((j.progress||0)*100);

    if (j.mode === "debug"){
      if (j.state === "downloading"){
        $("d_bar").style.width = pct+"%"; $("d_state").innerHTML='<span class="spinner"></span>Laddar ner KB-Whisper (engångs)…'; $("d_time").textContent=pct+"%"; poll();
      } else if (j.state === "loading_model"){
        $("d_state").innerHTML='<span class="spinner"></span>Startar modellen…'; $("d_time").textContent=""; poll();
      } else if (j.state === "working"){
        $("d_bar").style.width = pct+"%"; $("d_state").innerHTML='<span class="spinner"></span>'+(j.stage||"Transkriberar…");
        $("d_time").textContent = j.duration ? (fmt(j.position)+" / "+fmt(j.duration)) : ""; poll();
      } else if (j.state === "analyzing"){
        $("d_bar").style.width = "100%"; $("d_state").innerHTML='<span class="spinner"></span>'+(j.stage||"Bearbetar…"); $("d_time").textContent=""; poll();
      } else if (j.state === "done"){
        doneData = j;
        $("d_frames").textContent = (j.frames_count||0) + " skärmbilder";
        $("d_instr").textContent = j.instruction || "";
        $("d_path").textContent = j.handoff_dir ? ("📁 " + j.handoff_dir) : "";
        $("d_openClaude").classList.toggle("hidden", !j.command);
        $("d_copyMsg").textContent = "";
        dShow("d_done");
      } else if (j.state === "error"){ dFail(j.error || "Något gick fel."); }
      else { poll(); }
      return;
    }

    if (j.mode === "diarize"){
      if (j.filename) $("s_name").textContent = j.filename;
      if (j.state === "downloading"){
        sQueueInfo(j);
        $("s_bar").style.width = pct+"%"; $("s_state").innerHTML='<span class="spinner"></span>Laddar ner KB-Whisper (engångs)…'; $("s_time").textContent=pct+"%"; poll();
      } else if (j.state === "loading_model"){
        sQueueInfo(j);
        $("s_state").innerHTML='<span class="spinner"></span>Startar modellen…'; $("s_time").textContent=""; poll();
      } else if (j.state === "working"){
        sQueueInfo(j);
        $("s_bar").style.width = pct+"%"; $("s_state").innerHTML='<span class="spinner"></span>Transkriberar…';
        $("s_time").textContent = j.duration ? (fmt(j.position)+" / "+fmt(j.duration)) : ""; poll();
      } else if (j.state === "diarizing"){
        sQueueInfo(j);
        $("s_bar").style.width = "100%"; $("s_state").innerHTML='<span class="spinner"></span>Identifierar talare…'; $("s_time").textContent=""; poll();
      } else if (j.state === "done"){
        $("s_bar").style.width="100%";
        sResults = j.results || [];
        if (sResults.length > 1){
          sRenderResults();
        } else {
          $("s_results").innerHTML=""; $("s_results").classList.add("hidden");
          $("s_downloadAll").classList.add("hidden");
          $("s_out").value=j.text; $("s_doneName").textContent=j.filename;
          const note=$("s_note");
          if (j.note){ note.textContent=j.note; note.classList.remove("hidden"); } else { note.textContent=""; note.classList.add("hidden"); }
        }
        sShow("s_done");
      } else if (j.state === "error"){ sFail(j.error || "Något gick fel."); }
      else { poll(); }
      return;
    }

    if (j.state === "downloading"){
      $("t_bar").style.width = pct+"%"; $("t_state").innerHTML='<span class="spinner"></span>Laddar ner KB-Whisper (engångs)…'; $("t_time").textContent=pct+"%"; poll();
    } else if (j.state === "loading_model"){
      $("t_state").innerHTML='<span class="spinner"></span>Startar modellen…'; $("t_time").textContent=""; poll();
    } else if (j.state === "working"){
      $("t_bar").style.width = pct+"%"; $("t_state").innerHTML='<span class="spinner"></span>Transkriberar…';
      $("t_time").textContent = j.duration ? (fmt(j.position)+" / "+fmt(j.duration)) : ""; poll();
    } else if (j.state === "done"){
      $("t_bar").style.width="100%"; $("t_out").value=j.text; $("t_doneName").textContent=j.filename; tShow("t_done");
    } else if (j.state === "error"){ tFail(j.error || "Något gick fel."); }
    else { poll(); }
  }, 900);
}

loadConfig();
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    if not os.environ.get("KB_NO_OPEN"):  # ev. launchers öppnar webbläsaren själva
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"\n  KB Transkribering körs på  http://127.0.0.1:{PORT}")
    print("  Stäng detta fönster (eller klicka 'Avsluta appen') för att avsluta.\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
