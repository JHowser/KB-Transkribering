#!/usr/bin/env python3
"""
KB Transkribering + Felsökning — helt lokalt. Ingen extern AI-tjänst, ingen API-nyckel.

Transkribering körs på GPU:n (Metal) via whisper.cpp. Inget ljud lämnar datorn.

Felsökningsläget (för webbplatser du bygger):
  1. Du spelar in HELA skärmen + din röst medan du går igenom sidan och pekar ut problem.
  2. Rösten transkriberas lokalt (svenska), med tidsstämplar.
  3. Skärmbilder klipps automatiskt ut ur videon vid varje tidsstämpel.
  4. Allt skrivs som en "handoff" direkt i din projektmapp:
        <projekt>/.felsokning/latest/
            context.md   – berättelse med tidsstämplar + instruktion till Claude
            frames/      – skärmbilderna
            recording.*  – hela inspelningen
  5. I Claude Code / Cowork säger du bara: "Läs .felsokning/latest/context.md och fixa
     problemen jag pekade ut." Claude läser koden + bilderna och åtgärdar (t.ex. med Opus).
"""

import os
import io
import json
import time
import shutil
import threading
import tempfile
import subprocess
import webbrowser
import urllib.request

from flask import Flask, request, jsonify, Response

MODEL_URL = "https://huggingface.co/KBLab/kb-whisper-large/resolve/main/ggml-model-q5_0.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MODEL_NAME = "kb-whisper-large-q5_0.bin"
CONFIG_PATH = os.path.join(HERE, ".felsokning_config.json")
SESSIONS_DIR = os.path.join(HERE, "sessions")
PORT = 8725
SR = 16000

# Skärmbilder: minsta avstånd mellan utklipp (så varje replik får en egen bild men
# nära dubbletter undviks), maxantal, och nedskalning.
MIN_FRAME_GAP = 0.8
MAX_FRAMES = 80
FRAME_MAX_WIDTH = 1000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB (skärmvideo)

JOB = {
    "state": "idle",   # idle | downloading | loading_model | working | analyzing | done | error
    "mode": "",        # transcribe | debug
    "stage": "",
    "progress": 0.0,
    "position": 0.0,
    "duration": 0.0,
    "text": "",           # ren transkribering (transkriberingsläge)
    "transcript": "",     # läsbart transkript (felsökningsläge)
    "handoff_dir": "",
    "instruction": "",    # meningen att ge Claude
    "command": "",        # färdigt CLI-kommando (om projektmapp satt)
    "frames_count": 0,
    "project_dir": "",
    "filename": "",
    "error": "",
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


# ── Modell-sökväg: egen mapp först, annars grannmappens modell, annars ladda ner ──

def _resolve_model_path():
    local = os.path.join(MODEL_DIR, MODEL_NAME)
    if os.path.exists(local) and not os.path.islink(local):
        return local
    if os.path.islink(local) and os.path.exists(os.path.realpath(local)):
        return os.path.realpath(local)
    sibling = os.path.abspath(os.path.join(HERE, "..", "model", MODEL_NAME))
    if os.path.exists(sibling):
        return sibling
    return local


MODEL_PATH = _resolve_model_path()


# ── Konfiguration (projektmapp) sparas lokalt ──

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


def _ssl_contexts():
    """certifi -> systemets -> overifierad (sista utväg). Alla krypterade."""
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
    dest = os.path.join(MODEL_DIR, MODEL_NAME)
    tmp = dest + ".part"
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
            os.replace(tmp, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise last_err


def _load_model():
    global _model, MODEL_PATH
    if _model is None:
        path = _resolve_model_path()
        if not os.path.exists(path):
            path = _download_model()
        MODEL_PATH = path
        from pywhispercpp.model import Model
        _model = Model(path, print_realtime=False, print_progress=False)
    return _model


def _decode_audio(path):
    """Avkoda ljudspåret till 16 kHz mono float32 med PyAV (ingen ffmpeg krävs)."""
    import av
    import numpy as np

    container = av.open(path)
    try:
        if not container.streams.audio:
            raise RuntimeError("Inspelningen innehåller inget ljudspår.")
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
            take(resampler.resample(None))
        except Exception:
            pass
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def _start(s):
    return (getattr(s, "t0", 0) or 0) / 100.0  # centisekunder


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


# ── Bildextraktion ur skärmvideon (PyAV + Pillow) ──

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


# ── Bakgrundsjobb ──

def _transcribe_only(path, filename):
    try:
        _set(state="downloading" if not os.path.exists(_resolve_model_path()) else "loading_model",
             mode="transcribe", stage="", progress=0.0, position=0.0, duration=0.0,
             text="", transcript="", error="", filename=filename)
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


def _prepare_handoff(video_path, session_dir, video_ext):
    """Transkribera -> klipp ut bilder -> skriv handoff i projektmappen."""
    try:
        cfg = _load_config()
        project = cfg.get("project_dir") or ""
        use_project = bool(project) and os.path.isdir(project)
        target_root = project if use_project else session_dir

        _set(state="downloading" if not os.path.exists(_resolve_model_path()) else "loading_model",
             mode="debug", stage="", progress=0.0, position=0.0, duration=0.0,
             text="", transcript="", handoff_dir="", instruction="", command="",
             frames_count=0, project_dir=(project if use_project else ""),
             error="", filename="Felsökningssession")

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
            command = 'cd "{}" && claude "{}"'.format(project, INSTRUCTION)
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
    """Öppna en native mappväljare (macOS) och spara valet."""
    try:
        script = 'POSIX path of (choose folder with prompt "Välj din projektmapp")'
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=180)
        path = (out.stdout or "").strip().rstrip("/")
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
            subprocess.run(["open", path])
            return jsonify(ok=True)
        except Exception as exc:  # noqa: BLE001
            return jsonify(ok=False, error=str(exc))
    return jsonify(ok=False, error="Mappen finns inte."), 400


@app.route("/open-claude", methods=["POST"])
def open_claude():
    """Bäst-effort: öppna Terminal i projektmappen och kör Claude Code."""
    data = request.get_json(silent=True) or {}
    command = data.get("command") or ""
    if not command:
        return jsonify(ok=False, error="Inget kommando."), 400
    try:
        esc = command.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(["osascript",
                        "-e", 'tell application "Terminal" to do script "{}"'.format(esc),
                        "-e", 'tell application "Terminal" to activate'])
        return jsonify(ok=True)
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=False, error=str(exc))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    with LOCK:
        if JOB["state"] in ("downloading", "loading_model", "working", "analyzing"):
            return jsonify(ok=False, error="En körning pågår redan."), 409
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Ingen fil mottagen."), 400
    suffix = os.path.splitext(f.filename)[1] or ".m4a"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    f.save(path)
    threading.Thread(target=_transcribe_only, args=(path, f.filename), daemon=True).start()
    return jsonify(ok=True)


@app.route("/analyze", methods=["POST"])
def analyze():
    with LOCK:
        if JOB["state"] in ("downloading", "loading_model", "working", "analyzing"):
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
        if JOB["state"] in ("downloading", "loading_model", "working", "analyzing"):
            return jsonify(ok=False), 409
        JOB.update(state="idle", mode="", stage="", progress=0.0, position=0.0, duration=0.0,
                   text="", transcript="", handoff_dir="", instruction="", command="",
                   frames_count=0, project_dir="", filename="", error="")
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
<title>KB Felsökning</title>
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
    font-size:14px; font-weight:600; padding:9px 10px; border-radius:8px; cursor:pointer; transition:all .12s ease}
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
</style>
</head>
<body>
  <main class="card">
    <div class="brand"><h1>Felsökning</h1><span class="tag">KB-Whisper · lokalt</span></div>

    <div class="tabs">
      <button id="tabD" class="active" onclick="setTab('d')">Felsökning</button>
      <button id="tabT" onclick="setTab('t')">Transkribera</button>
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

    <div class="foot">Allt lokalt · KB-Whisper large · GPU (Metal) · <a id="quit">Avsluta appen</a></div>
  </main>

<script>
const $ = id => document.getElementById(id);
function fmt(s){ s = Math.max(0, Math.round(s||0)); const m = Math.floor(s/60), x = s%60; return m+":"+String(x).padStart(2,"0"); }

function setTab(which){
  $("tabT").classList.toggle("active", which==="t");
  $("tabD").classList.toggle("active", which==="d");
  $("paneT").classList.toggle("hidden", which!=="t");
  $("paneD").classList.toggle("hidden", which!=="d");
}

/* ── Projektmapp ── */
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
    setPermHint("Den här webbläsaren stöder inte skärminspelning här. Öppna appen i Google Chrome (adress: http://127.0.0.1:8725).", true);
    return;
  }
  btn.disabled = true; btn.textContent = "Väljer fönster att dela…";
  try{
    screenStream = await navigator.mediaDevices.getDisplayMedia({video:{frameRate:10}, audio:false});
  }catch(e){
    resetStartBtn();
    setPermHint("Skärmdelningen avbröts. Klicka Starta igen och välj fönstret eller fliken med din webbplats. På Mac kan du behöva tillåta webbläsaren under Systeminställningar → Integritet & säkerhet → Skärminspelning.", true);
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
  $("d_copyMsg").textContent = j.ok ? "Öppnar Claude Code i Terminal…" : ("Kunde inte öppna: " + (j.error||""));
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
    if not os.environ.get("KB_NO_OPEN"):  # .app-launcharen öppnar webbläsaren själv
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"\n  KB Felsökning körs på  http://127.0.0.1:{PORT}")
    print("  Stäng detta fönster (eller klicka 'Avsluta appen') för att avsluta.\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
