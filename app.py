#!/usr/bin/env python3
"""
KB Transkribering — lokal transkribering av svenska möten med KB-Whisper.
Körs på GPU:n (Metal) via whisper.cpp. Inget ljud lämnar datorn.
"""

import os
import threading
import tempfile
import webbrowser
import urllib.request

from flask import Flask, request, jsonify, Response

# q5_0 = liten nedladdning (~1,1 GB), full noggrannhet, snabb på Metal.
MODEL_URL = "https://huggingface.co/KBLab/kb-whisper-large/resolve/main/ggml-model-q5_0.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "kb-whisper-large-q5_0.bin")
PORT = 8723
SR = 16000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

JOB = {
    "state": "idle",   # idle | downloading | loading_model | working | done | error
    "progress": 0.0,
    "position": 0.0,
    "duration": 0.0,
    "text": "",
    "filename": "",
    "error": "",
}
LOCK = threading.Lock()
_model = None


def _set(**kw):
    with LOCK:
        JOB.update(kw)


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
            return
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
        from pywhispercpp.model import Model
        # whisper.cpp byggs med Metal som standard på Apple Silicon -> körs på GPU.
        _model = Model(MODEL_PATH, print_realtime=False, print_progress=False)
    return _model


def _decode(path):
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


def _transcribe(path, filename):
    try:
        _set(state="downloading" if not os.path.exists(MODEL_PATH) else "loading_model",
             progress=0.0, position=0.0, duration=0.0, text="", error="", filename=filename)

        if not os.path.exists(MODEL_PATH):
            _download_model()
            _set(state="loading_model", progress=0.0)

        model = _load_model()

        audio = _decode(path)
        import numpy as np
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


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    with LOCK:
        if JOB["state"] in ("downloading", "loading_model", "working"):
            return jsonify(ok=False, error="En transkribering pågår redan."), 409
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="Ingen fil mottagen."), 400
    suffix = os.path.splitext(f.filename)[1] or ".m4a"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    f.save(path)
    threading.Thread(target=_transcribe, args=(path, f.filename), daemon=True).start()
    return jsonify(ok=True)


@app.route("/status")
def status():
    with LOCK:
        return jsonify(dict(JOB))


@app.route("/reset", methods=["POST"])
def reset():
    with LOCK:
        if JOB["state"] in ("downloading", "loading_model", "working"):
            return jsonify(ok=False), 409
        JOB.update(state="idle", progress=0.0, position=0.0, duration=0.0,
                   text="", filename="", error="")
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
    --display:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
    --ui:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--paper); color:var(--ink); font-family:var(--ui);
    display:flex; align-items:center; justify-content:center; padding:32px; -webkit-font-smoothing:antialiased;}
  .card{width:100%; max-width:560px; background:var(--card); border:1px solid var(--line);
    border-radius:14px; padding:34px 34px 30px; box-shadow:0 1px 0 #fff inset, 0 18px 50px -28px rgba(22,37,43,.35);}
  .brand{display:flex; align-items:baseline; gap:10px; margin-bottom:4px}
  .brand h1{font-family:var(--display); font-weight:600; font-size:25px; letter-spacing:.2px; margin:0}
  .brand .tag{font-size:12px; color:var(--muted); letter-spacing:.04em; text-transform:uppercase}
  .sub{color:var(--muted); font-size:13.5px; margin:0 0 24px; line-height:1.5}
  .drop{border:1.5px dashed var(--line); border-radius:12px; background:#fff; padding:46px 24px;
    text-align:center; cursor:pointer; transition:border-color .15s ease, background .15s ease;}
  .drop:hover{border-color:var(--accent); background:#FAFCFD}
  .drop.over{border-color:var(--accent); background:var(--accent-soft)}
  .drop .icon{width:42px; height:42px; margin:0 auto 14px; color:var(--accent)}
  .drop .big{font-size:15.5px; font-weight:550}
  .drop .small{font-size:12.5px; color:var(--muted); margin-top:6px}
  input[type=file]{display:none}
  .work .name{font-size:14px; font-weight:550; margin-bottom:18px; word-break:break-all}
  .track{height:8px; background:var(--accent-soft); border-radius:99px; overflow:hidden}
  .bar{height:100%; width:0%; background:var(--accent); border-radius:99px; transition:width .4s ease}
  .work .meta{display:flex; justify-content:space-between; margin-top:12px; font-size:13px; color:var(--muted)}
  .work .meta .state{color:var(--ink)}
  .spinner{display:inline-block; width:13px; height:13px; margin-right:7px; vertical-align:-2px;
    border:2px solid var(--accent-soft); border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .result .name{font-size:13px; color:var(--muted); margin-bottom:10px; word-break:break-all}
  textarea{width:100%; height:300px; resize:vertical; border:1px solid var(--line); border-radius:10px;
    padding:14px 16px; font-family:var(--ui); font-size:14px; line-height:1.65; color:var(--ink); background:#fff;}
  textarea:focus{outline:none; border-color:var(--accent)}
  .actions{display:flex; gap:10px; margin-top:16px}
  button{font-family:var(--ui); font-size:14px; font-weight:550; cursor:pointer; border-radius:9px;
    padding:11px 16px; border:1px solid var(--line); background:#fff; color:var(--ink); transition:background .12s ease, border-color .12s ease;}
  button:hover{background:#F2F5F6}
  button.primary{background:var(--accent); border-color:var(--accent); color:#fff}
  button.primary:hover{background:#1A4F62}
  .grow{flex:1}
  .err{color:var(--danger); font-size:13.5px; line-height:1.5; margin-top:4px}
  .hidden{display:none}
  .foot{margin-top:22px; font-size:11.5px; color:var(--muted); text-align:center; letter-spacing:.02em}
</style>
</head>
<body>
  <main class="card">
    <div class="brand"><h1>Transkribering</h1><span class="tag">KB-Whisper</span></div>
    <p class="sub">Släpp in en mötesinspelning så får du tillbaka texten. Allt körs lokalt på din dator.</p>

    <section id="idle">
      <label class="drop" id="drop" for="file">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>
        </svg>
        <div class="big">Dra in din ljudfil</div>
        <div class="small">eller klicka för att välja · m4a, mp3, wav, mp4</div>
      </label>
      <input type="file" id="file" accept="audio/*,video/mp4,.m4a,.mp3,.wav,.mp4">
    </section>

    <section id="working" class="work hidden">
      <div class="name" id="workName"></div>
      <div class="track"><div class="bar" id="bar"></div></div>
      <div class="meta">
        <span class="state" id="workState"><span class="spinner"></span>Förbereder…</span>
        <span id="workTime"></span>
      </div>
    </section>

    <section id="done" class="result hidden">
      <div class="name" id="doneName"></div>
      <textarea id="out" spellcheck="false"></textarea>
      <div class="actions">
        <button class="primary grow" id="download">Ladda ner .txt</button>
        <button id="copy">Kopiera</button>
        <button id="again">Ny fil</button>
      </div>
    </section>

    <section id="error" class="hidden">
      <p class="err" id="errMsg"></p>
      <div class="actions"><button id="retry" class="grow">Försök igen</button></div>
    </section>

    <div class="foot">Bearbetning sker offline · KB-Whisper large · körs på GPU (Metal)</div>
  </main>

<script>
const $ = id => document.getElementById(id);
const show = id => ["idle","working","done","error"].forEach(s => $(s).classList.toggle("hidden", s!==id));
const drop = $("drop"), fileInput = $("file");
["dragenter","dragover"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.add("over");}));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.remove("over");}));
drop.addEventListener("drop", ev => { const f = ev.dataTransfer.files[0]; if (f) upload(f); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); });
function fmt(s){ s = Math.max(0, Math.round(s||0)); const m = Math.floor(s/60), x = s%60; return m+":"+String(x).padStart(2,"0"); }
let currentName = "transkribering";
async function upload(file){
  currentName = file.name.replace(/\.[^.]+$/, "");
  $("workName").textContent = file.name;
  $("bar").style.width = "0%";
  $("workState").innerHTML = '<span class="spinner"></span>Förbereder…';
  $("workTime").textContent = "";
  show("working");
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/transcribe", {method:"POST", body:fd});
  if (!r.ok){ const j = await r.json().catch(()=>({})); fail(j.error || "Kunde inte starta."); return; }
  poll();
}
let timer;
function poll(){
  clearTimeout(timer);
  timer = setTimeout(async () => {
    let j;
    try { j = await (await fetch("/status")).json(); } catch { return poll(); }
    const pct = Math.round((j.progress||0)*100);
    if (j.state === "downloading"){
      $("bar").style.width = pct + "%";
      $("workState").innerHTML = '<span class="spinner"></span>Laddar ner KB-Whisper (engångs)…';
      $("workTime").textContent = pct + "%";
      poll();
    } else if (j.state === "loading_model"){
      $("workState").innerHTML = '<span class="spinner"></span>Startar modellen…';
      $("workTime").textContent = "";
      poll();
    } else if (j.state === "working"){
      $("bar").style.width = pct + "%";
      $("workState").innerHTML = '<span class="spinner"></span>Transkriberar…';
      $("workTime").textContent = j.duration ? (fmt(j.position) + " / " + fmt(j.duration)) : "";
      poll();
    } else if (j.state === "done"){
      $("bar").style.width = "100%"; $("out").value = j.text; $("doneName").textContent = j.filename; show("done");
    } else if (j.state === "error"){
      fail(j.error || "Något gick fel.");
    } else { poll(); }
  }, 900);
}
function fail(msg){ $("errMsg").textContent = msg; show("error"); }
$("download").addEventListener("click", () => {
  const blob = new Blob([$("out").value], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = currentName + ".txt"; a.click(); URL.revokeObjectURL(a.href);
});
$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("out").value);
  $("copy").textContent = "Kopierat ✓"; setTimeout(() => $("copy").textContent = "Kopiera", 1500);
});
async function reset(){ await fetch("/reset", {method:"POST"}); fileInput.value=""; show("idle"); }
$("again").addEventListener("click", reset);
$("retry").addEventListener("click", reset);
</script>
</body>
</html>"""


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"\n  KB Transkribering körs på  http://127.0.0.1:{PORT}")
    print("  Stäng detta fönster för att avsluta.\n")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
