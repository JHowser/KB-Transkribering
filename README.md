# KB Transkribering

Lokal transkribering av svenska inspelningar på din Mac med **KB-Whisper (large)**. Körs
på din GPU via Metal (whisper.cpp) — **inget ljud lämnar datorn**. Innehåller även ett
**felsökningsläge** (`KB-Felsökning/`) som spelar in skärm + röst medan du går igenom en
webbplats, transkriberar med tidsstämplar, klipper ut skärmbilder vid rätt ögonblick och
skapar en färdig handoff till Claude.

## Kom igång

1. **Installera Python 3** (en gång): https://www.python.org/downloads/
2. Dubbelklicka på **Starta Transkribering.command** (första gången: högerklicka → **Öppna**).
3. Ett fönster öppnas i webbläsaren. Påbörja skärminspelning och röstinspelning genom att
   trycka på **"Starta inspelning"**, eller gör endast en transkribering genom att ladda upp
   en röstinspelning (m4a, mp3, wav, mp4).

Första körningen bygger motorn. Se till att modellen finns på plats först (se nedan).

## Ladda ner KB-Whisper-modellen

Modellen ligger **inte** i det här repot (den är ca 1,1 GB och för stor för GitHub).
Ladda ner den själv:

1. Gå till modellsidan hos KBLab: **https://huggingface.co/KBLab/kb-whisper-large**
2. Hämta den kvantiserade **whisper.cpp / GGML**-varianten (`q5_0`) — filen heter
   `kb-whisper-large-q5_0.bin`.
3. Lägg filen i mappen **`model/`** i projektet, så att sökvägen blir:

   ```
   model/kb-whisper-large-q5_0.bin
   ```

Efter det fungerar transkriberingen helt lokalt, utan internet.

## Viktigt för Mac-användare: stäng av Rosetta

För att appen ska fungera och kunna kopplas till **Genvägar (Shortcuts)** måste den köras
native (Apple Silicon), inte via Rosetta:

1. Högerklicka på appen i **Finder** → **Visa info** (Get Info).
2. **Avmarkera** rutan **"Öppna med Rosetta"** ("Open using Rosetta").
3. Starta appen på nytt.

Om "Öppna med Rosetta" är ikryssad kan kopplingen till Genvägar sluta fungera.

## Filer

- `Starta Transkribering.command` — det du dubbelklickar på.
- `app.py` — själva programmet.
- `model/` — här läggs den nedladdade modellen (se ovan).
- `KB-Felsökning/` — felsökningsläget (skärm + röst → handoff till Claude).
- `.venv/` — skapas automatiskt (rör den inte).

> `model/`, `.venv/` och inspelade `sessions/` är undantagna från git via `.gitignore`.
