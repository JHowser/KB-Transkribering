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

## Talaridentifiering: vem sa vad?

Appen märker automatiskt repliker per talare ("Talare 1", "Talare 2" …) med
tidsstämplar:

```
Talare 1 (00:12): Ska vi börja med budgeten?
Talare 2 (00:18): Ja, jag har siffrorna här…
```

Diarisering ("vem talar när") är **språkoberoende** — den arbetar på röstens klang,
inte på orden — så det finns ingen svensk-specifik talarmodell. KB-Whisper sköter
orden, en allmän diariseringsmodell (`pyannote/speaker-diarization-3.1`) sköter
talarna, och de vävs ihop på tidsstämplarna. Allt körs lokalt.

### Riktiga namn i stället för "Talare 1" (valfritt)

Under släppytan finns **"Lägg till röstprover (valfritt)"**. Där kan du lägga till
en person i taget med **namn** + en **kort inspelning (10–20 sek) där bara den
personen pratar**. Då märks replikerna med riktiga namn i stället för generiska
etiketter. Lägg till personerna innan du släpper in mötesinspelningen. Talare utan
säker matchning behåller "Talare N". Röstjämförelsen sker lokalt med ECAPA-TDNN
(`speechbrain/spkrec-ecapa-voxceleb`).

### Hugging Face-token (engångs)

Diariseringsmodellen är gratis och körs lokalt, men "gated" på Hugging Face — du
behöver ett konto och en token en gång. Ljudet lämnar aldrig datorn; token används
bara för att ladda ner modellvikterna första gången.

1. Skapa ett gratis konto på <https://huggingface.co/join> (hoppa över om du har ett).
2. Godkänn villkoren på **båda** modellsidorna (logga in och tryck på knappen):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. Skapa en **read**-token: <https://huggingface.co/settings/tokens> → **New token**
   → typ **Read** → kopiera värdet (börjar med `hf_`).
4. Gör token tillgänglig på **ett** av två sätt:
   - **Enklast:** skapa filen `hf_token.txt` bredvid `app.py` och klistra in enbart
     token i den. (Filen är undantagen från git.)
   - **Eller** sätt en miljövariabel innan du startar appen:

     ```
     export HF_TOKEN=hf_din_token_här
     ```

Utan token fungerar vanlig transkribering som vanligt — appen visar då texten utan
talaretiketter och en kort notis, i stället för att sluta fungera.

> Första gången laddar `pyannote` och röstmodellen ner sina vikter (några hundra MB).
> Det sker en gång; därefter körs allt offline.

## Viktigt för Mac-användare: stäng av Rosetta

För att appen ska fungera och kunna kopplas till **Genvägar (Shortcuts)** måste den köras
native (Apple Silicon), inte via Rosetta:

1. Högerklicka på appen i **Finder** → **Visa info** (Get Info).
2. **Avmarkera** rutan **"Öppna med Rosetta"** ("Open using Rosetta").
3. Starta appen på nytt.

Om "Öppna med Rosetta" är ikryssad kan kopplingen till Genvägar sluta fungera.

## Filer

- `Starta Transkribering.command` — det du dubbelklickar på.
- `app.py` — själva programmet (Flask-server + webbgränssnitt).
- `diarize.py` — talardiarisering (pyannote).
- `enroll.py` — röstprover och namnmatchning (ECAPA-TDNN).
- `merge.py` — slår ihop ord och talare på tidsstämplar och renderar utskriften.
- `requirements.txt` — Python-beroenden (installeras automatiskt i `.venv/`).
- `model/` — här läggs den nedladdade modellen (se ovan); även talarmodellernas cache.
- `KB-Felsökning/` — felsökningsläget (skärm + röst → handoff till Claude).
- `.venv/` — skapas automatiskt (rör den inte).

> `model/`, `.venv/`, `hf_token.txt` och inspelade `sessions/` är undantagna från git
> via `.gitignore`.
