# KB Transkribering

Lokal transkribering av svenska inspelningar på din Mac med **KB-Whisper (large)**. Körs
på din GPU via Metal (whisper.cpp) — **inget ljud lämnar datorn**. Allt samlas i **en app
med tre flikar**:

- **Felsökning** — spela in skärm + röst medan du går igenom en webbplats, peka ut problemen
  högt. Rösten transkriberas med tidsstämplar, skärmbilder klipps ut vid rätt ögonblick och
  en färdig handoff skrivs i din projektmapp, redo för Claude att läsa och åtgärda.
- **Transkribera** — dra in en mötesinspelning (m4a, mp3, wav, mp4) och få tillbaka ren text.
- **Talaridentifiering** — som Transkribera, men texten märks per talare ("Talare 1",
  "Talare 2" …) med diarisering. Valfria röstprover ger riktiga namn (se nedan).

## Kom igång

1. **Installera Python 3** (en gång): https://www.python.org/downloads/
2. Dubbelklicka på **Starta Transkribering.command** (första gången: högerklicka → **Öppna**).
3. Ett fönster öppnas i webbläsaren med de tre flikarna. Välj **Felsökning** för att spela in
   skärm + röst, eller **Transkribera** / **Talaridentifiering** för att ladda upp en
   röstinspelning (m4a, mp3, wav, mp4).

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

Fliken **Talaridentifiering** märker automatiskt repliker per talare ("Talare 1",
"Talare 2" …) med tidsstämplar:

```
Talare 1 (00:12): Ska vi börja med budgeten?
Talare 2 (00:18): Ja, jag har siffrorna här…
```

Diarisering ("vem talar när") är **språkoberoende** — den arbetar på röstens klang,
inte på orden — så det finns ingen svensk-specifik talarmodell. KB-Whisper sköter
orden, en allmän diariseringsmodell (`pyannote/speaker-diarization-3.1`) sköter
talarna, och de vävs ihop på tidsstämplarna. Allt körs lokalt.

### Riktiga namn i stället för "Talare 1" (valfritt)

Under släppytan i fliken **Talaridentifiering** finns **"Lägg till röstprover (valfritt)"**. Där kan du lägga till
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

## Felsökning: skärm + röst → handoff till Claude

Fliken **Felsökning** hjälper dig felsöka en webbplats du bygger — helt lokalt, ingen
API-nyckel. Appen förbereder allt Claude behöver; du låter sedan Claude (t.ex. Opus i
Claude Code eller Cowork) läsa koden och åtgärda.

1. **Välj projektmapp** en gång (knappen **Välj mapp**) — peka på mappen för webbplatsen
   du jobbar med. Sparas lokalt i `.felsokning_config.json`.
2. Klicka **Starta inspelning**, tillåt mikrofon, och välj **fönstret eller fliken med din
   webbplats** att dela.
3. Klicka runt och **beskriv felen högt** medan du går igenom sidan. Skärm + röst spelas in
   i en enda synkad fil.
4. Klicka **Stoppa & skapa handoff**. Appen (allt lokalt) transkriberar din röst med
   tidsstämplar, klipper ut skärmbilder ur videon vid de ögonblick du pratade och skriver en
   **handoff** i din projektmapp:
   ```
   <projekt>/.felsokning/latest/
       context.md    – berättelsen med tidsstämplar + instruktion till Claude
       frames/       – skärmbilderna
       recording.*   – hela inspelningen
   ```
5. Ge det till Claude i ditt projekt: klicka **📋 Kopiera instruktion till Claude** och
   klistra in i Claude Code / Cowork, **Öppna i Claude Code** (startar Claude Code i
   projektmappen åt dig), eller **Öppna mappen** för att se filerna.

**Bra att veta:** använd helst **Google Chrome** — skärminspelning fungerar smidigast där.
Första gången kan du behöva tillåta webbläsaren under Systeminställningar → Integritet &
säkerhet → Skärminspelning. Dela **fönstret med webbplatsen**, inte appfliken. Väljer du
ingen projektmapp hamnar handoffen i appens egen `sessions/`-mapp istället.

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
- `sessions/` — reservmapp för felsökningens handoff om ingen projektmapp valts.
- `.felsokning_config.json` — din valda projektmapp för felsökningsläget (skapas när du väljer).
- `.venv/` — skapas automatiskt (rör den inte).

> `model/`, `.venv/`, `hf_token.txt`, `.felsokning_config.json` och inspelade `sessions/`
> är undantagna från git via `.gitignore`.
