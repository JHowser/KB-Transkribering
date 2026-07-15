# KB Transkribering (GPU-version)

Transkriberar svenska möten lokalt på din dator med KB-Whisper (large).
Fungerar på **Mac** och **Windows**. På Mac körs modellen på GPU:n via Metal
(whisper.cpp); på Windows körs den på CPU. Inget ljud lämnar datorn.

Appen har tre flikar:
- **Felsökning** — spela in skärm + röst för en webbplats → färdig handoff till Claude.
- **Transkribera** — dra in en ljudfil → ren text.
- **Talaridentifiering** — text märkt per talare ("Talare 1", "Talare 2" …).
  Flera mötesinspelningar kan laddas upp samtidigt; de köas och körs en i taget.

Vill du **bara spela in** skärm + röst (utan att installera något)? Dubbelklicka på
**Spela in.html** — den öppnas i webbläsaren och sparar en videofil i Hämtade filer
när du stoppar. Ingen Python behövs.

## Så här gör du

1. **En gång:** installera Python 3 om du inte redan har det:
   https://www.python.org/downloads/ (ladda ner, dubbelklicka, klicka igenom).

2. Starta appen:
   - **Mac:** dubbelklicka på **Starta Transkribering.command**.
     - Första gången: högerklicka istället → **Öppna** → **Öppna**
       (macOS frågar en gång om du litar på filen).
     - Saknar du Apples utvecklarverktyg dyker en ruta upp – klicka **Installera**,
       vänta tills den är klar, och dubbelklicka sedan på filen igen.
   - **Windows:** dubbelklicka på **Starta Transkribering.bat**.
     - Kryssa i **"Add Python to PATH"** när du installerar Python (steg 1).
     - Skulle Windows visa en varning ("Windows SmartScreen"), klicka
       **Mer information** → **Kör ändå**.

3. Ett fönster öppnas i webbläsaren med de tre flikarna. Dra in din ljudfil under
   **Transkribera** eller **Talaridentifiering**, eller spela in en webbplats under
   **Felsökning**. Klart.

## Bra att veta

- **Första körningen** bygger motorn och laddar ner modellen (ca 1,1 GB).
  Det tar några minuter. Därefter går det snabbt och fungerar utan internet.
- En timmes möte tar ungefär 5–10 minuter på Mac (GPU). På Windows körs det på
  CPU och tar längre tid.
- Stöder m4a (röstmemo), mp3, wav och mp4. Ingen ffmpeg behövs.
- Avsluta genom att stänga terminalfönstret som öppnades.

## Filer

- `Starta Transkribering.command` — det du dubbelklickar på (Mac).
- `Starta Transkribering.bat` — det du dubbelklickar på (Windows).
- `Spela in.html` — fristående inspelare (skärm + röst) som öppnas direkt i webbläsaren.
- `app.py` — själva programmet.
- `model/` — modellen sparas här efter första nedladdningen.
- `.venv/` — skapas automatiskt (rör den inte).
