# KB Transkribering (GPU-version)

Transkriberar svenska möten lokalt på din Mac med KB-Whisper (large).
Körs på din GPU via Metal (whisper.cpp). Inget ljud lämnar datorn.

## Så här gör du

1. **En gång:** installera Python 3 om du inte redan har det:
   https://www.python.org/downloads/ (ladda ner, dubbelklicka, klicka igenom).

2. Dubbelklicka på **Starta Transkribering.command**.
   - Första gången: högerklicka istället → **Öppna** → **Öppna**
     (macOS frågar en gång om du litar på filen).
   - Saknar du Apples utvecklarverktyg dyker en ruta upp – klicka **Installera**,
     vänta tills den är klar, och dubbelklicka sedan på filen igen.

3. Ett fönster öppnas i webbläsaren. Dra in din ljudfil. Klart.

## Bra att veta

- **Första körningen** bygger motorn och laddar ner modellen (ca 1,1 GB).
  Det tar några minuter. Därefter går det snabbt och fungerar utan internet.
- En timmes möte tar ungefär 5–10 minuter (körs på GPU:n).
- Stöder m4a (röstmemo), mp3, wav och mp4. Ingen ffmpeg behövs.
- Avsluta genom att stänga terminalfönstret som öppnades.

## Filer

- `Starta Transkribering.command` — det du dubbelklickar på.
- `app.py` — själva programmet.
- `model/` — modellen sparas här efter första nedladdningen.
- `.venv/` — skapas automatiskt (rör den inte).
