# KB Felsökning

Lokal transkribering (KB-Whisper large, körs på din GPU via Metal — inget ljud lämnar
datorn) plus ett **felsökningsläge** för webbplatser du bygger. **Ingen API-nyckel, ingen
extern AI-tjänst.** Appen förbereder allt Claude behöver; du låter sedan Claude (t.ex. Opus
i Claude Code eller Cowork) läsa koden och åtgärda.

Den här mappen är en kopia av originalappen KB-Transkribering. Originalet i föräldermappen
är orört. Modellen delas via symlänk i `model/`, så ingen ny 1,1 GB-nedladdning behövs.

## Så här startar du (snabbt, utan att öppna Claude)

Appen är helt fristående och körs på **http://127.0.0.1:8725**. Den behöver inte att
Claude eller något annat är öppet. Allt är native (ingen Rosetta behövs).

1. **Enklast och mest pålitligt:** dubbelklicka på **Starta Felsökning.command**.
   Ett litet terminalfönster visar att servern startar och webbläsaren öppnas på
   felsökningsfliken. Stäng terminalfönstret för att avsluta.
2. **KB Felsökning.app** gör samma sak utan terminal (startar servern i bakgrunden och
   öppnar webbläsaren). Kan startas från **Spotlight** (⌘Space → skriv "KB Felsökning")
   eller läggas i **Dock** (dra dit den från mappen).

### Tangentbordsgenväg (⇧⌘K)

En genväg i **Genvägar** kör "Öppna app → KB Felsökning". Den är kopplad till **⇧⌘K** —
tryck det var som helst för att starta/öppna felsökningsfliken.

> **Om top-row-ikon:** din macOS-version av Genvägar saknar "Fäst i menyraden", så en äkta
> menyradsikon går tyvärr inte att få utan Rosetta eller en tredjepartsstartare. ⇧⌘K och
> Spotlight ger dig ändå en snabb tangentbordsstart.

**Om appen inte startar** (t.ex. "kunde inte köras – behörigheter" eller "kunde inte
öppnas"): filens körrättighet kan ha fallit bort. Öppna Terminal och kör en gång:
```
chmod +x "/Applications/KB-Transkribering/KB-Felsökning/Starta Felsökning.command"
chmod +x "/Applications/KB-Transkribering/KB-Felsökning/KB Felsökning.app/Contents/MacOS/launch"
```

## Fliken "Felsökning" — så funkar det

1. **Välj projektmapp** en gång (knappen **Välj mapp**) — peka på mappen för webbplatsen
   du jobbar med. Sparas lokalt i `.felsokning_config.json`.
2. Klicka **Starta inspelning**, tillåt mikrofon, och välj **fönstret eller fliken med din
   webbplats** att dela.
3. Klicka runt och **beskriv felen högt** medan du går igenom sidan. Skärm + röst spelas in
   i en enda synkad fil.
4. Klicka **Stoppa & skapa handoff**. Appen (allt lokalt):
   - transkriberar din röst med tidsstämplar,
   - klipper ut skärmbilder ur videon vid de ögonblick du pratade,
   - skriver en **handoff** i din projektmapp:
     ```
     <projekt>/.felsokning/latest/
         context.md    – berättelsen med tidsstämplar + instruktion till Claude
         frames/       – skärmbilderna
         recording.*   – hela inspelningen
     ```
5. Ge det till Claude i ditt projekt:
   - klicka **📋 Kopiera instruktion till Claude** och klistra in i Claude Code / Cowork, eller
   - klicka **Öppna i Claude Code** (startar Claude Code i projektmappen åt dig), eller
   - klicka **Öppna mappen** för att se filerna.

   Claude läser `context.md`, öppnar de relevanta skärmbilderna, korsrefererar med din kod
   och åtgärdar — med en stark modell (t.ex. Opus). Ett billigare/snabbt steg (att hitta
   tidsstämplarna) görs redan lokalt av appen, så modellen kan lägga kraften på själva fixen.

### Bra att veta

- **Använd helst Google Chrome** — skärminspelning fungerar smidigast där. `.app`-startaren
  öppnar Chrome om den finns. Första gången kan du behöva tillåta webbläsaren under
  Systeminställningar → Integritet & säkerhet → Skärminspelning.
- Dela **fönstret med webbplatsen**, inte appfliken (annars blir det bild-i-bild).
- Inget lämnar datorn. Mappen `.felsokning/` läggs utanför git automatiskt (egen `.gitignore`),
  så inspelningen råkar inte checkas in.
- Väljer du ingen projektmapp hamnar handoffen i appens egen `sessions/`-mapp istället, och
  appen visar den exakta sökvägen.
- Avsluta appen via länken **Avsluta appen** längst ner, eller stäng fliken (servern kan
  ligga kvar i bakgrunden tills du avslutar den).

## Fliken "Transkribera"

Som originalet: dra in en ljudfil (m4a, mp3, wav, mp4) och få tillbaka texten. Allt lokalt.

## Filer

- `KB Felsökning.app` — snabbstart (Dock/Spotlight/menyrad/tangentbord).
- `Starta Felsökning.command` — engångsinstallet / manuell start i terminal.
- `app.py` — själva programmet (port 8725).
- `model/` — symlänk till originalappens modell.
- `sessions/` — reservmapp för handoff om ingen projektmapp valts.
- `.felsokning_config.json` — din valda projektmapp (skapas när du väljer).
- `.venv/` — skapas automatiskt (rör den inte).
