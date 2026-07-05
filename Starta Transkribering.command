#!/bin/bash
# Dubbelklicka denna fil för att starta KB Transkribering.
cd "$(dirname "$0")" || exit 1
clear
echo "──────────────────────────────"
echo "  KB Transkribering"
echo "──────────────────────────────"

if ! command -v python3 >/dev/null 2>&1; then
  echo
  echo "  Python 3 saknas. Installera härifrån (dubbelklicka installeraren):"
  echo "  https://www.python.org/downloads/"
  echo "  Kör sedan denna fil igen."
  echo
  read -r -p "  Tryck Enter för att stänga."
  exit 1
fi

# whisper.cpp behöver Apples kompilatorverktyg för att byggas första gången.
if ! xcode-select -p >/dev/null 2>&1; then
  echo
  echo "  Engångssteg: installera Apples utvecklarverktyg."
  echo "  Ett fönster dyker upp – klicka på \"Installera\" och vänta tills det är klart."
  echo "  Kör sedan denna fil igen."
  echo
  xcode-select --install >/dev/null 2>&1
  read -r -p "  Tryck Enter för att stänga."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo
  echo "  Förbereder appen. Detta görs bara en gång och kan ta några minuter…"
  python3 -m venv .venv || { echo "  Kunde inte skapa miljö."; read -r; exit 1; }
  ./.venv/bin/pip install --upgrade pip >/dev/null
  ./.venv/bin/pip install flask pywhispercpp av numpy certifi || { echo "  Installationen misslyckades."; read -r; exit 1; }
  echo "  Klart."
fi

# Säkerställ rotcertifikat även om miljön redan fanns sedan tidigare.
./.venv/bin/python -c "import certifi" >/dev/null 2>&1 || ./.venv/bin/pip install certifi >/dev/null 2>&1

echo
echo "  Startar… ett fönster öppnas i webbläsaren."
echo "  (Första transkriberingen laddar ner modellen en gång, ca 1,1 GB.)"
echo
./.venv/bin/python app.py
