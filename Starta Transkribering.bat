@echo off
rem Dubbelklicka denna fil for att starta KB Transkribering pa Windows.
setlocal enabledelayedexpansion
cd /d "%~dp0"
cls
echo --------------------------------
echo   KB Transkribering
echo --------------------------------

rem Hitta Python 3 (py-launchern foredras, annars python pa PATH).
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo   Python 3 saknas. Installera harifran ^(bocka i "Add Python to PATH"^):
  echo   https://www.python.org/downloads/
  echo   Kor sedan denna fil igen.
  echo.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo.
  echo   Forbereder appen. Detta gors bara en gang och kan ta nagra minuter...
  echo   ^(Talaridentifiering kraver nagra storre bibliotek - ha talamod.^)
  %PY% -m venv .venv
  if errorlevel 1 (
    echo   Kunde inte skapa miljo.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   Installationen misslyckades. Om felet ror ett bygge kan du behova
    echo   "Microsoft C++ Build Tools":
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    pause
    exit /b 1
  )
  echo   Klart.
)

rem Sakerstall rotcertifikat aven om miljon redan fanns sedan tidigare.
".venv\Scripts\python.exe" -c "import certifi" >nul 2>&1 || ".venv\Scripts\python.exe" -m pip install certifi >nul 2>&1

rem Sakerstall Pillow (felsokningslaget klipper ut skarmbilder) for aldre miljoer.
".venv\Scripts\python.exe" -c "import PIL" >nul 2>&1 || ".venv\Scripts\python.exe" -m pip install pillow >nul 2>&1

rem Sakerstall talaridentifieringens bibliotek for miljoer som skapades innan funktionen fanns.
".venv\Scripts\python.exe" -c "import pyannote.audio" >nul 2>&1
if errorlevel 1 (
  echo   Forbereder talaridentifiering ^(engangsnedladdning, kan ta nagra minuter^)...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt >nul 2>&1
)

echo.
echo   Startar... ett fonster oppnas i webblasaren.
echo   ^(Forsta transkriberingen laddar ner modellen en gang, ca 1,1 GB.^)
echo.
".venv\Scripts\python.exe" app.py
