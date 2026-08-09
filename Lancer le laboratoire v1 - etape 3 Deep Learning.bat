@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CineProfile - Laboratoire v1 - Etape 3 Deep Learning

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [Laboratoire v1] Le fichier .venv\Scripts\python.exe est introuvable.
    echo Verifie que ce BAT se trouve bien a la racine de F:\CineProfile.
    goto :error
)

echo.
echo [Laboratoire v1] Verification du module semantique...
".venv\Scripts\python.exe" -c "import cineprofile.semantic_arena"
if errorlevel 1 goto :repair
goto :run

:repair
echo.
echo [Laboratoire v1] Mise a jour du paquet local dans le venv existant...
".venv\Scripts\python.exe" -m pip install -e . --no-build-isolation --disable-pip-version-check
if errorlevel 1 goto :error

:run
echo.
echo [Laboratoire v1] Cette etape compare MiniLM, E5 Large et BGE-M3.
echo [Laboratoire v1] Jusqu'a environ 4.5 Gio seront telecharges une seule fois.
echo [Laboratoire v1] Le calcul local peut durer longtemps sur CPU.
set "CINEPROFILE_SEMANTIC_DEVICE=cpu"
".venv\Scripts\python.exe" run_semantic_arena.py
if errorlevel 1 goto :error

echo.
echo [Laboratoire v1] Le rapport semantique se trouve dans data\logs.
pause
goto :end

:error
echo.
echo [Laboratoire v1] Impossible de terminer l'etape 3.
pause

:end
endlocal

