@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CineProfile - Laboratoire v1 - Etape 2 MovieLens

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [Laboratoire v1] Le fichier .venv\Scripts\python.exe est introuvable.
    echo Verifie que ce BAT se trouve bien a la racine de F:\CineProfile.
    goto :error
)

echo.
echo [Laboratoire v1] Verification du module MovieLens...
".venv\Scripts\python.exe" -c "import cineprofile.movielens_arena"
if errorlevel 1 goto :repair
goto :run

:repair
echo.
echo [Laboratoire v1] Mise a jour du paquet local dans le venv existant...
".venv\Scripts\python.exe" -m pip install -e . --no-build-isolation --disable-pip-version-check
if errorlevel 1 goto :error

:run
echo.
echo [Laboratoire v1] Le premier lancement peut etre long.
echo [Laboratoire v1] Environ 239 Mio seront telecharges une seule fois.
".venv\Scripts\python.exe" run_movielens_arena.py
if errorlevel 1 goto :error

echo.
echo [Laboratoire v1] Le rapport MovieLens se trouve dans data\logs.
pause
goto :end

:error
echo.
echo [Laboratoire v1] Impossible de terminer l'etape 2.
pause

:end
endlocal
