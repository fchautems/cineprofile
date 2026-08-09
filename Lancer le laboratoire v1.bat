@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CineProfile - Laboratoire v1

rem Utilise l'environnement existant de CineProfile.
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [Laboratoire v1] Le fichier .venv\Scripts\python.exe est introuvable.
    echo Verifie que ce BAT se trouve bien a la racine de F:\CineProfile.
    goto :error
)

echo.
echo [Laboratoire v1] Environnement Python detecte.
echo [Laboratoire v1] Verification de la version locale...
".venv\Scripts\python.exe" -c "import cineprofile; import cineprofile.arena; print('[Laboratoire v1] CineProfile ' + cineprofile.__version__)"
if errorlevel 1 goto :repair
goto :run

:repair
echo.
echo [Laboratoire v1] Mise a jour du paquet local dans le venv existant...
".venv\Scripts\python.exe" -m pip install -e . --no-build-isolation --disable-pip-version-check
if errorlevel 1 goto :error

:run
echo.
echo [Laboratoire v1] L'application et la base source ne seront pas modifiees.
echo [Laboratoire v1] Le premier calcul semantique peut durer plusieurs minutes.
".venv\Scripts\python.exe" run_arena.py
if errorlevel 1 goto :error

echo.
echo [Laboratoire v1] Le rapport se trouve dans data\logs.
pause
goto :end

:error
echo.
echo [Laboratoire v1] Impossible de terminer le test.
echo Consulte data\logs\cineprofile.log si le fichier existe.
pause

:end
endlocal
