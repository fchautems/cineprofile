@echo off
setlocal
cd /d "%~dp0"
title CineProfile

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [CineProfile] Creation de l'environnement Python...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 goto :error
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo.
    echo [CineProfile] Installation initiale des composants...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto :error
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :error
)

echo.
echo [CineProfile] Verification des composants locaux...
".venv\Scripts\python.exe" -c "import fastembed, httpx, jinja2, pandas, plotly, scipy, sklearn, streamlit; import cineprofile; print('[CineProfile] Version locale : ' + cineprofile.__version__)"
if errorlevel 1 goto :repair
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :repair
goto :launch

:repair
echo.
echo [CineProfile] Installation des composants manquants...
".venv\Scripts\python.exe" -m pip install --upgrade setuptools wheel
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -e . --disable-pip-version-check
if errorlevel 1 goto :error

:launch
echo.
echo [CineProfile] Ouverture de l'interface...
echo [CineProfile] Pour arreter l'application, ferme cette fenetre.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false
goto :end

:error
echo.
echo Impossible de demarrer CineProfile.
echo Verifie que Python 3.11 ou plus recent est installe.
pause

:end
endlocal
