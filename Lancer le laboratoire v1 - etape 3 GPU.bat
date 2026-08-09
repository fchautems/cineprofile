@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CineProfile - Laboratoire v1 - Etape 3 GPU

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [Laboratoire v1] Le venv principal est introuvable.
    echo Verifie que ce BAT se trouve bien a la racine de F:\CineProfile.
    goto :error
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo.
    echo [Laboratoire v1] Le pilote NVIDIA ou nvidia-smi est introuvable.
    echo Mets a jour le pilote NVIDIA avant d'utiliser le calcul GPU.
    goto :error
)

echo.
echo [Laboratoire v1] Carte NVIDIA detectee :
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if not exist ".venv-gpu\Scripts\python.exe" (
    echo.
    echo [Laboratoire v1] Creation de l'environnement GPU separe...
    ".venv\Scripts\python.exe" -m venv ".venv-gpu"
    if errorlevel 1 goto :error
)

if not exist ".venv-gpu\cineprofile_gpu_cuda12_ready" (
    echo.
    echo [Laboratoire v1] Installation locale de CUDA 12, cuDNN et FastEmbed GPU...
    echo [Laboratoire v1] Cette preparation n'est faite qu'une seule fois.
    ".venv-gpu\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-gpu.txt
    if errorlevel 1 goto :error
    ".venv-gpu\Scripts\python.exe" -m pip install --disable-pip-version-check -e . --no-deps
    if errorlevel 1 goto :error
    type nul > ".venv-gpu\cineprofile_gpu_cuda12_ready"
)

set "CINEPROFILE_SEMANTIC_DEVICE=cuda"
echo.
echo [Laboratoire v1] Verification de CUDAExecutionProvider...
".venv-gpu\Scripts\python.exe" -c "from cineprofile.semantic import embedding_execution_providers; print('[Laboratoire v1] ONNX : ' + ', '.join(embedding_execution_providers()))"
if errorlevel 1 goto :error

echo.
echo [Laboratoire v1] Lancement du calcul sur la GTX 1070.
echo [Laboratoire v1] Les modeles seront charges un par un dans les 8 Gio de VRAM.
".venv-gpu\Scripts\python.exe" run_semantic_arena.py
if errorlevel 1 goto :error

echo.
echo [Laboratoire v1] Le rapport semantique se trouve dans data\logs.
pause
goto :end

:error
echo.
echo [Laboratoire v1] Impossible de terminer l'etape 3 sur GPU.
echo [Laboratoire v1] Le venv principal de CineProfile n'a pas ete modifie.
pause

:end
endlocal
