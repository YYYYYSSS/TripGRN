@echo off
REM Run DeepFGRN Fig.1 DoRothEA-ABC helpers inside WSL env deepfgrn_wsl (TF GPU).
REM Usage:
REM   tools\wsl_run_deepfgrn_dorothea.bat smoke
REM   tools\wsl_run_deepfgrn_dorothea.bat run --gse GSE156632 --fold 1
REM   tools\wsl_run_deepfgrn_dorothea.bat run --gse ALL --fold ALL
REM   tools\wsl_run_deepfgrn_dorothea.bat summarize

setlocal
if "%~1"=="" (
  echo Usage: %~nx0 smoke^|run^|summarize [args...]
  exit /b 2
)

wsl -d Ubuntu-22.04 -- bash -lc "source /home/yhr/activate_deepfgrn_wsl.sh && python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_dorothea.py %*"
exit /b %ERRORLEVEL%
