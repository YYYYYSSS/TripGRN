@echo off
REM Run DeepFGRN Fig.1 helpers inside WSL env deepfgrn_wsl (TF GPU).
REM Usage:
REM   tools\wsl_run_deepfgrn.bat smoke
REM   tools\wsl_run_deepfgrn.bat run --gse GSE156632 --fold 1
REM   tools\wsl_run_deepfgrn.bat summarize

setlocal
if "%~1"=="" (
  echo Usage: %~nx0 smoke^|run^|summarize [args...]
  exit /b 2
)

wsl -d Ubuntu-22.04 -- bash -lc "source /home/yhr/activate_deepfgrn_wsl.sh && python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_kegg.py %*"
exit /b %ERRORLEVEL%
