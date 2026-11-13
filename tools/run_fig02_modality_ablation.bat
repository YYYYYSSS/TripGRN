@echo off
REM Fig.2 GSE253718 modality ablation (7 stages x 10 folds)
REM Protocol: conv + no-gene-disjoint + raw STRING (see docs/RUN_PROTOCOL_FIG02.md)
setlocal EnableExtensions
cd /d "%~dp0.."
call conda activate project_GRN
if errorlevel 1 ( echo [ERROR] conda activate project_GRN failed & exit /b 1 )

set "LAUNCHER=fig02_modality_ablation\scripts\run_fig02_modality_ablation.py"
set "LOG=fig02_modality_ablation\experiments\run_fig02_modality_ablation.log"
if not exist "fig02_modality_ablation\experiments" mkdir "fig02_modality_ablation\experiments"

>>"%LOG%" echo ==== Fig.2 start %DATE% %TIME% ====
echo [fig02] smoke
python -u "%LAUNCHER%" smoke
if errorlevel 1 goto :fail

echo [fig02] status
python -u "%LAUNCHER%" status

echo [fig02] run ALL stages / ALL folds (conv, no gene-disjoint, raw PPI)
python -u "%LAUNCHER%" run --stage ALL --fold ALL
if errorlevel 1 goto :fail

echo [fig02] summarize
python -u "%LAUNCHER%" summarize
if errorlevel 1 goto :fail

echo [fig02] ALL DONE %DATE% %TIME%
>>"%LOG%" echo ALL DONE %DATE% %TIME%
exit /b 0

:fail
echo [fig02] FAILED rc=%ERRORLEVEL%
>>"%LOG%" echo FAILED rc=%ERRORLEVEL% %DATE% %TIME%
exit /b 1
