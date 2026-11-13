@echo off
REM Fig.3 TCGA bulk cross-cancer (12 cancers x 10 folds)
REM ORIGINAL protocol: conv encoder + no gene-disjoint + CDS/TSS + raw STRING
REM see docs/RUN_PROTOCOL_FIG03.md
setlocal EnableExtensions
cd /d "%~dp0.."
call conda activate project_GRN
if errorlevel 1 ( echo [ERROR] conda activate project_GRN failed & exit /b 1 )

set "LAUNCHER=fig03_bulk_cross_cancer\scripts\run_fig03_bulk_cross_cancer.py"
set "LOG=fig03_bulk_cross_cancer\experiments\run_fig03_bulk_cross_cancer.log"
if not exist "fig03_bulk_cross_cancer\experiments" mkdir "fig03_bulk_cross_cancer\experiments"

>>"%LOG%" echo ==== Fig.3 start %DATE% %TIME% ====
echo [fig03] smoke
python -u "%LAUNCHER%" smoke
if errorlevel 1 goto :fail

echo [fig03] status
python -u "%LAUNCHER%" status

echo [fig03] run ALL cancers / ALL folds (original: conv, no-gene-disjoint)
python -u "%LAUNCHER%" run --cancer ALL --fold ALL --expression-encoder conv
if errorlevel 1 goto :fail

echo [fig03] summarize
python -u "%LAUNCHER%" summarize
if errorlevel 1 goto :fail

echo [fig03] ALL DONE %DATE% %TIME%
>>"%LOG%" echo ALL DONE %DATE% %TIME%
exit /b 0

:fail
echo [fig03] FAILED rc=%ERRORLEVEL%
>>"%LOG%" echo FAILED rc=%ERRORLEVEL% %DATE% %TIME%
exit /b 1
