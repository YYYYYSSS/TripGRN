@echo off
REM Fig.1 remaining binary baselines (Trip/Deep KEGG+DO and DeepTGI-KEGG already done).
REM Protocol (keep consistent with prior Fig.1 agreement):
REM   - 9 tissue GSEs only (exclude GSE145825)
REM   - shared epithelial cv_splits_10fold.json; cell split 5:2:3; NO cell downsample
REM   - genes = current gold-standard GRN-mapped set (KEGG / DoRothEA); --max-genes 0
REM   - scMGATGRN / GENELink: zero-pad shorter splits for equal GAT input_dim
REM Order: summarize DeepTGI-KEGG -> scMGAT/GENELink KEGG -> DeepTGI/scMGAT/GENELink DoRothEA
REM
REM Usage:
REM   tools\run_fig01_remaining_baselines.bat
REM Resume after shutdown / Ctrl+C: run the SAME bat again.
REM   Folds with unseen_samples_test_metrics.json are skipped.
REM   The fold killed mid-run has no metrics and restarts from epoch 1.

setlocal EnableExtensions
cd /d "%~dp0.."
if errorlevel 1 (
  echo [ERROR] cannot cd to repo root
  exit /b 1
)

set "LOGDIR=fig01_pancancer_benchmark\experiments"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOG=%LOGDIR%\run_fig01_remaining_baselines.log"
set "LAUNCHER=fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py"

call conda activate project_GRN
if errorlevel 1 (
  echo [ERROR] conda activate project_GRN failed
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not on PATH after activating project_GRN
  exit /b 1
)

>>"%LOG%" echo ============================================================
>>"%LOG%" echo Fig.1 remaining baselines  %DATE% %TIME%
>>"%LOG%" echo Repo: %CD%
>>"%LOG%" echo Log:  %LOG%
>>"%LOG%" echo Protocol: GRN-mapped genes; full CV cells; no downsample; 9 GSEs
>>"%LOG%" echo mode: resume-safe
>>"%LOG%" echo ============================================================

echo [fig01-remaining] repo=%CD%
echo [fig01-remaining] log=%LOG%
echo [fig01-remaining] env=project_GRN
echo [fig01-remaining] genes=GRN-mapped  cells=full CV  no downsample  GSE145825 excluded
echo [fig01-remaining] resume: skip completed folds; interrupted fold restarts
echo.

echo ========== checkpoint status ==========
>>"%LOG%" echo ========== checkpoint status ========== %DATE% %TIME%
python -u "%LAUNCHER%" status --method ALL --network all
if errorlevel 1 goto :fail
echo.

REM Console shows live progress; step markers also go to LOG.
REM To tee full stdout into LOG as well, re-run under: powershell -Command "& {...} *>&1 | Tee-Object -FilePath log"

echo ========== smoke KEGG ==========
>>"%LOG%" echo ========== smoke KEGG ========== %DATE% %TIME%
python -u "%LAUNCHER%" smoke --method ALL --network kegg
if errorlevel 1 goto :fail

echo ========== smoke DoRothEA ==========
>>"%LOG%" echo ========== smoke DoRothEA ========== %DATE% %TIME%
python -u "%LAUNCHER%" smoke --method ALL --network dorothea
if errorlevel 1 goto :fail

echo ========== summarize DeepTGI KEGG ==========
>>"%LOG%" echo ========== summarize DeepTGI KEGG ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method DeepTGI --network kegg --gse ALL
if errorlevel 1 goto :fail

echo ========== run scMGATGRN KEGG ALL/ALL ==========
>>"%LOG%" echo ========== run scMGATGRN KEGG ALL/ALL ========== %DATE% %TIME%
python -u "%LAUNCHER%" run --method scMGATGRN --network kegg --gse ALL --fold ALL
if errorlevel 1 goto :fail
echo ========== summarize scMGATGRN KEGG ==========
>>"%LOG%" echo ========== summarize scMGATGRN KEGG ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method scMGATGRN --network kegg --gse ALL
if errorlevel 1 goto :fail

echo ========== run GENELink KEGG ALL/ALL ==========
>>"%LOG%" echo ========== run GENELink KEGG ALL/ALL ========== %DATE% %TIME%
python -u "%LAUNCHER%" run --method GENELink --network kegg --gse ALL --fold ALL
if errorlevel 1 goto :fail
echo ========== summarize GENELink KEGG ==========
>>"%LOG%" echo ========== summarize GENELink KEGG ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method GENELink --network kegg --gse ALL
if errorlevel 1 goto :fail

echo ========== run DeepTGI DoRothEA ALL/ALL ==========
>>"%LOG%" echo ========== run DeepTGI DoRothEA ALL/ALL ========== %DATE% %TIME%
python -u "%LAUNCHER%" run --method DeepTGI --network dorothea --gse ALL --fold ALL
if errorlevel 1 goto :fail
echo ========== summarize DeepTGI DoRothEA ==========
>>"%LOG%" echo ========== summarize DeepTGI DoRothEA ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method DeepTGI --network dorothea --gse ALL
if errorlevel 1 goto :fail

echo ========== run scMGATGRN DoRothEA ALL/ALL ==========
>>"%LOG%" echo ========== run scMGATGRN DoRothEA ALL/ALL ========== %DATE% %TIME%
python -u "%LAUNCHER%" run --method scMGATGRN --network dorothea --gse ALL --fold ALL
if errorlevel 1 goto :fail
echo ========== summarize scMGATGRN DoRothEA ==========
>>"%LOG%" echo ========== summarize scMGATGRN DoRothEA ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method scMGATGRN --network dorothea --gse ALL
if errorlevel 1 goto :fail

echo ========== run GENELink DoRothEA ALL/ALL ==========
>>"%LOG%" echo ========== run GENELink DoRothEA ALL/ALL ========== %DATE% %TIME%
python -u "%LAUNCHER%" run --method GENELink --network dorothea --gse ALL --fold ALL
if errorlevel 1 goto :fail
echo ========== summarize GENELink DoRothEA ==========
>>"%LOG%" echo ========== summarize GENELink DoRothEA ========== %DATE% %TIME%
python -u "%LAUNCHER%" summarize --method GENELink --network dorothea --gse ALL
if errorlevel 1 goto :fail

echo.
echo [fig01-remaining] ALL DONE %DATE% %TIME%
>>"%LOG%" echo [fig01-remaining] ALL DONE %DATE% %TIME%
exit /b 0

:fail
echo.
echo [fig01-remaining] FAILED rc=%ERRORLEVEL%  see %LOG%
>>"%LOG%" echo [fig01-remaining] FAILED %DATE% %TIME% rc=%ERRORLEVEL%
exit /b 1
