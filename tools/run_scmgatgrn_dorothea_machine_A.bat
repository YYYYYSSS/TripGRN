@echo off
REM scMGATGRN DoRothEA — Machine A (this box)
REM GSEs: GSE132465 GSE148071 GSE156632 GSE159115  (40 folds)
REM Machine B takes: GSE176031 GSE232237 GSE253718 GSE288479
REM Resume-safe: skips folds with unseen_samples_test_metrics.json
setlocal EnableExtensions
cd /d "%~dp0.."
call conda activate project_GRN
if errorlevel 1 ( echo [ERROR] conda activate project_GRN failed & exit /b 1 )

set "LAUNCHER=fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py"
set "LOG=fig01_pancancer_benchmark\experiments\run_scmgat_dorothea_machine_A.log"
>>"%LOG%" echo ==== Machine A start %DATE% %TIME% ====

for %%G in (GSE132465 GSE148071 GSE156632 GSE159115) do (
  echo ========== scMGATGRN DoRothEA %%G ==========
  >>"%LOG%" echo ========== %%G ========== %DATE% %TIME%
  python -u "%LAUNCHER%" run --method scMGATGRN --network dorothea --gse %%G --fold ALL
  if errorlevel 1 goto :fail
)

echo [Machine A] ALL DONE %DATE% %TIME%
>>"%LOG%" echo [Machine A] ALL DONE %DATE% %TIME%
exit /b 0

:fail
echo [Machine A] FAILED rc=%ERRORLEVEL%
>>"%LOG%" echo FAILED rc=%ERRORLEVEL% %DATE% %TIME%
exit /b 1
