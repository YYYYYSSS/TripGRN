@echo off
REM scMGATGRN DoRothEA — Machine B (other GPU box)
REM GSEs: GSE176031 GSE232237 GSE253718 GSE288479  (40 folds)
REM Prerequisite sync (from Machine A repo):
REM   1) code: grn\  fig01_pancancer_benchmark\scripts\  (or whole repo)
REM   2) GRN:  shared\data\network\grn\DoRothEA_human_ABC_activation_inhibition_with_symbols.csv
REM   3) expr+cv for EACH GSE below:
REM        fig01_pancancer_benchmark\data\expression\epithelial\{GSE}\
REM          values.npy  genes.npy  cells.npy  cv_splits_10fold.json
REM   4) external: E:\project\project_scMGATGRN\scMGATGRN-main\Code  (or set SCMGATGRN_CODE)
REM   5) conda env project_GRN with same deps (torch+cuda, etc.)
REM After finish: copy back
REM   fig01_pancancer_benchmark\experiments\dorothea_abc_epithelial_10cv\scMGATGRN\{GSE}\
REM onto Machine A, then summarize.
setlocal EnableExtensions
cd /d "%~dp0.."
call conda activate project_GRN
if errorlevel 1 ( echo [ERROR] conda activate project_GRN failed & exit /b 1 )

set "LAUNCHER=fig01_pancancer_benchmark\scripts\run_fig01_binary_baselines.py"
set "LOG=fig01_pancancer_benchmark\experiments\run_scmgat_dorothea_machine_B.log"
>>"%LOG%" echo ==== Machine B start %DATE% %TIME% ====

for %%G in (GSE176031 GSE232237 GSE253718 GSE288479) do (
  echo ========== scMGATGRN DoRothEA %%G ==========
  >>"%LOG%" echo ========== %%G ========== %DATE% %TIME%
  python -u "%LAUNCHER%" run --method scMGATGRN --network dorothea --gse %%G --fold ALL
  if errorlevel 1 goto :fail
)

echo [Machine B] ALL DONE %DATE% %TIME%
>>"%LOG%" echo [Machine B] ALL DONE %DATE% %TIME%
exit /b 0

:fail
echo [Machine B] FAILED rc=%ERRORLEVEL%
>>"%LOG%" echo FAILED rc=%ERRORLEVEL% %DATE% %TIME%
exit /b 1
