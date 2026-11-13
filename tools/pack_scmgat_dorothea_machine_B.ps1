# Pack data + minimal code for Machine B scMGATGRN DoRothEA split.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\pack_scmgat_dorothea_machine_B.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Out = Join-Path $Root "tools\_pack_scmgat_dorothea_B"
$Gses = @("GSE176031", "GSE232237", "GSE253718", "GSE288479")

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# Code slices
$copyDirs = @(
  "grn",
  "fig01_pancancer_benchmark\scripts"
)
foreach ($d in $copyDirs) {
  $src = Join-Path $Root $d
  $dst = Join-Path $Out $d
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  robocopy $src $dst /E /XD __pycache__ .git experiments /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}

# GRN
$grnSrc = "shared\data\network\grn\DoRothEA_human_ABC_activation_inhibition_with_symbols.csv"
$grnDst = Join-Path $Out $grnSrc
New-Item -ItemType Directory -Force -Path (Split-Path $grnDst) | Out-Null
Copy-Item (Join-Path $Root $grnSrc) $grnDst -Force

# Expression + CV per GSE
foreach ($g in $Gses) {
  $src = Join-Path $Root "fig01_pancancer_benchmark\data\expression\epithelial\$g"
  $dst = Join-Path $Out "fig01_pancancer_benchmark\data\expression\epithelial\$g"
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  foreach ($f in @("values.npy", "genes.npy", "cells.npy", "cv_splits_10fold.json")) {
    Copy-Item (Join-Path $src $f) (Join-Path $dst $f) -Force
  }
}

# Helper bats + README
New-Item -ItemType Directory -Force -Path (Join-Path $Out "tools") | Out-Null
Copy-Item (Join-Path $Root "tools\run_scmgatgrn_dorothea_machine_B.bat") (Join-Path $Out "tools\run_scmgatgrn_dorothea_machine_B.bat") -Force
@"
Machine B pack for scMGATGRN DoRothEA
GSEs: $($Gses -join ', ')

On Machine B:
1. Merge this folder into a TripGRN重构 clone (or use as repo root if grn/ present).
2. Install/activate conda env project_GRN (torch cuda).
3. Place scMGATGRN Code at E:\project\project_scMGATGRN\scMGATGRN-main\Code
   OR set env SCMGATGRN_CODE to that Code directory.
4. From repo root: tools\run_scmgatgrn_dorothea_machine_B.bat
5. After done, copy back:
   fig01_pancancer_benchmark\experiments\dorothea_abc_epithelial_10cv\scMGATGRN\{GSE}\
   for each of: $($Gses -join ', ')
"@ | Set-Content (Join-Path $Out "README_MACHINE_B.txt") -Encoding UTF8

Write-Host "Packed -> $Out"
Write-Host "Zip it (optional): Compress-Archive -Path `"$Out\*`" -DestinationPath `"$Root\tools\scmgat_dorothea_machine_B.zip`""
