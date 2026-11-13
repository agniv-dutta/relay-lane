# prod.ps1 - build frontend and serve everything from FastAPI
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location "$Root\frontend"
npm run build
Set-Location $Root
python -m uvicorn relay.api:app --host 127.0.0.1 --port 8000
