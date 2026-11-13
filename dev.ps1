# dev.ps1 - start stand-in, relay API, and Vite dev server
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# stand-in on 8001
$env:STANDIN_BASE_URL = "http://localhost:8001"
Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","standin.server:app","--host","127.0.0.1","--port","8001" -WorkingDirectory $Root
Start-Sleep -Seconds 2

# relay API on 8000
Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","relay.api:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root
Start-Sleep -Seconds 2

Set-Location "$Root\frontend"
npm run dev
