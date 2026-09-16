param(
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
)

$ErrorActionPreference = "Stop"

Get-Content -LiteralPath ".env" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null

$tables = @(
    "fixtures",
    "gameweeks",
    "player_identity_history",
    "player_market_history",
    "player_match_stats",
    "player_understat_history",
    "players",
    "team_history",
    "teams"
)

foreach ($table in $tables) {
    $out = (Resolve-Path "data\raw").Path + "\$table.csv"
    $sql = "COPY (SELECT * FROM processed.$table) TO STDOUT WITH CSV HEADER"
    & $PsqlPath $env:DATABASE_URL -X -v ON_ERROR_STOP=1 -q -c $sql | Out-File -FilePath $out -Encoding utf8
    "Exported processed.$table to data\raw\$table.csv" | Write-Output
}
