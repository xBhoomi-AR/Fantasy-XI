param(
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
)

$ErrorActionPreference = "Stop"

Get-Content -LiteralPath ".env" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}

New-Item -ItemType Directory -Force -Path "reports" | Out-Null

function Invoke-PsqlCsv {
    param(
        [string]$Sql,
        [string]$OutFile
    )
    & $PsqlPath $env:DATABASE_URL -X -v ON_ERROR_STOP=1 --csv -q -c "BEGIN READ ONLY; SET LOCAL statement_timeout = '60s'; $Sql; ROLLBACK;" | Out-File -FilePath $OutFile -Encoding utf8
}

Invoke-PsqlCsv @"
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name
"@ "reports\supabase_tables.csv"

Invoke-PsqlCsv @"
SELECT table_schema, table_name, ordinal_position, column_name, data_type, udt_name, is_nullable
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name, ordinal_position
"@ "reports\supabase_columns.csv"

Invoke-PsqlCsv @"
SELECT tc.table_schema, tc.table_name, tc.constraint_type, kcu.column_name,
       ccu.table_schema AS foreign_table_schema,
       ccu.table_name AS foreign_table_name,
       ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
LEFT JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name
 AND ccu.table_schema = tc.table_schema
WHERE tc.table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY tc.table_schema, tc.table_name, tc.constraint_type, kcu.column_name
"@ "reports\supabase_constraints.csv"

$tables = Import-Csv "reports\supabase_tables.csv" | Where-Object { $_.table_type -in @("BASE TABLE", "VIEW") }
$countRows = New-Object System.Collections.Generic.List[object]
foreach ($table in $tables) {
    $schema = $table.table_schema.Replace('"', '""')
    $name = $table.table_name.Replace('"', '""')
    $sql = "SELECT '$($table.table_schema)' AS table_schema, '$($table.table_name)' AS table_name, COUNT(*) AS row_count FROM ""$schema"".""$name"""
    $tmp = & $PsqlPath $env:DATABASE_URL -X -v ON_ERROR_STOP=1 --csv -q -c "BEGIN READ ONLY; SET LOCAL statement_timeout = '60s'; $sql; ROLLBACK;"
    $tmp | Select-Object -Skip 1 | ConvertFrom-Csv -Header table_schema,table_name,row_count | ForEach-Object { $countRows.Add($_) }
}
$countRows | Export-Csv -Path "reports\supabase_row_counts.csv" -NoTypeInformation -Encoding utf8

$columns = Import-Csv "reports\supabase_columns.csv"
$signalRows = New-Object System.Collections.Generic.List[object]
$tokens = @("season", "gameweek", "gw", "round", "event", "player", "element", "team", "fixture", "opponent", "status", "chance", "selected", "ownership", "price", "understat", "xg", "xa", "points", "minutes")
foreach ($col in $columns) {
    $lower = $col.column_name.ToLowerInvariant()
    $isSignal = $false
    foreach ($token in $tokens) {
        if ($lower.Contains($token)) { $isSignal = $true; break }
    }
    if (-not $isSignal) { continue }

    $schema = $col.table_schema.Replace('"', '""')
    $name = $col.table_name.Replace('"', '""')
    $column = $col.column_name.Replace('"', '""')
    $sql = @"
SELECT '$($col.table_schema)' AS table_schema,
       '$($col.table_name)' AS table_name,
       '$($col.column_name)' AS column_name,
       COUNT(*) FILTER (WHERE "$column" IS NOT NULL) AS non_null,
       COUNT(DISTINCT "$column") AS distinct_count,
       MIN("$column"::text) AS min_text,
       MAX("$column"::text) AS max_text
FROM "$schema"."$name"
"@
    try {
        $tmp = & $PsqlPath $env:DATABASE_URL -X -v ON_ERROR_STOP=1 --csv -q -c "BEGIN READ ONLY; SET LOCAL statement_timeout = '60s'; $sql; ROLLBACK;"
        $tmp | Select-Object -Skip 1 | ConvertFrom-Csv -Header table_schema,table_name,column_name,non_null,distinct_count,min_text,max_text | ForEach-Object { $signalRows.Add($_) }
    }
    catch {
        $signalRows.Add([PSCustomObject]@{
            table_schema = $col.table_schema
            table_name = $col.table_name
            column_name = $col.column_name
            non_null = ""
            distinct_count = ""
            min_text = "ERROR"
            max_text = $_.Exception.Message
        })
    }
}
$signalRows | Export-Csv -Path "reports\supabase_signal_columns.csv" -NoTypeInformation -Encoding utf8

"Supabase audit CSV files written to reports\." | Write-Output
