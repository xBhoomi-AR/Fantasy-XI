from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

from urllib.parse import urlparse

import pg8000.native
from dotenv import load_dotenv

from fpl_predictor.paths import REPORTS_DIR, ensure_dirs


def serialise(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def connect():
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        parsed = urlparse(dsn)
        return pg8000.native.Connection(
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/"),
            ssl_context=True,
        )
    return pg8000.native.Connection(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT") or 5432),
        ssl_context=True,
    )


def fetchall(cur, query, params=None):
    return cur.run(query, **(params or {}))


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def audit_table(cur, schema: str, table: str, columns: list[dict]) -> dict:
    qname = f"{quote_ident(schema)}.{quote_ident(table)}"
    row_count = cur.run(f"SELECT COUNT(*) AS n FROM {qname}")[0]["n"]

    interesting = [
        c["column_name"]
        for c in columns
        if any(
            token in c["column_name"].lower()
            for token in [
                "season",
                "gameweek",
                "gw",
                "round",
                "event",
                "player",
                "element",
                "team",
                "fixture",
                "opponent",
                "status",
                "chance",
                "selected",
                "ownership",
                "price",
                "understat",
                "xg",
                "xa",
            ]
        )
    ]

    summaries = {}
    for col in interesting[:24]:
        try:
            row = cur.run(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE {quote_ident(col)} IS NOT NULL) AS non_null,
                    COUNT(DISTINCT {quote_ident(col)}) AS distinct_count,
                    MIN({quote_ident(col)}::text) AS min_text,
                    MAX({quote_ident(col)}::text) AS max_text
                FROM {qname}
                """
            )[0]
            summaries[col] = {k: serialise(v) for k, v in row.items()}
        except Exception as exc:
            cur.run("ROLLBACK")
            cur.run("BEGIN READ ONLY")
            summaries[col] = {"error": str(exc)}

    sample_rows = []
    if row_count:
        safe_cols = [quote_ident(c["column_name"]) for c in columns[:20]]
        try:
            rows = cur.run(f"SELECT {', '.join(safe_cols)} FROM {qname} LIMIT 3")
            sample_rows = [{k: serialise(v) for k, v in row.items()} for row in rows]
        except Exception as exc:
            sample_rows = [{"error": str(exc)}]
            cur.run("ROLLBACK")
            cur.run("BEGIN READ ONLY")

    return {
        "schema": schema,
        "table": table,
        "row_count": row_count,
        "columns": columns,
        "interesting_column_summaries": summaries,
        "sample_rows": sample_rows,
    }


def main() -> None:
    ensure_dirs()
    report_path = REPORTS_DIR / "supabase_audit.md"
    json_path = REPORTS_DIR / "supabase_audit.json"

    conn = connect()
    try:
        conn.run("BEGIN READ ONLY")
        conn.run("SET LOCAL statement_timeout = '45s'")

        tables = fetchall(
                conn,
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
                """,
            )
        all_columns = fetchall(
                conn,
                """
                SELECT table_schema, table_name, column_name, ordinal_position, data_type, udt_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name, ordinal_position
                """,
            )
        constraints = fetchall(
                conn,
                """
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
                ORDER BY tc.table_schema, tc.table_name, tc.constraint_type
                """,
            )

        columns_by_table = {}
        for col in all_columns:
            columns_by_table.setdefault((col["table_schema"], col["table_name"]), []).append(col)

        table_audits = []
        for table in tables:
            if table["table_type"] not in {"BASE TABLE", "VIEW"}:
                continue
            table_audits.append(
                audit_table(
                    conn,
                    table["table_schema"],
                    table["table_name"],
                    columns_by_table.get((table["table_schema"], table["table_name"]), []),
                )
            )

        audit = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tables": tables,
            "constraints": constraints,
            "table_audits": table_audits,
        }
    finally:
        conn.close()

    json_path.write_text(json.dumps(audit, indent=2, default=serialise), encoding="utf-8")

    lines = [
        "# Supabase Read-Only Schema/Data Audit",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "Credentials were loaded locally from `.env`; no credential values are stored in this report.",
        "",
        "## Tables",
        "",
        "| Schema | Table | Type | Rows | Useful signals spotted |",
        "|---|---|---:|---:|---|",
    ]
    for item in table_audits:
        signals = ", ".join(item["interesting_column_summaries"].keys()) or "-"
        lines.append(f"| {item['schema']} | {item['table']} | table/view | {item['row_count']} | {signals[:220]} |")

    lines.extend(["", "## Columns"])
    for item in table_audits:
        lines.append("")
        lines.append(f"### {item['schema']}.{item['table']} ({item['row_count']} rows)")
        lines.append("")
        lines.append("| Column | Type | Nullable |")
        lines.append("|---|---|---|")
        for col in item["columns"]:
            lines.append(f"| {col['column_name']} | {col['data_type']} | {col['is_nullable']} |")
        if item["interesting_column_summaries"]:
            lines.append("")
            lines.append("Interesting column coverage:")
            for col, summary in item["interesting_column_summaries"].items():
                lines.append(
                    f"- `{col}`: non-null={summary.get('non_null')}, distinct={summary.get('distinct_count')}, "
                    f"min={summary.get('min_text')}, max={summary.get('max_text')}"
                )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
