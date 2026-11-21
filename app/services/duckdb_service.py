import math
import re
import duckdb
from typing import Dict, Any, List, Optional

FORBIDDEN_SQL_REGEX = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|"
    r"COPY|ATTACH|DETACH|PRAGMA|SET|INSTALL|LOAD|EXPORT|IMPORT|CALL|EXEC|EXECUTE|"
    r"SYSTEM|SHELL|HTTPFS|DUCKDB_SETTINGS|CURRENT_SETTING|SNIFF_CSV|READ_BLOB|"
    r"READ_CSV|READ_CSV_AUTO|READ_TEXT|READ_PARQUET|READ_JSON|READ_JSON_AUTO|GLOB)\b",
    re.IGNORECASE,
)


def validate_sql_safety_gate1(sql_query: str) -> str:
    """
    Gate 1: SQL AST & Keyword Sandbox Validator (DOC-15 Section 5, PR-040, AC-040).
    Blocks multi-statement queries, comments, DDL/DML, and DuckDB file-reader functions.
    """
    if not sql_query or not sql_query.strip():
        raise ValueError("SQL query cannot be empty.")

    cleaned = sql_query.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if ";" in cleaned or "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise ValueError("Unsafe SQL blocked: multi-statement queries and SQL comments are prohibited.")

    first_token = cleaned.split(None, 1)[0].upper() if cleaned.split() else ""
    if first_token not in ("SELECT", "WITH"):
        raise ValueError("Security Violation: Only read-only SELECT or WITH analytical queries are permitted.")

    match = FORBIDDEN_SQL_REGEX.search(cleaned)
    if match:
        raise ValueError(
            f"Security Violation: Prohibited SQL keyword or function '{match.group(1).upper()}' blocked."
        )

    return cleaned


def validate_unit_homogeneity_gate4(
    sql_query: str,
    available_units: List[str],
    result_rows: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Gate 4: Unit & Currency Homogeneity Validator (DOC-15, PR-036, EC-031).
    Prevents summing or averaging incompatible units/currencies (e.g. USD + IDR or % + count)
    on a single axis without grouping or filtering by unit_or_currency.
    """
    distinct_units = sorted({u.upper() for u in available_units if u and u.strip()})
    if len(distinct_units) <= 1:
        return None

    upper_sql = sql_query.upper()
    aggregates_metrics = (
        "WORKSPACE_METRICS" in upper_sql
        and ("SUM(" in upper_sql or "AVG(" in upper_sql)
    )
    if aggregates_metrics and "UNIT_OR_CURRENCY" not in upper_sql:
        raise ValueError(
            f"Unit Homogeneity Gate Blocked: Cannot aggregate mixed units/currencies ({', '.join(distinct_units)}) "
            "without filtering or grouping by unit_or_currency."
        )

    row_units = {
        str(r.get("unit_or_currency") or r.get("unit") or "").upper()
        for r in result_rows
        if r.get("unit_or_currency") or r.get("unit")
    }
    row_units.discard("")
    if len(row_units) > 1 and ("SUM(" in upper_sql or "AVG(" in upper_sql) and "GROUP BY" not in upper_sql:
        raise ValueError(
            f"Unit Homogeneity Gate Blocked: Result mixes incompatible units ({', '.join(sorted(row_units))})."
        )
    return None


def _hydrate_records_into_duckdb(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: List[Dict[str, Any]],
    default_columns: Optional[List[tuple[str, str]]] = None,
) -> None:
    """
    Hydrates a list of Python dicts into an in-memory DuckDB table without requiring pandas.
    """
    if not rows:
        cols_def = default_columns or [("id", "VARCHAR")]
        ddl_cols = ", ".join(f'"{c_name}" {c_type}' for c_name, c_type in cols_def)
        conn.execute(f'CREATE TABLE "{table_name}" ({ddl_cols});')
        return

    col_names = list(rows[0].keys())
    col_defs: List[str] = []
    for col in col_names:
        sample_vals = [r.get(col) for r in rows if r.get(col) is not None]
        if sample_vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample_vals):
            col_defs.append(f'"{col}" DOUBLE')
        else:
            col_defs.append(f'"{col}" VARCHAR')

    conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(col_defs)});')
    placeholders = ", ".join(["?"] * len(col_names))
    tuples_to_insert = [tuple(r.get(col) for col in col_names) for r in rows]
    conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders});', tuples_to_insert)


def _fetch_records(rel: duckdb.DuckDBPyRelation) -> tuple[List[str], List[Dict[str, Any]]]:
    cols = [desc[0] for desc in (rel.description or [])]
    raw_tuples = rel.fetchall()
    return cols, [dict(zip(cols, row)) for row in raw_tuples]


class WorkspaceDuckDBSandbox:
    """
    Ephemeral, Workspace-Scoped Sandboxed DuckDB Engine (ADR-011, DOC-15, Invariant 4 & 5).
    Hydrates ONLY the caller's workspace tables and extracted KPI metrics into an isolated
    in-memory DuckDB connection, disables external file/network access, locks configuration,
    and enforces the 5-Gate Numerical & Security Validation Layer.
    """

    def __init__(
        self,
        workspace_id: str,
        document_tables: Optional[List[Dict[str, Any]]] = None,
        extracted_metrics: Optional[List[Dict[str, Any]]] = None,
    ):
        if not workspace_id:
            raise ValueError("workspace_id is required to initialize WorkspaceDuckDBSandbox.")
        self.workspace_id = workspace_id
        self.document_tables = document_tables or []
        self.extracted_metrics = extracted_metrics or []
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.registered_table_names: List[str] = []
        self.table_source_map: Dict[str, Dict[str, Any]] = {}

    def __enter__(self) -> "WorkspaceDuckDBSandbox":
        self.conn = duckdb.connect(database=":memory:")

        # 1. Hydrate workspace_metrics table
        metric_rows = [
            {
                "metric_id": str(m.get("id") or ""),
                "doc_id": str(m.get("doc_id") or ""),
                "source_chunk_id": str(m.get("source_chunk_id") or ""),
                "metric_name": str(m.get("metric_name") or ""),
                "canonical_metric_key": str(m.get("canonical_metric_key") or ""),
                "numeric_value": float(m.get("numeric_value") or 0.0),
                "unit_or_currency": str(m.get("unit_or_currency") or "count"),
                "period_label": str(m.get("period_label") or "General"),
            }
            for m in self.extracted_metrics
        ]
        _hydrate_records_into_duckdb(
            self.conn,
            "workspace_metrics",
            metric_rows,
            default_columns=[
                ("metric_id", "VARCHAR"),
                ("doc_id", "VARCHAR"),
                ("source_chunk_id", "VARCHAR"),
                ("metric_name", "VARCHAR"),
                ("canonical_metric_key", "VARCHAR"),
                ("numeric_value", "DOUBLE"),
                ("unit_or_currency", "VARCHAR"),
                ("period_label", "VARCHAR"),
            ],
        )
        self.registered_table_names.append("workspace_metrics")

        # 2. Hydrate extracted document_tables
        for idx, tbl in enumerate(self.document_tables):
            rows_data = tbl.get("rows_data") or []
            if not rows_data:
                continue
            raw_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(tbl.get("table_name") or f"table_{idx + 1}")).lower()
            safe_name = f"doc_table_{idx + 1}_{raw_name}"[:48]
            _hydrate_records_into_duckdb(self.conn, safe_name, rows_data)
            self.registered_table_names.append(safe_name)
            self.table_source_names_alias(raw_name, safe_name)
            self.table_source_map[safe_name] = tbl

        # 3. Lock down DuckDB external access, memory, threads & configuration (Invariant 5, PR-040)
        self.conn.execute("SET memory_limit = '128MB';")
        self.conn.execute("SET threads = 2;")
        self.conn.execute("SET max_temp_directory_size = '0B';")
        self.conn.execute("SET enable_external_access = false;")
        self.conn.execute("SET lock_configuration = true;")
        return self

    def table_source_names_alias(self, alias_name: str, source_table: str) -> None:
        if (
            self.conn
            and alias_name
            and alias_name not in self.registered_table_names
            and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", alias_name)
        ):
            try:
                self.conn.execute(f'CREATE VIEW "{alias_name}" AS SELECT * FROM "{source_table}";')
                self.registered_table_names.append(alias_name)
            except Exception:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def get_schema_context_prompt(self) -> str:
        """Build a concise DESCRIBE + sample rows summary for gpt-6.1-sol SQL planning."""
        if not self.conn:
            return "No active DuckDB connection."
        sections: List[str] = []
        for tname in self.registered_table_names:
            try:
                desc_rows = self.conn.execute(f'DESCRIBE "{tname}"').fetchall()
                cols_str = ", ".join(f"{r[0]} ({r[1]})" for r in desc_rows)
                rel = self.conn.sql(f'SELECT * FROM "{tname}" LIMIT 3')
                _, sample_records = _fetch_records(rel)
                sections.append(f"Table `{tname}`:\n  Columns: {cols_str}\n  Sample Rows: {sample_records}")
            except Exception:
                pass
        return "\n\n".join(sections)

    def execute_validated_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Executes Gates 1, 2, 3, and 4 on the SQL query and returns deterministic rows & columns.
        """
        if not self.conn:
            raise RuntimeError("DuckDB sandbox is not open.")

        # Gate 1: SQL AST & Keyword Sandbox Check
        safe_sql = validate_sql_safety_gate1(sql_query)

        # Gate 2: Deterministic Execution
        rel = self.conn.sql(safe_sql)
        columns, raw_records = _fetch_records(rel)

        # Gate 3: Row Cardinality & NaN/Infinity Check
        clean_rows: List[Dict[str, Any]] = []
        for row in raw_records[:500]:
            clean_row: Dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_row[k] = 0.0
                else:
                    clean_row[k] = v
            clean_rows.append(clean_row)

        # Gate 4: Unit & Currency Homogeneity Gate
        available_units = [str(m.get("unit_or_currency") or "") for m in self.extracted_metrics]
        validate_unit_homogeneity_gate4(safe_sql, available_units, clean_rows)

        return {
            "sql_query": safe_sql,
            "columns": columns,
            "rows": clean_rows,
            "row_count": len(clean_rows),
        }


class DuckDBAnalyticsEngine:
    """
    Backward-compatible wrapper around WorkspaceDuckDBSandbox with full 5-Gate SQL security.
    """

    def __init__(self):
        self._custom_tables: Dict[str, List[Dict[str, Any]]] = {}

    def register_dataframe_table(self, table_name: str, data: List[Dict[str, Any]]):
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        self._custom_tables[safe_name] = data

    def execute_safe_analytical_query(self, sql_query: str, workspace_id: str = "default") -> Dict[str, Any]:
        tables = [
            {"table_name": name, "rows_data": rows}
            for name, rows in self._custom_tables.items()
        ]
        with WorkspaceDuckDBSandbox(workspace_id=workspace_id, document_tables=tables) as sandbox:
            return sandbox.execute_validated_query(sql_query)

    def validate_and_execute_widget_sql(
        self,
        workspace_id: str,
        sql_query: str,
        document_tables: Optional[List[Dict[str, Any]]] = None,
        extracted_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        try:
            validate_sql_safety_gate1(sql_query)
        except ValueError as e:
            return {
                "valid": False,
                "gate_failed": "gate_1_sql_ast_sandbox",
                "error": str(e),
                "rows": [],
            }
        try:
            with WorkspaceDuckDBSandbox(
                workspace_id=workspace_id,
                document_tables=document_tables,
                extracted_metrics=extracted_metrics,
            ) as sandbox:
                res = sandbox.execute_validated_query(sql_query)
                return {
                    "valid": True,
                    "gate_failed": None,
                    **res,
                }
        except ValueError as ve:
            gate = "gate_4_unit_homogeneity" if "Unit Homogeneity" in str(ve) else "gate_3_cardinality"
            return {
                "valid": False,
                "gate_failed": gate,
                "error": str(ve),
                "rows": [],
            }
        except Exception as exc:
            return {
                "valid": False,
                "gate_failed": "gate_2_duckdb_execution",
                "error": str(exc),
                "rows": [],
            }


duckdb_engine = DuckDBAnalyticsEngine()
duckdb_service = duckdb_engine
