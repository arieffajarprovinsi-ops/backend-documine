import asyncio
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel

from app.ai.contracts import AITaskType, DashboardGenerationPlan
from app.ai.gateway import ai_gateway
from app.core.hardening import analytics_report_cache
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.models.schemas import DashboardWidget, EChartsOption
from app.services.duckdb_service import WorkspaceDuckDBSandbox
from app.services.analytics_report_service import build_dynamic_analytics_report

router = APIRouter(prefix="/analytics", tags=["analytics"])


class AnalyticsReportBuildRequest(BaseModel):
    scope_mode: Literal["single_file", "all_files", "folder"] = "all_files"
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    folder_name: Optional[str] = None
    focus_prompt: Optional[str] = None
    force_refresh: bool = False


class DashboardWidgetPatchRequest(BaseModel):
    title: Optional[str] = None
    is_pinned: Optional[bool] = None
    display_order: Optional[int] = None


class DashboardGenerateRequest(BaseModel):
    prompt: Optional[str] = None
    document_id: Optional[str] = None
    chart_type: Optional[str] = None


def _detect_corpus_profile(
    docs: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> str:
    """
    Classify the user's repository into Corpus Profiles 0-3 (DOC-14 Section 2, PR-033, PR-034).
    """
    if not docs:
        return "empty_corpus"
    has_quant = len(tables) >= 1 or len(metrics) >= 2
    has_qual = any(d.get("category") in ("academic_paper", "contract_legal", "sop_policy", "book_chapter") for d in docs)
    if has_quant and has_qual:
        return "mixed_corpus"
    if has_quant:
        return "quantitative_corpus"
    return "qualitative_corpus"


def _build_corpus_intelligence_widget(docs: List[Dict[str, Any]], workspace_id: str) -> DashboardWidget:
    """
    Deterministic Corpus Intelligence Widget for Qualitative Corpora (DOC-14, PR-034, PR-038).
    Never invents fake sales/Carnot numbers when documents contain only text/qualitative data.
    Plots real Topic & Document Category distribution across the user's repository.
    """
    topic_counts: Dict[str, int] = {}
    for d in docs:
        for t in (d.get("main_topics") or []):
            if t:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        cat = d.get("category") or "general_document"
        if not (d.get("main_topics") or []):
            topic_counts[cat] = topic_counts.get(cat, 0) + 1

    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    if not sorted_topics:
        sorted_topics = [("Uploaded Documents", len(docs))]

    x_labels = [k for k, _ in sorted_topics]
    y_values = [v for _, v in sorted_topics]

    return DashboardWidget(
        id="widget_corpus_topics",
        workspace_id=workspace_id,
        title="Document Topic & Category Distribution",
        subtitle="Deterministic breakdown across your active workspace documents",
        description=(
            f"Showing distribution across {len(sorted_topics)} primary topics from {len(docs)} documents in your repository. "
            "Upload documents with numeric tables or metrics to enable quantitative comparison charts."
        ),
        widget_type="echarts_chart",
        chart_type="bar",
        status="active",
        is_auto_generated=True,
        unit_or_currency="documents",
        sql_query="SELECT topic_name, COUNT(doc_id) AS document_count FROM workspace_topics GROUP BY topic_name ORDER BY document_count DESC LIMIT 8",
        echarts_spec=EChartsOption(
            title={"text": "Document Topic Distribution", "left": "center"},
            tooltip={"trigger": "axis"},
            xAxis={"type": "category", "data": x_labels},
            yAxis={"type": "value", "name": "Documents"},
            series=[{"name": "Documents", "type": "bar", "data": y_values}],
        ),
        insight_summary=(
            f"Most prominent topic in your workspace is '{x_labels[0]}' ({y_values[0]} documents)."
        ),
        source_doc_ids=[str(d["id"]) for d in docs[:10]],
        source_citations=[
            {
                "document_id": str(d["id"]),
                "document_title": d.get("title"),
                "page_number": 1,
            }
            for d in docs[:5]
        ],
    )


def _build_default_quantitative_sql(
    sandbox: WorkspaceDuckDBSandbox,
    tables: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> tuple[str, str, str, str]:
    """
    Build a deterministic read-only DuckDB SELECT query over the workspace's actual tables or metrics.
    Returns (sql_query, x_col, y_col, unit_label).
    """
    if metrics:
        # Group by homogenous unit (most frequent unit_or_currency in workspace_metrics)
        unit_counts: Dict[str, int] = {}
        for m in metrics:
            u = str(m.get("unit_or_currency") or "count")
            unit_counts[u] = unit_counts.get(u, 0) + 1
        primary_unit = sorted(unit_counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        safe_unit = primary_unit.replace("'", "''")
        sql = (
            f"SELECT metric_name AS label, ROUND(AVG(numeric_value), 2) AS value, "
            f"unit_or_currency AS unit "
            f"FROM workspace_metrics "
            f"WHERE unit_or_currency = '{safe_unit}' "
            f"GROUP BY metric_name, unit_or_currency "
            f"ORDER BY value DESC LIMIT 12"
        )
        return sql, "label", "value", primary_unit

    # Otherwise query the first structured table with a numeric column
    for idx, tbl in enumerate(tables):
        cols = tbl.get("columns_schema") or []
        if not cols:
            continue
        raw_name = tbl.get("table_name") or f"table_{idx + 1}"
        tname = next(
            (n for n in sandbox.registered_table_names if n.endswith(raw_name.lower()) or n == raw_name.lower()),
            sandbox.registered_table_names[-1],
        )
        cat_col = cols[0]["name"]
        num_col = next((c["name"] for c in cols if c.get("inferred_type") == "numeric"), cols[-1]["name"])
        unit = next((c.get("unit_or_currency") for c in cols if c["name"] == num_col), None) or "value"
        sql = (
            f'SELECT CAST("{cat_col}" AS VARCHAR) AS label, '
            f'TRY_CAST("{num_col}" AS DOUBLE) AS value '
            f"FROM {tname} "
            f'WHERE TRY_CAST("{num_col}" AS DOUBLE) IS NOT NULL '
            f"LIMIT 20"
        )
        return sql, "label", "value", unit

    raise ValueError("No quantitative tables or metrics available.")


@router.post("/build-report")
async def build_analytics_report(
    req: AnalyticsReportBuildRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    On-demand Dynamic Analytics Report Builder:
    - Single File (`single_file`): deep multi-chart or qualitative insight breakdown of 1 document
    - Folder (`folder`): synthesis across all documents in a repository folder
    - All Files (`all_files`): full workspace synthesis across every document
    """
    try:
        return await build_dynamic_analytics_report(
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            scope_mode=req.scope_mode,
            document_id=req.document_id,
            document_ids=req.document_ids,
            folder_name=req.folder_name,
            focus_prompt=req.focus_prompt,
            force_refresh=req.force_refresh,
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.get("/dashboard")
async def get_personalized_dashboard(
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Pillar B: Personalized Adaptive Dashboard Home (DOC-14, PR-033..PR-039).
    Parallelized off the event loop and cached cleanly per workspace.
    """
    cache_key = f"an:{ctx.workspace_id}:dashboard_home"
    cached = analytics_report_cache.get(cache_key)
    if cached is not None:
        return cached

    docs, tables, metrics, relationships, saved_widgets = await asyncio.gather(
        asyncio.to_thread(supabase_db.list_workspace_documents, ctx.workspace_id),
        asyncio.to_thread(supabase_db.get_workspace_tables, ctx.workspace_id),
        asyncio.to_thread(supabase_db.get_workspace_metrics, ctx.workspace_id),
        asyncio.to_thread(supabase_db.list_workspace_relationships, ctx.workspace_id),
        asyncio.to_thread(supabase_db.list_dashboard_widgets, ctx.workspace_id),
    )

    profile = _detect_corpus_profile(docs, tables, metrics)

    auto_widgets: List[Dict[str, Any]] = []
    if docs:
        auto_widgets.append(_build_corpus_intelligence_widget(docs, ctx.workspace_id).model_dump())

    def _run_duckdb_quant() -> Optional[Dict[str, Any]]:
        with WorkspaceDuckDBSandbox(ctx.workspace_id, tables, metrics) as sandbox:
            sql, x_col, y_col, unit_label = _build_default_quantitative_sql(sandbox, tables, metrics)
            exec_res = sandbox.execute_validated_query(sql)
            rows = exec_res["rows"]
            if not rows:
                return None
            x_vals = [str(r.get(x_col, "")) for r in rows]
            y_vals = [float(r.get(y_col) or 0.0) for r in rows]
            return DashboardWidget(
                id="widget_auto_quantitative",
                workspace_id=ctx.workspace_id,
                title=f"Verified Metrics Overview ({unit_label})",
                subtitle="Computed deterministically via DuckDB from your documents",
                description=f"Showing {len(rows)} verified metrics in {unit_label}.",
                widget_type="echarts_chart" if len(rows) >= 2 else "kpi_card_group",
                chart_type="bar" if len(rows) >= 2 else "kpi",
                unit_or_currency=unit_label,
                sql_query=exec_res["sql_query"],
                echarts_spec=EChartsOption(
                    title={"text": f"Metrics ({unit_label})", "left": "center"},
                    tooltip={"trigger": "axis"},
                    dataset={"source": rows},
                    xAxis={"type": "category", "data": x_vals},
                    yAxis={"type": "value", "name": unit_label},
                    series=[{"name": unit_label, "type": "bar", "data": y_vals}],
                ),
                kpi_payload={"rows": rows, "unit": unit_label},
                insight_summary=(
                    f"Peak value recorded at '{x_vals[0]}' with {y_vals[0]} {unit_label}."
                ),
                source_doc_ids=sorted({str(m.get("doc_id")) for m in metrics if m.get("doc_id")})
                or [str(d["id"]) for d in docs[:3]],
            ).model_dump()

    if tables or len(metrics) >= 1:
        try:
            quant_dump = await asyncio.to_thread(_run_duckdb_quant)
            if quant_dump:
                auto_widgets.insert(0, quant_dump)
        except Exception as e:
            print(f"[Dashboard] Auto quantitative widget notice: {e}")

    result = {
        "workspace_id": ctx.workspace_id,
        "detected_corpus_profile": profile,
        "corpus_profile": profile,
        "corpus_telemetry": {
            "total_documents": len(docs),
            "total_pages": sum(int(d.get("page_count") or 1) for d in docs),
            "total_tables": len(tables),
            "total_verified_metrics": len(metrics),
            "total_relationships": len(relationships),
        },
        "auto_widgets": auto_widgets,
        "saved_widgets": saved_widgets,
        "widgets": auto_widgets + saved_widgets,
    }
    analytics_report_cache.set(cache_key, result, ttl_seconds=45.0)
    return result


@router.post("/generate-dashboard", response_model=DashboardWidget)
async def generate_dynamic_dashboard(
    prompt: Optional[str] = Query(
        default=None,
        description="Natural language request for data visualization",
    ),
    document_id: Optional[str] = Query(default=None),
    chart_type: str = Query(default="bar"),
    body: Optional[DashboardGenerateRequest] = Body(default=None),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Stage A-D Personalized Dashboard Widget Generator (DOC-15, ADR-008, ADR-011, ADR-012).
    Uses gpt-6.1-terra to plan read-only DuckDB SQL and ECharts spec, then executes and validates
    deterministically via the 5-Gate WorkspaceDuckDBSandbox.
    """
    if body:
        prompt = prompt or body.prompt
        document_id = document_id or body.document_id
        chart_type = body.chart_type or chart_type
    prompt = prompt or "Show a comparison of key metrics from my documents"
    doc_ids = [document_id] if document_id else None
    docs = (
        [d for d in [supabase_db.get_workspace_document(ctx.workspace_id, document_id)] if d]
        if document_id
        else supabase_db.list_workspace_documents(ctx.workspace_id)
    )
    if document_id and not docs:
        raise HTTPException(status_code=404, detail="Document not found in your workspace.")

    tables = supabase_db.get_workspace_tables(ctx.workspace_id, doc_ids=doc_ids)
    metrics = supabase_db.get_workspace_metrics(ctx.workspace_id, doc_ids=doc_ids)

    # Honest handling when the repository/document has zero structured tables and < 1 metric (PR-034, PR-038)
    if not tables and len(metrics) < 1:
        if docs:
            return _build_corpus_intelligence_widget(docs, ctx.workspace_id)
        return DashboardWidget(
            id="widget_insufficient_data",
            workspace_id=ctx.workspace_id,
            title="No Quantitative Data Available",
            subtitle="Upload a PDF or spreadsheet with tables or numeric metrics to generate charts",
            description="Your workspace does not yet contain structured tables or numeric metrics that can be queried.",
            widget_type="insufficient_data_card",
            chart_type="info",
            sql_query=None,
            echarts_spec=EChartsOption(series=[]),
            insight_summary="No numeric series were found to visualize.",
        )

    with WorkspaceDuckDBSandbox(ctx.workspace_id, tables, metrics) as sandbox:
        schema_context = sandbox.get_schema_context_prompt()
        planned_sql: Optional[str] = None
        x_col = "label"
        y_col = "value"
        unit_label = "value"
        widget_title = "Document Quantitative Analysis"
        widget_subtitle = prompt[:90]
        narrative = ""
        chosen_chart_type = chart_type if chart_type in ("bar", "line", "pie", "scatter", "area") else "bar"
        explicit_chart_requested = body is not None and body.chart_type in ("bar", "line", "pie", "scatter", "area")

        if ai_gateway.is_openai_configured:
            try:
                plan = await ai_gateway.generate_structured(
                    task_type=AITaskType.DASHBOARD_SQL_PLANNING,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    instructions=(
                        "You are the DocuMind Analytical SQL Planner. Given the available DuckDB "
                        "table schemas and sample rows, output a read-only SELECT query that answers the user's "
                        "analytical request. NEVER mix incompatible units or currencies in a single SUM/AVG without "
                        "filtering or grouping by unit_or_currency. Do NOT output hardcoded data arrays."
                    ),
                    input_blocks=(
                        f"AVAILABLE DUCKDB SCHEMA:\n{schema_context}\n\n"
                        f"PREFERRED CHART TYPE: {chosen_chart_type}\n"
                        f"USER REQUEST: {prompt}"
                    ),
                    response_model=DashboardGenerationPlan,
                    scope_key=f"dash:{document_id or 'all'}",
                )
                if not plan.data_sufficient or not plan.sql_query:
                    return DashboardWidget(
                        workspace_id=ctx.workspace_id,
                        title=plan.widget_title or "Data Kuantitatif Tidak Mencukupi",
                        subtitle=plan.widget_subtitle,
                        description=plan.insufficient_reason or plan.narrative_insight,
                        widget_type="insufficient_data_card",
                        chart_type="info",
                        sql_query=None,
                        echarts_spec=EChartsOption(series=[]),
                        insight_summary=plan.insufficient_reason or plan.narrative_insight,
                    )
                planned_sql = plan.sql_query
                widget_title = plan.widget_title
                widget_subtitle = plan.widget_subtitle
                narrative = plan.narrative_insight
                if plan.series_specs:
                    first_s = plan.series_specs[0]
                    x_col = first_s.x_dimension
                    y_col = first_s.y_dimension
                    unit_label = first_s.unit_label
                    if not explicit_chart_requested:
                        chosen_chart_type = first_s.type
            except Exception as plan_err:
                print(f"[Analytics] SQL planner fallback to deterministic SQL: {plan_err}")

        if not planned_sql:
            planned_sql, x_col, y_col, unit_label = _build_default_quantitative_sql(sandbox, tables, metrics)

        try:
            exec_res = sandbox.execute_validated_query(planned_sql)
        except ValueError as sec_err:
            raise HTTPException(status_code=400, detail=str(sec_err)) from sec_err
        except Exception:
            # 1-step deterministic fallback query if AI SQL had a column alias mismatch
            planned_sql, x_col, y_col, unit_label = _build_default_quantitative_sql(sandbox, tables, metrics)
            exec_res = sandbox.execute_validated_query(planned_sql)

        rows = exec_res["rows"]
        cols = exec_res["columns"]
        if rows and x_col not in rows[0] and len(cols) >= 1:
            x_col = cols[0]
        if rows and y_col not in rows[0] and len(cols) >= 2:
            y_col = cols[1]

        x_vals = [str(r.get(x_col, "")) for r in rows]
        y_vals: List[float] = []
        for r in rows:
            try:
                y_vals.append(round(float(r.get(y_col) or 0.0), 4))
            except (ValueError, TypeError):
                y_vals.append(0.0)

        if not narrative:
            narrative = (
                f"Dihitung secara deterministik oleh DuckDB ({len(rows)} baris hasil) dalam satuan {unit_label}."
            )

        # Gate 5: Provenance & Source Citations attachment
        source_doc_ids = sorted(
            {str(m.get("doc_id")) for m in metrics if m.get("doc_id")}
            | {str(t.get("doc_id")) for t in tables if t.get("doc_id")}
        )
        source_citations = [
            {
                "document_id": str(d["id"]),
                "document_title": d.get("title"),
                "page_number": 1,
            }
            for d in docs
            if str(d["id"]) in source_doc_ids
        ]

        if chosen_chart_type == "pie":
            pie_data = [
                {"name": x_vals[i], "value": y_vals[i]}
                for i in range(min(len(x_vals), len(y_vals)))
            ]
            echarts_option = EChartsOption(
                title={"text": widget_title, "left": "center"},
                tooltip={"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                legend={"orient": "vertical", "left": "left"},
                dataset={"source": rows},
                series=[
                    {
                        "name": unit_label,
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": True,
                        "data": pie_data,
                    }
                ],
            )
        else:
            echarts_option = EChartsOption(
                title={"text": widget_title, "left": "center"},
                tooltip={"trigger": "axis"},
                dataset={"source": rows},
                xAxis={"type": "category", "data": x_vals},
                yAxis={"type": "value", "name": unit_label},
                series=[
                    {
                        "name": unit_label,
                        "type": chosen_chart_type if chosen_chart_type != "area" else "line",
                        "data": y_vals,
                        "smooth": chosen_chart_type in ("line", "area"),
                        **({"areaStyle": {"opacity": 0.25}} if chosen_chart_type == "area" else {}),
                    }
                ],
            )

        widget_model = DashboardWidget(
            workspace_id=ctx.workspace_id,
            title=widget_title,
            subtitle=widget_subtitle,
            description=narrative,
            widget_type="echarts_chart" if len(rows) >= 2 else "kpi_card_group",
            chart_type=chosen_chart_type if len(rows) >= 2 else "kpi",
            unit_or_currency=unit_label,
            sql_query=exec_res["sql_query"],
            echarts_spec=echarts_option,
            kpi_payload={"rows": rows, "unit": unit_label},
            insight_summary=narrative,
            source_citations=source_citations,
            source_doc_ids=source_doc_ids,
            source_table_ids=[str(t["id"]) for t in tables if t.get("id")],
        )

        saved_row = supabase_db.save_dashboard_widget(ctx.workspace_id, widget_model.model_dump())
        widget_model.id = str(saved_row["id"])
        return widget_model


@router.patch("/widgets/{widget_id}")
async def update_dashboard_widget(
    widget_id: str,
    req: DashboardWidgetPatchRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    widgets = supabase_db.list_dashboard_widgets(ctx.workspace_id)
    target = next((w for w in widgets if str(w.get("id")) == widget_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Widget not found in your workspace.")

    if req.title is not None:
        target["title"] = req.title.strip()
    if req.is_pinned is not None:
        target["is_pinned"] = req.is_pinned
    if req.display_order is not None:
        target["display_order"] = req.display_order

    saved = supabase_db.save_dashboard_widget(ctx.workspace_id, target)
    return saved


@router.delete("/widgets/{widget_id}")
async def delete_dashboard_widget(
    widget_id: str,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    deleted = supabase_db.delete_dashboard_widget(ctx.workspace_id, widget_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Widget not found in your workspace.")
    return {"deleted": True, "id": widget_id}
