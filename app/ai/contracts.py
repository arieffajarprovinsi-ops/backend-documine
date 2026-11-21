from enum import Enum
from typing import TypeVar, Type, Optional, AsyncIterator, Protocol, List, Literal, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class AITaskType(str, Enum):
    """
    Semantic task taxonomy for DocuMind's Provider-Flexible AIGateway (ADR-008, ADR-017).
    Domain services request an AITaskType rather than hardcoding provider/model names.
    """

    EMBEDDING_DOCUMENT = "embedding_document"
    EMBEDDING_QUERY = "embedding_query"
    VISION_OCR_AND_TABLES = "vision_ocr_and_tables"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    QUERY_REWRITE_AND_ROUTING = "query_rewrite_and_routing"
    GROUNDED_CHAT_STANDARD = "grounded_chat_standard"
    GROUNDED_CHAT_DEEP = "grounded_chat_deep"
    DASHBOARD_SQL_PLANNING = "dashboard_sql_planning"
    CROSS_DOC_SYNTHESIS = "cross_doc_synthesis"
    STUDY_QUIZ_GENERATION = "study_quiz_generation"
    AUDIO_BRIEFING_SCRIPT = "audio_briefing_script"


class ModelRoutePolicy(BaseModel):
    task_type: AITaskType
    primary_provider: str = "openai"
    primary_model: str
    fallback_provider: str = "openai"
    fallback_model: str
    service_tier: Literal["auto", "flex", "batch"] = "auto"
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None
    dimensions: Optional[int] = None
    store: bool = False


class AIUsageTelemetry(BaseModel):
    workspace_id: str
    user_id: Optional[str] = None
    task_type: AITaskType
    provider: str
    model: str
    service_tier: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0


class LLMProviderAdapter(Protocol):
    provider_name: str

    async def generate_structured(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]],
        response_model: Type[T],
        reasoning_effort: Optional[str] = None,
        service_tier: str = "auto",
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> tuple[T, Dict[str, Any]]:
        ...

    async def generate_text(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]],
        reasoning_effort: Optional[str] = None,
        service_tier: str = "auto",
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        ...

    async def stream_text(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]],
        reasoning_effort: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        ...


class EmbeddingProviderAdapter(Protocol):
    provider_name: str

    async def embed_texts(
        self,
        *,
        model: str,
        texts: List[str],
        dimensions: int = 1536,
    ) -> tuple[List[List[float]], Dict[str, Any]]:
        ...


# =============================================================================
# Core Pydantic Structured Output Schemas (DOC-10, DOC-15, DOC-17, ADR-009, ADR-012)
# Invariant 3 & 4: LLMs NEVER invent bounding-box floats or chart data arrays.
# =============================================================================


class OCRRegionBlock(BaseModel):
    text: str = Field(description="Transcribed paragraph or section text from the scanned page")
    approx_y0: float = Field(default=0.1, description="Coarse normalized top position (0.0 to 1.0)")
    approx_y1: float = Field(default=0.9, description="Coarse normalized bottom position (0.0 to 1.0)")


class ExtractedTableColumnSchema(BaseModel):
    name: str
    inferred_type: Literal["numeric", "varchar", "date"] = "varchar"
    unit_or_currency: Optional[str] = None


class ExtractedTableSchema(BaseModel):
    table_title: str
    columns: List[ExtractedTableColumnSchema]
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class OCRPageExtraction(BaseModel):
    page_number: int
    blocks: List[OCRRegionBlock] = Field(default_factory=list)
    visual_chart_descriptions: List[str] = Field(default_factory=list)
    extracted_tables: List[ExtractedTableSchema] = Field(default_factory=list)


class ExtractedEntityItem(BaseModel):
    name: str
    entity_type: Literal[
        "organization", "person", "location", "technology", "metric_name", "regulation", "other"
    ] = "other"
    canonical_name: str = Field(description="Lowercase normalized entity identifier for cross-document matching")


class ExtractedKPIItem(BaseModel):
    metric_name: str = Field(description="Clear label, e.g. 'Q3 2026 Total Revenue' or 'Peak Coulombic Efficiency'")
    numeric_value: float
    unit_or_currency: str = Field(description="Standardized unit, e.g. 'USD', 'IDR', '%', 'mAh/g', 'count'")
    period_or_context: Optional[str] = Field(
        default=None, description="Time period or experimental condition, e.g. 'Q3 2026' or '45C'"
    )
    source_chunk_id: str = Field(description="Exact chunk_id where this number is stated")
    verbatim_quote: str = Field(description="Exact substring from the chunk containing the number")


class KeyFindingItem(BaseModel):
    headline: str
    explanation: str
    source_chunk_id: str


class DocumentUnderstandingPayload(BaseModel):
    inferred_title: str
    document_type: Literal[
        "academic_paper",
        "financial_report",
        "lab_report",
        "contract_legal",
        "sop_policy",
        "presentation_slides",
        "spreadsheet_dataset",
        "book_chapter",
        "general_document",
    ] = "general_document"
    primary_domain: str = Field(description="High-level domain, e.g. 'Battery Electrochemistry', 'Corporate Finance'")
    language_code: Literal["id", "en", "mixed", "other"] = "id"
    executive_summary: str = Field(
        description="Comprehensive 3-5 sentence summary of the document's core purpose and findings"
    )
    main_topics: List[str] = Field(default_factory=list, description="5 to 10 normalized topic tags")
    named_entities: List[ExtractedEntityItem] = Field(default_factory=list)
    key_dates: List[str] = Field(default_factory=list)
    key_findings: List[KeyFindingItem] = Field(default_factory=list)
    extracted_kpis: List[ExtractedKPIItem] = Field(default_factory=list)


class EChartsSeriesSpec(BaseModel):
    type: Literal["bar", "line", "pie", "scatter", "area"] = "bar"
    name: str
    x_dimension: str = Field(description="Column name from SQL SELECT result for X axis / category")
    y_dimension: str = Field(description="Column name from SQL SELECT result for Y axis / numeric value")
    unit_label: str = Field(description="Unit or currency symbol for tooltips and axis, e.g. 'USD ($M)', '%'")


class DashboardGenerationPlan(BaseModel):
    data_sufficient: bool = Field(
        description="True ONLY if the available schema contains valid comparable data to answer the query."
    )
    insufficient_reason: Optional[str] = None
    unit_consistency_verified: bool = Field(
        description="True if all aggregated numbers share the same unit/currency or are grouped separately by unit."
    )
    discrepancy_warning: Optional[str] = None
    widget_title: str
    widget_subtitle: str
    widget_type: Literal[
        "echarts_chart", "kpi_card_group", "comparison_table", "insufficient_data_card"
    ] = "echarts_chart"
    sql_query: Optional[str] = Field(
        default=None,
        description="Read-only DuckDB SELECT query. Must select explicit column aliases matching x_dimension and y_dimension.",
    )
    series_specs: List[EChartsSeriesSpec] = Field(default_factory=list)
    narrative_insight: str = Field(
        description="Concise 2-sentence analytical explanation of what the query measures, with [cite:CHUNK_ID] references."
    )
    source_table_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)


class SynthesisComparisonRow(BaseModel):
    dimension_name: str
    per_document_findings: Dict[str, str] = Field(
        description="Mapping of doc_id -> concise finding with [cite:CHUNK_ID]"
    )
    consensus_or_divergence: str


class ContradictionAlertItem(BaseModel):
    topic_or_metric: str
    doc_a_id: str
    doc_a_claim: str
    doc_a_chunk_id: str
    doc_b_id: str
    doc_b_claim: str
    doc_b_chunk_id: str
    severity: Literal["high", "medium", "low"] = "medium"
    explanation: str


class CrossDocumentSynthesisResponse(BaseModel):
    synthesis_headline: str
    executive_synthesis: str = Field(description="Comprehensive cross-document narrative with [cite:CHUNK_ID] tags")
    key_agreements: List[str] = Field(default_factory=list)
    comparison_matrix: List[SynthesisComparisonRow] = Field(default_factory=list)
    contradiction_alerts: List[ContradictionAlertItem] = Field(default_factory=list)
    emergent_takeaways: List[str] = Field(default_factory=list)


class QuizQuestionSchema(BaseModel):
    id: str
    question_type: Literal["multiple_choice", "open_ended"] = "multiple_choice"
    question: str
    options: List[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str
    source_chunk_id: str = Field(
        description="Exact chunk_id from which this question was derived (backend resolves page & bounding_boxes)"
    )


class QuizAssessmentPayload(BaseModel):
    quiz_title: str
    questions: List[QuizQuestionSchema]


class GraphAutoLinkItem(BaseModel):
    source_doc_id: str
    target_doc_id: str
    relationship_type: Literal[
        "shared_topics",
        "shared_entities",
        "semantic_similarity",
        "explicit_reference",
        "shared_metric_lineage",
        "contradiction_detected",
        "composite_related",
    ] = "composite_related"
    confidence: float = Field(default=0.78, ge=0.45, le=0.99)
    explanation: str = Field(
        description="Concise human-readable reason citing specific shared themes, schedules, entities, or metrics in both documents."
    )


class GraphAutoLinkPlan(BaseModel):
    summary: str = Field(description="High-level summary of discovered workspace connections")
    links: List[GraphAutoLinkItem] = Field(default_factory=list)


class DialogueSegmentSchema(BaseModel):
    speaker: Literal["Alex", "Sarah", "Host (Budi)", "Expert (Siti)"]
    text: str
    source_chunk_id: Optional[str] = None


class AudioBriefingScriptPayload(BaseModel):
    title: str
    duration_estimate_seconds: int = 90
    dialogue: List[DialogueSegmentSchema]
