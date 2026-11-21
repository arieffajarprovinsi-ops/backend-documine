from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum


class CognitiveDepth(str, Enum):
    ELI5 = "eli5"
    HIGH_SCHOOL = "high_school"
    ACADEMIC = "academic"


class BoundingBox(BaseModel):
    """
    Normalized coordinate system (0.0 - 1.0 relative to page width and height).
    Top-left is (0, 0), compatible with web DOM viewport and PDF.js canvas overlay.
    """

    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)
    page_number: int = Field(ge=1, description="1-indexed page number")

    @property
    def x0(self) -> float:
        return self.left

    @property
    def y0(self) -> float:
        return self.top

    @property
    def x1(self) -> float:
        return min(1.0, round(self.left + self.width, 4))

    @property
    def y1(self) -> float:
        return min(1.0, round(self.top + self.height, 4))


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    document_id: str
    doc_id: Optional[str] = None
    workspace_id: Optional[str] = None
    chunk_index: int
    content: str
    text_content: Optional[str] = None
    page_number: int
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    token_count: Optional[int] = None
    has_table: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    id: str
    workspace_id: Optional[str] = None
    title: str
    original_filename: Optional[str] = None
    file_type: str = "pdf"
    mime_type: Optional[str] = "application/pdf"
    page_count: int = 1
    file_size_bytes: int = 0
    processing_status: Literal["pending", "queued", "processing", "ready", "partial_ready", "completed", "failed"] = (
        "pending"
    )
    status: Optional[str] = "queued"
    category: Optional[str] = "general_document"
    primary_domain: Optional[str] = None
    language_code: Optional[str] = "id"
    summary: Optional[str] = None
    key_findings: List[Dict[str, Any]] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    user_tags: List[str] = Field(default_factory=list)
    has_tables: bool = False
    table_count: int = 0
    metric_count: int = 0
    chunk_count: Optional[int] = None
    is_duplicate: bool = False
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    processed_at: Optional[str] = None


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    user_tags: Optional[List[str]] = None


# --- ECharts & Personalized Dashboard Schemas (Pillar B - DOC-14, DOC-15) ---
class EChartsSeries(BaseModel):
    name: Optional[str] = None
    type: Literal["bar", "line", "pie", "radar", "scatter", "area"] = "bar"
    data: List[Any] = Field(default_factory=list)
    smooth: Optional[bool] = False


class EChartsOption(BaseModel):
    title: Optional[Dict[str, Any]] = None
    tooltip: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"trigger": "axis"})
    legend: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"top": "bottom"})
    dataset: Optional[Dict[str, Any]] = None
    xAxis: Optional[Dict[str, Any]] = None
    yAxis: Optional[Dict[str, Any]] = None
    series: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardWidget(BaseModel):
    id: Optional[str] = None
    workspace_id: Optional[str] = None
    title: str
    subtitle: Optional[str] = None
    description: str
    widget_type: Literal["echarts_chart", "kpi_card_group", "comparison_table", "insufficient_data_card"] = (
        "echarts_chart"
    )
    chart_type: Literal["bar", "line", "pie", "radar", "timeline", "scatter", "area", "kpi", "info"] = "bar"
    status: Literal["active", "stale_source_deleted", "needs_refresh"] = "active"
    is_auto_generated: bool = True
    is_pinned: bool = False
    unit_or_currency: Optional[str] = None
    sql_query: Optional[str] = None
    echarts_spec: EChartsOption
    kpi_payload: Dict[str, Any] = Field(default_factory=dict)
    insight_summary: str = Field(description="Actionable explanation derived from the data")
    discrepancy_warning: Optional[str] = None
    source_citations: List[Dict[str, Any]] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)
    source_table_ids: List[str] = Field(default_factory=list)


# --- Interactive Quiz Schemas (P2 Studio Extension) ---
class QuizOption(BaseModel):
    label: str  # "A", "B", "C", "D"
    text: str


class QuizQuestion(BaseModel):
    question_id: str
    question_text: str
    options: List[QuizOption]
    correct_option: str  # "A", "B", "C", "D"
    explanation: str
    source_chunk_id: Optional[str] = None
    source_page: int
    proof_quote: str = Field(description="Exact sentence from document proving this answer")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)


class QuizResponse(BaseModel):
    document_id: str
    title: str
    questions: List[QuizQuestion]


# --- Chat, Search & Citations Schemas (Pillar C - DOC-12, DOC-13) ---
class ChatCitation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    document_title: Optional[str] = None
    page_number: int
    quote: str
    bounding_boxes: List[BoundingBox]


class ChatRequest(BaseModel):
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    scope_mode: Optional[Literal["single_doc", "multi_doc", "repository"]] = None
    deep_analysis: bool = False
    message: str
    depth: CognitiveDepth = CognitiveDepth.HIGH_SCHOOL
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    answer: str
    retrieval_status: Literal["grounded", "insufficient_evidence", "overview_fallback"] = "grounded"
    model_used: Optional[str] = None
    citations: List[ChatCitation] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)


class HybridSearchRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    match_count: int = Field(default=10, ge=1, le=50)
    min_vector_similarity: float = Field(default=0.28, ge=0.0, le=1.0)
