from typing import Dict
from app.core.config import settings
from app.ai.contracts import AITaskType, ModelRoutePolicy


def build_default_routing_registry() -> Dict[AITaskType, ModelRoutePolicy]:
    """
    Authoritative Quality-Optimized Task Routing Table (DECIDED: ADR-006, ADR-008, ADR-017, DOC-30).
    - Embeddings: text-embedding-3-large (dimensions=1536)
    - Stage 6/7 Vision OCR & Complex Tables: gpt-6.1-terra (auto low-latency)
    - Stage 11 Single-Pass Document Understanding & KPIs: gpt-6.1-terra (auto low-latency)
    - Standard Grounded Chat Q&A: gpt-6.1-terra (auto SSE)
    - Deep / Multi-Doc Chat Q&A: gpt-6.1-sol (reasoning="medium")
    - Personalized Dashboard SQL & ECharts Planning: gpt-6.1-sol (reasoning="low")
    - Cross-Document Comparative Synthesis: gpt-6.1-sol (reasoning="medium")
    - Query Rewriting, Compaction & Audio Briefing Script: gpt-6-luna (reasoning="none")
    """
    return {
        AITaskType.EMBEDDING_DOCUMENT: ModelRoutePolicy(
            task_type=AITaskType.EMBEDDING_DOCUMENT,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_embedding_model,
            fallback_provider=settings.ai_default_provider,
            fallback_model="text-embedding-3-small",
            service_tier="auto",
            dimensions=settings.openai_embedding_dimensions,
            store=False,
        ),
        AITaskType.EMBEDDING_QUERY: ModelRoutePolicy(
            task_type=AITaskType.EMBEDDING_QUERY,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_embedding_model,
            fallback_provider=settings.ai_default_provider,
            fallback_model="text-embedding-3-small",
            service_tier="auto",
            dimensions=settings.openai_embedding_dimensions,
            store=False,
        ),
        AITaskType.VISION_OCR_AND_TABLES: ModelRoutePolicy(
            task_type=AITaskType.VISION_OCR_AND_TABLES,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_balanced,  # gpt-6.1-terra
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_deep,     # gpt-6.1-sol
            service_tier="auto",
            reasoning_effort="none",
            store=settings.openai_store_completions,
        ),
        AITaskType.DOCUMENT_UNDERSTANDING: ModelRoutePolicy(
            task_type=AITaskType.DOCUMENT_UNDERSTANDING,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_balanced,  # gpt-6.1-terra
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_fast,     # gpt-6-luna
            service_tier="auto",
            reasoning_effort="low",
            store=settings.openai_store_completions,
        ),
        AITaskType.QUERY_REWRITE_AND_ROUTING: ModelRoutePolicy(
            task_type=AITaskType.QUERY_REWRITE_AND_ROUTING,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_fast,      # gpt-6-luna
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_balanced, # gpt-6.1-terra
            service_tier="auto",
            reasoning_effort="none",
            store=settings.openai_store_completions,
        ),
        AITaskType.GROUNDED_CHAT_STANDARD: ModelRoutePolicy(
            task_type=AITaskType.GROUNDED_CHAT_STANDARD,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_balanced,  # gpt-6.1-terra
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_fast,     # gpt-6-luna
            service_tier="auto",
            reasoning_effort="low",
            store=settings.openai_store_completions,
        ),
        AITaskType.GROUNDED_CHAT_DEEP: ModelRoutePolicy(
            task_type=AITaskType.GROUNDED_CHAT_DEEP,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_deep,      # gpt-6.1-sol
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_balanced, # gpt-6.1-terra
            service_tier="auto",
            reasoning_effort="medium",
            store=settings.openai_store_completions,
        ),
        AITaskType.DASHBOARD_SQL_PLANNING: ModelRoutePolicy(
            task_type=AITaskType.DASHBOARD_SQL_PLANNING,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_deep,      # gpt-6.1-sol
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_balanced, # gpt-6.1-terra
            service_tier="auto",
            reasoning_effort="low",
            store=settings.openai_store_completions,
        ),
        AITaskType.CROSS_DOC_SYNTHESIS: ModelRoutePolicy(
            task_type=AITaskType.CROSS_DOC_SYNTHESIS,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_deep,      # gpt-6.1-sol
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_balanced, # gpt-6.1-terra
            service_tier="auto",
            reasoning_effort="medium",
            store=settings.openai_store_completions,
        ),
        AITaskType.STUDY_QUIZ_GENERATION: ModelRoutePolicy(
            task_type=AITaskType.STUDY_QUIZ_GENERATION,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_balanced,  # gpt-6.1-terra
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_fast,     # gpt-6-luna
            service_tier="auto",
            reasoning_effort="low",
            store=settings.openai_store_completions,
        ),
        AITaskType.AUDIO_BRIEFING_SCRIPT: ModelRoutePolicy(
            task_type=AITaskType.AUDIO_BRIEFING_SCRIPT,
            primary_provider=settings.ai_default_provider,
            primary_model=settings.openai_model_fast,      # gpt-6-luna
            fallback_provider=settings.ai_default_provider,
            fallback_model=settings.openai_model_balanced, # gpt-6.1-terra
            service_tier="auto",
            reasoning_effort="none",
            store=settings.openai_store_completions,
        ),
    }
