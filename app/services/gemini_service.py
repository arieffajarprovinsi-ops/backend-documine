from typing import List, Dict, Any
from app.ai.gateway import ai_gateway
from app.services.retrieval_service import citation_resolver
from app.models.schemas import (
    CognitiveDepth,
    QuizResponse,
    QuizQuestion,
    QuizOption,
    ChatResponse,
    BoundingBox,
)


class DeprecatedAIShimService:
    """
    Deprecated compatibility shim replacing the prototype GeminiIntelligenceService.
    All production AI calls now route through `app.ai.gateway.ai_gateway` (OpenAI default).
    """

    @property
    def has_active_api(self) -> bool:
        return ai_gateway.is_openai_configured

    def generate_chat_answer_with_citations(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        depth: CognitiveDepth = CognitiveDepth.HIGH_SCHOOL,
    ) -> ChatResponse:
        if not context_chunks:
            return ChatResponse(
                answer="Insufficient evidence in the selected document scope to answer this query.",
                retrieval_status="insufficient_evidence",
                citations=[],
                suggested_followups=[],
            )
        top = context_chunks[0]
        cid = str(top.get("chunk_id") or top.get("id") or "0")
        txt = (top.get("text_content") or top.get("content") or "").strip()
        raw_answer = f"Based on the retrieved document passage: {txt[:280]} [cite:{cid}]."
        cleaned, citations = citation_resolver.resolve_answer_citations(raw_answer, context_chunks)
        return ChatResponse(
            answer=cleaned,
            retrieval_status="grounded",
            citations=citations,
            suggested_followups=["What are the main findings in this document?"],
        )

    def generate_adaptive_quiz(
        self,
        document_id: str,
        document_title: str,
        document_text: str,
    ) -> QuizResponse:
        snippet = (document_text or document_title).strip()[:160]
        return QuizResponse(
            document_id=document_id,
            title=f"Study Quiz: {document_title}",
            questions=[
                QuizQuestion(
                    question_id="q1",
                    question_text=f"Which statement is directly supported by '{document_title}'?",
                    options=[
                        QuizOption(label="A", text=snippet[:110] or document_title),
                        QuizOption(label="B", text="Unrelated external claim not present in the document."),
                        QuizOption(label="C", text="Contradictory assumption outside the repository scope."),
                        QuizOption(label="D", text="None of the above."),
                    ],
                    correct_option="A",
                    explanation=f"Verified directly from '{document_title}'.",
                    source_page=1,
                    proof_quote=snippet,
                    bounding_boxes=[BoundingBox(left=0.08, top=0.12, width=0.84, height=0.10, page_number=1)],
                )
            ],
        )


gemini_service = DeprecatedAIShimService()
