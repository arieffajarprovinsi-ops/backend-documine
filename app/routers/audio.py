from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.contracts import AITaskType, AudioBriefingScriptPayload
from app.ai.gateway import ai_gateway, format_untrusted_chunk

router = APIRouter(prefix="/audio", tags=["audio"])


class PodcastDialogueTurn(BaseModel):
    speaker: str = Field(description="Host name ('Alex' / 'Host (Budi)' or 'Sarah' / 'Expert (Siti)')")
    text: str = Field(description="Natural conversational dialogue grounded in the document")
    source_chunk_id: Optional[str] = None


class PodcastScriptResponse(BaseModel):
    title: str
    duration_estimate: str
    script: List[PodcastDialogueTurn]
    takeaway_summary: str


class AudioBriefingRequest(BaseModel):
    document_id: Optional[str] = None


@router.post("/briefing-script", response_model=PodcastScriptResponse)
async def generate_podcast_briefing_script(
    document_id: Optional[str] = Query(default=None),
    body: Optional[AudioBriefingRequest] = Body(default=None),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Stage 6 P2 Studio Extension: Grounded 2-Host Audio Briefing Script Generator (DOC-17).
    Generates a conversational dialogue strictly grounded in the user's workspace document.
    """
    target_doc_id = document_id or (body.document_id if body else None)
    if not target_doc_id:
        docs = supabase_db.list_workspace_documents(ctx.workspace_id)
        if not docs:
            raise HTTPException(
                status_code=404,
                detail={"code": "ERR_DOCUMENT_NOT_FOUND", "message": "No documents found in workspace."},
            )
        doc = docs[0]
        target_doc_id = str(doc["id"])
    else:
        doc = supabase_db.get_workspace_document(ctx.workspace_id, target_doc_id)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ERR_DOCUMENT_NOT_FOUND",
                    "message": f"Document '{target_doc_id}' not found in workspace.",
                },
            )

    doc_title = doc.get("title", "Workspace Document")
    doc_summary = (doc.get("summary") or "").strip()
    doc_topics = doc.get("main_topics") or []
    chunks = supabase_db.get_document_chunks(ctx.workspace_id, target_doc_id)

    if ai_gateway.is_openai_configured and chunks:
        try:
            passages = "\n\n".join(
                format_untrusted_chunk(
                    chunk_id=str(c["id"]),
                    text_content=c.get("text_content") or c.get("content") or "",
                    page_number=int(c.get("page_number", 1)),
                    doc_title=doc_title,
                    doc_id=target_doc_id,
                )
                for c in chunks[:10]
            )
            payload = await ai_gateway.generate_structured(
                task_type=AITaskType.AUDIO_BRIEFING_SCRIPT,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                instructions=(
                    "You are the DocuMind Audio Briefing Scriptwriter. Create an engaging 4-to-6 turn dialogue "
                    "between 'Host (Budi)' and 'Expert (Siti)' discussing the key findings of the provided document "
                    "passages. Ground every point in the actual document text and reference source_chunk_id."
                ),
                input_blocks=f"Document Title: {doc_title}\nExecutive Summary: {doc_summary}\n\n{passages}",
                response_model=AudioBriefingScriptPayload,
                scope_key=f"audio:{target_doc_id}",
            )
            mins = max(1, payload.duration_estimate_seconds // 60)
            secs = payload.duration_estimate_seconds % 60
            return PodcastScriptResponse(
                title=payload.title or f"Audio Briefing: {doc_title}",
                duration_estimate=f"{mins:02d}:{secs:02d}",
                script=[
                    PodcastDialogueTurn(
                        speaker="Alex" if "Host" in seg.speaker else "Sarah",
                        text=seg.text,
                        source_chunk_id=seg.source_chunk_id,
                    )
                    for seg in payload.dialogue
                ],
                takeaway_summary=doc_summary or f"Key takeaways synthesized from {doc_title}.",
            )
        except Exception as e:
            print(f"[AudioRouter] OpenAI briefing fallback notice: {e}")

    # Deterministic document-grounded fallback script (zero hardcoded physics/Carnot text)
    first_chunk_text = (
        (chunks[0].get("text_content") or chunks[0].get("content") or "").strip()[:240]
        if chunks
        else doc_summary[:240]
    )
    second_chunk_text = (
        (chunks[1].get("text_content") or chunks[1].get("content") or "").strip()[:240]
        if len(chunks) > 1
        else first_chunk_text
    )
    first_cid = str(chunks[0]["id"]) if chunks else None
    second_cid = str(chunks[1]["id"]) if len(chunks) > 1 else first_cid
    topics_label = ", ".join(doc_topics[:4]) if doc_topics else (doc.get("primary_domain") or "core findings")

    return PodcastScriptResponse(
        title=f"Audio Briefing: {doc_title}",
        duration_estimate="01:30",
        script=[
            PodcastDialogueTurn(
                speaker="Alex",
                text=f"Today we are reviewing '{doc_title}'. What are its primary focus and key findings?",
                source_chunk_id=first_cid,
            ),
            PodcastDialogueTurn(
                speaker="Sarah",
                text=(
                    f"This document centers on {topics_label}. "
                    f"The core overview explains: {doc_summary or first_chunk_text}"
                ),
                source_chunk_id=first_cid,
            ),
            PodcastDialogueTurn(
                speaker="Alex",
                text="What supporting evidence or specific details are highlighted within the body sections?",
                source_chunk_id=second_cid,
            ),
            PodcastDialogueTurn(
                speaker="Sarah",
                text=f"Specifically noted inside the source section: \"{second_chunk_text}\"",
                source_chunk_id=second_cid,
            ),
        ],
        takeaway_summary=doc_summary or f"Executive briefing for '{doc_title}' covering {topics_label}.",
    )
