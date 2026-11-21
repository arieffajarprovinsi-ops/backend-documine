import asyncio
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.contracts import AITaskType, AudioBriefingScriptPayload
from app.ai.gateway import ai_gateway, format_untrusted_chunk
from app.core.config import settings

router = APIRouter(prefix="/audio", tags=["audio"])


class PodcastDialogueTurn(BaseModel):
    speaker: str = Field(description="Host name ('Alex' or 'Sarah')")
    text: str = Field(description="Natural conversational dialogue grounded in the document")
    source_chunk_id: Optional[str] = None


class PodcastScriptResponse(BaseModel):
    title: str
    duration_estimate: str
    script: List[PodcastDialogueTurn]
    takeaway_summary: str


class AudioBriefingRequest(BaseModel):
    document_id: Optional[str] = None


class SynthesizeSegmentInput(BaseModel):
    speaker: str
    text: str
    source_chunk_id: Optional[str] = None


class AudioSynthesizeRequest(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    script: List[SynthesizeSegmentInput] = Field(min_length=1, max_length=8)


class SynthesizedAudioTurn(BaseModel):
    speaker: str
    voice: str
    text: str
    audio_base64: Optional[str] = None
    content_type: str = "audio/mpeg"
    source_chunk_id: Optional[str] = None


class AudioSynthesizeResponse(BaseModel):
    model_used: str
    fallback_browser_tts: bool = False
    segments: List[SynthesizedAudioTurn]


@router.post("/briefing-script", response_model=PodcastScriptResponse)
async def generate_podcast_briefing_script(
    document_id: Optional[str] = Query(default=None),
    body: Optional[AudioBriefingRequest] = Body(default=None),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Studio 2-Host Audio Briefing Script Generator:
    Produces natural conversational dialogue strictly grounded in the user's document.
    """
    target_doc_id = document_id or (body.document_id if body else None)
    if not target_doc_id:
        docs = await asyncio.to_thread(supabase_db.list_workspace_documents, ctx.workspace_id)
        if not docs:
            raise HTTPException(
                status_code=404,
                detail={"code": "ERR_DOCUMENT_NOT_FOUND", "message": "No documents found in workspace."},
            )
        doc = docs[0]
        target_doc_id = str(doc["id"])
    else:
        doc = await asyncio.to_thread(supabase_db.get_workspace_document, ctx.workspace_id, target_doc_id)
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
    chunks = await asyncio.to_thread(supabase_db.get_document_chunks, ctx.workspace_id, target_doc_id)

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
                    "You are the DocuMind Audio Briefing Scriptwriter. Create an engaging 4-to-6 turn conversational dialogue "
                    "between 'Alex' (host) and 'Sarah' (analyst) summarizing the key findings, numbers, and implications "
                    "of the document passages. Match the primary language of the document (English or Indonesian), keep each "
                    "turn concise (2-3 crisp spoken sentences), and populate source_chunk_id."
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
                        speaker="Alex" if ("alex" in seg.speaker.lower() or "host" in seg.speaker.lower()) else "Sarah",
                        text=seg.text,
                        source_chunk_id=seg.source_chunk_id,
                    )
                    for seg in payload.dialogue
                ],
                takeaway_summary=doc_summary or f"Key takeaways synthesized from {doc_title}.",
            )
        except Exception as e:
            print(f"[AudioRouter] OpenAI briefing fallback notice: {e}")

    first_chunk_text = (
        " ".join((chunks[0].get("text_content") or chunks[0].get("content") or "").split())[:240]
        if chunks
        else doc_summary[:240]
    )
    second_chunk_text = (
        " ".join((chunks[1].get("text_content") or chunks[1].get("content") or "").split())[:240]
        if len(chunks) > 1
        else first_chunk_text
    )
    first_cid = str(chunks[0]["id"]) if chunks else None
    second_cid = str(chunks[1]["id"]) if len(chunks) > 1 else first_cid
    topics_label = ", ".join(doc_topics[:4]) if doc_topics else (doc.get("primary_domain") or "core findings")

    return PodcastScriptResponse(
        title=f"Audio Briefing: {doc_title}",
        duration_estimate="01:15",
        script=[
            PodcastDialogueTurn(
                speaker="Alex",
                text=f"Welcome to DocuMind Audio Briefing. Today we are reviewing {doc_title}. What is the main takeaway?",
                source_chunk_id=first_cid,
            ),
            PodcastDialogueTurn(
                speaker="Sarah",
                text=(
                    f"This document covers {topics_label}. "
                    f"Here is the core overview: {doc_summary or first_chunk_text}"
                ),
                source_chunk_id=first_cid,
            ),
            PodcastDialogueTurn(
                speaker="Alex",
                text="Are there specific details or evidence readers should pay attention to?",
                source_chunk_id=second_cid,
            ),
            PodcastDialogueTurn(
                speaker="Sarah",
                text=f"Yes, the source passages highlight this point directly: {second_chunk_text}",
                source_chunk_id=second_cid,
            ),
        ],
        takeaway_summary=doc_summary or first_chunk_text or f"Grounded summary of {doc_title}.",
    )


@router.post("/synthesize", response_model=AudioSynthesizeResponse)
async def synthesize_podcast_audio(
    req: AudioSynthesizeRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Generate spoken MP3 audio segments using OpenAI's cost-effective 2026 TTS model
    (`gpt-4o-mini-tts` with automatic `tts-1` fallback). Returns `fallback_browser_tts=True`
    if OpenAI TTS is unavailable so the UI seamlessly plays with browser speech synthesis.
    """
    turns = req.script[:6]
    if not ai_gateway.is_openai_configured:
        return AudioSynthesizeResponse(
            model_used="browser-speech",
            fallback_browser_tts=True,
            segments=[
                SynthesizedAudioTurn(
                    speaker=t.speaker,
                    voice="onyx" if "alex" in t.speaker.lower() else "nova",
                    text=t.text,
                    audio_base64=None,
                    source_chunk_id=t.source_chunk_id,
                )
                for t in turns
            ],
        )

    model_used = settings.openai_tts_model
    segments_out: List[SynthesizedAudioTurn] = []

    async def _synth_turn(turn: SynthesizeSegmentInput) -> SynthesizedAudioTurn:
        is_alex = "alex" in turn.speaker.lower() or "host" in turn.speaker.lower()
        voice = "onyx" if is_alex else "nova"
        tone_hint = (
            "Speak naturally in a warm, clear podcast host tone."
            if is_alex
            else "Speak naturally in an articulate, insightful co-host analyst tone."
        )
        audio_bytes, usage_meta = await ai_gateway.generate_speech(
            text=turn.text,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            voice=voice,
            instructions=tone_hint,
        )
        nonlocal model_used
        model_used = str(usage_meta.get("model") or model_used)
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return SynthesizedAudioTurn(
            speaker="Alex" if is_alex else "Sarah",
            voice=voice,
            text=turn.text,
            audio_base64=b64,
            content_type="audio/mpeg",
            source_chunk_id=turn.source_chunk_id,
        )

    try:
        results = await asyncio.gather(*(_synth_turn(t) for t in turns))
        segments_out = list(results)
        return AudioSynthesizeResponse(
            model_used=model_used,
            fallback_browser_tts=False,
            segments=segments_out,
        )
    except Exception as exc:
        print(f"[AudioRouter] OpenAI TTS fallback to browser speech: {exc}")
        return AudioSynthesizeResponse(
            model_used="browser-speech",
            fallback_browser_tts=True,
            segments=[
                SynthesizedAudioTurn(
                    speaker=t.speaker,
                    voice="onyx" if "alex" in t.speaker.lower() else "nova",
                    text=t.text,
                    audio_base64=None,
                    source_chunk_id=t.source_chunk_id,
                )
                for t in turns
            ],
        )
