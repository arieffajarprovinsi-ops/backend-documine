import asyncio
import base64
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.db.supabase_client import supabase_db
from app.ai.contracts import AITaskType, StorytellingScriptPayload
from app.ai.gateway import ai_gateway, format_untrusted_chunk
from app.core.config import settings

router = APIRouter(prefix="/audio", tags=["audio"])

SUPPORTED_STORY_VOICES = {
    "marin",
    "coral",
    "sage",
    "cedar",
    "ash",
    "ballad",
    "shimmer",
    "fable",
}


class StoryChapterTurn(BaseModel):
    speaker: str = Field(description="Chapter label or narrator heading")
    text: str = Field(description="Smooth single-narrator storytelling prose grounded in the document")
    source_chunk_id: Optional[str] = None


class StorytellingScriptResponse(BaseModel):
    title: str
    duration_estimate: str
    script: List[StoryChapterTurn]
    takeaway_summary: str
    style: str = "storytelling"


class AudioBriefingRequest(BaseModel):
    document_id: Optional[str] = None
    style: str = "storytelling"


class SynthesizeSegmentInput(BaseModel):
    speaker: str
    text: str
    source_chunk_id: Optional[str] = None


class AudioSynthesizeRequest(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    voice: str = "marin"
    style: str = "storytelling"
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
    voice_used: str = "marin"
    fallback_browser_tts: bool = False
    segments: List[SynthesizedAudioTurn]


@router.post("/briefing-script", response_model=StorytellingScriptResponse)
async def generate_storytelling_script(
    document_id: Optional[str] = Query(default=None),
    style: str = Query(default="storytelling"),
    body: Optional[AudioBriefingRequest] = Body(default=None),
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Immersive Document Storytelling Generator (powered by fast gpt-6-luna):
    Crafts a cohesive, engaging single-narrator story organized into clear chapters,
    strictly grounded in the user's document in its native language.
    """
    target_doc_id = document_id or (body.document_id if body else None)
    req_style = ((body.style if body and body.style else style) or "storytelling").strip().lower()

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
    lang_code = (doc.get("language_code") or "id").lower()
    chunks = await asyncio.to_thread(supabase_db.get_document_chunks, ctx.workspace_id, target_doc_id)

    style_guidance = {
        "storytelling": (
            "Write as an immersive, warm narrative story that draws the listener in from an intriguing opening hook "
            "through the core journey, key figures/turning points, and a memorable closing takeaway."
        ),
        "documentary": (
            "Write like a crisp, vivid documentary narrator guiding the listener clearly through the context, "
            "verified facts, quantitative highlights, and real-world implications."
        ),
        "executive": (
            "Write as a polished, cohesive single-voice narrative briefing connecting purpose, core evidence, "
            "and strategic takeaways smoothly without bullet-point stiffness."
        ),
    }.get(req_style, "Write as an engaging, natural single-voice narrative story.")

    if ai_gateway.is_openai_configured and chunks:
        try:
            max_samples = min(7, len(chunks))
            if len(chunks) <= max_samples:
                sampled = chunks
            else:
                step = (len(chunks) - 1) / float(max(1, max_samples - 1))
                idxs = sorted({round(i * step) for i in range(max_samples)})
                sampled = [chunks[i] for i in idxs if 0 <= i < len(chunks)]

            passages = "\n\n".join(
                format_untrusted_chunk(
                    chunk_id=str(c["id"]),
                    text_content=((c.get("text_content") or c.get("content") or "").strip())[:520],
                    page_number=int(c.get("page_number", 1)),
                    doc_title=doc_title,
                    doc_id=target_doc_id,
                )
                for c in sampled
            )
            payload = await ai_gateway.generate_structured(
                task_type=AITaskType.AUDIO_BRIEFING_SCRIPT,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                instructions=(
                    "You are the DocuMind Master Storyteller. Transform the provided document passages into a "
                    "captivating 3-to-4 chapter single-narrator audio story (NOT a two-person Q&A podcast). "
                    f"{style_guidance} "
                    "Use the exact primary language of the document (natural Indonesian if Indonesian, English if English). "
                    "Write smooth spoken sentences free of raw markdown symbols or citation brackets so it sounds "
                    "completely human when read aloud, and attach source_chunk_id for each chapter."
                ),
                input_blocks=f"Document Title: {doc_title}\nExecutive Summary: {doc_summary}\n\n{passages}",
                response_model=StorytellingScriptPayload,
                scope_key=f"story:{target_doc_id}:{req_style}",
            )
            mins = max(1, payload.duration_estimate_seconds // 60)
            secs = payload.duration_estimate_seconds % 60
            return StorytellingScriptResponse(
                title=payload.title or doc_title,
                duration_estimate=f"{mins:02d}:{secs:02d}",
                script=[
                    StoryChapterTurn(
                        speaker=ch.chapter_title or f"Chapter {idx + 1}",
                        text=ch.narration,
                        source_chunk_id=ch.source_chunk_id,
                    )
                    for idx, ch in enumerate(payload.chapters[:5])
                    if (ch.narration or "").strip()
                ],
                takeaway_summary=payload.takeaway_summary or doc_summary or f"Core story from {doc_title}.",
                style=req_style,
            )
        except Exception as e:
            print(f"[AudioRouter] OpenAI storytelling fallback notice: {e}")

    first_chunk_text = (
        " ".join((chunks[0].get("text_content") or chunks[0].get("content") or "").split())[:280]
        if chunks
        else doc_summary[:280]
    )
    second_chunk_text = (
        " ".join((chunks[1].get("text_content") or chunks[1].get("content") or "").split())[:280]
        if len(chunks) > 1
        else first_chunk_text
    )
    first_cid = str(chunks[0]["id"]) if chunks else None
    second_cid = str(chunks[1]["id"]) if len(chunks) > 1 else first_cid
    topics_label = ", ".join(doc_topics[:4]) if doc_topics else (doc.get("primary_domain") or "topik utama")

    if lang_code == "id":
        fallback_chapters = [
            StoryChapterTurn(
                speaker="Babak 1 · Konteks Cerita",
                text=f"Mari kita telusuri isi dari {doc_title}. Dokumen ini mengangkat fokus utama seputar {topics_label}. {doc_summary or first_chunk_text}",
                source_chunk_id=first_cid,
            ),
            StoryChapterTurn(
                speaker="Babak 2 · Detail & Bukti Penting",
                text=f"Memasuki bagian inti dokumen, tercatat detail penting berikut: {second_chunk_text}",
                source_chunk_id=second_cid,
            ),
        ]
    else:
        fallback_chapters = [
            StoryChapterTurn(
                speaker="Chapter 1 · Opening Context",
                text=f"Let us explore the story inside {doc_title}, which centers around {topics_label}. {doc_summary or first_chunk_text}",
                source_chunk_id=first_cid,
            ),
            StoryChapterTurn(
                speaker="Chapter 2 · Core Evidence",
                text=f"Looking closer at the key passages, the document highlights: {second_chunk_text}",
                source_chunk_id=second_cid,
            ),
        ]

    return StorytellingScriptResponse(
        title=doc_title,
        duration_estimate="01:20",
        script=fallback_chapters,
        takeaway_summary=doc_summary or first_chunk_text or f"Narrative overview of {doc_title}.",
        style=req_style,
    )


@router.post("/synthesize", response_model=AudioSynthesizeResponse)
async def synthesize_storytelling_audio(
    req: AudioSynthesizeRequest,
    ctx: WorkspaceContext = Depends(get_verified_workspace_context),
):
    """
    Synthesize cohesive single-narrator Storytelling audio chapters using OpenAI's natural voices
    (`marin`, `coral`, `sage`, `cedar`, `ash`, `ballad`, `shimmer`) over HD TTS (`tts-1-hd` / `gpt-4o-mini-tts`).
    """
    turns = req.script[:6]
    chosen_voice = (req.voice or getattr(settings, "openai_tts_default_voice", "marin") or "marin").strip().lower()
    if chosen_voice not in SUPPORTED_STORY_VOICES:
        chosen_voice = "marin"

    if not ai_gateway.is_openai_configured:
        return AudioSynthesizeResponse(
            model_used="browser-speech",
            voice_used=chosen_voice,
            fallback_browser_tts=True,
            segments=[
                SynthesizedAudioTurn(
                    speaker=t.speaker,
                    voice=chosen_voice,
                    text=t.text,
                    audio_base64=None,
                    source_chunk_id=t.source_chunk_id,
                )
                for t in turns
            ],
        )

    model_used = settings.openai_tts_model
    tone_instruction = (
        "Narrate smoothly, warmly, and naturally like a captivating human audiobook storyteller, "
        "matching the exact native pronunciation of the text without rushing."
    )

    async def _synth_chapter(turn: SynthesizeSegmentInput) -> SynthesizedAudioTurn:
        audio_bytes, usage_meta = await ai_gateway.generate_speech(
            text=turn.text,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            voice=chosen_voice,
            instructions=tone_instruction,
        )
        nonlocal model_used
        model_used = str(usage_meta.get("model") or model_used)
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return SynthesizedAudioTurn(
            speaker=turn.speaker,
            voice=chosen_voice,
            text=turn.text,
            audio_base64=b64,
            content_type="audio/mpeg",
            source_chunk_id=turn.source_chunk_id,
        )

    try:
        results = await asyncio.gather(*(_synth_chapter(t) for t in turns))
        return AudioSynthesizeResponse(
            model_used=model_used,
            voice_used=chosen_voice,
            fallback_browser_tts=False,
            segments=list(results),
        )
    except Exception as exc:
        print(f"[AudioRouter] OpenAI TTS fallback to browser speech: {exc}")
        return AudioSynthesizeResponse(
            model_used="browser-speech",
            voice_used=chosen_voice,
            fallback_browser_tts=True,
            segments=[
                SynthesizedAudioTurn(
                    speaker=t.speaker,
                    voice=chosen_voice,
                    text=t.text,
                    audio_base64=None,
                    source_chunk_id=t.source_chunk_id,
                )
                for t in turns
            ],
        )
