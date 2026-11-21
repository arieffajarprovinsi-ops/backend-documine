from typing import Optional, Dict, Any
from fastapi import BackgroundTasks
from app.core.config import settings
from app.services.ingestion_service import ingestion_service


async def arq_ingest_document_task(
    ctx: Dict[str, Any],
    *,
    workspace_id: str,
    doc_id: str,
    job_id: str,
    local_file_path: str,
    user_id: Optional[str] = None,
    expected_version: int = 1,
) -> Dict[str, Any]:
    """
    arq background worker entry point for Railway documind-worker service (ADR-014, ADR-018).
    """
    return await ingestion_service.execute_pipeline(
        workspace_id=workspace_id,
        doc_id=doc_id,
        job_id=job_id,
        local_file_path=local_file_path,
        user_id=user_id,
        expected_version=expected_version,
    )


async def dispatch_ingestion_job(
    *,
    background_tasks: Optional[BackgroundTasks],
    workspace_id: str,
    doc_id: str,
    job_id: str,
    local_file_path: str,
    user_id: Optional[str] = None,
    expected_version: int = 1,
    run_inline_if_no_bg: bool = True,
) -> str:
    """
    Dual-mode ingestion dispatcher:
    1. If REDIS_URL is configured and reachable, enqueues onto the durable arq Redis queue.
    2. Otherwise falls back seamlessly to FastAPI BackgroundTasks (or inline execution in tests)
       using the exact same 6-checkpoint public.processing_jobs state machine.
    """
    if settings.redis_url and settings.enable_arq_worker:
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            await pool.enqueue_job(
                "arq_ingest_document_task",
                workspace_id=workspace_id,
                doc_id=doc_id,
                job_id=job_id,
                local_file_path=local_file_path,
                user_id=user_id,
                expected_version=expected_version,
            )
            await pool.close()
            return "arq_redis"
        except Exception as e:
            print(f"[WorkerDispatcher] Redis unavailable ({e}); falling back to BackgroundTasks.")

    if background_tasks is not None:
        background_tasks.add_task(
            ingestion_service.execute_pipeline,
            workspace_id=workspace_id,
            doc_id=doc_id,
            job_id=job_id,
            local_file_path=local_file_path,
            user_id=user_id,
            expected_version=expected_version,
        )
        return "background_tasks"

    if run_inline_if_no_bg:
        await ingestion_service.execute_pipeline(
            workspace_id=workspace_id,
            doc_id=doc_id,
            job_id=job_id,
            local_file_path=local_file_path,
            user_id=user_id,
            expected_version=expected_version,
        )
        return "inline"

    return "queued"
