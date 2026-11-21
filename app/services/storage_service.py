import os
import re
from typing import Optional
from app.core.config import settings
from app.db.supabase_client import supabase_db


def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename or "document.pdf").strip()
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return safe[:180] or "document.pdf"


class TenantStorageService:
    """
    Multi-tenant binary file storage service (ADR-004, PR-008).
    Stores files under: workspaces/{workspace_id}/documents/{doc_id}/{safe_filename}
    Synchronizes with private Supabase Storage bucket ('documents') and keeps a local
    tenant-isolated cache file under settings.upload_dir for fast C++ PyMuPDF access.
    """

    def __init__(self):
        self._bucket_checked = False
        os.makedirs(settings.upload_dir, exist_ok=True)

    def _ensure_bucket(self) -> None:
        if self._bucket_checked or not supabase_db.is_connected:
            return
        try:
            buckets = supabase_db.client.storage.list_buckets()
            existing_names = {
                getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else None)
                for b in (buckets or [])
            }
            if settings.supabase_storage_bucket not in existing_names:
                supabase_db.client.storage.create_bucket(
                    settings.supabase_storage_bucket,
                    options={"public": False},
                )
        except Exception:
            pass
        finally:
            self._bucket_checked = True

    def build_storage_path(self, workspace_id: str, doc_id: str, filename: str) -> str:
        if not workspace_id or not doc_id:
            raise ValueError("workspace_id and doc_id are required to build tenant storage path.")
        safe_name = sanitize_filename(filename)
        return f"workspaces/{workspace_id}/documents/{doc_id}/{safe_name}"

    def get_local_cache_path(self, storage_path: str) -> str:
        normalized = storage_path.replace("\\", "/").lstrip("/")
        if ".." in normalized.split("/"):
            raise ValueError("Invalid storage path traversal attempt.")
        full_path = os.path.join(settings.upload_dir, *normalized.split("/"))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    def verify_tenant_ownership_of_path(self, workspace_id: str, storage_path: str) -> None:
        normalized = (storage_path or "").replace("\\", "/").lstrip("/")
        expected_prefix = f"workspaces/{workspace_id}/"
        if not normalized.startswith(expected_prefix):
            raise PermissionError(
                f"Storage path '{storage_path}' does not belong to workspace '{workspace_id}'."
            )

    def save_document_bytes(
        self,
        *,
        workspace_id: str,
        doc_id: str,
        filename: str,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
    ) -> tuple[str, str]:
        """
        Save file bytes to both local tenant cache and remote Supabase Storage bucket.
        Returns (storage_path, local_cache_path).
        """
        storage_path = self.build_storage_path(workspace_id, doc_id, filename)
        local_path = self.get_local_cache_path(storage_path)

        with open(local_path, "wb") as f:
            f.write(file_bytes)

        if supabase_db.is_connected:
            self._ensure_bucket()
            try:
                supabase_db.client.storage.from_(settings.supabase_storage_bucket).upload(
                    path=storage_path,
                    file=file_bytes,
                    file_options={"content-type": mime_type, "upsert": "true"},
                )
            except Exception as e:
                print(f"[TenantStorageService] Remote storage upload notice ({storage_path}): {e}")

        return storage_path, local_path

    def resolve_local_file_for_reading(self, workspace_id: str, storage_path: str) -> Optional[str]:
        """
        Verify tenant ownership and return a local filesystem path for PyMuPDF / FileResponse.
        If the local cache file is missing (e.g. after container restart), downloads from Supabase Storage.
        """
        self.verify_tenant_ownership_of_path(workspace_id, storage_path)
        local_path = self.get_local_cache_path(storage_path)
        if os.path.isfile(local_path):
            return local_path

        if supabase_db.is_connected:
            try:
                data = supabase_db.client.storage.from_(settings.supabase_storage_bucket).download(storage_path)
                if data:
                    with open(local_path, "wb") as f:
                        f.write(data)
                    return local_path
            except Exception as e:
                print(f"[TenantStorageService] Remote download failed for {storage_path}: {e}")

        return None

    def delete_document_file(self, workspace_id: str, storage_path: str) -> bool:
        """Delete file from both Supabase Storage and local cache after verifying workspace ownership."""
        self.verify_tenant_ownership_of_path(workspace_id, storage_path)
        local_path = self.get_local_cache_path(storage_path)
        removed_local = False
        if os.path.isfile(local_path):
            try:
                os.remove(local_path)
                removed_local = True
            except OSError:
                pass

        if supabase_db.is_connected:
            try:
                supabase_db.client.storage.from_(settings.supabase_storage_bucket).remove([storage_path])
            except Exception:
                pass

        return removed_local


storage_service = TenantStorageService()
