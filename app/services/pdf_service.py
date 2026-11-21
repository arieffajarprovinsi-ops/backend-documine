import base64
import csv
import io
import os
import re
import uuid
import pymupdf as fitz
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.models.schemas import BoundingBox, DocumentChunk


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace("\x00", " ").strip()


def _normalize_numeric_string(raw_val: str) -> str:
    s = (raw_val or "").strip()
    # Remove isolated dots/commas placed on separate lines by PDF kerning (e.g. "Rp 37000\n. .")
    s = re.sub(r"[\r\n]+[\.\,\s]+$", "", s)
    s = re.sub(r"\s+[\.\,](?=\s|$)", "", s)
    # Remove currency symbols, percent, and spaces
    s = re.sub(r"[$%\s]|Rp\.?|USD|IDR", "", s, flags=re.IGNORECASE)
    # Handle Indonesian thousand separator dots (e.g. 1.250.000 or 37.000)
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    else:
        s = s.strip(".,")
    return s


def _infer_column_type_and_unit(col_name: str, values: List[Any]) -> Dict[str, Any]:
    non_empty = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    unit = None
    col_lower = (col_name or "").lower()
    if "usd" in col_lower or "$" in col_lower:
        unit = "USD"
    elif (
        "idr" in col_lower
        or "rp" in col_lower
        or "harga" in col_lower
        or "biaya" in col_lower
        or "total" in col_lower
        or any("rp" in str(v).lower() for v in non_empty[:10])
    ):
        unit = "IDR"
    elif "%" in col_lower or "pct" in col_lower or "percent" in col_lower or "efisiensi" in col_lower:
        unit = "%"

    numeric_hits = 0
    for val in non_empty[:25]:
        cleaned = _normalize_numeric_string(val)
        try:
            float(cleaned)
            numeric_hits += 1
        except ValueError:
            pass

    inferred_type = "numeric" if (non_empty and numeric_hits >= max(1, int(len(non_empty[:25]) * 0.6))) else "varchar"
    return {
        "name": re.sub(r"[^a-zA-Z0-9_]", "_", col_name.strip() or "col").strip("_").lower() or "col",
        "display_name": col_name.strip() or "Column",
        "inferred_type": inferred_type,
        "unit_or_currency": unit,
    }


def _try_parse_numeric(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if not s:
        return None
    cleaned = _normalize_numeric_string(s)
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned):
        try:
            f = float(cleaned)
            return int(f) if f.is_integer() else round(f, 6)
        except ValueError:
            return _clean_text(s.replace("\n", " "))
    return _clean_text(s.replace("\n", " "))


class PDFProcessorService:
    """
    Deterministic Document & PDF Processing Engine using PyMuPDF (fitz), csv, and openpyxl/pandas (DOC-10).
    Handles Stages 1, 3, 4, 5, 7, and 8 of the 15-Stage Ingestion Pipeline:
    - Stage 1: Magic-byte, size, encryption, and format validation (.pdf, .csv, .xlsx)
    - Stage 3: Page geometry & image-density profiling
    - Stage 4: Sub-millimeter digital text & normalized bounding-box extraction
    - Stage 5: Per-page scanned-document & visual-chart detection gate
    - Stage 7: Deterministic PDF table grid extraction (find_tables) & spreadsheet parsing
    - Stage 8: Layout-aware semantic chunking (300-600 tokens, 15% overlap, multi-BBox preservation)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx"}

    @classmethod
    def validate_upload_bytes(cls, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """
        Stage 1 Deterministic Validation (PR-012, EC-001..EC-005).
        Raises ValueError with a clear user-facing error message if invalid.
        """
        if not filename:
            raise ValueError("Filename is required.")
        ext = os.path.splitext(filename.lower())[1]
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: PDF (.pdf), CSV (.csv), and Excel (.xlsx)."
            )

        size_bytes = len(file_bytes or b"")
        if size_bytes == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"File size ({size_bytes / (1024 * 1024):.1f} MB) exceeds the {settings.max_upload_size_mb} MB limit."
            )

        if ext == ".pdf":
            if not file_bytes[:1024].lstrip().startswith(b"%PDF-"):
                raise ValueError("Invalid PDF file: missing %PDF- magic header.")
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
            except Exception as exc:
                raise ValueError(f"Corrupted or unreadable PDF file: {exc}") from exc

            try:
                if doc.is_encrypted or doc.needs_pass:
                    raise ValueError(
                        "Password-protected or encrypted PDFs cannot be processed. Please upload an unlocked PDF."
                    )
                page_count = len(doc)
                if page_count <= 0:
                    raise ValueError("PDF file contains 0 pages.")
                title = _clean_text(doc.metadata.get("title") or "") or os.path.basename(filename)
                author = _clean_text(doc.metadata.get("author") or "")
            finally:
                doc.close()

            return {
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "page_count": page_count,
                "file_size_bytes": size_bytes,
                "inferred_title": title,
                "author": author,
            }

        if ext == ".xlsx":
            if not file_bytes.startswith(b"PK\x03\x04"):
                raise ValueError("Invalid Excel (.xlsx) file: corrupt ZIP container header.")
            return {
                "file_type": "xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "page_count": 1,
                "file_size_bytes": size_bytes,
                "inferred_title": os.path.basename(filename),
                "author": "",
            }

        # .csv
        return {
            "file_type": "csv",
            "mime_type": "text/csv",
            "page_count": 1,
            "file_size_bytes": size_bytes,
            "inferred_title": os.path.basename(filename),
            "author": "",
        }

    @staticmethod
    def get_document_metadata(file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path.lower())[1]
        if ext != ".pdf":
            return {
                "page_count": 1,
                "title": os.path.basename(file_path),
                "author": "",
                "file_size_bytes": os.path.getsize(file_path) if os.path.isfile(file_path) else 0,
            }
        doc = fitz.open(file_path)
        meta = {
            "page_count": len(doc),
            "title": _clean_text(doc.metadata.get("title") or "") or os.path.basename(file_path),
            "author": _clean_text(doc.metadata.get("author") or ""),
            "file_size_bytes": os.path.getsize(file_path),
        }
        doc.close()
        return meta

    @staticmethod
    def render_page_png_base64(file_path: str, page_number: int, dpi: int = 150) -> Optional[str]:
        """Stage 6 Helper: Render a 1-indexed PDF page to a 150 DPI base64 PNG for gpt-6.1-terra Vision OCR."""
        doc = fitz.open(file_path)
        try:
            idx = page_number - 1
            if idx < 0 or idx >= len(doc):
                return None
            pix = doc[idx].get_pixmap(dpi=dpi, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("ascii")
        finally:
            doc.close()

    @classmethod
    def analyze_pdf_pages_and_blocks(
        cls,
        file_path: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Executes Stages 3, 4, 5, and 7 (PDF tables) in a single pass over the PDF:
        Returns:
          1. page_profiles: list of dicts for public.document_pages
          2. page_blocks: dict mapping page_number (1-indexed) -> list of positioned text blocks
          3. extracted_tables: list of dicts for public.document_tables
        """
        doc = fitz.open(file_path)
        page_profiles: List[Dict[str, Any]] = []
        page_blocks: Dict[int, List[Dict[str, Any]]] = {}
        extracted_tables: List[Dict[str, Any]] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                rect = page.rect
                w = float(rect.width or 612.0)
                h = float(rect.height or 792.0)
                page_area = max(1.0, w * h)

                raw_blocks = page.get_text("blocks")
                positioned_blocks: List[Dict[str, Any]] = []
                digital_word_count = 0
                image_area = 0.0

                for b in raw_blocks:
                    x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    block_type = int(b[6]) if len(b) >= 7 else 0
                    if block_type == 1:
                        image_area += max(0.0, (x1 - x0) * (y1 - y0))
                        continue

                    text = _clean_text(str(b[4]))
                    if not text:
                        continue

                    words = text.split()
                    digital_word_count += len(words)

                    norm_left = round(max(0.0, min(1.0, x0 / w)), 4)
                    norm_top = round(max(0.0, min(1.0, y0 / h)), 4)
                    norm_width = round(max(0.0, min(1.0 - norm_left, (x1 - x0) / w)), 4)
                    norm_height = round(max(0.0, min(1.0 - norm_top, (y1 - y0) / h)), 4)

                    positioned_blocks.append(
                        {
                            "text": text,
                            "word_count": len(words),
                            "bbox": BoundingBox(
                                left=norm_left,
                                top=norm_top,
                                width=norm_width,
                                height=norm_height,
                                page_number=page_num,
                            ),
                        }
                    )

                image_area_ratio = round(min(1.0, image_area / page_area), 4)
                needs_ocr = bool(digital_word_count < 20 and (image_area_ratio > 0.20 or len(raw_blocks) == 0))

                page_profiles.append(
                    {
                        "page_number": page_num,
                        "width_pt": round(w, 2),
                        "height_pt": round(h, 2),
                        "digital_word_count": digital_word_count,
                        "image_area_ratio": image_area_ratio,
                        "needs_ocr": needs_ocr,
                        "ocr_status": "pending" if needs_ocr else "not_needed",
                    }
                )
                page_blocks[page_num] = positioned_blocks

                # Stage 7: Deterministic table grid detection on digital PDF pages
                if hasattr(page, "find_tables"):
                    try:
                        found = page.find_tables()
                        tables_list = getattr(found, "tables", None) or []
                        for t_idx, tbl in enumerate(tables_list):
                            grid = tbl.extract()
                            if not grid or len(grid) < 2:
                                continue
                            header_row = [
                                _clean_text(str(c or f"col_{i+1}")) or f"col_{i+1}"
                                for i, c in enumerate(grid[0])
                            ]
                            data_rows_raw = grid[1:]
                            if not any(any(cell for cell in r) for r in data_rows_raw):
                                continue

                            col_schemas: List[Dict[str, Any]] = []
                            seen_names: Dict[str, int] = {}
                            for c_i, raw_col_name in enumerate(header_row):
                                col_vals = [r[c_i] if c_i < len(r) else None for r in data_rows_raw]
                                col_info = _infer_column_type_and_unit(raw_col_name, col_vals)
                                base_name = col_info["name"]
                                if base_name in seen_names:
                                    seen_names[base_name] += 1
                                    col_info["name"] = f"{base_name}_{seen_names[base_name]}"
                                else:
                                    seen_names[base_name] = 1
                                col_schemas.append(col_info)

                            typed_rows: List[Dict[str, Any]] = []
                            for r in data_rows_raw:
                                row_dict: Dict[str, Any] = {}
                                non_null_count = 0
                                for c_i, col_spec in enumerate(col_schemas):
                                    cell_val = r[c_i] if c_i < len(r) else None
                                    parsed_val = (
                                        _try_parse_numeric(cell_val)
                                        if col_spec["inferred_type"] == "numeric"
                                        else _clean_text(str(cell_val or ""))
                                    )
                                    if parsed_val not in (None, ""):
                                        non_null_count += 1
                                    row_dict[col_spec["name"]] = parsed_val
                                if non_null_count > 0:
                                    typed_rows.append(row_dict)

                            if not typed_rows:
                                continue

                            t_bbox = getattr(tbl, "bbox", None)
                            norm_tbl_bbox = (
                                {
                                    "left": round(max(0.0, min(1.0, float(t_bbox[0]) / w)), 4),
                                    "top": round(max(0.0, min(1.0, float(t_bbox[1]) / h)), 4),
                                    "width": round(max(0.0, min(1.0, float(t_bbox[2] - t_bbox[0]) / w)), 4),
                                    "height": round(max(0.0, min(1.0, float(t_bbox[3] - t_bbox[1]) / h)), 4),
                                    "page_number": page_num,
                                }
                                if t_bbox and len(t_bbox) == 4
                                else {"left": 0.05, "top": 0.1, "width": 0.9, "height": 0.4, "page_number": page_num}
                            )

                            extracted_tables.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "table_name": f"pdf_p{page_num}_t{t_idx + 1}",
                                    "table_title": f"Page {page_num} Table {t_idx + 1}",
                                    "page_number": page_num,
                                    "bbox": norm_tbl_bbox,
                                    "columns_schema": col_schemas,
                                    "rows_data": typed_rows,
                                    "row_count": len(typed_rows),
                                    "extraction_method": "pymupdf",
                                }
                            )
                    except Exception:
                        pass
        finally:
            doc.close()

        return page_profiles, page_blocks, extracted_tables

    @classmethod
    def extract_spreadsheet_tables_and_blocks(
        cls,
        file_path: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Stage 7 & 8 for .csv and .xlsx files:
        Parses tabular rows into public.document_tables AND generates readable text blocks
        so spreadsheet contents are searchable via Hybrid Search and AI Assistant.
        """
        ext = os.path.splitext(file_path.lower())[1]
        rows_grid: List[List[Any]] = []

        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if any(str(c).strip() for c in row):
                        rows_grid.append(row)
        elif ext == ".xlsx":
            try:
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
                sheet = wb.active
                for row in sheet.iter_rows(values_only=True):
                    if row and any(c is not None and str(c).strip() != "" for c in row):
                        rows_grid.append([c for c in row])
                wb.close()
            except Exception as e:
                print(f"[PDFProcessorService] openpyxl parse warning: {e}")

        if not rows_grid:
            return (
                [
                    {
                        "page_number": 1,
                        "width_pt": 612.0,
                        "height_pt": 792.0,
                        "digital_word_count": 0,
                        "image_area_ratio": 0.0,
                        "needs_ocr": False,
                        "ocr_status": "not_needed",
                    }
                ],
                {1: []},
                [],
            )

        header = [_clean_text(str(c or f"col_{i+1}")) or f"col_{i+1}" for i, c in enumerate(rows_grid[0])]
        data_rows = rows_grid[1:] if len(rows_grid) > 1 else []

        col_schemas: List[Dict[str, Any]] = []
        seen_names: Dict[str, int] = {}
        for c_i, raw_col_name in enumerate(header):
            col_vals = [r[c_i] if c_i < len(r) else None for r in data_rows]
            col_info = _infer_column_type_and_unit(raw_col_name, col_vals)
            base_name = col_info["name"]
            if base_name in seen_names:
                seen_names[base_name] += 1
                col_info["name"] = f"{base_name}_{seen_names[base_name]}"
            else:
                seen_names[base_name] = 1
            col_schemas.append(col_info)

        typed_rows: List[Dict[str, Any]] = []
        text_lines: List[str] = [f"Spreadsheet Columns: {', '.join(c['display_name'] for c in col_schemas)}"]
        for r_idx, r in enumerate(data_rows[:1000]):
            row_dict: Dict[str, Any] = {}
            parts: List[str] = []
            for c_i, col_spec in enumerate(col_schemas):
                cell_val = r[c_i] if c_i < len(r) else None
                parsed_val = (
                    _try_parse_numeric(cell_val)
                    if col_spec["inferred_type"] == "numeric"
                    else _clean_text(str(cell_val or ""))
                )
                row_dict[col_spec["name"]] = parsed_val
                if parsed_val not in (None, ""):
                    parts.append(f"{col_spec['display_name']}: {parsed_val}")
            if parts:
                typed_rows.append(row_dict)
                if r_idx < 200:
                    text_lines.append(f"Row {r_idx + 1} — " + "; ".join(parts))

        full_text = "\n".join(text_lines)
        words = full_text.split()
        blocks = [
            {
                "text": full_text,
                "word_count": len(words),
                "bbox": BoundingBox(left=0.05, top=0.05, width=0.9, height=0.9, page_number=1),
            }
        ]
        extracted_tables = [
            {
                "id": str(uuid.uuid4()),
                "table_name": "sheet_1",
                "table_title": os.path.basename(file_path),
                "page_number": 1,
                "bbox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9, "page_number": 1},
                "columns_schema": col_schemas,
                "rows_data": typed_rows,
                "row_count": len(typed_rows),
                "extraction_method": "spreadsheet",
            }
        ]
        page_profiles = [
            {
                "page_number": 1,
                "width_pt": 612.0,
                "height_pt": 792.0,
                "digital_word_count": len(words),
                "image_area_ratio": 0.0,
                "needs_ocr": False,
                "ocr_status": "not_needed",
            }
        ]
        return page_profiles, {1: blocks}, extracted_tables

    @classmethod
    def build_layout_chunks(
        cls,
        *,
        document_id: str,
        workspace_id: Optional[str] = None,
        page_blocks: Dict[int, List[Dict[str, Any]]],
        pages_with_tables: Optional[set[int]] = None,
        target_words: int = 260,
        overlap_words: int = 40,
    ) -> List[DocumentChunk]:
        """
        Stage 8 Layout-Aware Semantic Chunking (DOC-10, PR-021):
        Groups adjacent blocks on each page into ~300-600 token (~260 word) chunks
        with ~15% overlap while preserving the exact array of constituent BoundingBox items.
        """
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        table_pages = pages_with_tables or set()

        for page_num in sorted(page_blocks.keys()):
            blocks = page_blocks[page_num]
            current_words: List[str] = []
            current_boxes: List[BoundingBox] = []

            for blk in blocks:
                text = blk.get("text", "").strip()
                if not text:
                    continue
                words = text.split()
                bbox: BoundingBox = blk["bbox"]

                current_words.extend(words)
                current_boxes.append(bbox)

                if len(current_words) >= target_words:
                    chunk_text = " ".join(current_words)
                    chunk_id = str(uuid.uuid4())
                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            document_id=document_id,
                            doc_id=document_id,
                            workspace_id=workspace_id,
                            chunk_index=global_chunk_idx,
                            content=chunk_text,
                            text_content=chunk_text,
                            page_number=page_num,
                            bounding_boxes=list(current_boxes),
                            token_count=int(len(current_words) * 1.3),
                            has_table=(page_num in table_pages),
                        )
                    )
                    global_chunk_idx += 1
                    # Keep 15% overlap tail for semantic continuity across paragraph breaks
                    current_words = current_words[-overlap_words:] if overlap_words > 0 else []
                    current_boxes = [bbox] if current_words else []

            if current_words:
                chunk_text = " ".join(current_words)
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        doc_id=document_id,
                        workspace_id=workspace_id,
                        chunk_index=global_chunk_idx,
                        content=chunk_text,
                        text_content=chunk_text,
                        page_number=page_num,
                        bounding_boxes=list(current_boxes),
                        token_count=int(len(current_words) * 1.3),
                        has_table=(page_num in table_pages),
                    )
                )
                global_chunk_idx += 1

        return chunks

    @classmethod
    def extract_chunks_with_bounding_boxes(
        cls,
        file_path: str,
        document_id: str,
        chunk_size_words: int = 250,
        workspace_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        ext = os.path.splitext(file_path.lower())[1]
        if ext in (".csv", ".xlsx"):
            _, page_blocks, tables = cls.extract_spreadsheet_tables_and_blocks(file_path)
        else:
            _, page_blocks, tables = cls.analyze_pdf_pages_and_blocks(file_path)
        table_pages = {int(t.get("page_number", 1)) for t in tables}
        return cls.build_layout_chunks(
            document_id=document_id,
            workspace_id=workspace_id,
            page_blocks=page_blocks,
            pages_with_tables=table_pages,
            target_words=chunk_size_words,
        )


pdf_processor = PDFProcessorService()
