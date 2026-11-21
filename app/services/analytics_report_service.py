import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.ai.contracts import AITaskType
from app.ai.gateway import ai_gateway
from app.core.hardening import analytics_report_cache
from app.db.supabase_client import supabase_db


class AnalyticsNarrativeEnrichment(BaseModel):
    domain_label: str = Field(..., description="Short English domain label (e.g. Finance & Revenue, Operations SOP, Academic Research)")
    executive_summary: str = Field(..., description="Concise 2-3 sentence grounded executive summary in English")
    key_insights: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Up to 4 structured insights with keys: title, detail, tag",
    )
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="3 concrete follow-up questions in English grounded in the document(s)",
    )


DOMAIN_RULES: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    (
        "finance",
        "Finance & Revenue",
        "Financial statements, revenue trends, cash flow, or business KPIs.",
        (
            "revenue", "pendapatan", "profit", "laba", "rugi", "biaya", "beban",
            "anggaran", "budget", "neraca", "kas", "cashflow", "idr", "rp ", "rupiah",
            "usd", "margin", "pajak", "tax", "invoice", "faktur", "transaksi", "roi",
            "arr", "ebitda", "penjualan", "sales", "omzet", "keuangan", "financial",
            "asset", "aset", "liabilitas", "ekuitas", "investasi",
        ),
    ),
    (
        "culinary",
        "Culinary & Recipe Guide",
        "Recipes, ingredient measurements, portions, and preparation steps.",
        (
            "resep", "masak", "memasak", "nasi", "beras", "bahan", "bumbu", "takaran",
            "sendok", "sdm", "sdt", "didih", "kukus", "goreng", "tumis", "panggang",
            "recipe", "cook", "cooking", "rice", "ingredient", "ingredients", "kitchen",
            "garam", "gula", "air ", "matang", "sajikan", "porsi", "kalori",
        ),
    ),
    (
        "technology",
        "Technology & Engineering",
        "Software architecture, technical specifications, system performance, or data engineering.",
        (
            "architecture", "arsitektur", "api", "endpoint", "server", "database",
            "sql", "latency", "throughput", "cloud", "docker", "kubernetes", "software",
            "engineering", "python", "typescript", "react", "algoritma", "algorithm",
            "sistem", "microservices", "deployment", "model", "machine learning",
        ),
    ),
    (
        "research",
        "Research & Academic",
        "Academic papers, methodology, experiments, and empirical research findings.",
        (
            "abstract", "abstrak", "metodologi", "methodology", "hipotesis", "hypothesis",
            "eksperimen", "experiment", "sampel", "sample", "penelitian", "research",
            "jurnal", "journal", "kesimpulan", "conclusion", "bab ", "skripsi", "tesis",
            "referensi", "literature", "signifikan", "korelasi", "variabel",
        ),
    ),
    (
        "legal",
        "Legal, Policy & SOPs",
        "Regulatory documents, contracts, compliance standards, or operating procedures.",
        (
            "pasal", "ayat", "perjanjian", "kontrak", "contract", "agreement", "clause",
            "kebijakan", "policy", "prosedur", "sop", "regulasi", "compliance", "audit",
            "kepatuhan", "hukum", "legal", "syarat", "ketentuan", "sanksi", "soc2",
        ),
    ),
    (
        "health",
        "Healthcare & Medicine",
        "Clinical notes, healthcare guidance, diagnostics, nutrition, or medical protocols.",
        (
            "pasien", "patient", "klinis", "clinical", "diagnosis", "terapi", "therapy",
            "obat", "medis", "medical", "kesehatan", "health", "dosis", "dosage",
            "gejala", "symptom", "rumah sakit", "dokter", "nutrisi",
        ),
    ),
    (
        "marketing",
        "Business, Market & Strategy",
        "Market research, customer retention, competitive landscape, and growth strategy.",
        (
            "market", "pasar", "pelanggan", "customer", "churn", "retention", "retensi",
            "kampanye", "campaign", "strategi", "strategy", "kompetitor", "competitor",
            "segment", "segmen", "brand", "pricing", "harga", "target", "growth",
            "pertumbuhan", "konversi", "conversion", "funnel",
        ),
    ),
]

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "dalam", "ini",
    "itu", "atau", "adalah", "sebagai", "oleh", "tidak", "akan", "dapat", "juga",
    "telah", "agar", "lebih", "saat", "jika", "bisa", "secara", "seperti", "serta",
    "bahwa", "antara", "tersebut", "karena", "hingga", "namun", "tetapi", "sudah",
    "masih", "harus", "sangat", "hanya", "semua", "setiap", "selama", "melalui",
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "been", "will", "into", "over", "such", "than", "then", "also",
    "about", "their", "there", "which", "when", "where", "while", "using", "used",
    "page", "halaman", "document", "dokumen", "file", "data", "table", "tabel",
}

TIME_COL_HINTS = (
    "date", "tanggal", "tgl", "month", "bulan", "year", "tahun", "quarter", "kuartal",
    "q1", "q2", "q3", "q4", "week", "minggu", "day", "hari", "period", "periode",
    "time", "waktu", "step", "tahap", "fase", "phase", "jan", "feb", "mar", "apr",
)

SHARE_COL_HINTS = (
    "%", "persen", "percent", "pct", "share", "porsi", "proporsi", "komposisi",
    "rasio", "ratio", "distribusi", "alokasi", "bobot", "weight",
)


def normalize_folder_name(raw: Optional[str]) -> str:
    if not raw or not str(raw).strip():
        return "General"
    cleaned = str(raw).strip()
    if cleaned.lower() in ("general_document", "general", "uncategorized"):
        return "General"
    return cleaned.replace("_", " ").title()


def _parse_numeric(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # Strip common currency & unit prefixes/suffixes
    cleaned = re.sub(r"(?i)^(rp\.?|idr|usd|\$|€|£)\s*", "", s)
    cleaned = re.sub(r"\s*(%|persen|pcs|unit|buah|kg|gr|gram|ml|liter|menit|jam|detik|juta|miliar|ribu|k|m|b)$", "", cleaned, flags=re.IGNORECASE).strip()
    # Handle Indonesian thousand dots vs decimals (e.g. 1.250.000 or 12,5)
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d+(,\d{1,4})", cleaned):
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        num = float(cleaned)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    except ValueError:
        return None


def _detect_domain_profile(
    docs: List[Dict[str, Any]],
    sample_texts: List[str],
    tables: List[Dict[str, Any]],
) -> Dict[str, str]:
    corpus_parts: List[str] = []
    for d in docs:
        corpus_parts.append(str(d.get("title") or ""))
        corpus_parts.append(str(d.get("category") or ""))
        corpus_parts.append(str(d.get("primary_domain") or ""))
        corpus_parts.append(str(d.get("summary") or ""))
        corpus_parts.extend(str(t) for t in (d.get("main_topics") or []))
    for tbl in tables[:6]:
        corpus_parts.append(str(tbl.get("table_name") or ""))
        for c in (tbl.get("columns_schema") or []):
            if isinstance(c, dict):
                corpus_parts.append(str(c.get("name") or ""))
    corpus_parts.extend(sample_texts[:24])

    haystack = " ".join(corpus_parts).lower()

    best_score = 0
    best_domain = (
        "general",
        "General Knowledge & Documents",
        "Overview of content, core topics, and key takeaways from your documents.",
    )
    for dom_id, label, tagline, keywords in DOMAIN_RULES:
        score = sum(haystack.count(kw) for kw in keywords)
        if score > best_score:
            best_score = score
            best_domain = (dom_id, label, tagline)

    return {
        "domain_id": best_domain[0],
        "domain_label": best_domain[1],
        "domain_tagline": best_domain[2],
    }


def _extract_top_terms(texts: List[str], existing_topics: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for t in existing_topics:
        cleaned = str(t).strip()
        if len(cleaned) >= 3 and cleaned.lower() not in STOPWORDS:
            counter[cleaned] += 3

    word_counts: Counter[str] = Counter()
    for txt in texts:
        for token in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]{3,24}", txt):
            low = token.lower()
            if low in STOPWORDS or low.isdigit():
                continue
            word_counts[low] += 1

    # Merge frequent words if existing_topics are sparse
    existing_lower = {k.lower() for k in counter}
    for word, cnt in word_counts.most_common(30):
        if word in existing_lower or cnt < 2:
            continue
        counter[word.title()] += cnt

    top = counter.most_common(limit)
    if not top:
        return []
    max_val = max(v for _, v in top) or 1
    return [
        {
            "label": k,
            "mentions": v,
            "weight": round((v / max_val) * 100),
        }
        for k, v in top
    ]


def _extract_qualitative_highlights(
    docs: List[Dict[str, Any]],
    chunks_by_doc: Dict[str, List[Dict[str, Any]]],
    max_items: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract grounded key findings, steps/procedures, and notable excerpts directly
    from real document chunks (with page provenance).
    """
    steps_or_sections: List[Dict[str, Any]] = []
    evidence_cards: List[Dict[str, Any]] = []
    seen_snippets: set[str] = set()

    step_regex = re.compile(r"^(?:langkah\s*\d+|tahap\s*\d+|step\s*\d+|\d+[\.\)]\s+|[-•*]\s+)(.+)$", re.IGNORECASE)

    for doc in docs:
        doc_id = str(doc["id"])
        doc_title = str(doc.get("title") or "Document")
        chunks = chunks_by_doc.get(doc_id, [])

        for ch in chunks[:20]:
            text = (ch.get("text_content") or ch.get("content") or "").strip()
            if not text:
                continue
            page_num = int(ch.get("page_number") or 1)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for ln in lines:
                m = step_regex.match(ln)
                if m and len(steps_or_sections) < max_items:
                    body = " ".join(m.group(1).strip().split())
                    sig = body[:45].lower()
                    if len(body) >= 14 and sig not in seen_snippets:
                        seen_snippets.add(sig)
                        detail = body[:260] + ("..." if len(body) > 260 else "")
                        steps_or_sections.append(
                            {
                                "title": body[:78] + ("..." if len(body) > 78 else ""),
                                "detail": detail,
                                "doc_id": doc_id,
                                "doc_title": doc_title,
                                "page_number": page_num,
                            }
                        )

            # High-signal factual sentences
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(lines))
            for sent in sentences:
                s = sent.strip()
                if len(s) < 35 or len(s) > 260:
                    continue
                sig = s[:48].lower()
                if sig in seen_snippets:
                    continue
                # Prioritize sentences with concrete facts, numbers, or conclusive cues
                has_signal = bool(
                    re.search(r"\d", s)
                    or any(
                        cue in s.lower()
                        for cue in (
                            "utama", "penting", "kesimpulan", "hasil", "tujuan",
                            "menunjukkan", "meningkat", "menurun", "total", "rata-rata",
                            "key", "finding", "conclusion", "result", "increase", "decrease",
                            "pastikan", "gunakan", "rekomendasi", "risiko", "tantangan",
                        )
                    )
                )
                if has_signal and len(evidence_cards) < max_items:
                    seen_snippets.add(sig)
                    evidence_cards.append(
                        {
                            "tag": "Key Finding" if re.search(r"\d", s) else "Content Highlight",
                            "title": s[:72] + ("..." if len(s) > 72 else ""),
                            "detail": s,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "page_number": page_num,
                        }
                    )

    # Backfill evidence cards from opening paragraphs if document has plain prose
    if len(evidence_cards) < 3:
        for doc in docs:
            doc_id = str(doc["id"])
            doc_title = str(doc.get("title") or "Document")
            for ch in chunks_by_doc.get(doc_id, [])[:6]:
                text = " ".join((ch.get("text_content") or ch.get("content") or "").split())
                if len(text) < 30:
                    continue
                snippet = text[:240] + ("..." if len(text) > 240 else "")
                sig = snippet[:48].lower()
                if sig not in seen_snippets:
                    seen_snippets.add(sig)
                    evidence_cards.append(
                        {
                            "tag": f"Summary p.{int(ch.get('page_number') or 1)}",
                            "title": doc_title,
                            "detail": snippet,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "page_number": int(ch.get("page_number") or 1),
                        }
                    )
                if len(evidence_cards) >= 5:
                    break

    return steps_or_sections[:max_items], evidence_cards[:max_items]


def _extract_inline_numeric_series(
    docs: List[Dict[str, Any]],
    chunks_by_doc: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    When a document has no extracted SQL tables or metrics yet, carefully mine genuine
    labeled numeric measurements from text (e.g. 'Beras: 500 gram', 'Pendapatan: Rp 120 juta',
    'Tahap 1: 15 menit', 'Margin: 32%') grouped strictly by homogeneous unit.
    Only returns charts when >= 3 distinct labeled items share the same real unit.
    """
    # Matches patterns like "Label: 500 gram", "Biaya Operasional - Rp 45.000", "Waktu Kukus (25 menit)"
    pair_pattern = re.compile(
        r"(?P<label>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 /&()-]{2,36}?)\s*(?::|=|-|–|sebesar|adalah)\s*"
        r"(?P<prefix>Rp\.?|IDR|USD|\$)?\s*(?P<num>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*"
        r"(?P<unit>%|persen|juta|miliar|ribu|gram|gr|kg|ml|liter|l\b|sdm|sdt|menit|jam|detik|hari|bulan|tahun|orang|unit|pcs|buah|kalori|kkal)?",
        re.IGNORECASE,
    )

    unit_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_labels_per_unit: Dict[str, set[str]] = defaultdict(set)

    for doc in docs:
        doc_id = str(doc["id"])
        doc_title = str(doc.get("title") or "Document")
        for ch in chunks_by_doc.get(doc_id, [])[:24]:
            text = ch.get("text_content") or ch.get("content") or ""
            page_num = int(ch.get("page_number") or 1)
            for ln in text.splitlines():
                line = ln.strip()
                if not line or len(line) > 220:
                    continue
                for m in pair_pattern.finditer(line):
                    raw_label = re.sub(r"^[\d\.\-\*\•\)\s]+", "", m.group("label")).strip()
                    if len(raw_label) < 3 or raw_label.lower() in STOPWORDS:
                        continue
                    prefix = (m.group("prefix") or "").strip().upper()
                    raw_unit = (m.group("unit") or "").strip().lower()
                    num = _parse_numeric(m.group("num"))
                    if num is None or num <= 0:
                        continue
                    # Ignore bare years or page numbers without units
                    if not prefix and not raw_unit:
                        continue
                    if raw_unit in ("tahun",) and 1900 <= num <= 2100:
                        continue

                    if prefix in ("RP", "RP.", "IDR"):
                        unit_key = f"IDR ({raw_unit})" if raw_unit in ("juta", "miliar", "ribu") else "IDR"
                    elif prefix in ("USD", "$"):
                        unit_key = f"USD ({raw_unit})" if raw_unit in ("juta", "miliar", "ribu") else "USD"
                    elif raw_unit in ("%", "persen"):
                        unit_key = "%"
                    elif raw_unit in ("gr", "gram"):
                        unit_key = "gram"
                    elif raw_unit in ("l", "liter"):
                        unit_key = "liter"
                    else:
                        unit_key = raw_unit

                    label_sig = raw_label.lower()
                    if label_sig in seen_labels_per_unit[unit_key]:
                        continue
                    seen_labels_per_unit[unit_key].add(label_sig)
                    unit_buckets[unit_key].append(
                        {
                            "label": raw_label[:32],
                            "value": round(num, 2),
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "page_number": page_num,
                        }
                    )

    mined_charts: List[Dict[str, Any]] = []
    ranked_units = sorted(unit_buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for idx, (unit_key, items) in enumerate(ranked_units[:2]):
        if len(items) < 3:
            continue
        top_items = sorted(items, key=lambda x: x["value"], reverse=True)[:10]
        labels = [it["label"] for it in top_items]
        values = [it["value"] for it in top_items]
        is_share = unit_key == "%" and len(top_items) <= 7
        is_duration = unit_key in ("menit", "jam", "detik")
        chart_type = "donut" if is_share else ("horizontal_bar" if len(top_items) >= 5 or is_duration else "bar")
        top_item = top_items[0]
        mined_charts.append(
            {
                "id": f"mined_chart_{idx + 1}",
                "title": f"Extracted Value Distribution ({unit_key})",
                "subtitle": f"Extracted directly from document text in {unit_key}",
                "chart_type": chart_type,
                "unit": unit_key,
                "labels": labels,
                "series": [{"name": unit_key, "data": values}],
                "rows": [
                    {"Item": it["label"], f"Value ({unit_key})": it["value"], "Source": f"{it['doc_title']} (p.{it['page_number']})"}
                    for it in top_items
                ],
                "takeaway": (
                    f"Highest value recorded for '{top_item['label']}' at {top_item['value']:g} {unit_key} "
                    f"(across {len(top_items)} items detected in {unit_key})."
                ),
                "source_docs": sorted({it["doc_id"] for it in top_items}),
                "source_labels": list(dict.fromkeys(f"{it['doc_title']} · Hal. {it['page_number']}" for it in top_items))[:4],
                "sql_query": None,
            }
        )
    return mined_charts


def _build_charts_from_tables(
    tables: List[Dict[str, Any]],
    doc_by_id: Dict[str, Dict[str, Any]],
    max_charts: int = 4,
) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    used_types: List[str] = []

    for idx, tbl in enumerate(tables):
        if len(charts) >= max_charts:
            break
        rows_data = tbl.get("rows_data") or []
        if not isinstance(rows_data, list) or len(rows_data) < 2:
            continue

        sample_row = rows_data[0]
        if not isinstance(sample_row, dict):
            continue

        col_names = list(sample_row.keys())
        if len(col_names) < 2:
            continue

        # Identify best label column (non-numeric) and numeric columns
        numeric_cols: List[str] = []
        label_col: Optional[str] = None
        for col in col_names:
            parsed_vals = [_parse_numeric(r.get(col)) for r in rows_data[:25]]
            valid_ratio = sum(1 for v in parsed_vals if v is not None) / max(len(parsed_vals), 1)
            if valid_ratio >= 0.65:
                numeric_cols.append(col)
            elif label_col is None:
                label_col = col

        if not numeric_cols:
            continue
        if label_col is None:
            label_col = col_names[0]
            numeric_cols = [c for c in numeric_cols if c != label_col]
        if not numeric_cols:
            continue

        sliced_rows = rows_data[:16]
        labels = [str(r.get(label_col) or f"Item {i + 1}").strip()[:34] for i, r in enumerate(sliced_rows)]

        # Pick up to 2 numeric series (or 3 if grouped comparison)
        selected_num_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1]
        series_list: List[Dict[str, Any]] = []
        for ncol in selected_num_cols:
            vals = [round(_parse_numeric(r.get(ncol)) or 0.0, 2) for r in sliced_rows]
            if any(v != 0 for v in vals):
                series_list.append({"name": str(ncol), "data": vals})

        if not series_list:
            continue

        # Infer unit from columns_schema
        schema_cols = tbl.get("columns_schema") or []
        unit = ""
        for sc in schema_cols:
            if isinstance(sc, dict) and sc.get("name") == selected_num_cols[0]:
                unit = str(sc.get("unit_or_currency") or "")
                break
        if not unit:
            first_col_low = selected_num_cols[0].lower()
            if "%" in first_col_low or "persen" in first_col_low or "rate" in first_col_low:
                unit = "%"
            elif any(k in first_col_low for k in ("rp", "idr")):
                unit = "IDR"
            elif any(k in first_col_low for k in ("usd", "$")):
                unit = "USD"
            else:
                unit = selected_num_cols[0]

        # Intelligent chart type selection for real visual variety!
        lbl_col_low = str(label_col).lower()
        num_col_low = str(selected_num_cols[0]).lower()
        all_positive = all(v >= 0 for v in series_list[0]["data"])
        is_time_axis = any(h in lbl_col_low for h in TIME_COL_HINTS) or any(
            re.search(r"(?i)^(q[1-4]|jan|feb|mar|apr|mei|may|jun|jul|agu|aug|sep|okt|oct|nov|des|dec|20\d{2}|tahap|step|bulan)", lb)
            for lb in labels[:4]
        )
        is_share_metric = any(h in num_col_low for h in SHARE_COL_HINTS) or unit == "%"

        if len(series_list) >= 2:
            chart_type = "line" if is_time_axis else "grouped_bar"
        elif is_time_axis and len(labels) >= 3:
            chart_type = "area"
        elif is_share_metric and all_positive and 2 <= len(labels) <= 7:
            chart_type = "donut"
        elif all_positive and 3 <= len(labels) <= 6 and "donut" not in used_types and len(charts) >= 1:
            chart_type = "donut"
        elif len(labels) >= 6 or max((len(lb) for lb in labels), default=0) > 15:
            chart_type = "horizontal_bar"
        else:
            chart_type = "bar"

        used_types.append(chart_type)

        doc_id = str(tbl.get("doc_id") or "")
        doc_title = str(doc_by_id.get(doc_id, {}).get("title") or "Document")
        page_num = int(tbl.get("page_number") or 1)
        raw_tname = str(tbl.get("table_name") or f"Table {idx + 1}").replace("_", " ").strip().title()

        primary_vals = series_list[0]["data"]
        max_idx = max(range(len(primary_vals)), key=lambda i: primary_vals[i])
        min_idx = min(range(len(primary_vals)), key=lambda i: primary_vals[i])
        total_val = round(sum(primary_vals), 2)

        if chart_type in ("area", "line") and len(primary_vals) >= 2:
            delta = primary_vals[-1] - primary_vals[0]
            trend_word = "increased" if delta > 0 else ("decreased" if delta < 0 else "remained stable")
            takeaway = (
                f"{selected_num_cols[0]} trend {trend_word} from {primary_vals[0]:g} ({labels[0]}) "
                f"to {primary_vals[-1]:g} ({labels[-1]}), peaking at {labels[max_idx]} ({primary_vals[max_idx]:g})."
            )
        elif chart_type == "donut" and total_val > 0:
            top_pct = round((primary_vals[max_idx] / total_val) * 100, 1)
            takeaway = (
                f"'{labels[max_idx]}' accounts for {top_pct}% of the total "
                f"({primary_vals[max_idx]:g} out of {total_val:g})."
            )
        else:
            takeaway = (
                f"Highest: '{labels[max_idx]}' ({primary_vals[max_idx]:g}) · "
                f"Lowest: '{labels[min_idx]}' ({primary_vals[min_idx]:g}) across {len(labels)} rows."
            )

        quoted_cols = ", ".join(f'"{c}"' for c in selected_num_cols)
        sql_table_name = tbl.get("table_name") or f"table_{idx + 1}"
        charts.append(
            {
                "id": f"tbl_chart_{idx + 1}",
                "title": f"{raw_tname}: {selected_num_cols[0]}",
                "subtitle": f"From {doc_title} · Page {page_num} ({len(rows_data)} rows)",
                "chart_type": chart_type,
                "unit": unit,
                "labels": labels,
                "series": series_list,
                "rows": sliced_rows[:12],
                "takeaway": takeaway,
                "source_docs": [doc_id] if doc_id else [],
                "source_labels": [f"{doc_title} · p.{page_num}"],
                "sql_query": f'SELECT "{label_col}", {quoted_cols} FROM "{sql_table_name}" LIMIT 16',
            }
        )

    return charts


def _build_charts_from_metrics(
    metrics: List[Dict[str, Any]],
    doc_by_id: Dict[str, Dict[str, Any]],
    existing_count: int,
    existing_types: Optional[set[str]] = None,
    max_charts: int = 3,
) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    if not metrics:
        return charts
    used_types = set(existing_types or set())

    # Group strictly by homogeneous unit_or_currency (Gate 4 Unit Homogeneity!)
    by_unit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        unit = str(m.get("unit_or_currency") or "value").strip() or "value"
        by_unit[unit].append(m)

    sorted_groups = sorted(by_unit.items(), key=lambda kv: len(kv[1]), reverse=True)
    for idx, (unit, grp) in enumerate(sorted_groups):
        if len(charts) >= max_charts:
            break
        # Deduplicate metric names cleanly (keep max/latest value)
        dedup: Dict[str, Dict[str, Any]] = {}
        for m in grp:
            name = str(m.get("metric_name") or m.get("canonical_metric_key") or "Metrik").strip()
            val = _parse_numeric(m.get("numeric_value"))
            if not name or val is None:
                continue
            period = str(m.get("period_label") or "").strip()
            display_label = f"{name} ({period})" if period and period.lower() not in ("general", "umum", "") else name
            if display_label not in dedup or abs(val) > abs(dedup[display_label]["value"]):
                dedup[display_label] = {
                    "label": display_label[:34],
                    "value": round(val, 2),
                    "period": period,
                    "doc_id": str(m.get("doc_id") or ""),
                }

        items = sorted(dedup.values(), key=lambda x: x["value"], reverse=True)[:12]
        if len(items) < 2:
            continue

        labels = [it["label"] for it in items]
        values = [it["value"] for it in items]
        periods = {it["period"] for it in items if it["period"] and it["period"].lower() not in ("general", "umum")}
        all_pos = all(v >= 0 for v in values)
        total_val = sum(values)
        looks_like_composition = (
            unit == "%"
            and all_pos
            and 3 <= len(items) <= 7
            and 92.0 <= total_val <= 108.0
            and not any(kw in " ".join(labels).lower() for kw in ("margin", "roe", "roa", "roi", "growth", "yoy", "cagr"))
        )

        if len(periods) >= 3:
            chart_type = "line"
        elif looks_like_composition and "donut" not in used_types:
            chart_type = "donut"
        elif "horizontal_bar" not in used_types:
            chart_type = "horizontal_bar"
        elif "bar" not in used_types:
            chart_type = "bar"
        elif (existing_count + len(charts)) % 2 == 1:
            chart_type = "horizontal_bar"
        else:
            chart_type = "bar"
        used_types.add(chart_type)

        src_docs = sorted({it["doc_id"] for it in items if it["doc_id"]})
        src_labels = [
            str(doc_by_id.get(did, {}).get("title") or "Document")
            for did in src_docs[:4]
        ]

        charts.append(
            {
                "id": f"metric_chart_{idx + 1}",
                "title": f"Verified Metrics Comparison ({unit})",
                "subtitle": f"{len(items)} verified metrics in {unit}",
                "chart_type": chart_type,
                "unit": unit,
                "labels": labels,
                "series": [{"name": unit, "data": values}],
                "rows": [
                    {
                        "Metric": it["label"],
                        f"Value ({unit})": it["value"],
                        "Document": str(doc_by_id.get(it["doc_id"], {}).get("title") or "-"),
                    }
                    for it in items
                ],
                "takeaway": (
                    f"Highest metric is '{items[0]['label']}' ({items[0]['value']:g} {unit}) "
                    f"while lowest is '{items[-1]['label']}' ({items[-1]['value']:g} {unit})."
                ),
                "source_docs": src_docs,
                "source_labels": src_labels,
                "sql_query": (
                    f"SELECT metric_name, ROUND(AVG(numeric_value), 2) AS value FROM workspace_metrics "
                    f"WHERE unit_or_currency = '{unit.replace(chr(39), '')}' GROUP BY metric_name ORDER BY value DESC LIMIT 12"
                ),
            }
        )

    return charts


def _build_all_files_portfolio_charts(
    docs: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Real cross-repository charts ONLY for All Files / Folder scopes with >= 2 documents.
    Plots genuine volume, structure, category distribution, and data richness across files.
    """
    if len(docs) < 2:
        return []

    charts: List[Dict[str, Any]] = []

    # 1. Category / Folder Distribution Donut
    folder_counts: Counter[str] = Counter()
    folder_pages: Counter[str] = Counter()
    for d in docs:
        f = normalize_folder_name(d.get("category"))
        folder_counts[f] += 1
        folder_pages[f] += int(d.get("page_count") or 1)

    if len(folder_counts) >= 2:
        sorted_folders = folder_counts.most_common(8)
        charts.append(
            {
                "id": "repo_folder_donut",
                "title": "Repository Breakdown by Category",
                "subtitle": f"Distribution of {len(docs)} files across {len(sorted_folders)} categories",
                "chart_type": "donut",
                "unit": "files",
                "labels": [k for k, _ in sorted_folders],
                "series": [{"name": "File Count", "data": [v for _, v in sorted_folders]}],
                "rows": [
                    {"Category": k, "Files": v, "Total Pages": folder_pages[k]}
                    for k, v in sorted_folders
                ],
                "takeaway": (
                    f"Largest category is '{sorted_folders[0][0]}' with {sorted_folders[0][1]} files "
                    f"({folder_pages[sorted_folders[0][0]]} pages)."
                ),
                "source_docs": [str(d["id"]) for d in docs[:10]],
                "source_labels": [f"{len(docs)} files in repository"],
                "sql_query": None,
            }
        )

    # 2. Per-File Depth & Data Density (Grouped Bar or Horizontal Bar)
    table_by_doc: Counter[str] = Counter(str(t.get("doc_id") or "") for t in tables)
    metric_by_doc: Counter[str] = Counter(str(m.get("doc_id") or "") for m in metrics)
    link_by_doc: Counter[str] = Counter()
    for r in relationships:
        s = str(r.get("source_doc_id") or "")
        t = str(r.get("target_doc_id") or "")
        if s:
            link_by_doc[s] += 1
        if t:
            link_by_doc[t] += 1

    ranked_docs = sorted(
        docs,
        key=lambda d: (
            int(d.get("page_count") or 1)
            + metric_by_doc[str(d["id"])] * 2
            + table_by_doc[str(d["id"])] * 3
        ),
        reverse=True,
    )[:10]

    doc_labels = [str(d.get("title") or "Document")[:26] for d in ranked_docs]
    pages_arr = [int(d.get("page_count") or 1) for d in ranked_docs]
    metrics_arr = [metric_by_doc[str(d["id"])] or int(d.get("metric_count") or 0) for d in ranked_docs]
    links_arr = [link_by_doc[str(d["id"])] for d in ranked_docs]

    has_multi_signals = any(m > 0 for m in metrics_arr) or any(l > 0 for l in links_arr)
    charts.append(
        {
            "id": "repo_file_comparison",
            "title": "Cross-File Volume & Data Depth Comparison",
            "subtitle": f"Comparing pages, metrics, and links across {len(ranked_docs)} files",
            "chart_type": "grouped_bar" if has_multi_signals else "horizontal_bar",
            "unit": "count",
            "labels": doc_labels,
            "series": (
                [
                    {"name": "Pages", "data": pages_arr},
                    {"name": "Metrics", "data": metrics_arr},
                    {"name": "Links", "data": links_arr},
                ]
                if has_multi_signals
                else [{"name": "Pages", "data": pages_arr}]
            ),
            "rows": [
                {
                    "Document": str(d.get("title") or "Document"),
                    "Pages": pages_arr[i],
                    "Tables": table_by_doc[str(d["id"])] or int(d.get("table_count") or 0),
                    "Metrics": metrics_arr[i],
                    "Links": links_arr[i],
                }
                for i, d in enumerate(ranked_docs)
            ],
            "takeaway": (
                f"Densest file in the repository is '{ranked_docs[0].get('title')}' "
                f"({pages_arr[0]} pages, {metrics_arr[0]} metrics, {links_arr[0]} links)."
            ),
            "source_docs": [str(d["id"]) for d in ranked_docs],
            "source_labels": [str(d.get("title") or "Document") for d in ranked_docs[:4]],
            "sql_query": None,
        }
    )

    return charts


async def build_dynamic_analytics_report(
    workspace_id: str,
    user_id: str,
    scope_mode: str = "all_files",
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    folder_name: Optional[str] = None,
    focus_prompt: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Build a comprehensive, domain-adaptive, 100% grounded analytics report for either:
    - single_file: deep breakdown of 1 chosen document
    - folder: synthesis of all files in a chosen repository folder
    - all_files: full workspace synthesis across EVERY uploaded document
    Automatically selects diverse chart types when quantitative data exists, and cleanly
    replaces charts with deep structured qualitative insights when no chart data exists.
    Backed by a workspace-fingerprinted TTL cache (Anti-Boros & fast repeat loads).
    """
    all_ws_docs = supabase_db.list_workspace_documents(workspace_id)
    if not all_ws_docs:
        return {
            "scope_mode": scope_mode,
            "scope_label": "Empty Workspace",
            "presentation_mode": "empty",
            "domain": {
                "domain_id": "empty",
                "domain_label": "No Documents Yet",
                "domain_tagline": "Upload documents first to generate dynamic analytics.",
            },
            "executive_summary": "No documents exist in this workspace yet.",
            "no_chart_reason": "Upload at least 1 PDF, CSV, or TXT file.",
            "what_is_inside": [],
            "kpis": [],
            "charts": [],
            "insights": [],
            "steps_or_sections": [],
            "top_topics": [],
            "file_digests": [],
            "suggested_questions": [],
            "target_doc_ids": [],
        }

    # 1. Resolve Target Scope Documents
    if scope_mode == "single_file" and document_id:
        target_docs = [d for d in all_ws_docs if str(d.get("id")) == str(document_id)]
        if not target_docs:
            raise ValueError("The selected document was not found in your workspace.")
        scope_label = str(target_docs[0].get("title") or "1 Document")
    elif scope_mode == "folder" and folder_name:
        target_norm = normalize_folder_name(folder_name).lower()
        target_docs = [
            d for d in all_ws_docs if normalize_folder_name(d.get("category")).lower() == target_norm
        ]
        if not target_docs:
            raise ValueError(f"No documents found in folder '{folder_name}'.")
        scope_label = f"Repo {normalize_folder_name(folder_name)} ({len(target_docs)} files)"
    elif document_ids:
        id_set = {str(x) for x in document_ids}
        target_docs = [d for d in all_ws_docs if str(d.get("id")) in id_set]
        if not target_docs:
            raise ValueError("Selected documents were not found.")
        scope_label = (
            str(target_docs[0].get("title"))
            if len(target_docs) == 1
            else f"{len(target_docs)} Selected Documents"
        )
    else:
        scope_mode = "all_files"
        target_docs = all_ws_docs
        scope_label = f"Full Repository ({len(target_docs)} files)"

    target_doc_ids = [str(d["id"]) for d in target_docs]
    doc_by_id = {str(d["id"]): d for d in target_docs}

    # 1b. Smart TTL Cache Check (Anti-Boros: skips redundant DB/LLM work when docs haven't changed)
    fp_parts = sorted(
        f"{d.get('id')}:{d.get('status')}:{d.get('table_count')}:{d.get('metric_count')}:{d.get('page_count')}:{d.get('category')}"
        for d in target_docs
    )
    fp_raw = f"{scope_mode}|{document_id or ''}|{folder_name or ''}|{(focus_prompt or '').strip().lower()}|{'|'.join(fp_parts)}"
    cache_key = f"an:{workspace_id}:{hashlib.sha256(fp_raw.encode('utf-8')).hexdigest()[:24]}"
    if not force_refresh:
        cached_report = analytics_report_cache.get(cache_key)
        if cached_report is not None:
            return {**cached_report, "cache_hit": True}

    # 2. Load ALL Real Grounded Data (Tables, Metrics, Chunks, Relationships)
    tables = supabase_db.get_workspace_tables(workspace_id, doc_ids=target_doc_ids)
    metrics = supabase_db.get_workspace_metrics(workspace_id, doc_ids=target_doc_ids)
    all_rels = supabase_db.list_workspace_relationships(workspace_id)
    target_id_set = set(target_doc_ids)
    scoped_rels = [
        r for r in all_rels
        if str(r.get("source_doc_id")) in target_id_set and str(r.get("target_doc_id")) in target_id_set
    ]

    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    sample_texts: List[str] = []
    existing_topics: List[str] = []
    for d in target_docs:
        did = str(d["id"])
        existing_topics.extend(str(t) for t in (d.get("main_topics") or []) if t)
        doc_chunks = supabase_db.get_document_chunks(workspace_id, did)
        chunks_by_doc[did] = doc_chunks
        for ch in doc_chunks[:8]:
            txt = (ch.get("text_content") or ch.get("content") or "").strip()
            if txt:
                sample_texts.append(txt)

    # 3. Domain & Content Understanding
    domain = _detect_domain_profile(target_docs, sample_texts, tables)
    top_topics = _extract_top_terms(sample_texts, existing_topics, limit=10)
    steps_or_sections, evidence_cards = _extract_qualitative_highlights(target_docs, chunks_by_doc, max_items=8)

    # 4. Build Real Multi-Type Charts
    charts: List[Dict[str, Any]] = []
    charts.extend(_build_charts_from_tables(tables, doc_by_id, max_charts=3))
    charts.extend(
        _build_charts_from_metrics(
            metrics,
            doc_by_id,
            existing_count=len(charts),
            existing_types={c["chart_type"] for c in charts},
            max_charts=2,
        )
    )

    # If still no structured charts from tables/metrics, attempt grounded inline numeric mining
    if not charts:
        charts.extend(_extract_inline_numeric_series(target_docs, chunks_by_doc))

    has_quantitative_data = len(charts) > 0

    # For All Files / Folder with >= 2 docs, enrich with real portfolio comparison charts
    if scope_mode != "single_file" and len(target_docs) >= 2:
        portfolio_charts = _build_all_files_portfolio_charts(target_docs, tables, metrics, scoped_rels)
        # Keep total charts crisp (up to 5)
        for pc in portfolio_charts:
            if len(charts) < 5:
                charts.append(pc)

    # Presentation mode: if single_file has ZERO quantitative charts -> replace charts cleanly with deep insights!
    presentation_mode = "charts_and_insights" if len(charts) > 0 else "insights_only"
    no_chart_reason: Optional[str] = None
    if presentation_mode == "insights_only":
        no_chart_reason = (
            f"This document focuses on qualitative/narrative content ({domain['domain_label']}) "
            "without numeric tables or metric series suitable for charting; the view automatically "
            "highlights In-Depth Insights, Key Takeaways, and Dominant Topics."
        )

    # 5. Build Grounded KPI Cards
    total_pages = sum(int(d.get("page_count") or 1) for d in target_docs)
    kpis: List[Dict[str, Any]] = []
    if scope_mode == "single_file":
        doc0 = target_docs[0]
        kpis = [
            {"label": "Detected Domain", "value": domain["domain_label"], "sub": normalize_folder_name(doc0.get("category")), "tone": "accent"},
            {"label": "Page Volume", "value": f"{total_pages} pp", "sub": str(doc0.get("file_type") or "PDF").upper(), "tone": "ink"},
            {"label": "Tables & Metrics", "value": f"{len(tables)} / {len(metrics)}", "sub": "Tables · Numeric Metrics", "tone": "sage"},
            {"label": "Core Concepts", "value": f"{len(top_topics)} Topics", "sub": top_topics[0]["label"] if top_topics else "Indexed", "tone": "neutral"},
        ]
    else:
        kpis = [
            {"label": "Files Analyzed", "value": str(len(target_docs)), "sub": f"{total_pages} total pages", "tone": "accent"},
            {"label": "Primary Domain", "value": domain["domain_label"], "sub": f"{len({normalize_folder_name(d.get('category')) for d in target_docs})} categories", "tone": "ink"},
            {"label": "Tables & Metrics", "value": f"{len(tables)} / {len(metrics)}", "sub": "Cross-file quantitative data", "tone": "sage"},
            {"label": "Cross-File Links", "value": f"{len(scoped_rels)} Links", "sub": f"{len(top_topics)} key topics", "tone": "neutral"},
        ]

    # 6. Build What-Is-Inside Badges & Grounded Executive Summary
    what_is_inside: List[str] = [domain["domain_label"]]
    if len(tables) > 0:
        what_is_inside.append(f"{len(tables)} data tables")
    if len(metrics) > 0:
        what_is_inside.append(f"{len(metrics)} numeric metrics")
    if not has_quantitative_data:
        what_is_inside.append("Narrative & qualitative content")
    if steps_or_sections:
        what_is_inside.append(f"{len(steps_or_sections)} structured sections")
    if scope_mode != "single_file" and scoped_rels:
        what_is_inside.append(f"{len(scoped_rels)} cross-document links")

    # Deterministic grounded executive summary
    top_topic_names = ", ".join(t["label"] for t in top_topics[:4])
    if scope_mode == "single_file":
        doc0 = target_docs[0]
        raw_sum = str(doc0.get("summary") or "").strip()
        if raw_sum and len(raw_sum) >= 30:
            executive_summary = raw_sum
        elif evidence_cards:
            executive_summary = (
                f"Document '{doc0.get('title')}' falls under {domain['domain_label']} "
                f"({total_pages} pages). {evidence_cards[0]['detail']}"
            )
        else:
            executive_summary = (
                f"Document '{doc0.get('title')}' is classified under {domain['domain_label']} "
                f"spanning {total_pages} pages focusing on {top_topic_names or 'key document information'}."
            )
    else:
        executive_summary = (
            f"Comprehensive synthesis across {len(target_docs)} documents ({total_pages} pages) with primary domain "
            f"{domain['domain_label']}. Extracted {len(tables)} structured tables, {len(metrics)} verified metrics, "
            f"and {len(scoped_rels)} cross-document links around core themes including {top_topic_names or 'multiple topics'}."
        )

    # 7. Cross-Document Relationship Insights (for All Files / Folder)
    insights: List[Dict[str, Any]] = []
    if scope_mode != "single_file" and scoped_rels:
        for rel in sorted(scoped_rels, key=lambda r: float(r.get("composite_confidence") or 0), reverse=True)[:3]:
            src_t = str(doc_by_id.get(str(rel.get("source_doc_id")), {}).get("title") or "Document A")
            tgt_t = str(doc_by_id.get(str(rel.get("target_doc_id")), {}).get("title") or "Document B")
            rtype = str(rel.get("relationship_type") or "semantic_similarity")
            tag = "Metric Contradiction" if rtype == "metric_contradiction" else "Cross-File Link"
            insights.append(
                {
                    "tag": tag,
                    "title": f"{src_t} <-> {tgt_t}",
                    "detail": str(rel.get("explanation") or "Both documents share key topics or entities."),
                    "doc_id": str(rel.get("source_doc_id") or ""),
                    "doc_title": src_t,
                    "page_number": 1,
                }
            )

    for ev in evidence_cards:
        if len(insights) >= 6:
            break
        insights.append(ev)

    # 8. Suggested Contextual Questions (for 1-click handoff to Chat!)
    if scope_mode == "single_file":
        t_name = str(target_docs[0].get("title") or "this file")
        suggested_questions = [
            f"Explain the core findings and conclusions from {t_name}",
            (
                f"Analyze the most critical metrics and trends in {t_name}"
                if has_quantitative_data
                else f"What are the most important steps or points to note in {t_name}?"
            ),
            f"Are there any key risks, caveats, or recommendations noted in {t_name}?",
        ]
    else:
        suggested_questions = [
            "Summarize the most critical findings across all files in my repository",
            "Compare key data, metrics, or notable differences across documents",
            f"What overarching patterns relate to {top_topics[0]['label']} across documents?" if top_topics else "Which documents are most strongly linked and why?",
        ]

    # 9. Optional Fail-Safe AI Narrative Enrichment (polishes summary & insights when OpenAI is live)
    if ai_gateway.is_openai_configured and sample_texts:
        try:
            excerpt_blob = "\n---\n".join(sample_texts[:6])[:3800]
            chart_titles = ", ".join(c["title"] for c in charts) or "No quantitative charts"
            enriched = await ai_gateway.generate_structured(
                task_type=AITaskType.DASHBOARD_SQL_PLANNING,
                workspace_id=workspace_id,
                user_id=user_id,
                instructions=(
                    "You are an expert DocuMind document analyst. Based on the raw document excerpts and the list of "
                    "deterministically computed charts, provide an accurate English executive summary, an appropriate "
                    "domain label (e.g., Finance & Revenue, Culinary Guide, Academic Research, Operations SOP), 3-4 crisp "
                    "insights grounded strictly in the source excerpts (DO NOT fabricate numbers or facts), and 3 follow-up questions."
                ),
                input_blocks=(
                    f"SCOPE: {scope_label}\n"
                    f"USER FOCUS: {focus_prompt or 'Comprehensive overview'}\n"
                    f"AVAILABLE CHARTS: {chart_titles}\n"
                    f"SOURCE EXCERPTS:\n{excerpt_blob}"
                ),
                response_model=AnalyticsNarrativeEnrichment,
                scope_key=f"analytics_report:{document_id or folder_name or 'all'}",
            )
            if enriched.domain_label:
                domain["domain_label"] = enriched.domain_label
            if enriched.executive_summary:
                executive_summary = enriched.executive_summary
            if enriched.key_insights:
                ai_insights = [
                    {
                        "tag": str(item.get("tag") or "AI Insight"),
                        "title": str(item.get("title") or "Key Finding"),
                        "detail": str(item.get("detail") or ""),
                        "doc_id": target_doc_ids[0] if len(target_doc_ids) == 1 else "",
                        "doc_title": scope_label,
                        "page_number": 1,
                    }
                    for item in enriched.key_insights[:4]
                    if item.get("detail")
                ]
                if ai_insights:
                    insights = ai_insights + insights[: max(0, 6 - len(ai_insights))]
            if enriched.suggested_questions:
                suggested_questions = [str(q) for q in enriched.suggested_questions[:3] if q]
        except Exception:
            # Cleanly keep deterministic grounded analysis when OpenAI is unavailable/expired
            pass

    # 10. Per-File Digests (vital for All Files & Folder views)
    table_counts = Counter(str(t.get("doc_id") or "") for t in tables)
    metric_counts = Counter(str(m.get("doc_id") or "") for m in metrics)
    rel_counts: Counter[str] = Counter()
    for r in scoped_rels:
        rel_counts[str(r.get("source_doc_id") or "")] += 1
        rel_counts[str(r.get("target_doc_id") or "")] += 1

    file_digests: List[Dict[str, Any]] = []
    for d in target_docs:
        did = str(d["id"])
        doc_texts = [
            (c.get("text_content") or c.get("content") or "").strip()
            for c in chunks_by_doc.get(did, [])[:4]
        ]
        doc_dom = _detect_domain_profile([d], doc_texts, [t for t in tables if str(t.get("doc_id")) == did])
        t_cnt = table_counts[did] or int(d.get("table_count") or 0)
        m_cnt = metric_counts[did] or int(d.get("metric_count") or 0)
        raw_s = str(d.get("summary") or "").strip()
        if not raw_s and doc_texts:
            raw_s = " ".join(doc_texts[0].split())[:170]
        file_digests.append(
            {
                "id": did,
                "title": str(d.get("title") or "Document"),
                "folder": normalize_folder_name(d.get("category")),
                "file_type": str(d.get("file_type") or "PDF").upper(),
                "domain_label": doc_dom["domain_label"],
                "page_count": int(d.get("page_count") or 1),
                "table_count": t_cnt,
                "metric_count": m_cnt,
                "link_count": rel_counts[did],
                "has_quant_data": t_cnt > 0 or m_cnt > 0,
                "summary": raw_s or "Indexed document ready for deeper analysis.",
                "top_topics": [str(t) for t in (d.get("main_topics") or [])[:4]],
            }
        )

    result = {
        "scope_mode": scope_mode,
        "scope_label": scope_label,
        "folder_name": folder_name,
        "focus_prompt": focus_prompt,
        "presentation_mode": presentation_mode,
        "has_quantitative_data": has_quantitative_data,
        "no_chart_reason": no_chart_reason,
        "domain": domain,
        "executive_summary": executive_summary,
        "what_is_inside": what_is_inside,
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
        "steps_or_sections": steps_or_sections,
        "top_topics": top_topics,
        "file_digests": file_digests,
        "suggested_questions": suggested_questions,
        "target_doc_ids": target_doc_ids,
        "cache_hit": False,
    }
    analytics_report_cache.set(cache_key, result, ttl_seconds=180.0)
    return result
