import json
import unittest
import uuid
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app
from app.core.workspace import WorkspaceContext, get_verified_workspace_context
from app.services.duckdb_service import duckdb_service


def _create_sample_financial_pdf_bytes() -> bytes:
    """
    Generates a genuine 2-page PDF in memory using PyMuPDF (fitz) with
    paragraphs, quantitative KPIs, and structured financial data.
    """
    doc = fitz.open()

    # Page 1: Executive Summary & Quantitative KPIs
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 60),
        "PT Nusantara Cloud Infrastructure - Q3 2026 Financial & Operational Report",
        fontsize=14,
    )
    page1.insert_text(
        (50, 100),
        (
            "Executive Summary: In Q3 2026, PT Nusantara Cloud Infrastructure achieved Total Revenue "
            "of 1250 USD million across enterprise cloud and AI data center regions in Jakarta and Surabaya. "
            "Operating Margin improved to 24.5 % driven by renewable geothermal power adoption."
        ),
        fontsize=11,
    )
    page1.insert_text(
        (50, 160),
        (
            "Key Risk & Governance Finding: Data center water cooling consumption decreased by 18.2 % "
            "year-over-year while server rack utilization reached 89.4 % across Tier-4 facilities."
        ),
        fontsize=11,
    )

    # Page 2: Quarterly Regional Breakdown
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 60),
        "Regional Data Center Performance Breakdown (FY 2026)",
        fontsize=13,
    )
    page2.insert_text(
        (50, 100),
        "Jakarta Region recorded 780 USD million revenue with 99.99 % uptime SLA compliance.",
        fontsize=11,
    )
    page2.insert_text(
        (50, 130),
        "Surabaya Region recorded 470 USD million revenue with 99.95 % uptime SLA compliance.",
        fontsize=11,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_sample_csv_bytes() -> bytes:
    csv_text = (
        "region,quarter,revenue_usd_m,operating_margin_pct,uptime_pct\n"
        "Jakarta,Q1_2026,710,22.1,99.98\n"
        "Jakarta,Q2_2026,745,23.4,99.99\n"
        "Jakarta,Q3_2026,780,25.0,99.99\n"
        "Surabaya,Q1_2026,410,19.8,99.94\n"
        "Surabaya,Q2_2026,440,21.2,99.95\n"
        "Surabaya,Q3_2026,470,23.6,99.95\n"
    )
    return csv_text.encode("utf-8")


class TestDocuMindBackendStages1To6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.user_a_id = str(uuid.uuid4())
        cls.workspace_a_id = str(uuid.uuid4())
        cls.user_b_id = str(uuid.uuid4())
        cls.workspace_b_id = str(uuid.uuid4())
        cls.active_ctx = WorkspaceContext(
            user_id=cls.user_a_id,
            email="alice@documind.test",
            workspace_id=cls.workspace_a_id,
            role="owner",
        )

    def setUp(self):
        # Default to User A context
        self.__class__.active_ctx = WorkspaceContext(
            user_id=self.user_a_id,
            email="alice@documind.test",
            workspace_id=self.workspace_a_id,
            role="owner",
        )
        app.dependency_overrides[get_verified_workspace_context] = lambda: self.__class__.active_ctx

    def tearDown(self):
        app.dependency_overrides.clear()

    def _switch_to_user_b(self):
        self.__class__.active_ctx = WorkspaceContext(
            user_id=self.user_b_id,
            email="bob@documind.test",
            workspace_id=self.workspace_b_id,
            role="owner",
        )

    def test_01_root_and_health_endpoints(self):
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        root_data = r_root.json()
        self.assertEqual(root_data["version"], "2.0.0")
        self.assertEqual(root_data["ai_provider"], "openai")

        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["status"], "healthy")

    def test_02_end_to_end_stages_1_to_6_workflow_and_tenant_isolation(self):
        # ---------------------------------------------------------------------
        # 1. Stage 2: Reject Corrupt PDF Upload (Stage 1 Validation Gate)
        # ---------------------------------------------------------------------
        corrupt_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("broken.pdf", b"NOT_A_REAL_PDF_HEADER_12345", "application/pdf")},
        )
        self.assertIn(corrupt_resp.status_code, (400, 422))

        # ---------------------------------------------------------------------
        # 2. Stage 2: Upload Genuine 2-Page PDF & Run 15-Stage Ingestion Pipeline
        # ---------------------------------------------------------------------
        pdf_bytes = _create_sample_financial_pdf_bytes()
        upload1_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("Nusantara_Cloud_Q3_2026_Report.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(upload1_resp.status_code, 200, upload1_resp.text)
        doc1 = upload1_resp.json()
        doc1_id = doc1["id"]
        self.assertEqual(doc1["workspace_id"], self.workspace_a_id)
        self.assertFalse(doc1.get("is_duplicate", False))
        self.assertEqual(doc1["page_count"], 2)

        # Verify SHA-256 Deduplication when uploading the exact same file bytes
        dup_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("Duplicate_Copy.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(dup_resp.status_code, 200)
        dup_doc = dup_resp.json()
        self.assertEqual(dup_doc["id"], doc1_id)
        self.assertTrue(dup_doc.get("is_duplicate", False))

        # ---------------------------------------------------------------------
        # 3. Stage 2: Upload Second Document (CSV Structured Dataset)
        # ---------------------------------------------------------------------
        csv_bytes = _create_sample_csv_bytes()
        upload2_resp = self.client.post(
            "/api/documents/upload",
            files={"file": ("Nusantara_Regional_Quarterly_Metrics.csv", csv_bytes, "text/csv")},
        )
        self.assertEqual(upload2_resp.status_code, 200, upload2_resp.text)
        doc2 = upload2_resp.json()
        doc2_id = doc2["id"]
        self.assertNotEqual(doc1_id, doc2_id)
        doc2_after_bg = self.client.get(f"/api/documents/{doc2_id}").json()
        self.assertTrue(doc2_after_bg.get("has_tables", False))

        # ---------------------------------------------------------------------
        # 4. Stage 2: Verify Repository Listing, Summary, Metadata PATCH & Chunks
        # ---------------------------------------------------------------------
        list_resp = self.client.get("/api/documents")
        self.assertEqual(list_resp.status_code, 200)
        docs_list = list_resp.json()
        self.assertEqual(len(docs_list), 2)

        summary_resp = self.client.get("/api/documents/summary")
        self.assertEqual(summary_resp.status_code, 200)
        self.assertEqual(summary_resp.json()["total_documents"], 2)

        patch_resp = self.client.patch(
            f"/api/documents/{doc1_id}",
            json={"category": "financial_report", "user_tags": ["q3-2026", "cloud-infra"]},
        )
        self.assertEqual(patch_resp.status_code, 200)
        self.assertIn("q3-2026", patch_resp.json()["user_tags"])

        chunks_resp = self.client.get(f"/api/documents/{doc1_id}/chunks")
        self.assertEqual(chunks_resp.status_code, 200)
        chunks_list = chunks_resp.json()
        self.assertGreaterEqual(len(chunks_list), 2)
        for ch in chunks_list:
            self.assertGreaterEqual(ch["page_number"], 1)
            for bbox in ch.get("bounding_boxes", []):
                self.assertGreaterEqual(bbox["left"], 0.0)
                self.assertLessEqual(bbox["left"], 1.0)
                self.assertGreaterEqual(bbox["top"], 0.0)
                self.assertLessEqual(bbox["top"], 1.0)
                self.assertGreaterEqual(bbox["width"], 0.0)
                self.assertLessEqual(bbox["width"], 1.0)
                self.assertGreaterEqual(bbox["height"], 0.0)
                self.assertLessEqual(bbox["height"], 1.0)

        status_resp = self.client.get(f"/api/documents/{doc1_id}/status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertIn(status_resp.json()["status"], ("ready", "partial_ready"))

        file_resp = self.client.get(f"/api/documents/{doc1_id}/file")
        self.assertEqual(file_resp.status_code, 200)
        self.assertTrue(file_resp.content.startswith(b"%PDF-"))

        # ---------------------------------------------------------------------
        # 5. Stage 1 & 2: Strict Multi-Tenant Isolation (User B Cannot Access User A)
        # ---------------------------------------------------------------------
        self._switch_to_user_b()
        b_list = self.client.get("/api/documents")
        self.assertEqual(b_list.status_code, 200)
        self.assertEqual(len(b_list.json()), 0)

        b_get_a = self.client.get(f"/api/documents/{doc1_id}")
        self.assertEqual(b_get_a.status_code, 404)

        b_file_a = self.client.get(f"/api/documents/{doc1_id}/file")
        self.assertEqual(b_file_a.status_code, 404)

        b_dash = self.client.get("/api/analytics/dashboard")
        self.assertEqual(b_dash.status_code, 200)
        self.assertEqual(b_dash.json()["corpus_profile"], "empty_corpus")

        # Switch back to User A
        self.setUp()

        # ---------------------------------------------------------------------
        # 6. Stage 3: Hybrid RRF Search (/api/search)
        # ---------------------------------------------------------------------
        search_resp = self.client.post(
            "/api/search",
            json={"query": "Jakarta Total Revenue Operating Margin", "match_count": 5},
        )
        self.assertEqual(search_resp.status_code, 200)
        search_data = search_resp.json()
        self.assertGreaterEqual(search_data["result_count"], 1)
        top_hit = search_data["results"][0]
        self.assertIn("rrf_score", top_hit)
        self.assertIn("bounding_boxes", top_hit)

        # ---------------------------------------------------------------------
        # 7. Stage 3: Evidence-Grounded Chat, SSE Streaming & Quiz (/api/chat)
        # ---------------------------------------------------------------------
        chat_resp = self.client.post(
            "/api/chat/query",
            json={
                "document_id": doc1_id,
                "message": "What was the Total Revenue and Operating Margin in Q3 2026?",
                "depth": "academic",
            },
        )
        self.assertEqual(chat_resp.status_code, 200, chat_resp.text)
        chat_data = chat_resp.json()
        self.assertEqual(chat_data["retrieval_status"], "grounded")
        self.assertNotIn("Carnot", chat_data["answer"])
        self.assertNotIn("R-134a", chat_data["answer"])
        self.assertGreaterEqual(len(chat_data["citations"]), 1)
        first_cit = chat_data["citations"][0]
        self.assertEqual(first_cit["document_id"], doc1_id)
        self.assertGreaterEqual(len(first_cit["bounding_boxes"]), 1)

        # Verify SSE Streaming endpoint (/api/chat/stream)
        stream_resp = self.client.post(
            "/api/chat/stream",
            json={
                "document_id": doc1_id,
                "message": "Summarize the regional performance in Jakarta and Surabaya.",
            },
        )
        self.assertEqual(stream_resp.status_code, 200)
        stream_body = stream_resp.text
        self.assertIn("event: status", stream_body)
        self.assertIn("event: token_delta", stream_body)
        self.assertIn("event: citations_verified", stream_body)

        # Verify Conversation Persistence
        conv_resp = self.client.get("/api/chat/conversations")
        self.assertEqual(conv_resp.status_code, 200)
        self.assertGreaterEqual(conv_resp.json()["count"], 1)

        # Verify Grounded Study Quiz Generation (/api/chat/quiz)
        quiz_resp = self.client.post(f"/api/chat/quiz?document_id={doc1_id}")
        self.assertEqual(quiz_resp.status_code, 200, quiz_resp.text)
        quiz_data = quiz_resp.json()
        self.assertEqual(quiz_data["document_id"], doc1_id)
        self.assertGreaterEqual(len(quiz_data["questions"]), 1)
        self.assertNotIn("Carnot", json.dumps(quiz_data))

        # ---------------------------------------------------------------------
        # 8. Stage 4: Personalized Dashboard & 5-Gate DuckDB Sandbox (/api/analytics)
        # ---------------------------------------------------------------------
        dash_resp = self.client.get("/api/analytics/dashboard")
        self.assertEqual(dash_resp.status_code, 200, dash_resp.text)
        dash_data = dash_resp.json()
        self.assertIn(dash_data["corpus_profile"], ("quantitative_corpus", "mixed_corpus"))
        self.assertGreaterEqual(len(dash_data["widgets"]), 1)

        gen_dash_resp = self.client.post(
            "/api/analytics/generate-dashboard",
            json={"prompt": "Compare revenue_usd_m by region and quarter"},
        )
        self.assertEqual(gen_dash_resp.status_code, 200, gen_dash_resp.text)
        gen_widget = gen_dash_resp.json()
        self.assertIn("echarts_spec", gen_widget)
        self.assertNotIn("R-134a", json.dumps(gen_widget))

        # Verify 5-Gate DuckDB Security Sandbox directly
        sandbox_check_copy = duckdb_service.validate_and_execute_widget_sql(
            workspace_id=self.workspace_a_id,
            sql_query="COPY (SELECT 1) TO 'pwn.csv'",
        )
        self.assertFalse(sandbox_check_copy["valid"])
        self.assertEqual(sandbox_check_copy["gate_failed"], "gate_1_sql_ast_sandbox")

        sandbox_check_read = duckdb_service.validate_and_execute_widget_sql(
            workspace_id=self.workspace_a_id,
            sql_query="SELECT * FROM read_csv_auto('/etc/passwd')",
        )
        self.assertFalse(sandbox_check_read["valid"])
        self.assertEqual(sandbox_check_read["gate_failed"], "gate_1_sql_ast_sandbox")

        # ---------------------------------------------------------------------
        # 9. Stage 5: Document Relationships & Cross-Document Synthesis
        # ---------------------------------------------------------------------
        recompute_resp = self.client.post("/api/relationships/recompute")
        self.assertEqual(recompute_resp.status_code, 200)

        rel_resp = self.client.get("/api/relationships")
        self.assertEqual(rel_resp.status_code, 200)
        self.assertEqual(rel_resp.json()["node_count"], 2)

        synth_resp = self.client.post(
            "/api/synthesis/compare",
            json={
                "document_ids": [doc1_id, doc2_id],
                "focus_query": "Compare regional revenue and margin metrics across both documents",
            },
        )
        self.assertEqual(synth_resp.status_code, 200, synth_resp.text)
        synth_data = synth_resp.json()
        self.assertEqual(synth_data["document_ids"], [doc1_id, doc2_id])
        self.assertGreaterEqual(len(synth_data["comparison_matrix"]), 2)
        self.assertNotIn("Carnot", json.dumps(synth_data))
        self.assertNotIn("R-134a", json.dumps(synth_data))

        # ---------------------------------------------------------------------
        # 10. Stage 6: Grounded Audio Briefing Script & Document Cascade Delete
        # ---------------------------------------------------------------------
        audio_resp = self.client.post(f"/api/audio/briefing-script?document_id={doc1_id}")
        self.assertEqual(audio_resp.status_code, 200, audio_resp.text)
        audio_data = audio_resp.json()
        self.assertGreaterEqual(len(audio_data["script"]), 2)
        self.assertNotIn("Carnot", json.dumps(audio_data))

        del_resp = self.client.delete(f"/api/documents/{doc2_id}")
        self.assertEqual(del_resp.status_code, 200)
        remaining_docs = self.client.get("/api/documents").json()
        self.assertEqual(len(remaining_docs), 1)
        self.assertEqual(remaining_docs[0]["id"], doc1_id)


if __name__ == "__main__":
    unittest.main()
