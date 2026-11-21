"""
DOCUMIND Complete End-to-End API Test Suite
Tests every single backend endpoint:
1. GET  / (Root Info)
2. GET  /health (Health Check)
3. POST /api/documents/upload (PDF Upload & PyMuPDF Coordinate Extraction)
4. GET  /api/documents/{id} (Document Status & Page Count)
5. GET  /api/documents/{id}/chunks (Chunks with Normalized Bounding Boxes)
6. POST /api/analytics/widget (DuckDB Deterministic SQL & ECharts Option)
7. POST /api/chat/message (Cognitive Copilot with Visual Citations & Depth Levels)
8. POST /api/chat/quiz (Adaptive Socratic Quiz linked to document proof quotes)
9. POST /api/synthesis/matrix (Cross-Document Comparative Synthesis Matrix)
10. POST /api/audio/briefing-script (Audio Briefing 2-Host Podcast Generator)
"""

import os
import sys
import fitz # PyMuPDF
import json
import time

# Ensure apps/api is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_sample_pdf(filename: str, title: str, content_paragraphs: list) -> str:
    """Generates a valid PDF with PyMuPDF for realistic testing."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=595, height=842) # A4 size
    page1.insert_text((50, 70), title, fontsize=18, color=(0.1, 0.2, 0.5))
    y = 120
    for p in content_paragraphs:
        page1.insert_textbox(fitz.Rect(50, y, 545, y + 100), p, fontsize=11, color=(0.1, 0.1, 0.1))
        y += 120

    # Page 2 (with table text)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 70), f"{title} - Data Pengujian Laboratorium", fontsize=14, color=(0.1, 0.2, 0.5))
    table_text = (
        "Tabel 1.1 Hasil Pengukuran Efisiensi Siklus Termal:\n"
        "Fluida A: Suhu Th=550K, Tc=300K, Efisiensi=45.4%\n"
        "Fluida B: Suhu Th=420K, Tc=280K, Efisiensi=33.3%\n"
        "Fluida C: Suhu Th=350K, Tc=290K, Efisiensi=17.1%"
    )
    page2.insert_textbox(fitz.Rect(50, 100, 545, 300), table_text, fontsize=10, color=(0.1, 0.1, 0.1))

    doc.save(filepath)
    doc.close()
    return filepath

def run_all_tests():
    print("=" * 75)
    print("       DOCUMIND BACKEND SUITE: TESTING ALL 10 ENDPOINTS")
    print("=" * 75)

    passed_count = 0
    total_count = 10

    # 1. GET /
    print("\n[TEST 1] Testing GET / (Root Info)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["app"] == "DOCUMIND API"
    assert data["status"] == "online"
    print(f"  -> SUCCESS! App: '{data['app']}', Model: '{data['model']}', Status: {data['status']}")
    passed_count += 1

    # 2. GET /health
    print("\n[TEST 2] Testing GET /health...")
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}
    print("  -> SUCCESS! Health check returned 200 OK.")
    passed_count += 1

    # Create 2 real PDFs for ingestion
    pdf1_path = create_sample_pdf(
        "sample_termo.pdf",
        "BAB 4: HUKUM KEDUA TERMODINAMIKA & MESIN CARNOT",
        [
            "Hukum kedua termodinamika menyatakan bahwa total entropi dari suatu sistem terisolasi tidak akan pernah berkurang seiring berjalannya waktu. Sebagian energi selalu terdegradasi menjadi kalor bersuhu rendah.",
            "Efisiensi mesin kalor ideal Carnot hanya bergantung pada suhu mutlak reservoir panas Th dan reservoir dingin Tc, dengan rumus eta = 1 - (Tc/Th). Tidak ada mesin yang melampaui efisiensi ini."
        ]
    )

    pdf2_path = create_sample_pdf(
        "sample_refrig.pdf",
        "BAB 5: SIKLUS REFRIGERASI KOMPRESI UAP",
        [
            "Siklus refrigerasi memindahkan kalor dari ruang bersuhu rendah ke lingkungan bersuhu tinggi menggunakan kerja mekanis kompresor.",
            "Refrigeran R-134a banyak diaplikasikan karena ramah lingkungan dibandingkan CFC, meskipun memiliki efisiensi termal terbatas pada temperatur kerja 350 Kelvin."
        ]
    )

    # 3. POST /api/documents/upload (Document 1)
    print("\n[TEST 3] Testing POST /api/documents/upload...")
    with open(pdf1_path, "rb") as f:
        res = client.post("/api/documents/upload", files={"file": ("sample_termo.pdf", f, "application/pdf")})
    assert res.status_code == 200
    doc1 = res.json()
    doc1_id = doc1["id"]
    assert doc1_id is not None
    assert doc1["page_count"] == 2
    print(f"  -> SUCCESS! Document uploaded: ID={doc1_id}, Title='{doc1['title']}', Pages={doc1['page_count']}")
    passed_count += 1

    # Upload Document 2 for synthesis test
    with open(pdf2_path, "rb") as f:
        res = client.post("/api/documents/upload", files={"file": ("sample_refrig.pdf", f, "application/pdf")})
    doc2 = res.json()
    doc2_id = doc2["id"]

    # 4. GET /api/documents/{id}
    print(f"\n[TEST 4] Testing GET /api/documents/{doc1_id}...")
    res = client.get(f"/api/documents/{doc1_id}")
    assert res.status_code == 200
    data = res.json()
    print(f"  -> SUCCESS! Document Status: {data['processing_status']}, File Size: {data['file_size_bytes']} bytes")
    passed_count += 1

    # 5. GET /api/documents/{id}/chunks
    print(f"\n[TEST 5] Testing GET /api/documents/{doc1_id}/chunks (Normalized Bounding Boxes)...")
    res = client.get(f"/api/documents/{doc1_id}/chunks")
    assert res.status_code == 200
    chunks = res.json()
    assert len(chunks) > 0, "Expected at least 1 chunk extracted"
    first_chunk = chunks[0]
    boxes = first_chunk["bounding_boxes"]
    assert len(boxes) > 0, "Expected bounding boxes"
    sample_box = boxes[0]
    print(f"  -> SUCCESS! Chunks extracted: {len(chunks)} chunks.")
    print(f"     First Chunk Preview: '{first_chunk['content'][:80]}...'")
    print(f"     Normalized Bounding Box: left={sample_box['left']}, top={sample_box['top']}, w={sample_box['width']}, h={sample_box['height']}, Page={sample_box['page_number']}")
    passed_count += 1

    # 6. POST /api/analytics/widget (DuckDB + ECharts)
    print("\n[TEST 6] Testing POST /api/analytics/widget (DuckDB In-Memory OLAP)...")
    res = client.post("/api/analytics/widget", json={"query": "Tampilkan distribusi beban belajar per bab"})
    assert res.status_code == 200
    widget = res.json()
    assert widget["chart_type"] == "bar"
    assert "echarts_spec" in widget
    series = widget["echarts_spec"]["series"]
    assert len(series) >= 2
    print(f"  -> SUCCESS! DuckDB SQL: '{widget['sql_query']}'")
    print(f"     Chart Title: '{widget['echarts_spec']['title']['text']}'")
    print(f"     ECharts Categories (X): {widget['echarts_spec']['xAxis']['data']}")
    print(f"     Series 1 Data: {series[0]['name']} -> {series[0]['data']}")
    print(f"     Insight Derivation: '{widget['insight_summary']}'")
    passed_count += 1

    # 7. POST /api/chat/message (Cognitive Copilot with Citations & Depth Levels)
    print("\n[TEST 7] Testing POST /api/chat/message with Cognitive Depth Slider...")
    for depth in ["eli5", "high_school", "academic"]:
        res = client.post("/api/chat/message", json={
            "document_id": doc1_id,
            "message": "Jelaskan apa itu efisiensi Mesin Carnot dan bagaimana perbandingannya?",
            "depth": depth
        })
        assert res.status_code == 200
        chat_data = res.json()
        assert len(chat_data["citations"]) > 0
        citation = chat_data["citations"][0]
        print(f"  -> Level [{depth.upper()}]: Answer snippet: '{chat_data['answer'][:90]}...'")
        print(f"     Visual Citation Attached: ID='{citation['citation_id']}', Hal={citation['page_number']}, Boxes Count={len(citation['bounding_boxes'])}")
    passed_count += 1

    # 8. POST /api/chat/quiz (Adaptive Socratic Quiz)
    print(f"\n[TEST 8] Testing POST /api/chat/quiz?document_id={doc1_id}...")
    res = client.post(f"/api/chat/quiz?document_id={doc1_id}")
    assert res.status_code == 200
    quiz_data = res.json()
    assert len(quiz_data["questions"]) >= 2
    q1 = quiz_data["questions"][0]
    print(f"  -> SUCCESS! Generated Quiz: '{quiz_data['title']}' ({len(quiz_data['questions'])} questions)")
    print(f"     Question 1: '{q1['question_text']}'")
    print(f"     Correct Option: [{q1['correct_option']}]")
    print(f"     Proof Quote in Document: '{q1['proof_quote']}' (Hal {q1['source_page']})")
    passed_count += 1

    # 9. POST /api/synthesis/matrix (Cross-Document Synthesis Matrix)
    print(f"\n[TEST 9] Testing POST /api/synthesis/matrix (Comparing {doc1_id} and {doc2_id})...")
    res = client.post("/api/synthesis/matrix", json={"document_ids": [doc1_id, doc2_id]})
    assert res.status_code == 200
    synth = res.json()
    assert len(synth["matrix"]) > 0
    item1 = synth["matrix"][0]
    print(f"  -> SUCCESS! Synthesis Matrix Generated: '{synth['title']}'")
    print(f"     Concept Compared: '{item1['concept']}'")
    print(f"     Doc A Stance: '{item1['doc_a_stance']}'")
    print(f"     Doc B Stance: '{item1['doc_b_stance']}'")
    print(f"     Analysis: '{item1['comparison_analysis']}' (Contradiction: {item1['is_contradiction']})")
    print(f"     Executive Summary: '{synth['executive_summary'][:100]}...'")
    passed_count += 1

    # 10. POST /api/audio/briefing-script (Audio Briefing 2-Host Podcast)
    print(f"\n[TEST 10] Testing POST /api/audio/briefing-script?document_id={doc1_id}...")
    res = client.post(f"/api/audio/briefing-script?document_id={doc1_id}")
    assert res.status_code == 200
    audio_data = res.json()
    assert len(audio_data["script"]) >= 2
    t1 = audio_data["script"][0]
    t2 = audio_data["script"][1]
    print(f"  -> SUCCESS! Podcast Script: '{audio_data['title']}' (Duration: {audio_data['duration_estimate']})")
    print(f"     Host {t1['speaker']}: '{t1['text']}'")
    print(f"     Host {t2['speaker']}: '{t2['text']}'")
    print(f"     Takeaway Summary: '{audio_data['takeaway_summary']}'")
    passed_count += 1

    print("\n" + "=" * 75)
    print(f"TEST RESULTS: {passed_count}/{total_count} ENDPOINTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == "__main__":
    run_all_tests()
