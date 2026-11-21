"""
Smoke test script to verify that all modules, schemas, and DuckDB analytics
run properly without import errors.
"""
import sys
import os

# Add apps/api to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports_and_schemas():
    print("Testing DOCUMIND Backend Modules...")

    from app.core.config import settings
    print(f"  [OK] Config loaded: {settings.app_name} (Model: {settings.default_model})")

    from app.models.schemas import BoundingBox, DocumentChunk, CognitiveDepth, DashboardWidget, QuizQuestion
    box = BoundingBox(left=0.1, top=0.2, width=0.5, height=0.05, page_number=1)
    assert box.left == 0.1
    print("  [OK] Pydantic models validated successfully.")

    from app.services.duckdb_service import duckdb_engine
    test_data = [{"kategori": "Aljabar", "soal": 10}, {"kategori": "Geometri", "soal": 15}]
    tbl = duckdb_engine.register_table_from_records("test_data", test_data)
    cols, rows = duckdb_engine.execute_safe_query(f"SELECT kategori, soal FROM {tbl} ORDER BY soal DESC")
    assert len(rows) == 2
    print(f"  [OK] DuckDB in-memory analytical query verified: {cols} -> {rows[0]}")

    from app.main import app
    print("  [OK] FastAPI app & all routers mounted cleanly.")
    print("\nAll Backend Sanity Checks PASSED! Ready for frontend integration.")

if __name__ == "__main__":
    test_imports_and_schemas()
