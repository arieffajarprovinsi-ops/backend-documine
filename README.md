# DOCUMIND Backend API Engine

Sistem backend kecerdasan dokumen, analitik DuckDB deterministik, dan asisten kognitif bertenaga **Google Gemini 3.8 Flash** & **PyMuPDF**.

---

## 🚀 Cara Menjalankan Backend

### Cara 1: Sekali Klik (Windows)
Cukup jalankan file `run.bat` di dalam folder `apps/api`:
```bash
run.bat
```
Script ini akan otomatis membuat virtual environment, memasang dependensi, dan menjalankan server FastAPI di `http://localhost:8000`.

### Cara 2: Manual Terminal
```bash
cd apps/api

# 1. Buat dan aktifkan virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# 2. Pasang dependensi
pip install -r requirements.txt

# 3. Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Setelah server aktif, buka dokumentasi interaktif API di browser:
👉 **Swagger UI:** `http://localhost:8000/docs`  
👉 **ReDoc:** `http://localhost:8000/redoc`

---

## 📡 Daftar Endpoint API Siap Pakai untuk Frontend

| Method | Endpoint | Fungsi | Keterangan untuk Frontend |
|---|---|---|---|
| `POST` | `/api/documents/upload` | Upload PDF & Ekstraksi Bounding Box | Kirim `multipart/form-data` dengan field `file`. Mengembalikan `id`, `page_count`, dan status. |
| `GET` | `/api/documents/{id}` | Cek status pemrosesan dokumen | Polling status: `pending` -> `processing` -> `completed`. |
| `GET` | `/api/documents/{id}/chunks` | Ambil semua chunk & koordinat visual | Menghasilkan array chunk dengan koordinat `bounding_boxes` (0.0 - 1.0) untuk highlight di PDF viewer. |
| `POST` | `/api/chat/message` | Tanya-jawab cerdas dengan sitasi visual | Body: `{ message, document_id, depth: "eli5" | "high_school" | "academic" }`. Mengembalikan jawaban + badge sitasi `[Hal. X]`. |
| `POST` | `/api/chat/quiz` | Generator Kuis Adaptif Sokratik | Query param: `?document_id=...`. Mengembalikan 4 soal kuis lengkap dengan kalimat pembuktian di PDF (`proof_quote`). |
| `POST` | `/api/analytics/widget` | Generator Widget ECharts Anti-Halusinasi | Body: `{ document_id, query }`. Menghasilkan opsi ECharts yang dihitung langsung oleh DuckDB. |
| `POST` | `/api/synthesis/matrix` | Matriks Sintesis Antardokumen (HOTS) | Body: `{ document_ids: ["id1", "id2"] }`. Membandingkan persamaan, perbedaan, dan kontradiksi 2 buku/dokumen. |
| `POST` | `/api/audio/briefing-script` | Naskah Podcast Audio Briefing 2 Host | Query param: `?document_id=...`. Mengembalikan naskah percakapan edukatif Alex & Sarah (NotebookLM style). |

---

## 🔑 Konfigurasi Environment (`.env`)

Salin file `.env.example` menjadi `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ALLOWED_ORIGINS=http://localhost:3000
```
Jika `GEMINI_API_KEY` belum diisi, backend tetap bisa berjalan dan memproses ekstraksi PDF serta eksekusi analitik DuckDB.
