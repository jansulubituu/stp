# AI Research Analysis Web

Ứng dụng tra cứu và trả kết quả phân tích bằng giao diện web.

## Stack

- Frontend: Next.js + React + TypeScript
- Backend: FastAPI + SQLAlchemy
- Database: SQLite khi chạy cục bộ, hoặc PostgreSQL qua `DATABASE_URL`
- AI/Search: Multi-agent prior-art pipeline, Groq/OpenAI-compatible LLM, Elasticsearch KNN, optional Jina embedding service

## Cấu trúc

```text
project/
  backend/
    app/
    requirements.txt
    .env
  frontend/
    app/
    lib/
    package.json
```

## Yêu cầu

- Python 3.12 trở lên
- Node.js 22 trở lên và npm
- PostgreSQL 16 trở lên nếu không dùng SQLite

## Cài đặt database

Backend đọc kết nối database từ biến `DATABASE_URL` trong file `backend/.env`.
Khi backend khởi động lần đầu, SQLAlchemy tự tạo bảng `analysis_records` nếu bảng chưa tồn tại.

Chọn một trong hai cách dưới đây.

### Cách 1: SQLite cho môi trường local

SQLite không cần cài dịch vụ database riêng. Tạo hoặc cập nhật file `backend/.env`:

```dotenv
DATABASE_URL=sqlite:///./analysis_app.db
```

Sau khi backend được chạy từ thư mục `backend`, file database sẽ nằm tại:

```text
backend/analysis_app.db
```

Để tạo lại database SQLite rỗng, dừng backend rồi xóa file database:

```powershell
# Windows PowerShell, chạy tại thư mục gốc project
Remove-Item .\backend\analysis_app.db
```

```bash
# macOS / Linux, chạy tại thư mục gốc project
rm ./backend/analysis_app.db
```

Chạy lại backend; bảng `analysis_records` sẽ được tạo lại tự động.

### Cách 2: PostgreSQL

#### Cài PostgreSQL

Windows:

1. Tải PostgreSQL installer từ https://www.postgresql.org/download/windows/.
2. Trong lúc cài đặt, giữ cổng mặc định `5432` và ghi nhớ mật khẩu tài khoản quản trị `postgres`.
3. Mở `SQL Shell (psql)` từ Start Menu sau khi cài xong, nhập user `postgres` ở bước đăng nhập.

macOS với Homebrew:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

#### Tạo user và database

Mở `psql` bằng tài khoản quản trị:

```bash
# Windows PowerShell, nếu lệnh psql đã có trong PATH
psql -U postgres

# macOS
psql postgres

# Ubuntu / Debian
sudo -u postgres psql
```

Trên Windows, nếu đang dùng ứng dụng `SQL Shell (psql)` từ Start Menu thì bỏ qua lệnh `psql -U postgres`; sau khi đăng nhập thành công, chạy trực tiếp phần lệnh SQL bên dưới.

Chạy các lệnh SQL sau:

```sql
CREATE USER analysis_user WITH PASSWORD 'analysis_password';
CREATE DATABASE analysis_app OWNER analysis_user;
\q
```

Tạo hoặc cập nhật `backend/.env`:

```dotenv
DATABASE_URL=postgresql+psycopg://analysis_user:analysis_password@localhost:5432/analysis_app
```

Mật khẩu trong ví dụ chỉ phù hợp để chạy local. Khi triển khai thực tế, đặt mật khẩu khác và không commit file `.env`.

#### Kiểm tra kết nối PostgreSQL

```bash
psql -h localhost -U analysis_user -d analysis_app
```

Trong `psql`, kiểm tra bảng sau khi backend đã được khởi động ít nhất một lần:

```sql
\dt
SELECT * FROM analysis_records ORDER BY created_at DESC LIMIT 5;
\q
```

#### Tạo lại database PostgreSQL rỗng

Thao tác này xóa toàn bộ lịch sử phân tích đang lưu. Dừng backend, đăng nhập bằng tài khoản quản trị rồi chạy:

```sql
DROP DATABASE IF EXISTS analysis_app;
DROP USER IF EXISTS analysis_user;
CREATE USER analysis_user WITH PASSWORD 'analysis_password';
CREATE DATABASE analysis_app OWNER analysis_user;
```

Khởi động lại backend để tạo lại bảng `analysis_records`.

### Lưu ý về schema

Dự án hiện chưa cấu hình Alembic/migration. `create_all()` chỉ tạo bảng còn thiếu, không tự cập nhật cấu trúc bảng đã có. Nếu model database thay đổi, cần reset database local hoặc bổ sung migration trước khi dùng dữ liệu cần giữ lại.

## Chạy backend

Mở terminal thứ nhất tại thư mục gốc của dự án:

### Windows PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sau khi cấu hình `DATABASE_URL` theo phần cài đặt database, để chạy đầy đủ tìm kiếm và phân tích AI, bổ sung các khóa tương ứng trong `backend/.env`:

```dotenv
AI_KEY=
ES_CLOUD_ID=
ES_API_KEY=
JINA_API_KEY=
GEMINI_API_KEY=
```

Pipeline backend hien tai dung service multi-agent trong `backend/pipeline-ma`. De chay luong moi, uu tien cau hinh cac bien sau trong `backend/.env`:

```dotenv
GROQ_API_KEY=
MULTIAGENT_LLM_PROVIDER=groq
MULTIAGENT_LLM_MODEL=openai/gpt-oss-120b
MULTIAGENT_AGENT2_LLM_MODEL=openai/gpt-oss-120b
MULTIAGENT_AGENT3_LLM_MODEL=openai/gpt-oss-120b

ES_CLOUD_ID=
ES_API_KEY=
BM25_INDEX=clef_ip_patents_v1_mini
KNN_INDEX=clef_ip_patents_v1_mini_jina
ES_VECTOR_FIELD=content_vector

MULTIAGENT_RETRIEVAL_BACKEND=es_knn
MULTIAGENT_KNN_EMBED_API_BASE=
MULTIAGENT_KNN_EMBED_API_KEY=EMPTY
MULTIAGENT_KNN_EMBED_MODEL=jina-embed-safe
```

Neu da set `JINA_API_KEY` va khong set `MULTIAGENT_KNN_EMBED_API_BASE`, pipeline se tu dong goi Jina API voi model `jina-embeddings-v3`. Neu khong co ca hai cau hinh nay, pipeline se thu load local `jinaai/jina-embeddings-v3`, can them cac dependency GPU/Transformers nang hon.

Không commit file `.env` có chứa API key.

Khởi động API từ thư mục `backend`:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Kiểm tra backend:

- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

## Chạy frontend

Giữ backend đang chạy tại `http://localhost:8000`. Mở terminal thứ hai tại thư mục gốc:

```bash
cd frontend
npm install
npm run dev
```

Mở ứng dụng tại http://localhost:3000.

Frontend hiện chuyển tiếp các request `/api/*` đến backend tại `http://localhost:8000`, vì vậy backend phải chạy trước và dùng đúng cổng `8000`.

## API chính

- `POST /api/search`: tìm các tài liệu ứng viên
- `POST /api/analyze-selected`: phân tích các tài liệu đã chọn
- `POST /api/analyze`: gửi câu hỏi và nhận kết quả phân tích
- `GET /api/history`: lấy lịch sử
- `GET /api/history/{id}`: lấy một kết quả
- `DELETE /api/history/{id}`: xóa một kết quả
