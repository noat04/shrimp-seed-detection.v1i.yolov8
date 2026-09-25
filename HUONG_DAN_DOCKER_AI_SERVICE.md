# Hướng dẫn đóng gói và tích hợp AI Service bằng Docker

## 1. Mục tiêu

Tài liệu này hướng dẫn đưa chức năng nhận diện tôm giống hiện tại vào một codebase sử dụng Docker và Docker Compose.

Kết luận quan trọng:

- Không cần train lại model khi chuyển sang Docker.
- File `best.pt` chứa trọng số model đã học và có thể chạy trong container.
- Không đưa `.venv`, dataset hoặc toàn bộ thư mục `runs` vào Docker image phục vụ dự đoán.
- AI Service nên là một container Python riêng.
- Backend gọi AI Service bằng tên service Docker, ví dụ `http://ai-service:8001`, không dùng `localhost` khi hai thành phần nằm trong hai container khác nhau.
- Việc huấn luyện nên tách khỏi container chạy API. Container runtime chỉ cần phục vụ dự đoán.

## 2. Kiến trúc khuyến nghị

```text
Người dùng
    ↓
Frontend container
    ↓
Backend container
    ↓ HTTP multipart/form-data
AI Service container
    ↓
models/shrimp_seed/v1/best.pt
```

Trong Docker Compose, các container cùng network có thể gọi nhau bằng tên service:

```text
Backend → http://ai-service:8001/predict
```

Từ máy host hoặc trình duyệt, dùng cổng đã publish:

```text
Máy host → http://localhost:8001/docs
```

## 3. Cấu trúc thư mục đề xuất

Giả sử codebase mới có cấu trúc:

```text
codebase-moi/
├── docker-compose.yml
├── frontend/
├── backend/
└── services/
    └── ai-service/
        ├── Dockerfile
        ├── .dockerignore
        ├── requirements.txt
        ├── ai_service/
        │   ├── __init__.py
        │   └── main.py
        └── models/
            └── shrimp_seed/
                └── v1/
                    └── best.pt
```

Model được đặt theo phiên bản để có thể giữ nhiều phiên bản:

```text
models/shrimp_seed/v1/best.pt
models/shrimp_seed/v2/best.pt
```

## 4. Bước 1 — Xác nhận model hiện tại

Tại dự án AI hiện tại, chạy:

```powershell
Test-Path ".\runs\detect\shrimp_seed_yolov8n\weights\best.pt"
```

Kết quả phải là:

```text
True
```

Model nguồn hiện tại:

```text
runs/detect/shrimp_seed_yolov8n/weights/best.pt
```

Không cần chạy lại `train_yolo.py` nếu file này vẫn còn và chức năng cần dùng vẫn là nhận diện `shrimp-seed`.

## 5. Bước 2 — Tạo thư mục AI Service trong codebase Docker

Ví dụ codebase Docker nằm tại:

```text
D:\MyProject
```

Tại thư mục dự án AI hiện tại, chạy:

```powershell
$DockerCodebase = "D:\MyProject"
$DockerAiService = "$DockerCodebase\services\ai-service"

New-Item -ItemType Directory -Force $DockerAiService
New-Item -ItemType Directory -Force "$DockerAiService\models\shrimp_seed\v1"
```

Thay `D:\MyProject` bằng đường dẫn codebase thật.

## 6. Bước 3 — Sao chép mã nguồn, requirements và model

Sao chép mã nguồn API:

```powershell
Copy-Item -LiteralPath ".\ai_service" `
  -Destination $DockerAiService `
  -Recurse
```

Sao chép dependencies:

```powershell
Copy-Item -LiteralPath ".\requirements.txt" `
  -Destination "$DockerAiService\requirements.txt"
```

Sao chép model tốt nhất:

```powershell
Copy-Item `
  -LiteralPath ".\runs\detect\shrimp_seed_yolov8n\weights\best.pt" `
  -Destination "$DockerAiService\models\shrimp_seed\v1\best.pt"
```

Kiểm tra:

```powershell
Test-Path "$DockerAiService\ai_service\main.py"
Test-Path "$DockerAiService\requirements.txt"
Test-Path "$DockerAiService\models\shrimp_seed\v1\best.pt"
```

Cả ba kết quả phải là `True`.

## 7. Bước 4 — Tạo Dockerfile cho AI Service

Tạo file:

```text
services/ai-service/Dockerfile
```

Nội dung đề xuất cho bản chạy CPU:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/shrimp_seed/v1/best.pt

WORKDIR /app

# OpenCV/Ultralytics có thể cần các thư viện hệ thống này khi import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tách bước copy requirements để Docker tận dụng build cache.
COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY ai_service ./ai_service
COPY models ./models

# Không chạy ứng dụng bằng tài khoản root.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Giải thích:

- `python:3.11-slim`: Python ổn định và image nền nhỏ hơn bản đầy đủ.
- `MODEL_PATH`: đường dẫn bên trong container, không phải đường dẫn Windows host.
- `COPY requirements.txt` trước mã nguồn: giúp cache bước cài thư viện.
- `COPY models ./models`: đóng gói `best.pt` vào image.
- `USER appuser`: giảm rủi ro so với chạy container bằng root.
- Không dùng `--reload` trong container chạy ổn định hoặc production.

## 8. Bước 5 — Tạo `.dockerignore`

Tạo file:

```text
services/ai-service/.dockerignore
```

Nội dung:

```dockerignore
.venv/
__pycache__/
*.py[cod]
*.log
.pytest_cache/
.mypy_cache/
.git/
.gitignore

train/
valid/
test/
runs/

README*
PROJECT_STATUS.md
HUONG_DAN_*.md
```

Không thêm dòng `models/` vào `.dockerignore`, vì Dockerfile cần copy `best.pt` vào image.

Dataset không nằm trong build context của cấu trúc đề xuất, nhưng các dòng loại trừ vẫn giúp tránh đóng gói nhầm nếu cấu trúc thay đổi sau này.

## 9. Bước 6 — Thêm AI Service vào Docker Compose

Trong file `docker-compose.yml` ở thư mục gốc codebase, thêm service:

```yaml
services:
  ai-service:
    build:
      context: ./services/ai-service
      dockerfile: Dockerfile
    image: shrimp-ai-service:1.0.0
    environment:
      MODEL_PATH: /app/models/shrimp_seed/v1/best.pt
    ports:
      - "8001:8001"
    restart: unless-stopped
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - >-
          import json, urllib.request;
          data = json.load(urllib.request.urlopen('http://127.0.0.1:8001/health'));
          assert data['model_exists']
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
```

Healthcheck trên kiểm tra cả hai điều kiện:

- API phản hồi được.
- File model tồn tại trong container.

`model_loaded` có thể vẫn là `false` trước lần dự đoán đầu tiên vì service hiện nạp model theo yêu cầu.

## 10. Bước 7 — Kết nối backend với AI Service

Nếu backend nằm trong cùng `docker-compose.yml`, backend phải gọi:

```text
http://ai-service:8001/predict
```

Không dùng:

```text
http://localhost:8001/predict
```

Trong container backend, `localhost` là chính container backend, không phải container AI.

Ví dụ cấu hình Compose cho backend:

```yaml
services:
  backend:
    build:
      context: ./backend
    environment:
      AI_SERVICE_URL: http://ai-service:8001
    depends_on:
      ai-service:
        condition: service_healthy

  ai-service:
    build:
      context: ./services/ai-service
    environment:
      MODEL_PATH: /app/models/shrimp_seed/v1/best.pt
    ports:
      - "8001:8001"
```

Backend gửi request:

```http
POST http://ai-service:8001/predict?conf=0.25&imgsz=640
Content-Type: multipart/form-data
```

Tên field chứa ảnh phải là:

```text
file
```

## 11. Bước 8 — Kiểm tra cấu hình Compose

Tại thư mục gốc codebase mới:

```powershell
cd "D:\MyProject"
docker compose config
```

Lệnh này kiểm tra cú pháp và hiển thị cấu hình Compose sau khi được hợp nhất.

Nếu không có lỗi, tiếp tục build.

## 12. Bước 9 — Build Docker image

```powershell
docker compose build ai-service
```

Lần build đầu tiên có thể lâu và image có thể lớn hơn 1 GB vì Ultralytics và PyTorch là các thư viện AI có dung lượng lớn.

Theo dõi các bước quan trọng:

```text
COPY requirements.txt
pip install -r requirements.txt
COPY ai_service
COPY models
```

Không cần sao chép `.venv` vào image. Dependencies được cài mới trong môi trường Linux của container.

## 13. Bước 10 — Khởi động container

```powershell
docker compose up -d ai-service
```

Kiểm tra trạng thái:

```powershell
docker compose ps
```

Xem log:

```powershell
docker compose logs -f ai-service
```

Khi thành công, log sẽ có nội dung tương tự:

```text
Uvicorn running on http://0.0.0.0:8001
```

Nhấn `Ctrl+C` để thoát chế độ xem log; container vẫn tiếp tục chạy vì đã dùng `-d`.

## 14. Bước 11 — Kiểm tra API từ máy host

Kiểm tra health:

```powershell
curl.exe http://localhost:8001/health
```

Kết quả mong muốn:

```json
{
  "status": "ok",
  "model_path": "/app/models/shrimp_seed/v1/best.pt",
  "model_exists": true,
  "model_loaded": false
}
```

Mở Swagger:

```text
http://localhost:8001/docs
```

Kiểm tra dự đoán bằng PowerShell:

```powershell
curl.exe -X POST `
  "http://localhost:8001/predict?conf=0.25&imgsz=640" `
  -F "file=@D:\Images\shrimp-test.jpg"
```

Đường dẫn sau ký tự `@` là đường dẫn ảnh trên máy host vì `curl.exe` đọc ảnh rồi gửi qua HTTP. Ảnh không cần được mount vào container.

Response thành công có mã HTTP `200` và dạng:

```json
{
  "filename": "shrimp-test.jpg",
  "image": {
    "width": 640,
    "height": 640
  },
  "count": 12,
  "detections": [
    {
      "class_id": 0,
      "class": "shrimp-seed",
      "confidence": 0.87,
      "box": {
        "x1": 120.2,
        "y1": 80.5,
        "x2": 180.9,
        "y2": 145.3
      }
    }
  ]
}
```

## 15. Bước 12 — Kiểm tra bên trong container

Kiểm tra model tồn tại:

```powershell
docker compose exec ai-service `
  python -c "from pathlib import Path; p=Path('/app/models/shrimp_seed/v1/best.pt'); print(p, p.exists(), p.stat().st_size if p.exists() else 0)"
```

Kiểm tra biến môi trường:

```powershell
docker compose exec ai-service `
  python -c "import os; print(os.getenv('MODEL_PATH'))"
```

Kết quả phải là:

```text
/app/models/shrimp_seed/v1/best.pt
```

## 16. Hai cách quản lý file model

### Cách A — Đóng gói model vào Docker image

Dockerfile sử dụng:

```dockerfile
COPY models ./models
```

Ưu điểm:

- Image tự chứa đầy đủ code và model.
- Triển khai sang máy khác đơn giản.
- Mỗi image tag gắn với một phiên bản model rõ ràng.
- Phù hợp production và báo cáo đồ án.

Nhược điểm:

- Thay model phải build lại image.
- Image lớn hơn.

Đây là cách khuyến nghị cho bản triển khai ổn định.

### Cách B — Mount model từ host

Compose:

```yaml
services:
  ai-service:
    build:
      context: ./services/ai-service
    environment:
      MODEL_PATH: /models/shrimp_seed/v1/best.pt
    volumes:
      - ./services/ai-service/models:/models:ro
    ports:
      - "8001:8001"
```

Ưu điểm:

- Có thể thay model mà không build lại image.
- Phù hợp môi trường phát triển.

Nhược điểm:

- Phụ thuộc file trên máy host.
- Khi chuyển máy phải sao chép cả thư mục model và giữ đúng đường dẫn mount.

Sau khi thay `best.pt`, phải restart container vì model cũ có thể đang nằm trong RAM:

```powershell
docker compose restart ai-service
```

Không nên vừa đóng gói model vào image vừa mount chồng lên cùng đường dẫn nếu không có lý do rõ ràng, vì volume sẽ che nội dung đã được copy trong image.

## 17. Cập nhật model mà không train lại

Nếu đã có một file `best.pt` mới được huấn luyện ở nơi khác, chỉ cần thay model và triển khai lại.

### Khi model được đóng gói trong image

1. Sao chép model mới vào:

```text
services/ai-service/models/shrimp_seed/v2/best.pt
```

2. Đổi `MODEL_PATH` trong Compose:

```yaml
environment:
  MODEL_PATH: /app/models/shrimp_seed/v2/best.pt
```

3. Tăng image tag:

```yaml
image: shrimp-ai-service:2.0.0
```

4. Build và khởi động lại:

```powershell
docker compose build ai-service
docker compose up -d ai-service
```

5. Kiểm tra `/health` và `/predict`.

Giữ lại `v1` để có thể quay về model cũ nếu `v2` hoạt động không tốt.

### Khi model được mount từ host

1. Thay file model hoặc tạo thư mục phiên bản mới.
2. Cập nhật `MODEL_PATH` nếu đường dẫn thay đổi.
3. Chạy:

```powershell
docker compose restart ai-service
```

## 18. Khi nào cần train lại?

Không cần train lại khi:

- Chuyển từ chạy trực tiếp sang Docker.
- Chuyển codebase hoặc đổi cấu trúc thư mục.
- Chuyển sang máy/server khác.
- Thay frontend, backend hoặc database.
- Đổi cổng API.
- Thêm một chức năng AI độc lập sử dụng model khác.

Cần train lại khi:

- Muốn model hiện tại nhận diện thêm class mới.
- Muốn nhận diện cả tôm và cua bằng cùng model.
- Bổ sung ảnh mới đã gán nhãn để cải thiện model.
- Model nhận diện kém trên camera, ánh sáng hoặc môi trường thực tế.
- Muốn cải thiện khả năng định vị bounding box.

Train nên được chạy bên ngoài container phục vụ API hoặc bằng một Docker image huấn luyện riêng. Không đưa dataset và công cụ train vào image runtime nếu không cần thiết.

## 19. Mở rộng thêm nhiều chức năng AI

Cấu trúc đề xuất:

```text
services/ai-service/
├── Dockerfile
├── requirements.txt
├── ai_service/
│   ├── main.py
│   ├── routers/
│   │   ├── shrimp_detection.py
│   │   ├── shrimp_quality.py
│   │   └── disease_detection.py
│   └── services/
│       ├── shrimp_detector.py
│       ├── quality_classifier.py
│       └── disease_detector.py
└── models/
    ├── shrimp_seed/v1/best.pt
    ├── shrimp_quality/v1/best.pt
    └── disease_detection/v1/best.pt
```

Các endpoint có thể là:

```text
GET  /health
POST /shrimp/detect
POST /shrimp/quality
POST /disease/detect
```

Mỗi model nên có:

- Module xử lý riêng.
- Đường dẫn cấu hình riêng.
- Thư mục phiên bản riêng.
- Test API riêng.

Khi thêm model mới, Docker image sẽ lớn hơn và container cần thêm RAM. Có thể tách từng model thành service riêng nếu dependencies xung đột hoặc nhu cầu mở rộng độc lập.

Ví dụ:

```text
shrimp-detection-service
shrimp-quality-service
disease-detection-service
```

Với quy mô đồ án nhỏ, một AI Service chứa nhiều router thường dễ quản lý hơn. Khi hệ thống lớn hoặc model nặng, tách service sẽ phù hợp hơn.

## 20. Có nên publish port 8001 không?

### Môi trường phát triển

Giữ:

```yaml
ports:
  - "8001:8001"
```

Điều này cho phép truy cập Swagger tại `http://localhost:8001/docs`.

### Môi trường production

Nếu chỉ backend cần gọi AI Service, có thể không publish cổng ra host:

```yaml
expose:
  - "8001"
```

Backend vẫn gọi được:

```text
http://ai-service:8001
```

Nhưng người ngoài Docker network không truy cập trực tiếp được AI Service. Đây là cấu hình an toàn hơn.

## 21. Lưu ý về CPU, RAM và dung lượng image

Model hiện tại chạy CPU, vì vậy không cần cấu hình NVIDIA GPU trong Compose.

Các lưu ý:

- Docker image có thể lớn hơn 1 GB vì PyTorch và Ultralytics.
- Không copy `.venv` để tránh làm image lớn thêm và gây lỗi khác hệ điều hành.
- Lần dự đoán đầu tiên có thể chậm hơn vì model được nạp vào RAM.
- Nhiều worker Uvicorn có thể nạp nhiều bản model và tăng lượng RAM sử dụng.
- Với model hiện tại, nên bắt đầu bằng một process Uvicorn như trong Dockerfile.
- Nếu chạy nhiều model trong cùng container, cần theo dõi RAM thực tế.

Kiểm tra tài nguyên:

```powershell
docker stats
```

## 22. Các lỗi thường gặp

### Build context không chứa model

Lỗi thường xảy ra khi Dockerfile dùng `COPY models ./models` nhưng thư mục `models` nằm ngoài `build.context`.

Đảm bảo:

```yaml
build:
  context: ./services/ai-service
```

và model nằm tại:

```text
services/ai-service/models/shrimp_seed/v1/best.pt
```

### `/health` trả `model_exists: false`

Kiểm tra:

```powershell
docker compose exec ai-service `
  python -c "import os; from pathlib import Path; p=Path(os.environ['MODEL_PATH']); print(p, p.exists())"
```

### Backend không gọi được `localhost:8001`

Trong container backend, đổi URL thành:

```text
http://ai-service:8001
```

### Lỗi import OpenCV hoặc `libGL.so`

Đảm bảo Dockerfile đã cài:

```text
libgl1
libglib2.0-0
```

Sau đó build lại:

```powershell
docker compose build --no-cache ai-service
```

### API trả `503 Model file not found`

Nguyên nhân là `MODEL_PATH` sai hoặc model chưa được copy/mount vào container.

### API trả `422 Unprocessable Entity`

Request chưa gửi ảnh bằng field `file` hoặc sai định dạng `multipart/form-data`.

### API trả `400 Uploaded file is not a valid image`

File tải lên không phải ảnh hợp lệ hoặc đã bị hỏng.

## 23. Checklist triển khai Docker

- [ ] `best.pt` tồn tại trong `services/ai-service/models/shrimp_seed/v1/`.
- [ ] `ai_service/main.py` đã được sao chép.
- [ ] `requirements.txt` đã được sao chép.
- [ ] Đã tạo `Dockerfile`.
- [ ] Đã tạo `.dockerignore`.
- [ ] `.dockerignore` không loại trừ thư mục `models`.
- [ ] Compose dùng đúng build context.
- [ ] `MODEL_PATH` là đường dẫn Linux bên trong container.
- [ ] `docker compose config` không báo lỗi.
- [ ] `docker compose build ai-service` thành công.
- [ ] Container ở trạng thái running/healthy.
- [ ] `/health` trả `model_exists: true`.
- [ ] `/predict` trả HTTP `200`.
- [ ] Backend gọi `http://ai-service:8001`, không gọi `localhost`.
- [ ] Đã thử bằng ảnh ngoài tập train.
- [ ] Đã kiểm tra RAM và CPU bằng `docker stats`.

## 24. Quy trình ngắn gọn

```text
1. Lấy best.pt đã huấn luyện
2. Tạo services/ai-service trong codebase Docker
3. Sao chép ai_service và requirements.txt
4. Đặt best.pt trong models/shrimp_seed/v1
5. Tạo Dockerfile và .dockerignore
6. Thêm ai-service vào docker-compose.yml
7. Đặt MODEL_PATH=/app/models/shrimp_seed/v1/best.pt
8. docker compose config
9. docker compose build ai-service
10. docker compose up -d ai-service
11. Kiểm tra /health và /predict
12. Cho backend gọi http://ai-service:8001
```

Việc đóng gói bằng Docker không yêu cầu train lại. Chỉ cần đưa đúng mã nguồn, dependencies và `best.pt` vào image hoặc mount model qua volume, sau đó cấu hình `MODEL_PATH` đúng với đường dẫn bên trong container.
