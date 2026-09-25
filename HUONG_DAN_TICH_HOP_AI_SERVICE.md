# Hướng dẫn chuyển và tích hợp AI Service vào codebase khác

> Nếu codebase đích được vận hành bằng Docker/Docker Compose, hãy ưu tiên tài liệu [HUONG_DAN_DOCKER_AI_SERVICE.md](./HUONG_DAN_DOCKER_AI_SERVICE.md). Tài liệu hiện tại phù hợp với cách chạy trực tiếp bằng Python trên máy host.

## 1. Mục tiêu

Tài liệu này hướng dẫn chuyển chức năng nhận diện tôm giống hiện tại sang một codebase khác, đồng thời chuẩn bị cấu trúc để có thể bổ sung thêm các chức năng AI sau này.

Kết luận quan trọng:

- Không cần huấn luyện lại model chỉ vì chuyển thư mục, chuyển codebase hoặc chuyển máy.
- Kiến thức model đã học nằm trong file `best.pt`.
- Khi triển khai để dự đoán, không cần mang theo toàn bộ dataset.
- Không nên sao chép thư mục `.venv`; hãy tạo môi trường Python mới trong codebase đích.
- Chỉ cần train lại khi muốn model học dữ liệu mới, nhận diện class mới hoặc cải thiện độ chính xác.

## 2. Thành phần hiện có

Các thành phần cần quan tâm trong dự án hiện tại:

```text
ai_service/
├── __init__.py
└── main.py

runs/detect/shrimp_seed_yolov8n/weights/
├── best.pt
└── last.pt

requirements.txt
```

Trong đó:

- `ai_service/main.py`: FastAPI nhận ảnh và gọi YOLO dự đoán.
- `best.pt`: model có kết quả tốt nhất trong quá trình huấn luyện.
- `last.pt`: model ở epoch cuối, không nhất thiết tốt bằng `best.pt`.
- `requirements.txt`: danh sách thư viện cần cài đặt.

Khi triển khai, nên sử dụng `best.pt`.

## 3. Kiến trúc khuyến nghị

Nếu codebase mới có frontend, backend và nhiều chức năng AI, nên giữ AI Service thành một service Python riêng:

```text
codebase-moi/
├── frontend/
├── backend/
└── services/
    └── ai-service/
        ├── ai_service/
        │   ├── __init__.py
        │   └── main.py
        ├── models/
        │   └── shrimp_seed/
        │       └── v1/
        │           └── best.pt
        ├── requirements.txt
        └── .venv/
```

Luồng hoạt động:

```text
Frontend
    ↓ gửi ảnh
Backend chính
    ↓ HTTP multipart/form-data
AI Service FastAPI
    ↓
Model YOLO best.pt
    ↓
JSON: count + confidence + bounding boxes
```

Nếu backend chính viết bằng Node.js, Java, .NET hoặc PHP, không nên cố gắng import trực tiếp code Python vào backend. Hãy chạy AI Service độc lập và giao tiếp qua HTTP.

## 4. Bước 1 — Xác nhận model nguồn

Tại thư mục dự án hiện tại, chạy:

```powershell
Test-Path ".\runs\detect\shrimp_seed_yolov8n\weights\best.pt"
```

Kết quả phải là:

```text
True
```

Có thể xem kích thước model:

```powershell
Get-Item ".\runs\detect\shrimp_seed_yolov8n\weights\best.pt" |
  Select-Object Name, Length, LastWriteTime
```

Nếu `best.pt` tồn tại thì không cần train lại chỉ để chuyển codebase.

## 5. Bước 2 — Chọn vị trí codebase đích

Ví dụ codebase mới nằm tại:

```text
D:\MyProject
```

AI Service sẽ được đặt tại:

```text
D:\MyProject\services\ai-service
```

Trong các lệnh bên dưới, hãy thay `D:\MyProject` bằng đường dẫn thật của codebase mới.

## 6. Bước 3 — Tạo cấu trúc thư mục đích

Đứng tại thư mục dự án AI hiện tại và chạy:

```powershell
$TargetAiService = "D:\MyProject\services\ai-service"

New-Item -ItemType Directory -Force $TargetAiService
New-Item -ItemType Directory -Force "$TargetAiService\models\shrimp_seed\v1"
```

Kết quả mong muốn:

```text
D:\MyProject\services\ai-service
└── models
    └── shrimp_seed
        └── v1
```

## 7. Bước 4 — Sao chép mã nguồn AI Service

Chạy từ thư mục dự án hiện tại:

```powershell
Copy-Item -LiteralPath ".\ai_service" `
  -Destination $TargetAiService `
  -Recurse
```

Sau khi sao chép, cần có:

```text
D:\MyProject\services\ai-service\ai_service\main.py
```

Sao chép file thư viện:

```powershell
Copy-Item -LiteralPath ".\requirements.txt" `
  -Destination "$TargetAiService\requirements.txt"
```

## 8. Bước 5 — Sao chép model đã huấn luyện

Chạy:

```powershell
Copy-Item `
  -LiteralPath ".\runs\detect\shrimp_seed_yolov8n\weights\best.pt" `
  -Destination "$TargetAiService\models\shrimp_seed\v1\best.pt"
```

Kiểm tra model tại codebase mới:

```powershell
Test-Path "$TargetAiService\models\shrimp_seed\v1\best.pt"
```

Kết quả phải là `True`.

Không cần sao chép `last.pt` trừ khi muốn lưu để đối chiếu.

## 9. Bước 6 — Những thư mục không cần sao chép

Đối với môi trường chỉ dùng để dự đoán, không bắt buộc sao chép:

```text
train/
valid/
test/
train_yolo.py
runs/detect/predict/
.venv/
```

Giải thích:

- `train`, `valid`, `test`: chỉ cần khi huấn luyện hoặc đánh giá model.
- `train_yolo.py`: chỉ dùng để train.
- `runs/detect/predict`: ảnh kết quả cũ, không cần cho API.
- `.venv`: gắn với Python và đường dẫn trên máy hiện tại, không nên mang sang codebase khác.

Nếu codebase mới vẫn cần hỗ trợ huấn luyện lại, có thể giữ thêm dataset, `data.yaml` và `train_yolo.py` trong một khu vực riêng như `training/`.

## 10. Bước 7 — Tạo môi trường Python mới

Đi tới AI Service mới:

```powershell
cd "D:\MyProject\services\ai-service"
```

Tạo môi trường ảo:

```powershell
python -m venv .venv
```

Kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script kích hoạt, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra các thư viện chính:

```powershell
python -c "import fastapi, uvicorn, ultralytics, PIL, multipart; print('Dependencies OK')"
```

Nếu terminal in ra `Dependencies OK` thì môi trường đã sẵn sàng.

## 11. Bước 8 — Cấu hình đường dẫn model

Code hiện tại mặc định tìm model ở:

```text
runs/detect/shrimp_seed_yolov8n/weights/best.pt
```

Trong cấu trúc mới, model nằm ở:

```text
models/shrimp_seed/v1/best.pt
```

Không cần sửa code ngay. File `main.py` đã hỗ trợ biến môi trường `MODEL_PATH`.

Tại thư mục `ai-service`, chạy:

```powershell
$env:MODEL_PATH = (Resolve-Path ".\models\shrimp_seed\v1\best.pt").Path
```

Kiểm tra giá trị:

```powershell
$env:MODEL_PATH
```

Lưu ý: biến môi trường được đặt theo cách này chỉ tồn tại trong cửa sổ PowerShell hiện tại. Khi mở terminal mới, cần đặt lại trước khi khởi động API.

## 12. Bước 9 — Khởi động AI Service

Đảm bảo terminal đang đứng tại:

```text
D:\MyProject\services\ai-service
```

Sau đó chạy:

```powershell
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```

Khi thành công, terminal sẽ hiển thị thông báo tương tự:

```text
Uvicorn running on http://0.0.0.0:8001
```

Giữ nguyên terminal này khi sử dụng API. Nhấn `Ctrl+C` khi muốn dừng service.

## 13. Bước 10 — Kiểm tra `/health`

Mở một terminal khác và chạy:

```powershell
curl.exe http://localhost:8001/health
```

Kết quả mong muốn:

```json
{
  "status": "ok",
  "model_path": "D:\\MyProject\\services\\ai-service\\models\\shrimp_seed\\v1\\best.pt",
  "model_exists": true,
  "model_loaded": false
}
```

Ý nghĩa:

- `model_exists: true`: service đã tìm thấy model.
- `model_loaded: false`: chưa có request dự đoán nên model chưa được nạp vào RAM.
- Sau request `/predict` đầu tiên, `model_loaded` sẽ chuyển thành `true`.

Nếu `model_exists` là `false`, kiểm tra lại `$env:MODEL_PATH` và vị trí `best.pt`.

## 14. Bước 11 — Kiểm tra dự đoán bằng Swagger

Mở trình duyệt:

```text
http://localhost:8001/docs
```

Thực hiện:

1. Mở endpoint `POST /predict`.
2. Nhấn **Try it out**.
3. Chọn ảnh JPG hoặc PNG tại trường `file`.
4. Đặt `conf = 0.25`.
5. Đặt `imgsz = 640`.
6. Nhấn **Execute**.

Nếu thành công, phần `Server response` sẽ hiển thị mã `200` và JSON như:

```json
{
  "filename": "shrimp.jpg",
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

## 15. Bước 12 — Kiểm tra dự đoán bằng PowerShell

Có thể kiểm tra không qua Swagger:

```powershell
curl.exe -X POST `
  "http://localhost:8001/predict?conf=0.25&imgsz=640" `
  -F "file=@D:\Images\shrimp-test.jpg"
```

Thay `D:\Images\shrimp-test.jpg` bằng đường dẫn ảnh thật.

API hiện không lưu ảnh tải lên và không trả ảnh đã vẽ bounding box. API chỉ trả JSON gồm số lượng và tọa độ box.

## 16. Bước 13 — Backend chính gọi AI Service

Backend cần gửi request:

```http
POST http://localhost:8001/predict?conf=0.25&imgsz=640
Content-Type: multipart/form-data
```

Tên trường chứa ảnh bắt buộc là:

```text
file
```

Backend nhận response và sử dụng:

- `count`: hiển thị hoặc lưu số lượng tôm.
- `detections`: danh sách đối tượng phát hiện.
- `confidence`: độ tin cậy của từng đối tượng.
- `box`: tọa độ để frontend vẽ bounding box trên ảnh gốc.

Nếu frontend gọi AI Service trực tiếp từ một domain hoặc port khác, có thể cần bổ sung cấu hình CORS cho FastAPI. Trong kiến trúc khuyến nghị, frontend gọi backend chính và backend chính gọi AI Service để việc bảo mật và quản lý dữ liệu rõ ràng hơn.

## 17. Bước 14 — Tổ chức khi bổ sung nhiều chức năng AI

Khi thêm chức năng AI mới, nên mở rộng cấu trúc thành:

```text
ai-service/
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
├── models/
│   ├── shrimp_seed/
│   │   └── v1/best.pt
│   ├── shrimp_quality/
│   │   └── v1/best.pt
│   └── disease_detection/
│       └── v1/best.pt
└── requirements.txt
```

Endpoint có thể được tổ chức thành:

```text
GET  /health
POST /shrimp/detect
POST /shrimp/quality
POST /disease/detect
```

Nguyên tắc:

- Mỗi chức năng AI nên có module xử lý riêng.
- Mỗi model nên có thư mục và phiên bản riêng.
- Không đặt toàn bộ logic vào `main.py` khi số chức năng tăng lên.
- Có thể nạp model theo yêu cầu để tránh chiếm RAM khi chức năng chưa được sử dụng.
- Giữ endpoint `/predict` cũ nếu backend hiện tại vẫn đang sử dụng để tránh làm hỏng tích hợp.

## 18. Có cần train lại khi thêm chức năng AI không?

### Không cần train lại model tôm hiện tại khi

- Chuyển sang codebase khác.
- Đổi đường dẫn thư mục.
- Chuyển sang máy khác.
- Xây dựng giao diện mới.
- Thêm database hoặc chức năng đăng nhập.
- Thêm một model AI độc lập cho chức năng khác.

Ví dụ: thêm model phân loại chất lượng tôm không bắt buộc phải train lại model phát hiện tôm. Hai model có thể hoạt động nối tiếp nhau.

### Cần train lại khi

- Muốn model hiện tại nhận diện thêm class mới.
- Muốn nhận diện cả tôm và cua bằng cùng một model.
- Bổ sung ảnh tôm mới đã gán nhãn để cải thiện độ chính xác.
- Camera, môi trường, nền hoặc ánh sáng thực tế khác nhiều so với dataset cũ.
- Model bỏ sót hoặc nhận nhầm quá nhiều trên dữ liệu thực tế.

Nếu bổ sung class mới vào cùng model YOLO, cần cập nhật dataset, nhãn và `data.yaml`, sau đó train lại.

## 19. Nếu muốn tiếp tục huấn luyện từ model hiện tại

Không nhất thiết phải bắt đầu lại từ `yolov8n.pt`. Có thể dùng `best.pt` làm điểm khởi đầu cho lần train tiếp theo.

Ví dụ bằng Python:

```python
from ultralytics import YOLO

model = YOLO("models/shrimp_seed/v1/best.pt")
model.train(
    data="training/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="shrimp_seed_v2",
)
```

Sau khi đánh giá phiên bản mới, lưu theo phiên bản:

```text
models/shrimp_seed/v1/best.pt
models/shrimp_seed/v2/best.pt
```

Không ghi đè ngay model cũ trước khi xác nhận model mới hoạt động tốt hơn.

## 20. Xử lý lỗi thường gặp

### Lỗi `503 Model file not found`

Nguyên nhân: đường dẫn model không đúng hoặc chưa đặt `MODEL_PATH`.

Kiểm tra:

```powershell
$env:MODEL_PATH
Test-Path $env:MODEL_PATH
```

### Lỗi `422 Unprocessable Entity`

Nguyên nhân thường gặp: request không gửi ảnh bằng trường tên `file` hoặc không dùng `multipart/form-data`.

### Lỗi `400 Uploaded file is not a valid image`

File tải lên không phải ảnh hợp lệ hoặc bị hỏng. Hãy thử ảnh JPG hoặc PNG khác.

### Lỗi không tìm thấy module `ai_service`

Hãy chạy Uvicorn từ thư mục chứa thư mục `ai_service`:

```powershell
cd "D:\MyProject\services\ai-service"
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```

### Port 8001 đã được sử dụng

Có thể chạy bằng port khác:

```powershell
uvicorn ai_service.main:app --host 0.0.0.0 --port 8002
```

Khi đó URL API sẽ là:

```text
http://localhost:8002
```

### Máy khác không truy cập được API

Kiểm tra:

- Uvicorn được chạy với `--host 0.0.0.0`.
- Windows Firewall cho phép port đang sử dụng.
- Backend gọi đúng địa chỉ IP của máy chạy AI Service, không dùng `localhost` nếu hai service nằm trên hai máy khác nhau.

## 21. Danh sách kiểm tra sau khi chuyển

Hoàn thành lần lượt các mục sau:

- [ ] Đã sao chép `ai_service/`.
- [ ] Đã sao chép `requirements.txt`.
- [ ] Đã sao chép `best.pt`.
- [ ] Không sao chép `.venv` cũ.
- [ ] Đã tạo `.venv` mới.
- [ ] Đã cài đặt thư viện.
- [ ] Đã đặt biến môi trường `MODEL_PATH`.
- [ ] `/health` trả `model_exists: true`.
- [ ] `/predict` trả HTTP `200`.
- [ ] Response có `count` và `detections`.
- [ ] Backend chính gọi được AI Service.
- [ ] Đã thử ít nhất một ảnh ngoài tập train.

## 22. Tóm tắt quy trình nhanh

```text
1. Xác nhận best.pt tồn tại
2. Tạo thư mục services/ai-service trong codebase mới
3. Sao chép ai_service/
4. Sao chép requirements.txt
5. Sao chép best.pt vào models/shrimp_seed/v1/
6. Tạo .venv mới
7. Cài requirements.txt
8. Đặt MODEL_PATH
9. Chạy Uvicorn
10. Kiểm tra /health
11. Kiểm tra /predict
12. Kết nối backend chính
```

Chuyển codebase không làm mất kiến thức của model. Chỉ cần bảo quản và cấu hình đúng file `best.pt`; không cần huấn luyện lại từ đầu.
