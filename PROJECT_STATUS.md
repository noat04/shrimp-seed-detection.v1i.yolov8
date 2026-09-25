# Trạng thái hiện tại của dự án nhận diện tôm giống

## 1. Tổng quan

Dự án sử dụng YOLOv8 để phát hiện và đếm tôm giống trong ảnh. Hiện tại dự án đã hoàn thành phần lõi AI và API ở mức demo.

Luồng xử lý chính:

```text
Ảnh đầu vào
    ↓
YOLOv8 phát hiện tôm giống
    ↓
Lọc kết quả theo độ tin cậy
    ↓
Trả về số lượng, độ tin cậy và tọa độ bounding box
```

Model hiện chỉ nhận diện một class:

```text
shrimp-seed
```

## 2. Dataset

Dataset được xuất từ Roboflow theo định dạng YOLOv8, gồm 300 ảnh có kích thước 640 × 640 pixel:

| Tập dữ liệu | Số ảnh | Số bounding box |
|---|---:|---:|
| Train | 210 | 1.632 |
| Validation | 60 | 459 |
| Test | 30 | 226 |
| Tổng cộng | 300 | 2.317 |

Cấu trúc thư mục:

```text
train/images     Ảnh dùng để huấn luyện
train/labels     Nhãn dùng để huấn luyện
valid/images     Ảnh dùng để đánh giá trong lúc huấn luyện
valid/labels     Nhãn validation
test/images      Ảnh dùng để kiểm tra cuối cùng
test/labels      Nhãn test
data.yaml        Cấu hình dataset và tên class
```

Nhãn YOLO của mỗi đối tượng có dạng:

```text
class_id x_center y_center width height
```

Các giá trị tọa độ và kích thước được chuẩn hóa trong khoảng từ 0 đến 1.

## 3. Model đã huấn luyện

Model YOLOv8n đã được huấn luyện thành công trong 30 epoch. File model tốt nhất nằm tại:

```text
runs/detect/shrimp_seed_yolov8n/weights/best.pt
```

Kết quả trên tập validation:

| Chỉ số | Kết quả | Ý nghĩa |
|---|---:|---|
| Precision | 82,8% | Tỷ lệ dự đoán đúng trong các bounding box mà model tạo ra |
| Recall | 85,1% | Khả năng phát hiện đủ các con tôm có trong ảnh |
| mAP50 | 85,6% | Chất lượng phát hiện ở ngưỡng IoU 0,5 |
| mAP50–95 | 27,8% | Chất lượng bounding box ở các ngưỡng nghiêm ngặt hơn |

Model đã nhận diện được phần lớn tôm giống, nhưng vị trí và kích thước bounding box vẫn có thể chưa thật sự sát trong một số trường hợp.

Model đang chạy bằng CPU Intel Core i5-11400H. Thời gian xử lý đo được khoảng 93 ms cho mỗi ảnh, tương đương khoảng 10 ảnh mỗi giây trong điều kiện thử nghiệm.

## 4. Những chức năng đã làm được

### 4.1. Nhận diện tôm trong ảnh

Dự án có thể:

- Nhận ảnh JPG hoặc PNG.
- Phát hiện vị trí từng con tôm giống.
- Trả độ tin cậy của từng phát hiện.
- Trả tọa độ bounding box theo pixel.
- Đếm tổng số tôm được phát hiện.
- Xử lý ảnh có kích thước khác 640 × 640.

Logic đếm hiện tại:

```text
Số tôm = số bounding box còn lại sau khi YOLO lọc kết quả
```

Số lượng phụ thuộc vào chất lượng model và ngưỡng `conf`. Ngưỡng thấp có thể phát hiện nhiều hơn nhưng dễ nhận nhầm; ngưỡng cao giảm nhận nhầm nhưng có thể bỏ sót tôm.

### 4.2. Dự đoán bằng dòng lệnh

Có thể xử lý một ảnh hoặc toàn bộ thư mục ảnh:

```powershell
yolo detect predict `
  model="runs/detect/shrimp_seed_yolov8n/weights/best.pt" `
  source="test/images" `
  imgsz=640 `
  conf=0.25 `
  save=True
```

Ảnh kết quả có bounding box được lưu trong thư mục như:

```text
runs/detect/predict
```

Nếu chạy nhiều lần, YOLO có thể tạo thêm các thư mục `predict2`, `predict3`, v.v.

### 4.3. AI API

Dự án có FastAPI để nhận ảnh và trả kết quả nhận diện.

Khởi động API:

```powershell
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```

Giao diện thử API:

```text
http://localhost:8001/docs
```

API hiện có hai endpoint:

```http
GET /health
```

Dùng để kiểm tra service, đường dẫn model và trạng thái nạp model.

```http
POST /predict
```

Dùng để tải ảnh lên và nhận kết quả phát hiện. Ví dụ response:

```json
{
  "filename": "sample.jpg",
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

Trong đó:

- `count`: số con tôm được phát hiện.
- `confidence`: độ tin cậy của từng phát hiện.
- `box`: tọa độ góc trên trái và góc dưới phải của bounding box.

API đã được kiểm tra và trả mã HTTP `200`. Với ảnh thử nghiệm đã kiểm tra, nhãn gốc có 12 đối tượng và model cũng dự đoán 12 đối tượng. Tuy nhiên, đây là ảnh thuộc tập train nên vẫn cần thử thêm bằng ảnh test và ảnh thực tế bên ngoài dataset.

## 5. Xử lý ảnh mới

### Chỉ muốn nhận diện

Có thể tải trực tiếp một ảnh bất kỳ lên endpoint `/predict`. Ảnh không cần nằm trong thư mục dự án và không cần có kích thước 640 × 640.

API hiện xử lý ảnh trong bộ nhớ và không tự lưu ảnh đã tải lên. Nếu không phát hiện được tôm, kết quả sẽ là:

```json
{
  "count": 0,
  "detections": []
}
```

### Muốn model học từ ảnh mới

Chỉ sao chép ảnh vào `train/images` là chưa đủ. Cần thực hiện:

1. Vẽ bounding box cho từng con tôm trong ảnh.
2. Tạo file nhãn YOLO cùng tên trong `train/labels`.
3. Chia dữ liệu hợp lý giữa train, validation và test.
4. Huấn luyện lại model.
5. Sử dụng file `best.pt` mới.

Ví dụ cặp ảnh và nhãn:

```text
train/images/anh-moi.jpg
train/labels/anh-moi.txt
```

Không nên thêm ảnh chưa gán nhãn vào `train/images`, vì YOLO có thể hiểu ảnh đó không chứa tôm.

## 6. Những chức năng chưa có

Hiện tại dự án chưa có:

- Giao diện website dành cho người dùng cuối.
- Chức năng hiển thị ảnh có bounding box trực tiếp từ API.
- Lưu ảnh tải lên hoặc ảnh kết quả.
- Database lưu lịch sử nhận diện.
- Đăng nhập và phân quyền người dùng.
- Nhận diện thời gian thực từ camera hoặc video qua API.
- Đo chiều dài tôm theo mm hoặc cm.
- Phân loại tôm khỏe, yếu, bệnh hoặc chất lượng.
- Nhận diện nhiều loại đối tượng khác nhau.
- Tự động huấn luyện lại khi thêm ảnh mới.
- Triển khai lên máy chủ để truy cập từ Internet.

## 7. Mức độ hoàn thiện

| Hạng mục | Trạng thái |
|---|---|
| Dataset YOLOv8 | Hoàn thành |
| Huấn luyện YOLOv8n | Hoàn thành |
| Model `best.pt` | Hoàn thành |
| Nhận diện ảnh | Hoàn thành |
| Đếm số tôm | Hoàn thành |
| Trả bounding box và confidence | Hoàn thành |
| API nhận ảnh | Hoàn thành |
| Trả kết quả JSON | Hoàn thành |
| Giao diện website | Chưa có |
| Lưu lịch sử vào database | Chưa có |
| Đo kích thước thực tế | Chưa có |
| Triển khai lên Internet | Chưa có |

## 8. Kết luận

Dự án hiện đã hoàn thành phần lõi AI ở mức demo: có thể nhận ảnh, phát hiện vị trí và đếm tôm giống, đồng thời cung cấp API để tích hợp với backend hoặc frontend.

Bước phát triển tiếp theo phù hợp là xây dựng giao diện tải ảnh, hiển thị ảnh có bounding box, lưu lịch sử phân tích và kiểm thử model bằng nhiều ảnh thực tế bên ngoài dataset.
