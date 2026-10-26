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
| Precision | 82,7% | Tỷ lệ dự đoán đúng trong các bounding box mà model tạo ra |
| Recall | 85,0% | Khả năng phát hiện đủ các con tôm có trong ảnh |
| mAP50 | 85,6% | Chất lượng phát hiện ở ngưỡng IoU 0,5 |
| mAP50–95 | 27,8% | Chất lượng bounding box ở các ngưỡng nghiêm ngặt hơn |

Model đã nhận diện được phần lớn tôm giống, nhưng vị trí và kích thước bounding box vẫn có thể chưa thật sự sát trong một số trường hợp.

Các con số trên là kết quả **validation** của checkpoint tốt nhất được chọn trong quá trình huấn luyện, không phải kết quả test. Căn cứ là log `runs/detect/shrimp_seed_yolov8n/results.csv`: dòng epoch 29 ghi Precision `0.82659`, Recall `0.84967`, mAP50 `0.85583`, mAP50–95 `0.27836`; các số trong bảng được đổi sang phần trăm và làm tròn một chữ số thập phân. `train_yolo.py` yêu cầu huấn luyện 30 epoch và Ultralytics lưu checkpoint tốt nhất tại `runs/detect/shrimp_seed_yolov8n/weights/best.pt`. README còn ghi kết quả đánh giá riêng trên test (Precision 79,90%, Recall 80,90%, mAP50 78,01%, mAP50–95 24,86%); nên báo cáo rõ đang nói về validation hay test.

### Căn cứ và ý nghĩa của số liệu

- Số ảnh được đếm từ các tệp trong `train/images`, `valid/images`, `test/images`; repo hiện có lần lượt 210, 60 và 30 ảnh.
- Số bounding box được tính bằng tổng số dòng nhãn không rỗng trong các thư mục `train/labels`, `valid/labels`, `test/labels`. Mỗi dòng YOLO biểu diễn một đối tượng: `class_id x_center y_center width height`. Kiểm tra các nhãn hiện có cho kết quả 1.632, 459 và 226 box. Ảnh không có dòng nhãn sẽ không đóng góp box.
- `data.yaml` chỉ định đường dẫn ba split và một class `shrimp-seed`; `README.roboflow.txt` cho biết dataset được xuất từ Roboflow, tự định hướng ảnh theo EXIF và resize ảnh về 640 × 640 bằng Stretch; không áp dụng augmentation khi xuất dataset.
- `results.csv` do Ultralytics sinh trong lúc validation cuối mỗi epoch. Nó là nguồn cho các metric validation; không thể suy ra những metric này chỉ từ số ảnh/box trong bảng dataset. Muốn đánh giá tổng quát hóa, dùng checkpoint `best.pt` chạy trên split test chưa dùng để chọn model.

Precision = TP / (TP + FP): trong các box model dự đoán, tỷ lệ box khớp một đối tượng thật. Recall = TP / (TP + FN): trong các đối tượng thật được gán nhãn, tỷ lệ model tìm thấy. TP/FP/FN được xác định bằng cách ghép dự đoán với nhãn thật theo class và độ chồng lấn IoU; dự đoán trùng nhau hoặc không khớp đủ mức IoU có thể bị tính là FP, đối tượng bị bỏ sót là FN.

IoU (Intersection over Union) = diện tích giao nhau của box dự đoán và box nhãn thật chia cho diện tích hợp của hai box. AP là diện tích dưới đường cong Precision–Recall cho một class khi thay đổi ngưỡng confidence; mAP là trung bình AP trên các class. Vì project chỉ có một class, mAP ở đây là AP của class `shrimp-seed`. `mAP50` tính tại IoU 0,50; `mAP50–95` lấy trung bình qua IoU 0,50, 0,55, …, 0,95, nên nghiêm ngặt hơn về độ khít của box. Vì thế mAP50–95 thấp hơn đáng kể cho thấy model thường tìm được tôm nhưng khung định vị còn chưa chính xác ở các chuẩn IoU cao. Các metric này đo detection, không trực tiếp đo sai số đếm; muốn báo cáo khả năng đếm cần thêm MAE/RMSE trên từng ảnh.

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

### 4.2. Cách YOLOv8 phát hiện tôm giống từ ảnh

#### Model và cách học

Project dùng biến thể **YOLOv8n (nano)** từ Ultralytics, khởi tạo bằng trọng số pretrained `yolov8n.pt`, sau đó fine-tune cho bài toán detection một class. Cấu hình thực tế nằm trong `train_yolo.py`, `data.yaml` và `runs/detect/shrimp_seed_yolov8n/args.yaml`: 30 epoch, ảnh 640 × 640, batch 8, pretrained bật, chạy CPU. Ultralytics/PyTorch xây dựng và huấn luyện mạng; Pillow đọc ảnh API; FastAPI nhận file và trả JSON.

Ở mức khái niệm, mạng gồm **backbone** trích xuất đặc trưng từ cạnh, hình dạng, màu sắc và cấu trúc ở nhiều mức; **neck** kết hợp đặc trưng độ phân giải cao/thấp để vừa nhận vật thể nhỏ vừa nhận ngữ cảnh; và **detection head** dự đoán class, độ tin cậy và tọa độ box tại nhiều vị trí/kích thước. YOLOv8 dùng head không anchor (anchor-free). Đây là mô tả kiến trúc chung của YOLOv8; cấu hình model và trọng số cụ thể mà project nạp là file `best.pt`.

Trong lúc huấn luyện, ảnh train đi cùng nhãn box YOLO để tính loss định vị box, phân loại class và phân bố cạnh box. Trọng số được cập nhật để giảm loss qua các batch và epoch. Tập validation dùng để theo dõi chất lượng và chọn checkpoint; nó không được dùng như ảnh train để cập nhật trọng số. Split test dành cho đánh giá cuối độc lập.

#### Từ ảnh đến JSON kết quả

Luồng xử lý một ảnh:

1. API đọc bytes ảnh tải lên bằng Pillow và gọi `convert("RGB")`. Ảnh màu được chuẩn hóa về ba kênh Red, Green, Blue; ảnh xám hoặc ảnh có kênh trong suốt cũng được chuyển thành ảnh RGB để đầu vào nhất quán.
2. Khi nhận ảnh PIL, Ultralytics chuyển ảnh sang mảng NumPy. Trong đường đi nội bộ của phiên bản đang dùng, mảng được đảo thứ tự kênh RGB thành BGR ở bước nạp ảnh, rồi predictor đổi BGR trở lại RGB trước khi tạo tensor. Đây là chuyển đổi nội bộ có cặp tương ứng; dữ liệu cuối cùng đưa vào model vẫn theo thứ tự RGB.
3. Ultralytics đổi kích thước ảnh để phù hợp với `imgsz` (mặc định 640). Với ảnh không vuông, LetterBox giữ nguyên tỷ lệ khung hình, thu nhỏ ảnh rồi thêm phần đệm (padding, thường màu xám) để đạt kích thước cần thiết thay vì kéo giãn làm méo con tôm. Dataset huấn luyện hiện có ảnh 640 × 640, nhưng ảnh tải lên API có thể có kích thước khác.
4. Mảng ảnh được đổi từ dạng `(cao, rộng, 3)` sang tensor dạng `(batch, 3, cao, rộng)`, là thứ tự chiều mà PyTorch/model sử dụng. Giá trị pixel ban đầu trong khoảng 0–255 được đổi sang số thực trong khoảng 0–1 bằng cách chia cho 255; model đang dùng float32 trên CPU.
5. Detection head sinh nhiều ứng viên. Mỗi ứng viên có class ID, score/confidence và tọa độ box. Vì chỉ có class `shrimp-seed`, class ID `0` được ánh xạ thành tên này; confidence là điểm tin cậy model gán cho phát hiện, không phải xác suất hiệu chuẩn đảm bảo đúng tuyệt đối.
6. Ultralytics áp dụng ngưỡng confidence `conf` (API mặc định `0.25`, có thể truyền giá trị 0–1) để bỏ ứng viên yếu, rồi dùng Non-Maximum Suppression (NMS) để ưu tiên box điểm cao và bỏ box chồng lấn quá nhiều, tránh mô tả lặp cùng một con.
7. Ultralytics ánh xạ tọa độ box trở lại kích thước ảnh gốc. `ai_service/main.py` đọc `xyxy` (x1,y1 góc trái trên; x2,y2 góc phải dưới), làm tròn hai chữ số, lấy confidence/class và tạo JSON. `count` bằng số box còn lại sau lọc/NMS; do đó đây là số dự đoán, không phải nhãn thật hay cam kết đếm tuyệt đối chính xác.
8. Sau khi suy luận thành công, API gửi bytes của ảnh gốc lên Cloudinary trong thread pool. Ảnh được lưu trong `shrimp-detection/originals`; ảnh được upload là ảnh gốc, không phải tensor đã resize/chuẩn hóa.

Các thư viện chính được khai báo trong `requirements.txt`: `ultralytics==8.3.184` (YOLOv8, inference, hậu xử lý và metric), `torch`/`torchvision` (tensor và thực thi mô hình, được kéo theo theo môi trường cài đặt), `fastapi` và `uvicorn` (HTTP API), `Pillow` (đọc/chuyển ảnh), `python-multipart` (nhận upload), cùng Cloudinary SDK để backup ảnh. Script huấn luyện gọi API Python của Ultralytics trực tiếp.

RGB mô tả thứ tự ba thành phần màu của từng pixel: đỏ (Red), xanh lá (Green) và xanh dương (Blue). Chẳng hạn, pixel `(255, 0, 0)` là đỏ thuần. Chuyển ảnh sang RGB không phải là bước nhận diện hay làm ảnh rõ hơn; mục đích là đưa ảnh về định dạng màu thống nhất. Các bước đổi kích thước, thêm padding, sắp xếp chiều tensor và chuẩn hóa pixel mới là phần tiền xử lý để model nhận ảnh.

```text
Ảnh gốc → RGB → resize/letterbox → tensor RGB, pixel 0–1 → YOLO dự đoán box, class, confidence
    → lọc confidence và NMS → chuyển tọa độ, đếm → upload ảnh gốc → JSON
```

`conf` thấp thường giữ lại nhiều dự đoán hơn, nhưng có thể tăng nhận nhầm. `conf` cao giảm dự đoán yếu, nhưng có thể làm bỏ sót tôm. NMS giúp tránh đếm một con nhiều lần khi model tạo các box gần trùng nhau.

### 4.3. Dự đoán bằng dòng lệnh

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

### 4.4. AI API

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

Dùng để kiểm tra service, đường dẫn model, trạng thái nạp model và cấu hình Cloudinary.

```http
POST /predict
```

Dùng để tải ảnh lên, nhận kết quả phát hiện và lưu ảnh gốc lên Cloudinary. Ví dụ response:

```json
{
  "filename": "sample.jpg",
  "image_saved": true,
  "image_url": "https://res.cloudinary.com/<cloud-name>/image/upload/...",
  "public_id": "shrimp-detection/originals/...",
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
- `image_saved`: xác nhận ảnh gốc đã được backup lên Cloudinary.
- `image_url`: URL HTTPS dùng để xem ảnh đã lưu.
- `public_id`: mã asset dùng để quản lý hoặc truy xuất ảnh trên Cloudinary.

Các kiểm thử tự động dùng mock Cloudinary bao phủ bốn trường hợp: thiếu cấu hình, upload thành công, lỗi upload và ảnh không hợp lệ. Chưa kiểm thử upload trực tiếp lên tài khoản Cloudinary thật; bước đó cần credentials hợp lệ trong biến môi trường.

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
