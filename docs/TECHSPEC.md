# TECHSPEC: Hệ thống phát hiện và đếm tôm giống bằng YOLO (YOLOv8)

| | |
|---|---|
| Tên dự án | Shrimp Seed Detection (AI Service cho hệ thống ChillShrimp) |
| Phiên bản tài liệu | 2.0 |
| Ngày cập nhật | 2026-11-03 |
| Nhánh git | `feature/cloudinary` (remote: `noat04/shrimp-seed-detection.v1i.yolov8`) |
| Trạng thái | Demo và nghiên cứu. Chưa đủ điều kiện dùng cho production (xem mục 19) |

Quy ước trong tài liệu: **[Đã xác minh]** nghĩa là nội dung được đọc trực tiếp từ mã nguồn, file cấu hình hoặc kết quả đo trong repo. **[Suy ra]** nghĩa là suy luận từ ảnh chụp màn hình hoặc tài liệu, vì phần đó nằm ngoài repo. **[Đề xuất]** là ngưỡng hoặc quy tắc do tài liệu này đề ra, chưa phải yêu cầu đã được chốt.

Bản 2.0 bổ sung: tiêu chí ảnh và vật thể cần đếm (mục 2), các tham số quan trọng (mục 9), cách huấn luyện từng bước (mục 11), cấu trúc dự án huấn luyện (mục 12), tiêu chí đánh giá đạt (mục 13).

---

## 1. Tổng quan

### 1.1. Mục tiêu
Tự động **phát hiện và đếm tôm giống** (post-larvae) trong ảnh chụp mẫu nước, thay cho việc đếm tay, để hỗ trợ kiểm tra chất lượng lô giống tại trang trại nuôi tôm.

### 1.2. Ý tưởng cốt lõi
- Dùng một mô hình phát hiện vật thể **YOLOv8** (1 lớp: `shrimp-seed`). Mỗi con tôm được phát hiện là một khung chữ nhật (bounding box) quanh **vùng đầu và thân sẫm** của con tôm (xem mục 2.4).
- **Số đếm = số khung phát hiện được** sau khi lọc theo ngưỡng tin cậy và loại khung trùng (NMS).
- Mô hình được đóng gói thành một **microservice HTTP (FastAPI)**. Hệ thống ChillShrimp gửi ảnh vào, nhận lại số đếm, danh sách khung và đường dẫn ảnh.
- Vì ảnh thực tế rất đa dạng (mật độ, màu tôm, độ phân giải), dữ liệu huấn luyện được **chia theo loại ảnh (A, B, C)**, cân bằng khi train và đo sai số riêng từng loại.
- Với ảnh mật độ dày, tùy chọn **suy luận theo ô** (tiling): cắt ảnh thành các ô nhỏ chồng lấn, chạy YOLO trên từng ô rồi gộp kết quả.

### 1.3. Phạm vi
**Trong phạm vi:** dataset và nhãn, huấn luyện mô hình, dịch vụ suy luận `/predict`, lưu ảnh gốc và ảnh đã vẽ khung lên Cloudinary, công cụ gán nhãn và mở rộng dữ liệu, đánh giá sai số đếm, kiểm thử đơn vị.

**Ngoài phạm vi (không có trong repo):** frontend và backend ChillShrimp, cơ sở dữ liệu nghiệp vụ, xác thực người dùng, tính mật độ (con/ml), hiệu chỉnh số đếm của người dùng. Video chưa được hỗ trợ (hướng làm: lấy mẫu khung hình rồi lấy trung vị số đếm).

### 1.4. Mục tiêu đếm

**Mục tiêu tổng quát:** với mỗi ảnh mẫu tôm giống, hệ thống trả về một số đếm **sát với số con thực tế nhìn thấy được trong ảnh**, đủ tin cậy để người dùng dùng làm số ban đầu và chỉ cần hiệu chỉnh nhỏ (cột "Số đếm hiệu chỉnh") thay vì đếm lại từ đầu.

**Định nghĩa "đếm đúng":**
- Mỗi con tôm giống **phân biệt được bằng mắt** trong ảnh được tính đúng một lần (không đếm trùng, không đếm bỏ sót).
- Con bị che kín hoàn toàn, hoặc ảnh quá mờ không phân biệt được cá thể, **không thuộc mục tiêu đếm** (cùng quy tắc với `LABELING_GUIDE.md`).
- Vật không phải tôm (logo, bóng, bọt, vết bẩn, thành chậu, chấm đen nhỏ rời) **không được đếm**.
- Số đếm là số con **trong ảnh**, không phải trong cả bể. Mật độ (con/ml) là bước tính của ChillShrimp: `số đếm ÷ thể tích mẫu`, nên sai số đếm truyền thẳng vào sai số mật độ.

**Chỉ số đo mục tiêu đếm:**

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| MAE | trung bình `abs(đếm − thật)` mỗi ảnh | Sai số tuyệt đối, hợp với ảnh thưa |
| Sai số tương đối | trung bình `abs(đếm − thật) ÷ thật` | Hợp với ảnh dày, nơi sai số tuyệt đối tự nhiên lớn |
| Bias | trung bình `(đếm − thật)` | Dương: đếm dư; âm: đếm thiếu. Bias lệch một chiều là dấu hiệu cần chỉnh `conf` hoặc dữ liệu |

Số thật được lấy từ **nhãn đã duyệt tay** hoặc **số đếm tay** của ảnh, trên tập ảnh **không dùng để train**. Ngưỡng đạt theo loại ảnh xem mục 13.3.

### 1.5. Bối cảnh sử dụng
Trong ảnh chụp giao diện ChillShrimp (trang "Kiểm tra chất lượng") **[Suy ra]**, người dùng tải ảnh AI. Hệ thống hiển thị: ảnh gốc, ảnh đã đánh dấu kết quả AI, "Số lượng AI đếm", "Số đếm hiệu chỉnh" (người dùng nhập), "Mật độ mẫu (con/ml)" (cần thể tích mẫu), "Độ tin cậy" (có vẻ là trung bình confidence của các khung), "Phiên bản model" (hiện `best.pt`), và lịch sử các lần kiểm tra.

---

## 2. Vật thể cần đếm và tiêu chí ảnh

### 2.1. Vật thể cần nhận biết
Mô hình chỉ có **một lớp**: `shrimp-seed` (tôm giống). Với mô hình, một "con tôm" là một vùng ảnh có các đặc điểm sau (suy ra từ ảnh mẫu và nhãn trong repo):

| Đặc điểm | Mô tả |
|---|---|
| Hình dạng | Thân thon dài, có **phần đầu-ngực** đậm màu, đuôi và râu mảnh, trong suốt |
| Màu sắc | Phụ thuộc loại ảnh: thân **sẫm** (đen, nâu đậm) ở loại A; thân **trong suốt có đốm cam đỏ** và mắt đen ở loại B; tôm rất nhỏ, tối ở loại C |
| Vùng được khung bao | Phần **đầu và thân sẫm**, không bao phần đuôi và râu mảnh (quy ước ở mục 2.4) |
| Nền | Sáng (trắng, xám nhạt, be) trong chậu, khay hoặc bể; nhìn từ trên xuống |
| Dễ nhầm | Chấm đen nhỏ rời (mắt, bọt, vết bẩn), bóng đổ, vật trang trí, thước, đồng xu trong ảnh |

Ví dụ lỗi nhận nhầm đã quan sát: mô hình cũ nhận **hai chấm đen (mắt)** của tôm loại B thành vật thể riêng và vẽ nhiều khung chồng lên nhau.

### 2.2. Các loại ảnh (domain)
Ảnh đầu vào được chia thành ba loại. Dữ liệu huấn luyện, nhãn và đánh giá đều tách theo loại này.

| Loại | Mô tả | Màu sắc | Mật độ | Ví dụ trong repo |
|---|---|---|---|---|
| **A** | Tôm **nhỏ, thân tối**, trong chậu hoặc khay, chen nhau | Thân nâu đen, nền trắng hoặc xám | **Rất dày**, trên 100 con mỗi ảnh (nhiều ảnh vài trăm con), có chồng dính | `real_001`, `real_004`, `real_007`, ảnh 600×337, 13 ảnh PL dày nhất (101 đến 237 con) |
| **B** | Tôm **to hơn, thân trong suốt**, đốm cam đỏ ở đầu-ngực, mắt đen rõ | Trong suốt kèm cam đỏ, nền trắng | **Thưa**, khoảng 10 đến 40 con (có ảnh vừa khoảng 70 đến 80 con) | `real_003`, ảnh 1050×700 trong `to_label/B/` |
| **C** | **Ảnh cận cảnh 640×640** từ dataset gốc, tôm rất nhỏ | Tôm nhỏ, tối, nền xám nhạt hoặc nền có thành chậu tối | **Thưa**, trung vị khoảng 8 box mỗi ảnh; có 24 ảnh gần như không có tôm | Thư mục `train/ valid/ test/` (tên ảnh `benur*`, `PL3*`) |
| D (tạm) | Ảnh tô mật độ vừa, thân mảnh, màu xám nâu nhạt | Xám nâu nhạt | 48 đến 124 con | `real_005`, `real_008`, `real_011`, `real_012`; chưa thuộc loại nào, đang để ngoài dataset huấn luyện |

Một kiểu ảnh chưa có loại: **thân trong suốt cam như B nhưng rất dày** (vài trăm con). Có thể cắt thành các mảnh nhỏ để dùng làm dữ liệu B.

### 2.3. Tiêu chí một ảnh được chấp nhận vào tập huấn luyện hoặc kiểm tra **[Đề xuất]**

**Chung cho mọi loại**
- **Đúng loại:** so sánh bằng mắt với ảnh mẫu của loại đó (mục 2.2). Ảnh giống nhưng thuộc loại khác không tính.
- **Đủ rõ:** đầu và mắt từng con còn phân biệt được khi phóng 100%. Ảnh quá nhỏ hoặc nén nặng bị loại (đã loại ảnh 192×128 và 299×168). Nên có cạnh dài từ khoảng 600 px trở lên.
- **Không có chữ chìm, logo, hình mờ** đè lên tôm.
- **Không trùng:** kiểm tra bằng băm ảnh với ảnh đã có (đã phát hiện nhiều ảnh gửi lại trùng ảnh cũ).
- **Rõ nguồn và giấy phép** (ghi trong `to_label/SOURCES.json`); ảnh từ internet phải kiểm tra giấy phép, CC BY thì phải ghi nguồn.
- **Đa dạng:** ít nhất 5 nguồn hoặc phiên chụp khác nhau cho mỗi loại; khác nền, ánh sáng, góc chụp, dụng cụ. Nhiều mảnh cắt từ một ảnh gốc chỉ tính là **một** ảnh nguồn.

**Số lượng:** mỗi loại cần tối thiểu **25 ảnh nguồn**: 18 để train, 2 để validation, 5 để giữ riêng làm test. Con số này là kết quả của cách chia dữ liệu trong `build_domain_dataset.py`, không phải kết luận từ nghiên cứu nào.

**Riêng từng loại**

| Loại | Mật độ chấp nhận | Màu và hình dạng | Ghi chú thêm |
|---|---|---|---|
| A | Trên 100 con mỗi ảnh | Thân sẫm, tôm nhỏ | Ảnh chậu **đầy, chồng dính** quan trọng hơn ảnh thưa; ảnh PL 640×640 là "A nhẹ" (ít chồng dính) |
| B | Khoảng 10 đến 40 con | Trong suốt, đốm cam đỏ, mắt đen | Nền sáng, thưa; ảnh quá dày thì cắt mảnh |
| C | Bất kỳ, thường thưa | Tôm nhỏ, tối | Đã có sẵn nhiều ảnh trong dataset gốc |

**Ảnh giữ riêng (test) của mỗi loại**
- Lấy từ nguồn hoặc phiên chụp **khác** ảnh train.
- Có **số đếm tay** hoặc nhãn đã duyệt tay đáng tin; **không** dùng nhãn nháp của mô hình làm số thật.
- Trải đều mật độ trong loại đó (ít, vừa, nhiều).
- Không đưa vào train ở bất kỳ vòng nào.

### 2.4. Quy ước nhãn
- **Một khung cho mỗi con** phân biệt được.
- Khung bao **vùng đầu và thân sẫm** (quy ước "vòng đầu", cùng quy ước với bộ PL Detection), **không** bao phần đuôi và râu mảnh.
- Con bị che một phần: chỉ bao phần nhìn thấy được. Con bị che kín: không gán.
- Khi cắt ảnh thành ô để train, khung được giữ nếu còn ít nhất **50%** diện tích nằm trong ô.
- Công cụ gán nhãn: `dataset_expansion/annotate.html`; quy tắc chi tiết: `dataset_expansion/LABELING_GUIDE.md`.
- **Cần lưu ý:** hướng dẫn gán nhãn cũ yêu cầu bao cả thân; nhãn `real_003` vẫn đang bao cả thân và cần gán lại theo quy ước vòng đầu (xem mục 19).

### 2.5. Nơi tìm ảnh
Tự chụp tại trại giống (tốt nhất); xin ảnh kiểm tra chất lượng từ nhà cung cấp giống; Roboflow Universe (Larvae Shrimp Counting, shirmp larvae count, Counting PL (Baby Shrimp)); bộ Penaeus_1k (nhãn điểm, hiện không tải được từ repo tác giả). Tất cả nguồn từ internet phải kiểm tra giấy phép và quy ước nhãn trước khi dùng.

---

## 3. Vấn đề và hạn chế cốt lõi

| Vấn đề | Chi tiết |
|---|---|
| Ảnh mật độ dày bị đếm thiếu | Ảnh 953×1006 chứa vài trăm con nhưng model đầu tiên chỉ đếm được 30 con **[Đã xác minh]**. |
| Nguyên nhân gốc | Model đầu tiên chỉ được train trên 300 ảnh gốc, trung bình khoảng 7,8 box mỗi ảnh; tôm rất nhỏ; ảnh bị thu về 640 px. Kết quả thay đổi mạnh theo **dữ liệu**, ít phụ thuộc vào kiến trúc (xem mục 10.4). |
| Lệch loại ảnh | Ảnh tôm to, trong suốt (loại B) hoàn toàn khác dữ liệu đã train nên bị đếm sai nặng (mô hình nhận mắt đen thành vật thể). |
| Chưa có số đếm thật cho ảnh dày | Thiếu tập ảnh chậu dày được đếm tay để đo độ đúng. |
| Hướng đang theo | Chia dữ liệu theo loại A, B, C, thêm ảnh dày (13 ảnh PL dày nhất, ảnh tự gán nhãn), cân bằng khi train, đo sai số riêng từng loại. |

---

## 4. Yêu cầu

### 4.1. Chức năng
| ID | Yêu cầu | Trạng thái |
|---|---|---|
| F1 | Nhận ảnh JPG/PNG qua `multipart/form-data`, trả số đếm và khung phát hiện | Có |
| F2 | Cho phép đổi `conf`, `imgsz`, `max_det` mỗi lần gọi | Có |
| F3 | Suy luận theo ô cho ảnh dày (`tile_size`, `tile_overlap`, `merge_iou`) | Có, mặc định tắt |
| F4 | Lưu ảnh gốc lên Cloudinary và trả URL, `public_id` | Có |
| F5 | Lưu ảnh đã vẽ khung lên Cloudinary và trả URL | Có |
| F6 | Kiểm tra sức khỏe dịch vụ, nạp model lười (lazy) | Có |
| F7 | Chọn model bằng biến môi trường `MODEL_PATH` | Có |
| F8 | Công cụ gán nhãn, cắt ô, dựng dataset theo loại, huấn luyện lại, đo sai số đếm | Có |
| F9 | Cho phép đổi ngưỡng NMS (`iou`) của YOLO qua API | Chưa (dùng mặc định 0,7) |
| F10 | Cờ báo số đếm bị cắt khi chạm `max_det` | Chưa |
| F11 | Chế độ "dày" gói sẵn tham số cho ảnh mật độ cao | Chưa |
| F12 | Xử lý video | Chưa |
| F13 | Tính mật độ con/ml | Ngoài phạm vi (do ChillShrimp tính) |

### 4.2. Phi chức năng
| Nhóm | Ghi nhận |
|---|---|
| Hiệu năng | Khoảng 93 ms mỗi ảnh trên CPU i5-11400H cho `yolov8n` (số đo trong `PROJECT_STATUS.md`). `yolov8s` lớn hơn nên chậm hơn, **chưa đo**. Suy luận theo ô tốn thời gian tỉ lệ với số ô |
| Độ chính xác | Xem mục 13 |
| Tính sẵn sàng | Một tiến trình uvicorn, một bản model nạp vào bộ nhớ. Không có cân bằng tải |
| Bảo mật | Credentials Cloudinary đọc từ biến môi trường. Chưa có xác thực, giới hạn kích thước upload hay CORS (xem mục 18) |
| Khả năng tái tạo | Phiên bản thư viện được ghim trong `requirements.txt`; `build_domain_dataset.py` dùng seed cố định (mặc định 0) nên cùng đầu vào cho cùng dataset |

---

## 5. Kiến trúc

### 5.1. Sơ đồ thành phần
```text
┌──────────────┐  ảnh   ┌───────────────────┐  POST /predict   ┌────────────────────────────┐
│  Người dùng  │ ─────▶ │ ChillShrimp       │ ───────────────▶ │ AI Service (FastAPI)       │
│  (trình      │        │ frontend+backend  │ ◀─────────────── │ ai_service/main.py         │
│   duyệt)     │ ◀───── │ [ngoài repo]      │   JSON kết quả   │  ├─ YOLO best.pt (lazy)    │
└──────────────┘ hiển   └───────────────────┘                  │  ├─ detect / detect_tiled  │
                 thị                                           │  ├─ merge_detections (NMS) │
                                                               │  └─ plot_detections        │
                                                               └─────────────┬──────────────┘
                                                                  upload     │  upload
                                                               ảnh gốc       ▼  ảnh có khung
                                                               ┌─────────────────────────────┐
                                                               │ Cloudinary                  │
                                                               │ shrimp-detection/originals  │
                                                               │ shrimp-detection/annotated  │
                                                               └─────────────────────────────┘
```
Sơ đồ chi tiết theo file: [project_flow.png](project_flow.png). Quy trình huấn luyện: [train_pipeline.png](train_pipeline.png). Cơ chế học: [train_explained.png](train_explained.png).

### 5.2. Nguyên tắc thiết kế
- **AI Service là dịch vụ độc lập** giao tiếp qua HTTP. Backend chính (bất kể ngôn ngữ) không import mã Python (theo `HUONG_DAN_TICH_HOP_AI_SERVICE.md`).
- **Không huấn luyện khi chạy.** Dịch vụ chỉ nạp file trọng số `best.pt` và suy luận.
- **Số đếm nằm ở dịch vụ AI, nghiệp vụ nằm ở ChillShrimp** (mật độ, hiệu chỉnh, lịch sử).
- **Mở rộng dữ liệu theo quy trình có duyệt tay**: nhãn do mô hình đề xuất chỉ là bản nháp, không được xem là ground truth.
- **Chia dữ liệu theo ảnh nguồn**: mọi bản tăng cường và mọi ô của một ảnh luôn nằm cùng một split.

### 5.3. Thành phần chính **[Đã xác minh]**
| Thành phần | File | Vai trò |
|---|---|---|
| API suy luận | `ai_service/main.py` | FastAPI: `/health`, `/predict`, nạp model, suy luận, ghép ô, vẽ khung, upload |
| Cấu hình Cloudinary | `ai_service/cloudinary_config.py` | Đọc `.env`, kiểm tra đủ ba biến, đặt `CLOUDINARY_CONFIGURED` |
| Huấn luyện ban đầu | `train_yolo.py` | Fine-tune `yolov8n.pt`, 30 epoch |
| Công cụ gán nhãn | `dataset_expansion/annotate.html` | Vẽ, sửa, xuất nhãn YOLO trên trình duyệt |
| Nhãn nháp | `dataset_expansion/draft_pending_labels.py` | Sinh nhãn đề xuất bằng model và suy luận theo ô |
| Cắt ô | `dataset_expansion/make_tiles.py` | Cắt ảnh lớn thành ô, chia split theo ảnh nguồn |
| Dựng dataset theo loại | `dataset_expansion/build_domain_dataset.py` | Dựng dataset cân bằng từ `domains_spec.json` |
| Cấu hình loại ảnh | `dataset_expansion/domains_spec.json` | Khai báo loại, kích thước ô, nguồn ảnh |
| Huấn luyện | `dataset_expansion/train_reviewed.py` | Kiểm tra dữ liệu rồi train (mặc định `yolov8s`, GPU) |
| Huấn luyện v2 cố định | `dataset_expansion/run_v2_training.py` | Train `yolov8n` trên `dataset_v2` |
| Đánh giá số đếm | `evaluate_counting.py` | MAE, bias cho nhiều cấu hình, trên dataset gốc |
| Đánh giá theo loại | `evaluate_domains.py` | MAE, sai số tương đối, bias riêng từng loại trên ảnh giữ riêng |
| Kiểm thử | `tests/` | `unittest`, mock Cloudinary |

---

## 6. Đầu vào

### 6.1. Request
```http
POST /predict?conf=0.25&imgsz=640&max_det=300&tile_size=0&tile_overlap=0.2&merge_iou=0.5
Content-Type: multipart/form-data
file=<ảnh JPG/PNG>
```
Tên field chứa ảnh bắt buộc là `file`. Các tham số query đều tùy chọn; ý nghĩa và giá trị khuyến nghị ở mục 9.

### 6.2. Đặc điểm ảnh đầu vào
- Định dạng: ảnh Pillow đọc được; được ép về RGB (`convert("RGB")`). Không có giới hạn kích thước hay dung lượng trong mã.
- **Ảnh huấn luyện gốc** có kích thước 640×640 (đã resize kiểu Stretch khi xuất từ Roboflow). **Ảnh thực tế** có thể khác (ví dụ 953×1006, 960×540, 600×337, 1050×700, 1200×800).
- Bối cảnh chụp: tôm giống trong chậu hoặc khay, nhìn từ trên xuống, nền sáng. Mật độ từ vài con đến vài trăm con. Các loại ảnh A, B, C xem mục 2.2.
- Tiền xử lý do Ultralytics làm: letterbox giữ tỷ lệ về `imgsz`, chuẩn hóa 0–1, tensor `(1, 3, H, W)`.

### 6.3. Cấu hình môi trường
| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `CLOUDINARY_CLOUD_NAME` | Có (để `/predict` chạy) | Tên cloud |
| `CLOUDINARY_API_KEY` | Có | API key |
| `CLOUDINARY_API_SECRET` | Có | API secret |
| `MODEL_PATH` | Không | Đường dẫn file `.pt`; mặc định `runs/detect/shrimp_seed_yolov8n/weights/best.pt` |

Cả ba biến Cloudinary được nạp từ `.env` bằng `python-dotenv`. Không đặt biến nào thì dịch vụ vẫn khởi động (`cloudinary_configured: false`). Đặt thiếu một hoặc hai biến thì **dịch vụ báo `RuntimeError` ngay khi khởi động**.

---

## 7. Đầu ra

### 7.1. Phản hồi thành công (HTTP 200) **[Đã xác minh]**
```json
{
  "filename": "sample.jpg",
  "image_saved": true,
  "annotated_image_saved": true,
  "image_url": "https://res.cloudinary.com/<cloud>/image/upload/...",
  "public_id": "shrimp-detection/originals/...",
  "annotated_image_url": "https://res.cloudinary.com/<cloud>/image/upload/...",
  "annotated_public_id": "shrimp-detection/annotated/...",
  "image": { "width": 953, "height": 1006 },
  "count": 220,
  "detections": [
    { "class_id": 0, "class": "shrimp-seed", "confidence": 0.87,
      "box": { "x1": 120.2, "y1": 80.5, "x2": 180.9, "y2": 145.3 } }
  ]
}
```
- `count = len(detections)`. Tọa độ khung tính bằng pixel của **ảnh gốc** (góc trái trên là `x1,y1`, góc phải dưới là `x2,y2`), làm tròn hai chữ số.
- `confidence` là điểm tin cậy của model cho từng khung, không phải xác suất đã hiệu chỉnh.
- Ảnh vẽ khung chỉ vẽ khung (không nhãn chữ, không confidence), lưu dạng PNG. Số khung vẽ luôn khớp `count`, kể cả suy luận theo ô.
- **Lưu ý:** khi `count` bằng đúng `max_det` thì kết quả gần như chắc chắn đã bị cắt; API chưa có cờ báo việc này.

### 7.2. Phản hồi lỗi
Thứ tự kiểm tra trong mã: đọc ảnh, kiểm tra Cloudinary, nạp model, suy luận, upload ảnh gốc, upload ảnh vẽ.

| Mã | Điều kiện | Nội dung |
|---|---|---|
| 400 | File không phải ảnh hợp lệ | `Uploaded file is not a valid image.` |
| 422 | Thiếu `file` hoặc tham số ngoài giới hạn | Do FastAPI tự sinh |
| 503 | Chưa cấu hình đủ biến Cloudinary | `detail.image_saved = false` (không chạy model) |
| 503 | Không tìm thấy file model | Gợi ý train hoặc đặt `MODEL_PATH` |
| 502 | Upload ảnh gốc thất bại | `detail.image_saved = false` (kết quả đếm bị bỏ, không trả về) |
| 502 | Ảnh gốc đã lưu nhưng upload ảnh vẽ thất bại | `image_saved = true`, kèm `image_url`, `public_id`, `annotated_image_saved = false`; ảnh gốc không bị xóa |

### 7.3. `GET /health`
```json
{ "status": "ok", "model_path": "...", "model_exists": true,
  "model_loaded": false, "cloudinary_configured": true }
```
`model_loaded` là `false` cho đến lần `/predict` đầu tiên. Trang gốc `/` không có nội dung (404). Khi dùng trình duyệt, mở `http://localhost:8001/docs` hoặc `/health` (không mở `0.0.0.0`, đó chỉ là địa chỉ để server lắng nghe).

---

## 8. Luồng xử lý suy luận

```text
Ảnh tải lên ─▶ Pillow đọc, ép RGB ─▶ kiểm tra Cloudinary ─▶ nạp model (lần đầu)
   │
   ├─ tile_size = 0 ─▶ YOLO predict toàn ảnh ─▶ lọc conf + NMS của Ultralytics
   │
   └─ tile_size > 0 ─▶ chia lưới ô chồng lấn (tile_origins)
                       ─▶ YOLO predict từng ô (ảnh BGR cho numpy)
                       ─▶ cộng offset (x0,y0) vào tọa độ khung
                       ─▶ merge_detections: NMS tham lam theo IoU ≥ merge_iou
   ▼
count = số khung ─▶ vẽ khung lên ảnh gốc (BGR → RGB → PNG)
   ─▶ upload ảnh gốc, upload ảnh vẽ (thread pool) ─▶ JSON
```

Chi tiết:
- **Lưới ô:** bước nhảy = `int(tile × (1 − overlap))`; ô cuối được dịch sát mép ảnh; nếu cạnh ảnh không lớn hơn `tile` thì chỉ có một ô.
- **Gộp khung:** sắp theo confidence giảm dần, bỏ khung có IoU ≥ `merge_iou` với khung đã giữ. Cách này chỉ dùng IoU, nên khung bị cắt ở biên ô có thể không bị loại.
- **Màu:** khi truyền mảng numpy cho Ultralytics phải là BGR; mã đã đảo kênh và có test (`TiledColorTests`) bảo vệ điều này.
- **Đồng bộ:** suy luận chạy trực tiếp trong hàm `async`, nên một request suy luận chặn vòng lặp sự kiện trong lúc chạy; riêng upload Cloudinary chạy trong thread pool.

---

## 9. Tham số quan trọng

### 9.1. Tham số suy luận (API `/predict`)

| Tham số | Mặc định trong mã | Tác dụng | Khuyến nghị hiện tại |
|---|---|---|---|
| `conf` | 0,25 | Ngưỡng điểm tin cậy; thấp hơn thì ra nhiều khung hơn nhưng dễ nhận nhầm | **0,25** với model `shrimp_domains_v3`. Hạ xuống 0,15 làm đếm dư rõ rệt (ví dụ `real_007` thành 309 khung) |
| `imgsz` | 640 | Kích thước ảnh đưa vào mô hình | **640**, khớp với lúc train. 1280 chưa được thử với v3 (với model cũ không cải thiện ổn định) |
| `max_det` | 300 | **Mức trần** số khung trả về; vượt quá bị cắt âm thầm | **1000** (ảnh dày nhất đã gặp khoảng 330 con; bài báo Penaeus_1k có ảnh 500 đến 800). Chưa đổi mặc định trong mã. Không cần ước lượng số tôm trước, chỉ cần đặt cao hơn số tôm có thể có |
| `tile_size` | 0 (tắt) | Cạnh ô cắt theo px | **0** với v3 (cắt ô làm đếm dư: `real_007` 103 con toàn ảnh so với 281 con ở ô 320). Cắt ô có ích với model cũ `yolov8n` ở ảnh dày |
| `tile_overlap` | 0,2 | Tỷ lệ chồng lấn giữa hai ô | Chỉ có tác dụng khi `tile_size` lớn hơn 0 |
| `merge_iou` | 0,5 | Ngưỡng IoU gộp khung trùng giữa các ô | Chỉ có tác dụng khi cắt ô. **Khác** với `iou` bên dưới |
| `iou` (YOLO NMS) | **Không có trong API**, Ultralytics dùng 0,7 | Ngưỡng loại khung trùng bên trong một lần dự đoán | Thử cho thấy hạ xuống 0,5 giảm khung đôi (v3 trên `real_007`: 165 xuống 153) nhưng **chưa nằm trong API** |

Bốn quy tắc dễ nhầm:
- `max_det` chỉ là mức trần, không phải số tôm mong đợi.
- `merge_iou` chỉ gộp khung giữa các ô; `iou` mới là ngưỡng NMS thường của YOLO.
- Với v3, cấu hình khuyến nghị là **toàn ảnh, `conf=0.25`, `imgsz=640`**.
- Cấu hình phải đi cùng model: cấu hình tối ưu của model cũ khác với v3.

### 9.2. Tham số huấn luyện (`dataset_expansion/train_reviewed.py`)

| Tham số | Giá trị | Tác dụng và lý do |
|---|---|---|
| Mô hình nền | `yolov8s.pt` (COCO) | Fine-tune từ trọng số tiền huấn luyện; `yolov8s` có khoảng 11 triệu tham số, đủ sức chứa nhiều loại ảnh hơn `yolov8n` (khoảng 3 triệu) |
| `epochs` | 100 (v1, v2), 60 (v3) | Số vòng học; epoch tốt nhất thường nằm giữa lần chạy (v2: epoch 50; v3: epoch 31) |
| `imgsz` | 640 | Kích thước ảnh lúc train; ô 320 px của loại A được phóng lên 640 |
| `batch` | 4 | **Batch 8 từng hết bộ nhớ GPU 4 GB** với ảnh dày (hàng trăm vật thể mỗi batch) |
| `device` | GPU nếu có | RTX 3050 Laptop 4 GB; CPU chậm hơn nhiều |
| `workers` | 2 | Số luồng nạp ảnh |
| `fliplr`, `flipud` | 0,5 và 0,5 | Lật ngang và dọc: ảnh nhìn từ trên xuống, không có hướng cố định |
| `degrees` | 0 | **Tắt xoay**: xoay một khung thẳng quanh con tôm mảnh nằm chéo làm khung lỏng |
| `hsv_h`, `hsv_s`, `hsv_v` | 0,015, 0,5, 0,4 | Đổi màu, độ bão hòa, độ sáng: chịu được điều kiện ánh sáng khác nhau |
| `scale` | 0,5 | Phóng to thu nhỏ ngẫu nhiên |
| `mosaic`, `close_mosaic` | 1,0 và 10 | Ghép 4 ảnh khi train; 10 epoch cuối tắt để học ảnh nguyên bản |
| `max_det` | 1000 | Trần số khung khi validation (ảnh dày) |
| `optimizer` | tự chọn (Ultralytics) | Log cho thấy chọn AdamW; `lr0` mặc định bị bỏ qua khi optimizer là auto |

### 9.3. Tham số dựng dataset (`build_domain_dataset.py`)

| Tham số | Mặc định | Tác dụng |
|---|---|---|
| `--holdout` | 5 | Số ảnh nguồn mỗi loại giữ riêng làm test (ảnh nguyên bản, không cắt ô) |
| `--valid` | 2 | Số ảnh nguồn mỗi loại dùng để validation |
| `--target` | kích thước loại lớn nhất | Số file train mỗi loại sau khi lặp; v3, v4: tự động; v5: 150 |
| `--max-repeat` | 8 | Số lần lặp tối đa của một file train |
| `--min-visible` | 0,5 | Tỷ lệ diện tích tối thiểu của khung phải nằm trong ô để giữ lại |
| `--seed` | 0 | Seed cố định cho việc chọn ảnh giữ riêng |
| `--lenient` | tắt | Loại nào quá ít ảnh thì cảnh báo và đưa hết vào train (không có ảnh test) |
| `"split"` trong spec | không | Ghim ảnh của một nguồn vào train, valid hoặc test |
| `"tile"` trong spec | 0 | Kích thước ô riêng cho từng loại (A đang là 320) |

---

## 10. Mô hình và dữ liệu

### 10.1. Mô hình
| Mục | Giá trị |
|---|---|
| Kiến trúc | YOLOv8 (Ultralytics 8.3.184): backbone, neck, head không anchor |
| Số lớp | 1 (`shrimp-seed`) |
| Model đang là mặc định của dịch vụ | `runs/detect/shrimp_seed_yolov8n/weights/best.pt` (`yolov8n`, khoảng 3 triệu tham số, 6,2 MB) |
| Model ứng viên tốt nhất hiện tại | `runs/detect/shrimp_domains_v3/weights/best.pt` (`yolov8s`, khoảng 11 triệu tham số, 22 MB). **Chưa thay vào dịch vụ**, thử bằng `MODEL_PATH` |
| Cách chọn `best.pt` | Epoch có chỉ số tổng hợp validation cao nhất (Ultralytics tính chủ yếu từ mAP50–95) |

### 10.2. Các bộ dữ liệu
| Bộ dữ liệu | Thư mục | Train / Valid / Test (ảnh) | Nguồn, giấy phép | Ghi chú |
|---|---|---|---|---|
| Gốc (loại C) | `train/ valid/ test/`, `data.yaml` | 210 / 60 / 30 | Roboflow `toan-minh-mshdd/shrimp-seed-detection-g1fon`, CC BY 4.0 | 2.317 box. Ảnh 640×640, tôm nhỏ, thưa (tối đa 37 box mỗi ảnh, trung vị khoảng 8) |
| `dataset_v2` | `dataset_v2/` | 210 / 60 / 30 (bản sao y hệt) | Như trên | Cộng thêm `pending/` (12 ảnh thực tế, nhãn nháp) |
| PL Detection | `external_datasets/` | 685 / 30 / 33 | Roboflow `teramina/pl-detection` v1, MIT | Nhãn **vòng đầu/trước**. 307 ảnh nguồn, 13 ảnh nguồn có từ 100 box trở lên |
| PL + ảnh tự gán nhãn | `dataset_pl_custom_v1/` | 691 / 31 / 34 | PL Detection (MIT) + 8 ảnh tự gán nhãn | 19.424 / 919 / 806 box. 8 ảnh tự gán nhãn có 1.128 box |
| 13 ảnh PL dày (loại A) | `to_label/A_pl_dense/` | 13 ảnh nguồn | PL Detection (MIT) | Mỗi ảnh 101 đến 237 box; tôm trải thưa hơn ảnh chậu thật |
| Ảnh chờ gán nhãn | `to_label/A/`, `to_label/B/` | 1 ảnh mỗi loại | Chưa rõ nguồn và giấy phép | Chưa có nhãn |
| Sao lưu sự cố | `recovery/accidental_extract_20261031_012945/` | 685 / 30 / 33 | | Không dùng để huấn luyện |

Nhãn YOLO: `class_id x_center y_center width height`, chuẩn hóa 0–1. Dataset gốc có 24 ảnh không có tôm (nhãn rỗng).

### 10.3. Các dataset theo loại A, B, C (do `build_domain_dataset.py` dựng)

Cách chia và tỷ lệ:
- **Chia theo ảnh nguồn:** mọi bản tăng cường (cùng tên trước `.rf.`) và mọi ô của một ảnh luôn thuộc cùng một split.
- Mỗi loại: **5 ảnh nguồn test** (nguyên bản), **2 ảnh nguồn valid**, phần còn lại train. Ảnh `real_007` được ghim vào test để có ít nhất một ảnh chậu thật được đo.
- Số ảnh test và valid **do khan hiếm dữ liệu quyết định**, không phải tỷ lệ chuẩn 70/20/10. Với 5 ảnh test, một ảnh sai đổi sai số trung bình đến 20%, nên kết quả chỉ mang tính tham khảo.
- **Cân bằng khi train:** mỗi loại được lặp file train để số file gần bằng `--target`, tối đa 8 lần. Lặp một ảnh nhiều lần chỉ là biện pháp tạm, mô hình có thể học thuộc ảnh đó (đã gặp ở loại B).

| Dataset | Loại A (ảnh nguồn tr/va/te) | Loại B | Loại C | Tập train | Tập valid | Tập test |
|---|---|---|---|---|---|---|
| `dataset_v3` | 3 / 0 / 0 (chỉ ảnh `real_*`) | 1 / 0 / 0 | 14 / 2 / 5 (21 ảnh nguồn) | 56 ảnh, 2.887 box | 5 ảnh, 26 box | 9 ảnh, 58 box |
| `dataset_v4` | 9 / 2 / 5 (thêm 13 ảnh PL dày) | 1 / 0 / 0 | 14 / 2 / 5 | 142 ảnh, 4.614 box | 23 ảnh, 497 box | 14 ảnh, 708 box |
| `dataset_v5` | 9 / 2 / 5 (A lặp 2 lần, 136 file) | 1 / 0 / 0 (lặp 8 lần) | 138 / 2 / 5 (257 file) | 401 ảnh, 9.590 box | 21 ảnh, 521 box | 15 ảnh, 707 box |

Trong `dataset_v5`, tỷ lệ theo số ảnh là khoảng 91,8% train, 4,8% valid và 3,4% test. Loại B **không có ảnh giữ riêng** vì chỉ có một ảnh nguồn.

### 10.4. Các lần huấn luyện đã có
| Run (`runs/detect/`) | Dataset | Mô hình, cấu hình | Kết quả (tập valid của chính run đó) |
|---|---|---|---|
| `shrimp_seed_yolov8n` | Gốc 210 ảnh | `yolov8n`, 30 epoch, imgsz 640, batch 8, CPU | Epoch 29: P 82,7%, R 85,0%, mAP50 85,6%, mAP50–95 27,8%. Trên test (theo README): P 79,9%, R 80,9%, mAP50 78,0%, mAP50–95 24,9% |
| `shrimp_seed_real_v2` | `dataset_v2` | `yolov8n`, 50 epoch, batch 8, CPU | Epoch 50: P 84,1%, R 81,6%, mAP50 84,0%, mAP50–95 28,5% (không cải thiện rõ) |
| `shrimp_pl_custom_v1` | `dataset_pl_custom_v1` | `yolov8n`, 50 epoch, batch 4, GPU | Epoch 34: P 93,3%, R 89,6%, mAP50 94,4%, mAP50–95 43,7%. Không ổn định trên ảnh thực tế (xem mục 13.4) |
| `shrimp_domains_v1` | `dataset_v3` | `yolov8s`, 100 epoch, batch 4, GPU | Epoch 74: P 58,6%, R 60,0%, mAP50 58,0%, mAP50–95 19,1% (valid chỉ 5 ảnh loại C) |
| `shrimp_domains_v2` | `dataset_v4` | `yolov8s`, 100 epoch, batch 4, GPU | Epoch 50: P 92,3%, R 84,7%, mAP50 91,8%, mAP50–95 41,9% |
| **`shrimp_domains_v3`** | `dataset_v5` | `yolov8s`, 60 epoch, batch 4, GPU | Epoch 31: P 90,2%, R 90,6%, mAP50 94,9%, mAP50–95 41,6% (epoch 60: P 94,4%, R 91,6%, mAP50 95,8%, mAP50–95 40,3%) |
| `shrimp_seed_yolov8n2` | | | Lần train dở, không có trọng số |

**Lưu ý khi so sánh:** mAP của các run **không so sánh trực tiếp được** vì mỗi run dùng tập valid khác nhau. Lần đầu train `shrimp_domains_v1` với batch 8 đã **hết bộ nhớ GPU** nên phải chạy lại với batch 4. So sánh có ý nghĩa phải dùng số đếm trên **cùng tập ảnh giữ riêng** (mục 13).

### 10.5. Vì sao dùng YOLOv8 mà chưa dùng YOLO26
- Dataset Roboflow ban đầu xuất theo YOLOv8 và thư viện đang dùng là `ultralytics 8.3.184`. **YOLO26 cần `ultralytics` từ 8.4.0 trở lên** (đã kiểm tra: bản 8.3.253 chưa có cấu hình YOLO26; từ 8.4.0 mới có).
- Kết quả của dự án phụ thuộc chủ yếu vào **dữ liệu**: cùng kiến trúc `yolov8s`, đổi dữ liệu làm sai số loại A giảm từ 28% xuống 8 đến 12% và đưa loại C về 11% (mục 13.4). Vì vậy kiến trúc được **cố định** để đo tác động của dữ liệu; đổi cả hai cùng lúc thì không biết nguyên nhân của cải thiện.
- Lợi thế chính của YOLO26 là suy luận không NMS và nhanh trên thiết bị biên; dịch vụ này chạy trên máy chủ nên tốc độ chưa phải nút thắt. Các tuyên bố về cải thiện cho vật thể nhỏ là của nhà phát hành, chưa có số liệu độc lập, và **chưa được thử trên dữ liệu của dự án**.
- Ở chế độ không NMS của YOLO26, `max_det` mặc định là 300 và tham số `iou` không còn là bước NMS thông thường; `end2end=False` chuyển về đầu có NMS.
- Hướng mở rộng cụ thể: cài `ultralytics` 8.4.0 trở lên trong môi trường riêng, train `yolo26s` trên cùng `dataset_v5`, so sai số đếm trên cùng tập test. Biến thể P2 (đầu độ phân giải cao cho vật thể nhỏ) cũng có sẵn trong bản hiện tại (`yolov8-p2.yaml`) và có thể thử ngay.

---

## 11. Cách huấn luyện từng bước

**Công cụ:** Python 3.12 và Ultralytics (huấn luyện, suy luận), PyTorch 2.7.1 với CUDA 11.8 (GPU), Pillow và NumPy (xử lý ảnh), trình duyệt (`annotate.html` để gán nhãn), PowerShell (chạy lệnh).

**Bước 0. Môi trường**
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # True nếu dùng được GPU
```
Nếu PowerShell chặn kích hoạt: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

**Bước 1. Thu ảnh** theo tiêu chí ở mục 2.3. Đặt ảnh chưa có nhãn vào `to_label/<loại>/images/` và ghi nguồn, giấy phép, số đếm tay vào `to_label/SOURCES.json`.

**Bước 2. Gán nhãn** (có thể lấy nhãn nháp của mô hình làm điểm xuất phát)
```powershell
python dataset_expansion/draft_pending_labels.py      # sinh nhãn NHÁP (không phải ground truth);
                                                      # script hiện cố định thư mục dataset_v2/pending,
                                                      # với ảnh mới cần sửa đường dẫn trong script
# rồi mở dataset_expansion/annotate.html bằng trình duyệt:
#   chọn ảnh, nạp nhãn nháp, chuột phải để xóa khung sai, kéo chuột vẽ khung thiếu, xuất file .txt
```
Gán theo quy ước vòng đầu (mục 2.4). Đặt nhãn đã duyệt vào thư mục `labels/` tương ứng.

**Bước 3. Khai báo nguồn theo loại** trong `dataset_expansion/domains_spec.json`: mỗi nguồn có `domain`, `images`, `labels`, tùy chọn `files`, `limit`, `split`. Mỗi loại có `tile` (A đang là 320).

**Bước 4. Dựng dataset**
```powershell
python dataset_expansion/build_domain_dataset.py --spec dataset_expansion/domains_spec.json `
    --out dataset_v5 --holdout 5 --valid 2 --target 150 --lenient --reviewed
```
Thư mục đích phải chưa có dữ liệu. Cờ `--reviewed` xác nhận bạn đã duyệt nhãn. Kết quả: `train/`, `valid/`, `test/`, `data.yaml`, `domain_manifest.json` (mỗi file thuộc loại nào, split nào, lặp bao nhiêu lần).

**Bước 5. Kiểm tra dataset**
```powershell
python dataset_expansion/train_reviewed.py --data dataset_v5/data.yaml --reviewed --check-only
```
Kiểm tra: `nc=1`, mỗi ảnh có nhãn, nhãn đúng 5 số, khung nằm trong ảnh, không trùng ảnh giữa các split, mỗi split có box.

**Bước 6. Huấn luyện**
```powershell
python dataset_expansion/train_reviewed.py --data dataset_v5/data.yaml --reviewed `
    --name shrimp_domains_v3 --epochs 60
```
Mặc định: `yolov8s.pt`, batch 4, GPU nếu có. Kết quả ở `runs/detect/shrimp_domains_v3/`: `weights/best.pt`, `weights/last.pt`, `results.csv`, đồ thị, ma trận nhầm lẫn, ảnh dự đoán mẫu. Tên run phải chưa tồn tại. Nếu GPU hết bộ nhớ, giảm `--batch`.

**Bước 7. Đánh giá số đếm theo loại**
```powershell
python evaluate_domains.py --dataset dataset_v5 `
    --model runs/detect/shrimp_seed_yolov8n/weights/best.pt `
    --model runs/detect/shrimp_domains_v3/weights/best.pt --tiles 0 320 --confs 0.25 0.15
```
Chạy trên ảnh test nguyên bản của từng loại, in MAE, sai số tương đối, bias và cấu hình tốt nhất cho từng loại. Loại không có ảnh giữ riêng sẽ được ghi chú "không đo được".

**Bước 8. Quyết định thay model** theo quy tắc ở mục 13.5, rồi thử với dịch vụ:
```powershell
$env:MODEL_PATH = "runs/detect/shrimp_domains_v3/weights/best.pt"
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```
Mở `http://localhost:8001/health` kiểm tra `model_path`, rồi thử `/docs`.

---

## 12. Cấu trúc dự án huấn luyện

```text
ảnh thô ─▶ to_label/<A|B>/images ─▶ [annotate.html: gán nhãn vòng đầu] ─▶ labels/*.txt
                                                    ▲
                     draft_pending_labels.py (nhãn NHÁP bằng model, phải duyệt tay)
                                                    │
dữ liệu có nhãn: train/ valid/ test/ (C), dataset_pl_custom_v1/pending (A, B), to_label/A_pl_dense (A)
        │
        ▼  dataset_expansion/domains_spec.json  (loại, kích thước ô, nguồn, ghim split)
build_domain_dataset.py ── chia theo ảnh nguồn ── cắt ô theo loại ── cân bằng bằng lặp file
        │
        ▼
dataset_vN/{train,valid,test}/{images,labels} + data.yaml + domain_manifest.json
        │
        ▼ train_reviewed.py  (validate ─▶ YOLO(yolov8s.pt).train(...))
runs/detect/<run>/weights/best.pt
        │
        ▼ evaluate_domains.py  (ảnh test nguyên bản, mỗi loại: MAE, sai số tương đối, bias)
quyết định thay model (mục 13.5) ─▶ MODEL_PATH ─▶ ai_service/main.py
```

**Cách các file nối với nhau**
- `ai_service/main.py` là trung tâm: `draft_pending_labels.py`, `make_tiles.py`, `build_domain_dataset.py`, `evaluate_counting.py`, `evaluate_domains.py` và cả hai file test đều import các hàm từ đó (`tile_origins`, `merge_detections`, `detect_tiled`...). Nhờ vậy lưới ô lúc train và lúc suy luận giống hệt nhau.
- `build_domain_dataset.py` dùng `read_rows` và `tile_labels` của `make_tiles.py`.
- `train_reviewed.py` cung cấp hàm `validate`, được `run_v2_training.py` dùng lại.

**Cấu trúc một dataset được dựng**
```text
dataset_v5/
├── data.yaml               (train/val/test, nc: 1, names: ['shrimp-seed'])
├── domain_manifest.json    (spec, tham số, tóm tắt từng loại, danh sách file kèm loại và split)
├── train/images, labels    (ô cắt và ảnh nguyên bản; tên bắt đầu bằng A__, B__, C__; bản lặp thêm __r1, __r2...)
├── valid/images, labels
└── test/images, labels     (ảnh nguyên bản, không cắt ô, dùng để đo số đếm)
```

---

## 13. Đánh giá chất lượng và tiêu chí đạt

### 13.1. Hai nhóm chỉ số
- **Phát hiện (Ultralytics):** Precision, Recall, mAP50, mAP50–95, tính trên tập valid trong lúc train. Dùng để chọn `best.pt`, chưa phải thước đo cuối.
- **Số đếm (quan trọng nhất cho sản phẩm):** MAE, sai số tương đối và bias, so số khung dự đoán với số box nhãn thật của từng ảnh test, tính **riêng từng loại**. mAP không đo trực tiếp sai số số đếm.

### 13.2. Quy trình đánh giá
1. Dùng ảnh test **nguyên bản** (không cắt ô) của mỗi loại, chưa từng tham gia train.
2. Số thật là số box của nhãn đã duyệt tay (hoặc số đếm tay).
3. Chạy mỗi cấu hình `(tile_size, conf)` và chọn cấu hình tốt nhất cho từng loại bằng `evaluate_domains.py`.
4. So model mới với model hiện tại **trên cùng tập ảnh**.
5. Nhìn bằng mắt ảnh vẽ khung để phát hiện lỗi mà con số che khuất: khung bắt chấm đen (mắt), khung đôi, khung quá to, con có nhìn thấy mà không có khung.

### 13.3. Ngưỡng đạt theo loại ảnh **[Đề xuất]**
Ngưỡng này do tài liệu đề ra, chờ xác nhận theo mức sai số mà nghiệp vụ chấp nhận (ví dụ sai số mật độ cho phép khi mua bán giống).

| Loại | Số con thật mỗi ảnh | Ngưỡng đạt (cấu hình đã chọn, tập test) | Kết quả hiện có (xem 13.4) |
|---|---|---|---|
| Thưa (loại B, C) | không quá khoảng 30 | MAE ≤ 1 con và abs(bias) ≤ 0,5 | C: sai số tương đối 11% (MAE 0,5), **đạt**; B: chưa đo được |
| Vừa | khoảng 30 đến 100 | Sai số tương đối ≤ 10% | Chưa có tập |
| Dày (loại A) | trên 100 | Sai số tương đối ≤ 15% | A: 12% trên 5 ảnh test (4 ảnh PL "dễ", 1 ảnh chậu thật); **chưa kết luận được** vì mẫu nhỏ |

Số lượng tối thiểu để một kết quả đáng tin: **5 ảnh test mỗi loại, trong đó có ảnh chậu thật** (không chỉ ảnh PL cắt sẵn); cùng một số thật phải được đếm tay, nhãn trên ảnh phải đầy đủ.

### 13.4. Kết quả đã đo **[Đã xác minh, do tôi chạy trong phiên làm việc]**

**Sai số tương đối theo loại, toàn ảnh, `conf=0.25`** (ảnh test của `dataset_v5` cho C, `dataset_v4/v5` cho A):

| Loại | Model cũ `yolov8n` | `shrimp_domains_v2` | **`shrimp_domains_v3`** |
|---|---:|---:|---:|
| A (5 ảnh test, trung bình 130 con) | 28% | **8%** | 12% |
| C (10 ảnh test, trung bình 5,7 con) | 19% (đã từng học các ảnh này) | 28% | **11%** |
| B | không có ảnh test | | |

**Đếm trên ảnh A chia theo từng ảnh** (toàn ảnh, `conf=0.25`; số trong ngoặc là số thật):
| Ảnh | Model cũ | v2 | v3 |
|---|---:|---:|---:|
| `real_007` (124) | 39 | 103 | 165 |
| 4 ảnh PL (109, 150, 146, 121) | 59, 148, 129, 103 | 122, 144, 145, 126 | 121, 162, 149, 129 |

**Nhận xét:**
- v3 khắc phục việc loại C bị lùi của v2 và giữ A tốt; vì vậy là **ứng viên tốt nhất**. v3 **không** tốt nhất ở loại A (v2: 8%, v3: 12%); chênh lệch chủ yếu đến từ ảnh `real_007`, nằm trong khoảng nhiễu của 5 ảnh.
- **Số thật của `real_007` (124) có thể chưa đầy đủ:** khi nhìn ảnh có khung, vẫn thấy nhiều đầu tôm sẫm không có khung. Cần đếm tay lại ảnh này, vì nếu số thật lớn hơn thì nhận xét "v3 đếm dư 33%" đảo thành "đếm thiếu".
- **Kết quả không ổn định giữa các lần train:** trên ảnh A 600×337 (chưa có số thật), số đếm là 135 (v2), 287 (v3), 47 (model cũ).
- **Cắt ô không giúp v3** (đếm dư), trong khi có ích với model cũ ở ảnh dày.
- Model `shrimp_pl_custom_v1` đo trên 8 ảnh thực tế có nhãn: sai số tương đối từ 48% đến 73% tùy cấu hình, đếm 0 con ở 2 ảnh dù nằm trong tập train. Nguyên nhân khả dĩ: lệch loại ảnh, ảnh của dự án chỉ chiếm khoảng 6% số box, và quy ước nhãn khác nhau.

**Tập valid gốc 60 ảnh (loại C, model `yolov8n` cũ):** MAE tốt nhất 0,77 con, bias +0,47 ở cấu hình mặc định; `conf=0.15` làm MAE tăng lên 3,18.

### 13.5. Quy tắc thay model đang chạy **[Đề xuất]**
Chỉ thay `MODEL_PATH` mặc định khi **đồng thời**:
1. Mỗi loại có ảnh test đạt ngưỡng ở mục 13.3, trên số ảnh tối thiểu (5 ảnh, có ảnh thật).
2. Không loại nào lùi quá 5 điểm phần trăm sai số tương đối so với model hiện tại trên cùng tập.
3. Có số đếm tay đáng tin cho ít nhất một ảnh chậu dày thật, và kết quả trên đó nằm trong ngưỡng.
4. Kết quả ổn định khi train lại (hai lần train cùng cấu hình không lệch quá 15% sai số tương đối trên ảnh dày).
5. Đo xong thời gian suy luận trên máy chạy dịch vụ (`yolov8s` chậm hơn `yolov8n`) và chấp nhận được.

**Hiện trạng:** chưa đủ điều kiện 3 và 4, nên `shrimp_domains_v3` chưa thay model mặc định.

### 13.6. Kiểm thử đơn vị
`python -m unittest discover tests -v` (lần chạy gần nhất ghi nhận 8 test qua; từ đó `test_tiled_counting.py` được thêm `AnnotationTests` và `TiledColorTests` **chưa chạy lại**):
- `test_cloudinary_predict.py`: 503 khi thiếu cấu hình, trường hợp thành công (mock Cloudinary).
- `test_tiled_counting.py`: `tile_origins`, `merge_detections`, vẽ ảnh kết quả, bảo toàn màu khi cắt ô.
Các test mock Cloudinary nên **chưa kiểm tra upload thật**. Các script mới (`build_domain_dataset.py`, `evaluate_domains.py`) mới được thử chạy tay, chưa có test tự động.

---

## 14. Lưu trữ

| Dữ liệu | Nơi lưu | Ghi chú |
|---|---|---|
| Ảnh gốc của mỗi lần nhận diện | Cloudinary `shrimp-detection/originals` | `public_id` do Cloudinary cấp, không ghi đè |
| Ảnh đã vẽ khung | Cloudinary `shrimp-detection/annotated` | PNG |
| Số đếm, tọa độ, lịch sử kiểm tra | **Không lưu trong repo này**; JSON trả về cho ChillShrimp, nơi lưu cùng `public_id` và URL **[Suy ra]** | Tài liệu `CLOUDINARY_DATA.md` khuyến nghị lưu có cấu trúc trong database của ChillShrimp |
| Trọng số mô hình | Hệ thống file: `runs/detect/<run>/weights/*.pt` | `.gitignore` bỏ qua `*.pt`, **trừ** `runs/detect/shrimp_seed_yolov8n/weights/best.pt`; các model mới (ví dụ `shrimp_domains_v3`) **không được git theo dõi**, cần sao lưu riêng |
| Dataset | Hệ thống file (`train/`, `valid/`, `test/`, `dataset_*`, `external_datasets/`, `to_label/`) | Phần lớn nằm trong git; ảnh chờ gán nhãn chưa rõ nguồn và giấy phép |
| Kết quả train | `runs/detect/<run>/` | Một số run đang được git theo dõi |
| Credentials | `.env` (đã bị Git bỏ qua) | Xem cảnh báo ở mục 18 |

Không có cơ sở dữ liệu, hàng đợi hay bộ nhớ đệm trong repo. Dịch vụ **không có trạng thái** ngoài việc giữ model trong bộ nhớ.

---

## 15. Công cụ và công nghệ

| Nhóm | Công cụ | Phiên bản | Vai trò |
|---|---|---|---|
| Ngôn ngữ | Python | 3.12.7 (máy phát triển); Docker mẫu dùng 3.11 | |
| Học sâu | Ultralytics YOLOv8 | 8.3.184 | Huấn luyện, suy luận, vẽ khung, metric |
| | PyTorch | 2.7.1+cu118 (máy phát triển) | Thực thi mô hình, CUDA |
| API | FastAPI / uvicorn / python-multipart | 0.116.1 / 0.35.0 / 0.0.20 | HTTP, upload |
| Ảnh | Pillow, NumPy | 11.3.0 / theo Ultralytics | Đọc ảnh, mảng, PNG |
| Lưu ảnh | Cloudinary SDK | 1.46.2 | Upload ảnh gốc và ảnh vẽ |
| Cấu hình | python-dotenv | 1.2.3 | Đọc `.env` |
| Dữ liệu | Roboflow (nguồn dataset), `annotate.html` (tự viết) | | Gán nhãn |
| Kiểm thử | `unittest`, `unittest.mock` | thư viện chuẩn | |
| Đóng gói | Docker (hướng dẫn và mẫu Dockerfile trong `AI_SERVICE_integration.zip`) | `python:3.11-slim`, cổng 8001 | |
| Môi trường | Git, GitHub, PowerShell, venv | | |

**Phần cứng phát triển:** Windows 11, CPU Intel i5-11400H, RAM 8 GB (thường gần đầy), GPU NVIDIA GeForce RTX 3050 Laptop 4 GB (batch 8 với ảnh dày hết bộ nhớ, nên mặc định batch 4).

---

## 16. Vận hành và triển khai

### 16.1. Chạy cục bộ
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # rồi điền 3 biến CLOUDINARY_*
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```
Kiểm tra: `http://localhost:8001/health`, giao diện thử `http://localhost:8001/docs`.

### 16.2. Gọi mẫu
```powershell
curl.exe -X POST "http://localhost:8001/predict?conf=0.25&imgsz=640&max_det=1000" -F "file=@anh.jpg"
```
Với model cũ (`yolov8n`) và ảnh dày có thể thử thêm `tile_size=320&tile_overlap=0.25`. Với `shrimp_domains_v3` không nên cắt ô.

### 16.3. Docker
`AI_SERVICE_integration.zip` chứa gói `AI_SERVICE/` (Dockerfile `python:3.11-slim`, `MODEL_PATH=/app/models/best.pt`, cổng 8001, `app/main.py`, `models/best.pt`). **Gói này là phiên bản cũ của `main.py`** (chưa có ảnh vẽ khung, tiling, `max_det`). Hướng dẫn chi tiết: `HUONG_DAN_DOCKER_AI_SERVICE.md` (trong cùng Docker Compose, backend gọi `http://ai-service:8001/predict`, không dùng `localhost`).

### 16.4. Thay model
Đặt `MODEL_PATH` tới file `.pt` mới rồi **khởi động lại dịch vụ** (model chỉ được nạp một lần, biến môi trường đặt trong PowerShell chỉ có hiệu lực trong cửa sổ đó). Chỉ thay mặc định sau khi đạt quy tắc ở mục 13.5.

---

## 17. Cấu trúc thư mục

```text
.
├── ai_service/              main.py, cloudinary_config.py        (dịch vụ AI)
├── dataset_expansion/       annotate.html, draft_pending_labels.py, make_tiles.py,
│                            build_domain_dataset.py, domains_spec.json,
│                            train_reviewed.py, run_v2_training.py, LABELING_GUIDE.md
├── to_label/                ảnh chờ gán nhãn: A/, B/, A_pl_dense/, SOURCES.json
├── tests/                   test_cloudinary_predict.py, test_tiled_counting.py
├── docs/                    tài liệu và sơ đồ (file này)
├── train/ valid/ test/      dataset gốc + data.yaml              (loại C; tên do Roboflow đặt)
├── dataset_v2/              bản sao dataset gốc + ảnh thực tế chờ duyệt
├── dataset_v3/ v4/ v5/      dataset theo loại A, B, C do build_domain_dataset.py dựng
├── external_datasets/       PL Detection (Teramina, MIT)
├── dataset_pl_custom_v1/    PL Detection + 8 ảnh tự gán nhãn
├── inspection/, recovery/   ảnh kiểm tra nhãn, bản sao lưu sự cố
├── runs/detect/             kết quả huấn luyện (shrimp_seed_yolov8n, ..., shrimp_domains_v1..v3)
├── train_yolo.py, evaluate_counting.py, evaluate_domains.py, requirements.txt, README.md
└── yolov8n.pt, yolov8s.pt, yolo11n.pt   (trọng số tiền huấn luyện; chưa thấy mã nào dùng yolo11n)
```

---

## 18. Bảo mật và rủi ro vận hành

| Mức | Rủi ro | Chi tiết | Hành động |
|---|---|---|---|
| **Cao** | **Lộ credentials Cloudinary** | Tại thời điểm kiểm tra (2026-11-02), `.env.example` (file được git theo dõi) chứa cloud name, API key và API secret có dạng thật. File nằm trong commit `f7ccb7a` và nhánh `feature/cloudinary` đã đồng bộ với GitHub. Nếu là khóa thật thì khóa đã nằm trong lịch sử git của remote | Đổi (rotate) API secret trên Cloudinary ngay; thay bằng giá trị mẫu; cân nhắc làm sạch lịch sử git. Nếu repo công khai thì coi khóa đã bị lộ |
| Trung bình | Không xác thực | `/predict` không có xác thực hay giới hạn tốc độ, ai gọi được cổng 8001 đều tải ảnh lên tài khoản Cloudinary của bạn | Chỉ mở cổng nội bộ giữa backend và AI Service; thêm khóa API nếu cần |
| Trung bình | Không giới hạn kích thước upload | Ảnh rất lớn kết hợp tiling có thể tốn RAM (máy dev chỉ 8 GB) | Thêm giới hạn dung lượng và kích thước ảnh |
| Trung bình | Suy luận chặn vòng lặp sự kiện | Hàm `async` gọi model đồng bộ; một request nặng làm các request khác chờ | Chạy suy luận trong thread pool hoặc dùng nhiều worker |
| Trung bình | Mất kết quả khi upload lỗi | Upload ảnh gốc lỗi trả 502 và không trả số đếm đã tính | Cân nhắc trả số đếm kèm cờ `image_saved=false` |
| Trung bình | Đếm bị cắt âm thầm | `max_det` mặc định 300 làm ảnh dày bị đếm thiếu mà không có cảnh báo | Đặt mặc định 1000 và thêm cờ `truncated` |
| Thấp | Chưa có CORS | Frontend gọi trực tiếp từ trình duyệt sẽ bị chặn; kiến trúc hiện tại gọi qua backend nên chưa ảnh hưởng | |
| Thấp | `Pillow` giải nén ảnh | Chưa đặt giới hạn `MAX_IMAGE_PIXELS` riêng | |

**Giấy phép dữ liệu:** dataset gốc theo CC BY 4.0 (cần ghi nguồn), PL Detection theo MIT. Ảnh tìm từ internet cần kiểm tra giấy phép từng nguồn; ảnh có chữ chìm bản quyền không dùng. Nếu phát hành mô hình hoặc báo cáo, cần ghi nhận nguồn.

---

## 19. Hạn chế đã biết

1. **Chưa có số đếm tay đáng tin cho ảnh chậu dày thật.** Số thật của `real_007` (124) có dấu hiệu chưa đầy đủ; ảnh A 600×337 và ảnh B 1050×700 chưa có số đếm. Vì vậy chưa kết luận chắc được về độ chính xác trên loại A và B.
2. **Loại B gần như chưa có dữ liệu:** chỉ có 1 ảnh có nhãn (`real_003`, lặp 8 lần khi train, và nhãn bao cả thân), nên không có ảnh test cho B.
3. **Loại A:** mới 3 ảnh chậu thật có nhãn; 13 ảnh PL dày là "A nhẹ" và nhãn của PL không đầy đủ (vẫn có con nhìn thấy mà chưa có khung).
4. **Quy ước nhãn chưa thống nhất hoàn toàn:** đã chọn quy ước vòng đầu, nhưng nhãn `real_003` vẫn bao cả thân và hướng dẫn `LABELING_GUIDE.md` cũ yêu cầu bao cả thân; cần gán lại và cập nhật hướng dẫn.
5. **Kết quả không ổn định giữa các lần train** trên ảnh dày thật (ảnh A 600×337: 47, 135, 287 con tùy model).
6. **Test nhỏ:** 5 ảnh test cho A, 10 ảnh cho C; một ảnh thay đổi sai số trung bình rất nhiều.
7. **Rò rỉ giữa các split trong dataset gốc:** ảnh cùng nguồn (hậu tố `.rf.<hash>`) có thể ở nhiều split; `build_domain_dataset.py` đã tránh việc này cho các dataset mới (chia theo ảnh nguồn) nhưng không sửa dataset gốc.
8. `yolov8s` chậm hơn `yolov8n` nhưng **chưa đo thời gian suy luận**.
9. API chưa cho đổi `iou`, `max_det` mặc định còn 300 và chưa có cờ `truncated`.
10. Upload Cloudinary và tích hợp frontend chưa được kiểm tra thật (test chỉ mock). Chưa có test tự động cho script dựng dataset và đánh giá theo loại.
11. Video chưa được hỗ trợ.

---

## 20. Lộ trình tiếp theo

Chi tiết: [NEXT_STEPS.md](NEXT_STEPS.md). Thứ tự ưu tiên:
1. **Xử lý credentials bị lộ** (mục 18).
2. **Đếm tay** ảnh `real_007`, ảnh A 600×337 và ảnh B 1050×700; rà lại nhãn `real_007`.
3. Gán nhãn theo quy ước vòng đầu: `real_003`, `to_label/A`, `to_label/B`; thu thêm ảnh B thưa và ảnh A chậu đầy (mục tiêu 25 ảnh nguồn mỗi loại, mục 2.3).
4. Dựng lại dataset (đưa thêm ảnh có nhãn vào `domains_spec.json`) và train lại; đo bằng `evaluate_domains.py`.
5. Thêm `iou` và cờ `truncated` vào `/predict`, đổi `max_det` mặc định lên 1000, thêm chế độ video (lấy mẫu khung hình, lấy trung vị).
6. Chỉ thay `MODEL_PATH` mặc định khi đạt quy tắc ở mục 13.5; thử YOLO26 hoặc YOLOv8-P2 trên cùng dữ liệu nếu cần.

---

## 21. Phụ lục

### 21.1. Thuật ngữ
| Thuật ngữ | Giải thích |
|---|---|
| Bounding box | Khung chữ nhật bao quanh một vật thể |
| Confidence | Điểm model tự gán cho một phát hiện, từ 0 đến 1 |
| NMS | Non-Maximum Suppression: giữ khung điểm cao nhất, bỏ khung chồng lấn |
| IoU | Giao chia hợp của hai khung, đo mức trùng nhau |
| mAP50 / mAP50–95 | Chất lượng phát hiện ở ngưỡng IoU 0,5 / trung bình các ngưỡng 0,5 đến 0,95 |
| MAE | Sai số tuyệt đối trung bình của số đếm mỗi ảnh |
| Sai số tương đối | Sai số tuyệt đối chia cho số thật, trung bình theo ảnh |
| Bias | Sai số có dấu trung bình: dương là đếm dư, âm là đếm thiếu |
| Tiling | Cắt ảnh lớn thành các ô nhỏ chồng lấn để phát hiện vật thể nhỏ |
| Domain, loại ảnh | Nhóm ảnh cùng kiểu (A, B, C) dùng để chia dữ liệu và đo riêng |
| Ground truth | Nhãn đúng do người duyệt; nhãn do model đề xuất không phải ground truth |
| Ảnh nguồn | Một ảnh gốc; mọi bản tăng cường và ô cắt của nó thuộc cùng một split |
| Vòng đầu | Quy ước nhãn: khung bao vùng đầu và thân sẫm của con tôm |

### 21.2. Tài liệu liên quan
- [README.md](../README.md): tổng quan nhanh, cài đặt, chạy.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): trạng thái chi tiết và giải thích từ ảnh đến JSON.
- [CLOUDINARY_DATA.md](CLOUDINARY_DATA.md), [ANNOTATED_IMAGE_INTEGRATION.md](ANNOTATED_IMAGE_INTEGRATION.md): lưu trữ Cloudinary và ảnh vẽ khung.
- [HUONG_DAN_TICH_HOP_AI_SERVICE.md](HUONG_DAN_TICH_HOP_AI_SERVICE.md), [HUONG_DAN_DOCKER_AI_SERVICE.md](HUONG_DAN_DOCKER_AI_SERVICE.md): tích hợp và Docker.
- [../dataset_expansion/LABELING_GUIDE.md](../dataset_expansion/LABELING_GUIDE.md): quy tắc gán nhãn.
