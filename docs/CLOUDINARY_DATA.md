# Dữ liệu có thể lưu trên Cloudinary

Trong dự án nhận diện tôm giống, Cloudinary có thể lưu các ảnh liên quan đến mỗi lần nhận diện.

## Ảnh gốc

Đây là ảnh khách tải lên API để nhận diện. Ảnh được giữ nguyên nội dung ban đầu và là loại dữ liệu sẽ được lưu mặc định để backup.

Luồng dự kiến: API kiểm tra ảnh, chạy YOLO dự đoán thành công, sau đó tải ảnh gốc lên Cloudinary. Cách này tránh lưu ảnh nếu quá trình dự đoán thất bại.

## Ảnh kết quả có bounding box

Đây là bản sao của ảnh gốc được vẽ bounding box, nhãn và confidence do YOLO dự đoán. Có thể bổ sung việc tạo và lưu loại ảnh này sau; API hiện chưa tạo hoặc tải ảnh kết quả lên Cloudinary.

## Dữ liệu nhận diện

Các thông tin như số lượng tôm, confidence, tọa độ bounding box, thời gian nhận diện và tên file hiện là dữ liệu JSON do API trả về. Chúng không phải ảnh và không được luồng hiện tại tự lưu trên Cloudinary.

Có thể gắn một số metadata với ảnh trên Cloudinary, nhưng nếu cần tìm kiếm, lọc hoặc lưu lịch sử nhận diện có cấu trúc, nên lưu các trường này trong database cùng với `public_id` và URL của ảnh Cloudinary.

## Trạng thái hiện tại

Endpoint `/predict` được cấu hình để lưu ảnh gốc vào Cloudinary sau khi YOLO dự đoán thành công. Cần đặt đủ ba biến `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY` và `CLOUDINARY_API_SECRET`; nếu thiếu cấu hình, endpoint trả HTTP 503. Nếu upload thất bại, endpoint trả HTTP 502 và báo rằng ảnh chưa được backup thành công.

## Cấu trúc thư mục Cloudinary đề xuất

Cloudinary dùng `folder` để tổ chức ảnh. Ban đầu, dự án chỉ cần lưu ảnh gốc:

```text
shrimp-detection/
└── originals/
    ├── ảnh-1
    ├── ảnh-2
    └── ...
```

Nếu sau này cần lưu thêm ảnh đã vẽ bounding box, có thể mở rộng thành:

```text
shrimp-detection/
├── originals/
└── annotated/
```

Ví dụ upload ảnh gốc bằng Cloudinary Python SDK:

```python
cloudinary.uploader.upload(
    image_stream,
    folder="shrimp-detection/originals",
    resource_type="image",
)
```

Mỗi lần upload để Cloudinary cấp `public_id` riêng nhằm tránh ghi đè ảnh cũ. Response từ `/predict` trả `public_id` và `image_url` (`secure_url`) để client lưu cùng kết quả nhận diện. Nếu cần phân loại thêm theo ngày, có thể tổ chức theo dạng `shrimp-detection/originals/2026/09/28`.

Folder trên Cloudinary là cách tổ chức asset theo đường dẫn logic, không phải thư mục filesystem trên máy chủ.

## Luồng upload trong API `/predict`

Khi đã cấu hình đủ credentials, API xử lý mỗi request theo thứ tự:

1. Đọc và kiểm tra ảnh tải lên.
2. Chạy YOLO để tạo danh sách detections.
3. Tải bytes của ảnh gốc lên `shrimp-detection/originals` bằng Cloudinary SDK trong thread pool.
4. Trả kết quả nhận diện cùng thông tin asset Cloudinary.

SDK upload là hàm đồng bộ; chạy nó trong thread pool giúp tránh chặn vòng xử lý bất đồng bộ của FastAPI.

Response thành công có thêm hai trường:

```json
{
  "image_saved": true,
  "image_url": "https://res.cloudinary.com/<cloud-name>/image/upload/...",
  "public_id": "shrimp-detection/originals/..."
}
```

`image_saved` xác nhận ảnh đã được lưu thành công. `image_url` là URL HTTPS (`secure_url`) của ảnh gốc, còn `public_id` là định danh asset để truy xuất hoặc quản lý ảnh sau này.

### Lỗi cấu hình hoặc upload

- HTTP 503: Cloudinary chưa được cấu hình đủ ba biến `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY` và `CLOUDINARY_API_SECRET`. Trường `detail.image_saved` là `false`.
- HTTP 502: YOLO đã dự đoán thành công nhưng upload ảnh gốc lên Cloudinary thất bại. Trường `detail.image_saved` là `false`, vì vậy client không nhầm kết quả này với một bản backup thành công.

Khi triển khai, cần cài dependencies trong `requirements.txt` và đặt credentials trong biến môi trường hoặc secrets của môi trường chạy service. Không commit file `.env` hay API secret lên Git.
