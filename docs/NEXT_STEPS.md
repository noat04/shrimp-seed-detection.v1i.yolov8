# Công việc cần làm tiếp theo

Cập nhật: 2026-10-26. Mục tiêu: đếm đúng hơn ở ảnh có mật độ tôm dày (ví dụ ảnh 953×1006, mặc định đếm được 30 con trong khi thực tế vài trăm con).

## Nguyên nhân gốc
Model đang chạy (`runs/detect/shrimp_seed_yolov8n`, `yolov8n`, 30 epoch) chỉ được train trên 300 ảnh gốc, trung bình khoảng 7,8 box mỗi ảnh, chưa từng thấy ảnh dày. Tham số suy luận (tiling, `conf`) chỉ là cách tạm.

## Số liệu đã đo (để so sánh sau này)
- Tập valid (60 ảnh, 459 box thật): baseline `conf=0.25` có MAE 0,77 con/ảnh, tốt nhất. `tile=320, conf=0.15` có MAE 2,00 (đếm dư).
- Ảnh dày 953×1006 (chưa có số đếm tay): baseline 30; `tile=320, conf=0.25` 220; `tile=320, conf=0.15` 765; `tile=320, conf=0.10` 1384 (chắc chắn dư).

## Danh sách việc

### A. Làm ngay
- [ ] **Khóa Cloudinary trong `.env.example`**: file chứa cloud name, API key, secret có dạng thật và `.gitignore` đang cho phép commit (`!.env.example`).
  - [ ] Đổi (rotate) secret trên trang Cloudinary.
  - [ ] Thay giá trị trong `.env.example` bằng chuỗi mẫu.
  - [ ] Mở `AI_SERVICE_integration.zip` kiểm tra xem có chứa khóa không.
- [ ] **Commit công việc chưa lưu** (sau khi xử lý khóa): repo mới có 2 commit; còn nhiều file chưa theo dõi (`ai_service/`, `tests/`, `dataset_expansion/`, `dataset_v2/`, `docs/`, `evaluate_counting.py`...).
  - [ ] Cập nhật `.gitignore`: log, `*.cache`, `*.pid`, `yolov8s.pt`, run train phụ.
  - [ ] Chia thành các commit nhỏ theo từng việc.

### B. Cần bạn làm (nút thắt lớn nhất)
- [ ] **Duyệt nhãn ảnh thực tế** bằng `dataset_expansion/annotate.html`.
  - Mở ảnh ở `dataset_v2/pending/images/`, nạp nhãn nháp ở `dataset_v2/pending/draft_labels/`, sửa, rồi xuất `.txt`.
  - Đặt ảnh và nhãn đã duyệt vào `reviewed/images/` và `reviewed/labels/` (cùng tên file).
  - Quy tắc: `dataset_expansion/LABELING_GUIDE.md`. `real_002` và `real_004` quá nhỏ, nên loại hoặc tìm ảnh gốc tốt hơn.
  - Mục tiêu: 40 đến 50 ảnh dày (nhiều lô, nhiều góc, nhiều điều kiện sáng). Bắt đầu với 5 đến 10 ảnh.
- [ ] **Đếm tay 5 đến 10 ảnh dày làm tập kiểm tra độc lập** (không dùng để train), gồm ảnh 953×1006. Ghi số đếm vào một file để so sánh.

### C. Train và đánh giá (sau khi có nhãn)
- [ ] Chạy `python dataset_expansion/make_tiles.py --images reviewed/images --labels reviewed/labels --out dataset_v3 --tile-size 320 --overlap 0.2 --include-base dataset_v2 --reviewed`.
- [ ] Chạy `python dataset_expansion/train_reviewed.py --data dataset_v3/data.yaml --reviewed` (mặc định `yolov8s`, 100 epoch, GPU RTX 3050 4 GB, batch 8).
- [ ] Chạy `evaluate_counting.py` (và so sánh trên tập đếm tay) giữa model mới và model đang chạy.
- [ ] Chọn `conf`, `tile_size`, `tile_overlap`, `merge_iou` theo kết quả. Chỉ đổi `MODEL_PATH` hoặc thay `best.pt` khi model mới tốt hơn rõ rệt, rồi khởi động lại service (service chỉ nạp model một lần).

### D. Làm song song trong lúc chờ nhãn
- [ ] Thêm chế độ "dày" vào `/predict` (tham số `mode=dense` hoặc biến môi trường); giá trị cụ thể chốt sau mục C.
- [ ] Viết script kiểm tra nhãn đã duyệt (số box mỗi ảnh, box quá nhỏ hoặc lớn, box chồng lấn bất thường).
- [ ] Chạy lại `python -m unittest discover tests -v`; có test mới (`AnnotationTests`) chưa chạy lại.
- [ ] Thử `/predict` thật với Cloudinary thật; kiểm tra `image_url`, `annotated_image_url` và việc frontend ChillShrimp hiển thị ảnh có box.

### E. Về sau
- [ ] Sửa điểm yếu dataset gốc: một số ảnh cùng nguồn nằm ở nhiều split; ảnh bị resize kiểu Stretch (nên giữ tỷ lệ).
- [ ] Nếu tôm chồng chéo quá nhiều mà detection vẫn không đủ, cân nhắc ước lượng mật độ (CSRNet, P2PNet) với nhãn chấm điểm.
- [ ] Gộp các file `.md` đang chồng lấn nội dung.
- [ ] Giữ cột "Số đếm hiệu chỉnh" cho người dùng sửa khi AI đếm sai.

## Ghi chú môi trường
- Python 3.12.7 trong `.venv`; PyTorch 2.7.1+cu118, CUDA dùng được (RTX 3050 Laptop 4 GB). RAM máy 8 GB, thường gần đầy, nên đóng bớt Chrome, ChatGPT, VS Code khi train.
- Kích hoạt môi trường: `Set-ExecutionPolicy -Scope Process RemoteSigned` rồi `.\.venv\Scripts\Activate.ps1`.
- Sơ đồ: `docs/train_pipeline.png` (toàn quy trình), `docs/train_explained.png` (cách mô hình học).
- Hiện không có tiến trình nào của dự án chạy ngầm. `dataset_v2/training.pid` là file cũ.
