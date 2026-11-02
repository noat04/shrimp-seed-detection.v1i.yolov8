# Thuyết trình: Huấn luyện mô hình đếm tôm giống từ đầu đến cuối

Tài liệu này là **bản ghi chú để sinh viên trình bày trước hội đồng**: dàn ý theo từng trang slide, lời nói mẫu, số liệu cần nhớ và phần trả lời câu hỏi. Số liệu lấy từ [TECHSPEC.md](TECHSPEC.md) (mục 10 và 13). Thời lượng gợi ý: 6 đến 8 phút trình bày và 3 đến 5 phút hỏi đáp.

## Câu tóm tắt (nói ngay ở đầu, khoảng 20 giây)
> "Em dạy máy tính nhìn ảnh tôm giống và đếm số con. Em cho máy xem nhiều ảnh đã được đánh dấu từng con để nó tự học cách nhận ra tôm. Sau đó em kiểm tra bằng những ảnh máy chưa từng thấy, rồi so số máy đếm với số em đếm tay."

**Cách nghĩ để giải thích cho người không chuyên:** dạy máy giống dạy một học sinh nhận biết tôm bằng ảnh có đáp án, rồi cho thi bằng đề chưa từng học.

---

## Dàn ý theo từng trang slide

### Slide 1. Bài toán và mục tiêu (khoảng 45 giây)
**Trên slide:** tên đề tài; một ảnh chậu tôm giống dày; dòng chữ "Đếm tôm giống bằng ảnh".

**Nói:**
- Đếm tôm giống bằng tay tốn công và dễ sai, nhất là khi vài trăm con trong một chậu.
- Mục tiêu: máy đếm số con trong ảnh để người dùng chỉ cần hiệu chỉnh nhỏ.
- Số đếm quyết định cả mật độ (con/ml), nên đếm sai sẽ kéo theo sai mật độ.

### Slide 2. Máy nhìn và đếm như thế nào (khoảng 1 phút)
**Trên slide:** hình [train_explained.png](train_explained.png) hoặc ảnh tôm có khung xanh.

**Nói:**
- Em dùng mô hình **YOLOv8**: nhìn ảnh một lần và vẽ **một khung quanh mỗi con tôm**.
- **Số đếm là số khung.**
- Mỗi khung có một điểm tin cậy; em chỉ giữ khung từ 0,25 trở lên và loại các khung chồng lên nhau.

### Slide 3. Dữ liệu: "đề học" của máy (khoảng 1 phút)
**Trên slide:** ba ảnh mẫu của ba loại A, B, C kèm một dòng mô tả mỗi loại.

| Loại | Đặc điểm |
|---|---|
| A | Tôm nhỏ, thân tối, **rất dày** (hàng trăm con, chồng dính) |
| B | Tôm to hơn, thân trong suốt, đốm cam, **thưa** |
| C | Ảnh cận cảnh 640×640, tôm nhỏ, thưa |

**Nói:**
- Mỗi con tôm được em hoặc nguồn dữ liệu vẽ một khung quanh **vùng đầu và thân sẫm** (quy ước "vòng đầu").
- Nguồn dữ liệu: bộ ảnh gốc từ Roboflow (300 ảnh, giấy phép CC BY 4.0), bộ PL Detection (giấy phép MIT) và ảnh em tự gán nhãn.
- Em chia ba loại vì chúng rất khác nhau: mô hình học từ một loại thì đếm sai nặng ở loại khác. Ví dụ ảnh tôm trong suốt bị nhận nhầm hai chấm mắt đen thành vật thể.

### Slide 4. Cách em huấn luyện (khoảng 1 phút 30 giây)
**Trên slide:** hình [train_pipeline.png](train_pipeline.png) và ba gạch đầu dòng: học chuyển giao, tăng cường dữ liệu, GPU.

**Nói:**
- **Học chuyển giao:** em dùng mô hình `yolov8s` đã học sẵn cách nhận vật thể nói chung (bộ COCO), rồi dạy thêm để nhận tôm. Cách này cần ít ảnh hơn nhiều so với học từ đầu.
- Mỗi vòng học (epoch) máy đoán, so với khung em vẽ, tính độ sai rồi tự chỉnh để lần sau đúng hơn. Em lặp 60 đến 100 vòng.
- **Tăng cường dữ liệu** để máy không học thuộc: lật ảnh, đổi độ sáng, ghép ảnh. Em tắt xoay ảnh vì xoay làm khung bị lỏng với con tôm nằm chéo.
- Phần cứng: GPU RTX 3050 4 GB, `batch=4` (batch 8 từng bị tràn bộ nhớ).
- Sau mỗi vòng, máy được kiểm tra trên tập valid (không học từ tập này) để chọn ra phiên bản tốt nhất.

### Slide 5. Cách chia dữ liệu để kiểm tra công bằng (khoảng 1 phút)
**Trên slide:** sơ đồ ba tập train, valid, test và một dòng "chia theo ảnh nguồn".

**Nói:**
- **Train** để học, **valid** để chọn phiên bản, **test** là bài thi cuối, ảnh nguyên bản chưa từng học.
- Em chia **theo ảnh nguồn**: các bản tăng cường và các mảnh cắt từ cùng một ảnh luôn nằm cùng một tập, để không "lộ đề".
- Mỗi loại em giữ riêng 5 ảnh test và 2 ảnh valid.
- Loại nào ít ảnh, em lặp lại ảnh đó khi học để cân bằng với loại nhiều ảnh.

### Slide 6. Cách đánh giá "đạt" (khoảng 1 phút)
**Trên slide:** bảng ngưỡng đạt.

| Nhóm ảnh | Ngưỡng đạt (đề xuất) |
|---|---|
| Thưa (không quá khoảng 30 con) | Sai trung bình không quá khoảng 1 con |
| Dày (trên 100 con) | Sai số tương đối không quá khoảng 15% |

**Nói:**
- Em không chỉ nhìn chỉ số phát hiện, mà so **số máy đếm với số thật** trên ảnh test, tính riêng từng loại.
- Hai chỉ số: sai số trung bình (MAE) và sai số tương đối (sai bao nhiêu phần trăm so với số thật).
- Các ngưỡng này em đề xuất và có thể điều chỉnh theo mức sai số mà thực tế chấp nhận.

### Slide 7. Kết quả (khoảng 1 phút 30 giây)
**Trên slide:** bảng kết quả và ảnh trước/sau.

| Loại ảnh | Mô hình đầu tiên (sai số tương đối) | Phiên bản cuối `shrimp_domains_v3` |
|---|---:|---:|
| C (10 ảnh test) | 19% | **11%** |
| A (5 ảnh test) | 28% | **12%** |

**Nói:**
- Mô hình đầu tiên đếm thiếu nặng ở ảnh dày: ảnh vài trăm con chỉ đếm được 30 con.
- Khi em thêm dữ liệu ảnh dày và chia theo loại, sai số giảm rõ rệt. Ví dụ ảnh chậu `real_007` (nhãn ghi 124 con): mô hình đầu đếm 39, các bản sau đếm 103 đến 165.
- Em nói rõ đây là tập test nhỏ: 5 ảnh loại A (trong đó 4 ảnh dễ) và 10 ảnh loại C.

### Slide 8. Triển khai (khoảng 30 giây)
**Trên slide:** sơ đồ ChillShrimp, AI Service và Cloudinary.

**Nói:**
- Mô hình chạy thành một dịch vụ web (FastAPI). Hệ thống ChillShrimp gửi ảnh vào, nhận lại số đếm, các khung và đường dẫn ảnh đã đánh dấu. Ảnh gốc và ảnh có khung lưu trên Cloudinary.

### Slide 9. Hạn chế và hướng phát triển (khoảng 1 phút, nên nói chủ động)
**Trên slide:** hai cột "Hạn chế" và "Hướng phát triển".

| Hạn chế | Hướng phát triển |
|---|---|
| Loại B rất ít dữ liệu (1 ảnh có nhãn), chưa đo được | Thu thêm ảnh A và B, mục tiêu 25 ảnh nguồn mỗi loại |
| Test nhỏ, thiếu số đếm tay cho ảnh chậu dày thật | Đếm tay thêm ảnh để có tập kiểm tra độc lập |
| Kết quả trên ảnh dày dao động giữa các lần train, nên chưa thay mô hình đang chạy | So sánh với YOLO26 hoặc YOLOv8-P2 trên cùng dữ liệu |
| Chưa xử lý video | Lấy mẫu khung hình rồi lấy trung vị số đếm |

### Slide 10. Kết luận (khoảng 20 giây)
**Nói:** "Điểm em rút ra là kết quả phụ thuộc chủ yếu vào dữ liệu và cách chia loại ảnh. Em đã có quy trình huấn luyện và đánh giá hoàn chỉnh, và sẽ tiếp tục bổ sung dữ liệu để cải thiện."

---

## Số liệu cần nhớ
- Dữ liệu gốc: 300 ảnh (210 train, 60 valid, 30 test), 2.317 khung; PL Detection: 748 ảnh.
- Dataset cuối `dataset_v5`: train 401 ảnh (9.590 khung), valid 21 ảnh, test 15 ảnh.
- Mô hình: `yolov8s` (khoảng 11 triệu tham số), `batch=4`, 60 đến 100 epoch, GPU RTX 3050 4 GB.
- Chỉ số phát hiện của v3 (tập valid): Precision 90,2%, Recall 90,6%, mAP50 94,9%.
- Sai số đếm: C 11%, A 12%; mô hình đầu tiên C 19%, A 28%.
- Ảnh chậu `real_007`: nhãn 124; đếm của mô hình đầu 39, v2 103, v3 165.

## Hỏi đáp dự kiến

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao ít ảnh (vài trăm) vẫn huấn luyện được? | Em dùng học chuyển giao từ mô hình đã học sẵn, chỉ dạy thêm phần nhận tôm nên cần ít ảnh hơn. Vì ít ảnh nên em kiểm tra kỹ bằng tập test riêng |
| Làm sao biết máy không học thuộc? | Tập test là ảnh nguyên bản chưa từng học và được chia theo ảnh nguồn nên không rò rỉ. Em so số đếm trên tập đó chứ không chỉ nhìn kết quả lúc học |
| Vì sao không dùng YOLO26 mới hơn? | Dataset và thư viện của em theo YOLOv8; YOLO26 cần nâng cấp thư viện từ bản 8.4.0. Quan trọng hơn, kết quả phụ thuộc chủ yếu vào dữ liệu: cùng kiến trúc, đổi dữ liệu làm sai số giảm mạnh. Em chưa thử YOLO26, đó là hướng phát triển |
| Độ chính xác bao nhiêu phần trăm? | Em báo sai số theo từng loại ảnh, không chỉ một con số: khoảng 11% ở ảnh cận cảnh và khoảng 12% ở ảnh dày, trên tập test nhỏ |
| Tại sao chia ba loại ảnh? | Vì ba loại rất khác về màu, kích thước, mật độ; một loại tốt chưa chắc tốt ở loại khác |
| Vì sao mô hình đếm thiếu ở ảnh dày? | Ban đầu dữ liệu chỉ có khoảng 8 con mỗi ảnh nên mô hình chưa từng thấy ảnh dày. Sau khi thêm ảnh dày, đếm tốt hơn nhiều |
| Tôm chồng lên nhau thì sao? | Con bị che kín thì không đếm được; đó là giới hạn của cách đếm bằng khung. Hướng khác là ước lượng mật độ |
| Dữ liệu lấy ở đâu, có vi phạm bản quyền không? | Bộ gốc CC BY 4.0 và PL Detection MIT, em có ghi nguồn; ảnh từ internet em chỉ dùng sau khi kiểm tra giấy phép |
| Tại sao chọn `yolov8s` mà không `yolov8n`? | `yolov8n` là bản em dùng đầu tiên; `yolov8s` lớn hơn (khoảng 11 triệu so với 3 triệu tham số) nên chứa được nhiều loại ảnh hơn, nhưng chạy chậm hơn trên CPU |
| Nếu mô hình đếm sai thì sao? | Hệ thống cho người dùng hiệu chỉnh số đếm, và em nêu rõ mức sai số của từng loại ảnh |

## Lưu ý khi trình bày
- **Đừng nói "chính xác 90%".** Hãy nói sai số theo từng loại và phạm vi test.
- **Đừng nói YOLOv8 tốt hơn YOLO26.** Em chưa so sánh.
- **Nói thẳng điểm yếu** (loại B, test nhỏ). Hội đồng thường đánh giá cao sự trung thực.
- **Số thật của `real_007` (124) có thể chưa đầy đủ**, vì ảnh còn nhiều con chưa có khung. Nếu bị hỏi, trả lời rằng em đang đếm tay lại.
- Chuẩn bị sẵn: [train_pipeline.png](train_pipeline.png), [train_explained.png](train_explained.png), và ảnh trước/sau (cùng một ảnh với mô hình đầu và v3).
- Trước buổi trình bày, nên chạy lại một lần `evaluate_domains.py` để chắc các số liệu trên slide khớp với máy của bạn.
