# Bien dataset hien tai thanh AI Service

Workspace hien tai dang la dataset YOLOv8. De dua vao du an, ta boc dataset/model nay thanh mot AI microservice rieng.

Luong xu ly mong muon:

```text
ReactJS upload anh
  -> Backend chinh
  -> AI Service /predict
  -> YOLOv8 tra bounding boxes + count
  -> Backend luu ket qua / tra ve frontend
```

## 1. Cai dat

Can cai Python 3.10 hoac 3.11 truoc. Khi cai tren Windows, hay tick `Add python.exe to PATH`.
Sau do dong terminal, mo lai terminal tai thu muc nay:

```powershell
python -m pip install -r requirements.txt
```

## 2. Train model

Chay:

```powershell
python train_yolo.py
```

Sau khi train xong, file model se nam o:

```text
runs/detect/shrimp_seed_yolov8n/weights/best.pt
```

Day la file can dua vao AI Service.

## 3. Chay AI Service

```powershell
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```

Kiem tra service:

```powershell
curl http://localhost:8001/health
```

Neu `model_exists` la `true`, service da thay file `best.pt`.

## 4. Test predict bang anh

Dung mot anh trong tap test:

```powershell
curl -X POST "http://localhost:8001/predict?conf=0.25" -F "file=@test/images/ten_file_anh.jpg"
```

Neu dung PowerShell va gap loi voi `curl`, hay dung `curl.exe`:

```powershell
curl.exe -X POST "http://localhost:8001/predict?conf=0.25" -F "file=@test/images/ten_file_anh.jpg"
```

Response mong muon:

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

## 5. Backend chinh goi AI Service

Backend chinh chi can gui multipart file den:

```text
POST http://localhost:8001/predict
```

Sau do lay:

- `count`: so luong tom dem duoc
- `detections`: danh sach bounding boxes
- `confidence`: do tin cay tung box
- `box`: toa do de frontend ve khung len anh

## 6. Do kich thuoc

Service hien tai moi tra bounding boxes theo pixel. De do kich thuoc thuc te, can them he so quy doi:

```text
real_length_mm = box_width_pixel * mm_per_pixel
```

`mm_per_pixel` nen lay tu:

- vat chuan trong anh, vi du thuoc do hoac grid
- camera co khoang cach chup co dinh
- calibration rieng cho tung thiet bi

Neu chua co calibration, chi nen bao cao kich thuoc theo pixel, chua nen ket luan mm/cm.

## 7. Muc do dap ung sau khi them service

Sau khi train duoc `best.pt` va chay duoc `/predict`, workspace se dap ung phan AI microservice muc demo:

- Co dataset YOLOv8
- Co script train model
- Co API nhan anh
- Co API tra count va bounding boxes

Van can bo sung them trong du an tong:

- Backend nghiep vu
- Frontend upload va hien thi ket qua
- Database luu lich su phan tich
- Calibration de do kich thuoc that
- Dataset cua giong neu can nhan dien ca tom va cua
