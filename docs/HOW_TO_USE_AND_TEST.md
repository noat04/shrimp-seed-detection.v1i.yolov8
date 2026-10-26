# Cach su dung va test dataset YOLOv8

Thu muc hien tai la dataset Roboflow da export theo dinh dang YOLOv8:

- `train/images`, `train/labels`: du lieu huan luyen
- `valid/images`, `valid/labels`: du lieu kiem tra trong luc train
- `test/images`, `test/labels`: du lieu test sau khi train
- `data.yaml`: cau hinh duong dan va ten class

## 1. Cai moi truong

Can cai Python 3.10 hoac 3.11 truoc. Khi cai tren Windows, hay tick `Add python.exe to PATH`.
Sau do dong terminal, mo lai terminal tai thu muc nay va chay:

```powershell
python -m pip install ultralytics opencv-python
```

Kiem tra YOLO da cai thanh cong:

```powershell
yolo version
```

## 2. Kiem tra dataset

Chay lenh sau de xem YOLO doc duoc `data.yaml` hay khong:

```powershell
yolo detect val model=yolov8n.pt data=data.yaml imgsz=640
```

Lenh nay dung model pretrained mac dinh de kiem tra pipeline doc dataset. Ket qua co the chua chinh xac vi model chua duoc train cho tom giong.

## 3. Train nhanh model

Train thu voi model nho de kiem tra luong chay:

```powershell
yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=640 batch=8 name=shrimp_seed_yolov8n
```

Sau khi train xong, model tot nhat nam o:

```text
runs/detect/shrimp_seed_yolov8n/weights/best.pt
```

## 4. Test model tren tap test

Danh gia model tren tap test:

```powershell
yolo detect val model=runs/detect/shrimp_seed_yolov8n/weights/best.pt data=data.yaml split=test imgsz=640
```

Can quan tam cac chi so:

- `precision`: du doan bounding box dung den dau
- `recall`: model bat duoc bao nhieu con tom trong anh
- `mAP50`: do chinh xac phat hien o nguong IoU 0.5
- `mAP50-95`: chi so nghiem ngat hon, dung de bao cao tot hon

## 5. Test tren mot anh bat ky

Chay detect tren anh trong tap test:

```powershell
yolo detect predict model=runs/detect/shrimp_seed_yolov8n/weights/best.pt source=test/images imgsz=640 conf=0.25 save=True
```

Anh ket qua co ve bounding box se nam trong:

```text
runs/detect/predict
```

## 6. Tich hop vao AI microservice

Sau khi co `best.pt`, AI service chi can:

1. Nhan anh tu backend.
2. Nap model `best.pt`.
3. Detect bounding boxes.
4. Tra JSON gom: class, confidence, toa do box, tong so tom dem duoc.

Vi du response mong muon:

```json
{
  "count": 12,
  "detections": [
    {
      "class": "shrimp-seed",
      "confidence": 0.87,
      "box": {
        "x1": 120,
        "y1": 80,
        "x2": 180,
        "y2": 145
      }
    }
  ]
}
```

## 7. Dieu kien de coi la test dat

Co the coi la dat muc demo khi:

- Train sinh ra duoc file `best.pt`
- Predict tren anh test co bounding box bao quanh tom
- So luong dem duoc gan dung voi anh mau
- Backend/API nhan anh va tra ve JSON bounding boxes
- Frontend hien thi lai anh kem khung va tong so tom
