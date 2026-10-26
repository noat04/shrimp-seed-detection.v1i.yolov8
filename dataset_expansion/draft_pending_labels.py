"""Generate UNREVIEWED proposals, never ground truth, for the 11 remaining images."""
from pathlib import Path
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ai_service.main import tile_origins, merge_detections, detection_to_dict


def main():
    pending=ROOT/"dataset_v2/pending"
    model=YOLO(str(ROOT/"runs/detect/shrimp_seed_yolov8n/weights/best.pt"))
    report=[]
    for path in sorted((pending/"images").glob("*.jpg")):
        if path.stem=="real_003": continue
        output=pending/"draft_labels"/(path.stem+".txt")
        if output.exists(): raise RuntimeError(f"Refusing to overwrite {output}")
        image=Image.open(path).convert("RGB"); w,h=image.size
        tile=128 if min(w,h)<200 else 320
        detections=[]
        # Ultralytics numpy input expects BGR; PIL image is RGB.
        pixels=np.asarray(image)[...,::-1].copy()
        for y in tile_origins(h,tile,0.25):
            for x in tile_origins(w,tile,0.25):
                result=model.predict(source=pixels[y:y+tile,x:x+tile],conf=0.20,
                    imgsz=640,max_det=2000,verbose=False,device="cpu")[0]
                for box in result.boxes:
                    det=detection_to_dict(box,result.names)
                    b=det["box"]
                    b.update(x1=max(0,min(w,b["x1"]+x)),x2=max(0,min(w,b["x2"]+x)),
                             y1=max(0,min(h,b["y1"]+y)),y2=max(0,min(h,b["y2"]+y)))
                    if b["x2"]>b["x1"] and b["y2"]>b["y1"]:detections.append(det)
        detections=merge_detections(detections,0.5)
        rows=[];draw=ImageDraw.Draw(image)
        for i,d in enumerate(detections,1):
            b=d["box"];x1,y1,x2,y2=[b[k] for k in ("x1","y1","x2","y2")]
            rows.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            draw.rectangle((x1,y1,x2,y2),outline="red",width=1)
            draw.text((x1,y1),str(i),fill="blue")
        output.write_text("\n".join(rows)+("\n" if rows else ""),encoding="utf-8")
        image.save(pending/"review"/(path.stem+"_model_draft.jpg"))
        record={"image":path.name,"status":"model_draft_unreviewed","proposals":len(rows),
          "conf":0.20,"tile_size":tile,"ground_truth":False,"body_extent_review_required":True}
        report.append(record);print(json.dumps(record),flush=True)
        (pending/"review/model_draft_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

if __name__=="__main__":main()
