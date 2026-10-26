# Annotated prediction images

POST /predict now uploads two images on every successful request:
- image_url / public_id: original image in shrimp-detection/originals.
- annotated_image_url / annotated_public_id: PNG with boxes and confidence in shrimp-detection/annotated.
- image_saved and annotated_image_saved: true on success.

Whole-image inference uses result.plot(). Tiled inference constructs a Results object from final merged detections and plots on the original image. Thus plotted boxes match the returned count, including zero detections. Plot output is converted from BGR to RGB before PNG encoding.

ChillShrimp backend integration (in the external repository):
```js
record.annotatedImageUrl = aiResponse.annotated_image_url;
```
Persist this field alongside the existing original image URL and detection data. Frontend displays the stored annotatedImageUrl as its image source. No ChillShrimp backend/frontend files are included or modified here.

Restart Uvicorn, submit a new prediction in /docs, then open annotated_image_url. Existing responses are not retroactively annotated. Request parameters remain unchanged. This does not change model accuracy.

If the original upload fails, the existing 502 behavior remains. If only the annotated upload fails, 502 detail contains image_saved=true, the original URL/public ID and annotated_image_saved=false. The original remains saved; no automatic rollback is performed. Successful requests now create two Cloudinary assets.

Validation: automated tests mock Cloudinary; live upload and frontend integration must be checked separately.
