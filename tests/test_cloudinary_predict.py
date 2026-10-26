"""Unit tests for the Cloudinary behavior of the prediction endpoint."""

from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
import numpy as np
from ultralytics.engine.results import Results

from ai_service import main


def make_test_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeDetector:
    def predict(self, **kwargs):
        return [Results(orig_img=np.asarray(kwargs["source"])[..., ::-1].copy(), path="",
                        names={0: "shrimp-seed"}, boxes=np.empty((0, 6), dtype=np.float32))]


class PredictCloudinaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_cloudinary_configuration_returns_503(self):
        with (
            patch.object(main, "CLOUDINARY_CONFIGURED", False),
            patch.object(main, "get_model") as get_model,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.predict(
                    UploadFile(filename="shrimp.jpg", file=BytesIO(make_test_image())),
                    conf=0.25,
                    imgsz=640,
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertFalse(raised.exception.detail["image_saved"])
        get_model.assert_not_called()

    async def test_successful_prediction_uploads_original_and_returns_asset(self):
        image_bytes = make_test_image()
        cloudinary_result = {
            "secure_url": "https://res.cloudinary.com/test/image/upload/sample.jpg",
            "public_id": "shrimp-detection/originals/sample",
        }
        with (
            patch.object(main, "CLOUDINARY_CONFIGURED", True),
            patch.object(main, "get_model", return_value=FakeDetector()),
            patch("cloudinary.uploader.upload", return_value=cloudinary_result) as upload,
        ):
            response = await main.predict(
                UploadFile(filename="shrimp.jpg", file=BytesIO(image_bytes)),
                conf=0.25,
                imgsz=640,
            )

        self.assertTrue(response["image_saved"])
        self.assertEqual(response["image_url"], cloudinary_result["secure_url"])
        self.assertEqual(response["public_id"], cloudinary_result["public_id"])
        self.assertEqual(response["count"], 0)
        self.assertEqual(upload.call_count, 2)
        self.assertEqual(response["annotated_image_url"], cloudinary_result["secure_url"])
        self.assertTrue(response["annotated_image_saved"])
        uploaded_stream = upload.call_args_list[0].args[0]
        self.assertEqual(uploaded_stream.getvalue(), image_bytes)
        self.assertEqual(upload.call_args_list[0].kwargs["folder"], "shrimp-detection/originals")

        self.assertEqual(upload.call_args_list[1].kwargs["folder"], "shrimp-detection/annotated")
        annotated = Image.open(upload.call_args_list[1].args[0])
        self.assertEqual(annotated.size, (8, 8))

    async def test_annotated_upload_failure_preserves_original_status(self):
        with (
            patch.object(main, "CLOUDINARY_CONFIGURED", True),
            patch.object(main, "get_model", return_value=FakeDetector()),
            patch("cloudinary.uploader.upload", side_effect=[
                {"secure_url": "https://example.com/original.jpg", "public_id": "original"},
                RuntimeError("annotated upload failed"),
            ]),
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.predict(UploadFile(filename="shrimp.jpg", file=BytesIO(make_test_image())),
                                   conf=0.25, imgsz=640)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertTrue(raised.exception.detail["image_saved"])
        self.assertFalse(raised.exception.detail["annotated_image_saved"])

    async def test_cloudinary_upload_error_returns_502_and_not_saved(self):
        with (
            patch.object(main, "CLOUDINARY_CONFIGURED", True),
            patch.object(main, "get_model", return_value=FakeDetector()),
            patch("cloudinary.uploader.upload", side_effect=RuntimeError("upload failed")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.predict(
                    UploadFile(filename="shrimp.jpg", file=BytesIO(make_test_image())),
                    conf=0.25,
                    imgsz=640,
                )

        self.assertEqual(raised.exception.status_code, 502)
        self.assertFalse(raised.exception.detail["image_saved"])

    async def test_invalid_image_returns_400_without_upload(self):
        with (
            patch.object(main, "CLOUDINARY_CONFIGURED", True),
            patch("cloudinary.uploader.upload") as upload,
        ):
            with self.assertRaises(HTTPException) as raised:
                await main.predict(
                    UploadFile(filename="not-an-image.jpg", file=BytesIO(b"not an image")),
                    conf=0.25,
                    imgsz=640,
                )

        self.assertEqual(raised.exception.status_code, 400)
        upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
