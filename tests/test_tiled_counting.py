"""Unit tests for tiled inference helpers."""

import unittest
from io import BytesIO
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
from PIL import Image

from ai_service import main


def det(x1, y1, x2, y2, confidence):
    return {"confidence": confidence, "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


class TileOriginsTests(unittest.TestCase):
    def test_small_image_uses_single_tile(self):
        self.assertEqual(main.tile_origins(300, 320, 0.2), [0])

    def test_last_tile_is_aligned_to_the_edge(self):
        origins = main.tile_origins(640, 320, 0.2)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 320)


class MergeDetectionsTests(unittest.TestCase):
    def test_duplicate_from_overlapping_tiles_is_removed(self):
        merged = main.merge_detections([det(0, 0, 10, 10, 0.4), det(1, 1, 11, 11, 0.6)], 0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["confidence"], 0.6)

    def test_separate_boxes_are_kept(self):
        merged = main.merge_detections([det(0, 0, 10, 10, 0.4), det(50, 50, 60, 60, 0.6)], 0.5)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()


class AnnotationTests(unittest.TestCase):
    def test_merged_boxes_plot_on_original_with_correct_colors(self):
        image = Image.new("RGB", (100, 80), (255, 0, 0))
        detection = {"class_id": 0, "class": "shrimp-seed", "confidence": 0.9,
                     "box": {"x1": 30, "y1": 40, "x2": 60, "y2": 65}}
        plotted = main.plot_detections(image, {0: "shrimp-seed"}, [detection])
        self.assertEqual(plotted.shape, (80, 100, 3))
        self.assertTrue(np.any(plotted[40, 30:61] != [0, 0, 255]))
        with patch("cloudinary.uploader.upload", return_value={}) as upload:
            main.upload_annotated_to_cloudinary(plotted)
        encoded = Image.open(BytesIO(upload.call_args.args[0].getvalue()))
        self.assertEqual(encoded.getpixel((0, 79)), (255, 0, 0))
        self.assertEqual(image.getpixel((30, 40)), (255, 0, 0))


class TiledColorTests(unittest.TestCase):
    def test_tiles_preserve_colors_expected_by_numpy_detector_input(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[:, :3] = [255, 30, 10]
        rgb[:, 3:] = [20, 80, 240]
        image = Image.fromarray(rgb)
        detector = Mock()
        detector.predict.return_value = [SimpleNamespace(names={0: "shrimp-seed"}, boxes=[])]
        names, detections = main.detect_tiled(detector, image, .25, 640, 300, 4, .25, .5)
        self.assertEqual(detector.predict.call_count, 2)
        for call, x in zip(detector.predict.call_args_list, [0, 2]):
            actual = call.kwargs["source"]
            np.testing.assert_array_equal(actual, rgb[:, x:x+4, ::-1])
        np.testing.assert_array_equal(np.asarray(image), rgb)
        self.assertEqual(names, {0: "shrimp-seed"})
        self.assertEqual(detections, [])
