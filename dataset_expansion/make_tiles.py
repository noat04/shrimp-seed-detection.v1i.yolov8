"""Cut large reviewed photos into overlapping tiles and build a train/valid/test dataset.

Splits are assigned per SOURCE photo (identical files share one split), so tiles of the same
photo never end up in different splits. Only pass photos whose labels were reviewed by hand;
model drafts in dataset_v2/pending/draft_labels are not ground truth.

Example:
  python dataset_expansion/make_tiles.py --images reviewed/images --labels reviewed/labels \
      --out dataset_v3 --tile-size 320 --overlap 0.2 --include-base dataset_v2 --reviewed
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_service.main import tile_origins  # same grid as tiled inference

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "valid", "test")


def assign_splits(groups, val_frac, test_frac, seed, forced):
    """groups: {sha: [stems]}. Returns {sha: split}; every split gets at least one group."""
    assignment = {}
    for sha, stems in groups.items():
        wanted = {forced[s] for s in stems if s in forced}
        if len(wanted) > 1:
            raise ValueError(f"Identical photos assigned to different splits: {stems}")
        if wanted:
            assignment[sha] = wanted.pop()

    free = sorted((s for s in groups if s not in assignment),
                  key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    total = len(groups)
    n_test = max(1, round(total * test_frac)) if test_frac else 0
    n_val = max(1, round(total * val_frac)) if val_frac else 0
    n_test -= sum(1 for v in assignment.values() if v == "test")
    n_val -= sum(1 for v in assignment.values() if v == "valid")
    for sha in free:
        if n_test > 0:
            assignment[sha], n_test = "test", n_test - 1
        elif n_val > 0:
            assignment[sha], n_val = "valid", n_val - 1
        else:
            assignment[sha] = "train"
    return assignment


def tile_labels(rows, width, height, x0, y0, tw, th, min_visible):
    kept = []
    for cls, cx, cy, w, h in rows:
        bx1, by1 = (cx - w / 2) * width, (cy - h / 2) * height
        bx2, by2 = (cx + w / 2) * width, (cy + h / 2) * height
        ix1, iy1, ix2, iy2 = max(bx1, x0), max(by1, y0), min(bx2, x0 + tw), min(by2, y0 + th)
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < min_visible * (bx2 - bx1) * (by2 - by1):
            continue
        kept.append(f"{int(cls)} {((ix1 + ix2) / 2 - x0) / tw:.6f} {((iy1 + iy2) / 2 - y0) / th:.6f} "
                    f"{(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}")
    return kept


def read_rows(label_path):
    rows = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            rows.append(tuple(map(float, line.split())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Folder of full-size reviewed photos")
    parser.add_argument("--labels", type=Path, required=True, help="Folder of reviewed YOLO labels (same stems)")
    parser.add_argument("--out", type=Path, required=True, help="New dataset folder (must not exist or be empty)")
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visible", type=float, default=0.5,
                        help="Keep a box in a tile only if at least this fraction of it is inside")
    parser.add_argument("--empty-ratio", type=float, default=0.0,
                        help="Fraction of label-less tiles to keep as negatives (0 = drop all)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assign", type=Path, help='JSON {"photo_stem": "train|valid|test"} to force splits (e.g. per lot)')
    parser.add_argument("--include-base", type=Path,
                        help="Existing dataset (train/valid/test with images+labels) copied in unchanged")
    parser.add_argument("--reviewed", action="store_true", help="Confirm the labels were manually reviewed")
    args = parser.parse_args()

    if not args.reviewed:
        parser.error("Manual review required: pass --reviewed only after checking the labels")
    if not (0 <= args.overlap < 1 and args.tile_size >= 64 and 0 < args.min_visible <= 1):
        parser.error("Invalid tile-size/overlap/min-visible")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} already exists and is not empty")

    photos = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        parser.error("No images found")
    groups = {}
    for photo in photos:
        if not (args.labels / f"{photo.stem}.txt").exists():
            parser.error(f"Missing label for {photo.name}")
        groups.setdefault(hashlib.sha256(photo.read_bytes()).hexdigest(), []).append(photo)
    if len(groups) < 1 + bool(args.val_frac) + bool(args.test_frac):
        parser.error("Not enough distinct source photos to fill every split")

    forced = json.loads(args.assign.read_text(encoding="utf-8")) if args.assign else {}
    unknown = set(forced) - {p.stem for p in photos}
    if unknown or not set(forced.values()) <= set(SPLITS):
        parser.error(f"--assign has unknown photos {sorted(unknown)} or invalid splits")
    assignment = assign_splits({sha: [p.stem for p in ps] for sha, ps in groups.items()},
                               args.val_frac, args.test_frac, args.seed, forced)

    for split in SPLITS:
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)

    keep_every = round(1 / args.empty_ratio) if args.empty_ratio else 0
    empty_seen = 0
    manifest = {}
    for sha, members in groups.items():
        split = assignment[sha]
        photo = members[0]  # duplicates of the same file are tiled once
        image = Image.open(photo).convert("RGB")
        width, height = image.size
        rows = read_rows(args.labels / f"{photo.stem}.txt")
        written = boxes = 0
        for y0 in tile_origins(height, args.tile_size, args.overlap):
            for x0 in tile_origins(width, args.tile_size, args.overlap):
                tw, th = min(args.tile_size, width - x0), min(args.tile_size, height - y0)
                lines = tile_labels(rows, width, height, x0, y0, tw, th, args.min_visible)
                if not lines:
                    empty_seen += 1
                    if not keep_every or empty_seen % keep_every:
                        continue
                name = f"{photo.stem}_x{x0}_y{y0}"
                image.crop((x0, y0, x0 + tw, y0 + th)).save(
                    args.out / split / "images" / f"{name}.jpg", quality=95)
                (args.out / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                written += 1
                boxes += len(lines)
        manifest[photo.stem] = {"split": split, "size": [width, height], "tiles": written, "boxes": boxes,
                                "duplicates": [p.name for p in members[1:]]}

    if args.include_base:
        for split in SPLITS:
            for sub in ("images", "labels"):
                src = args.include_base / split / sub
                for item in src.iterdir():
                    if item.is_file() and item.suffix.lower() in (*IMAGE_SUFFIXES, ".txt"):
                        shutil.copy2(item, args.out / split / sub / item.name)

    (args.out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['shrimp-seed']\n")
    (args.out / "tiles_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split in SPLITS:
        n_img = len(list((args.out / split / "images").iterdir()))
        sources = [s for s, m in manifest.items() if m["split"] == split]
        print(f"{split}: {n_img} images; new sources: {sources}")


if __name__ == "__main__":
    main()
