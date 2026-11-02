"""Build one training dataset from several image domains, balanced and with held-out photos per domain.

A spec file (see domains_spec.json) lists the domains and where their reviewed images/labels live.
For every domain this script:
  1. holds out whole photos for test (and valid), choosing by SOURCE PHOTO so augmented copies and
     tiles of one photo never cross splits;
  2. cuts the remaining photos into tiles when the domain asks for it (tile > 0);
  3. repeats each domain's training files so small domains are not drowned by large ones.
The test split keeps whole, full-size photos so counting can be measured on real inputs
(see evaluate_domains.py). Only pass labels that were reviewed by hand.

Example:
  python dataset_expansion/build_domain_dataset.py --spec dataset_expansion/domains_spec.json \
      --out dataset_v3 --holdout 5 --valid 2 --reviewed
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
from ai_service.main import tile_origins  # same tile grid as /predict
from dataset_expansion.make_tiles import read_rows, tile_labels

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def group_of(stem):
    """Roboflow augmentations share the name before '.rf.'; treat them as one source photo."""
    return stem.split(".rf.")[0]


def rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def collect(spec, seed):
    items, seen = [], {}
    for src in spec["sources"]:
        domain = src["domain"]
        if domain not in spec["domains"]:
            raise SystemExit(f"Source uses unknown domain {domain!r}")
        image_dir, label_dir = ROOT / src["images"], ROOT / src["labels"]
        wanted = set(src["files"]) if src.get("files") else None
        by_group = {}
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if wanted is not None and path.stem not in wanted:
                continue
            by_group.setdefault(group_of(path.stem), []).append(path)
        if wanted is not None and not by_group:
            raise SystemExit(f"None of the listed files exist in {image_dir}")
        groups = sorted(by_group, key=lambda g: rank(seed, g))
        if src.get("limit"):
            groups = groups[: src["limit"]]
        for group in groups:
            for path in by_group[group]:
                label = label_dir / f"{path.stem}.txt"
                if not label.is_file():
                    raise SystemExit(f"Missing label for {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    print(f"skip duplicate {path.name} (same bytes as {seen[digest]})")
                    continue
                seen[digest] = path.name
                items.append({"domain": domain, "image": path, "label": label,
                              "group": f"{domain}:{group}", "source": src["images"],
                              "fixed": src.get("split")})
    return items


def assign_splits(items, holdout, valid, lenient, seed):
    by_domain = {}
    for item in items:
        by_domain.setdefault(item["domain"], {}).setdefault(item["group"], []).append(item)
    for domain, groups in sorted(by_domain.items()):
        # a source may pin its photos to a split with "split" in the spec; the rest is chosen at random
        fixed = {g: members[0]["fixed"] for g, members in groups.items() if members[0]["fixed"]}
        for g, split in fixed.items():
            if split not in ("train", "valid", "test"):
                raise SystemExit(f"Invalid fixed split {split!r} for {g}")
            for item in groups[g]:
                item["split"] = split
        order = sorted((g for g in groups if g not in fixed), key=lambda g: rank(seed, g))
        n_hold = max(0, holdout - sum(1 for s in fixed.values() if s == "test"))
        n_valid = max(0, valid - sum(1 for s in fixed.values() if s == "valid"))
        n_train_fixed = sum(1 for s in fixed.values() if s == "train")
        if len(order) + n_train_fixed < n_hold + n_valid + 1:
            msg = (f"domain {domain}: {len(groups)} source photo(s) but {holdout} test + {valid} valid + "
                   f">=1 train are required")
            if not lenient:
                raise SystemExit(msg + ". Add photos or lower --holdout/--valid (or use --lenient).")
            print(f"WARNING {msg}; --lenient: the unpinned ones go to train, NO extra held-out photos for this domain")
            n_hold = n_valid = 0
        for i, group in enumerate(order):
            split = "test" if i < n_hold else "valid" if i < n_hold + n_valid else "train"
            for item in groups[group]:
                item["split"] = split
    return by_domain


def write_item(item, out, tile, min_visible, name_prefix):
    """Write one photo (whole or tiled) into out/<split>; return [(path, boxes)]."""
    split = item["split"]
    image = Image.open(item["image"]).convert("RGB")
    width, height = image.size
    rows = read_rows(item["label"])
    base = f"{name_prefix}{item['image'].stem}"
    images_dir, labels_dir = out / split / "images", out / split / "labels"
    written = []
    if split == "test" or not tile:
        item["image"].parent  # whole image, copied byte for byte
        shutil.copy2(item["image"], images_dir / f"{base}{item['image'].suffix.lower()}")
        shutil.copy2(item["label"], labels_dir / f"{base}.txt")
        written.append((images_dir / f"{base}{item['image'].suffix.lower()}", len(rows)))
        return written
    for y0 in tile_origins(height, tile, 0.2):
        for x0 in tile_origins(width, tile, 0.2):
            tw, th = min(tile, width - x0), min(tile, height - y0)
            lines = tile_labels(rows, width, height, x0, y0, tw, th, min_visible)
            if not lines:
                continue
            name = f"{base}_x{x0}_y{y0}"
            image.crop((x0, y0, x0 + tw, y0 + th)).save(images_dir / f"{name}.jpg", quality=95)
            (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            written.append((images_dir / f"{name}.jpg", len(lines)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=5, help="Whole photos per domain kept for test")
    parser.add_argument("--valid", type=int, default=2, help="Photos per domain used for validation")
    parser.add_argument("--max-repeat", type=int, default=8, help="Largest number of times a train file is repeated")
    parser.add_argument("--target", type=int, help="Train files per domain after repeating (default: size of the largest domain)")
    parser.add_argument("--no-balance", action="store_true", help="Do not repeat any files")
    parser.add_argument("--min-visible", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lenient", action="store_true", help="Warn instead of failing when a domain has too few photos")
    parser.add_argument("--reviewed", action="store_true", help="Confirm the labels were manually reviewed")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Manual review required: pass --reviewed only after checking the labels")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} already exists and is not empty")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    items = collect(spec, args.seed)
    by_domain = assign_splits(items, args.holdout, args.valid, args.lenient, args.seed)
    for split in ("train", "valid", "test"):
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)

    files, train_files = [], {d: [] for d in by_domain}
    for item in items:
        domain = item["domain"]
        tile = int(spec["domains"][domain].get("tile", 0))
        for path, boxes in write_item(item, args.out, tile, args.min_visible, f"{domain}__"):
            files.append({"file": path.relative_to(args.out).as_posix(), "domain": domain, "split": item["split"],
                          "group": item["group"], "source": item["source"], "boxes": boxes, "repeat": 0})
            if item["split"] == "train":
                train_files[domain].append(files[-1])

    sizes = {d: len(v) for d, v in train_files.items() if v}
    target = args.target or (max(sizes.values()) if sizes else 0)
    repeats = {}
    for domain, entries in train_files.items():
        if not entries:
            continue
        repeats[domain] = 1 if args.no_balance else max(1, min(args.max_repeat, round(target / len(entries))))
        for entry in list(entries):
            src = args.out / entry["file"]
            for k in range(1, repeats[domain]):
                copy = src.with_name(f"{src.stem}__r{k}{src.suffix}")
                shutil.copy2(src, copy)
                shutil.copy2(args.out / "train/labels" / f"{src.stem}.txt", args.out / "train/labels" / f"{copy.stem}.txt")
                files.append({**entry, "file": copy.relative_to(args.out).as_posix(), "repeat": k})
            entry["repeat"] = 0

    (args.out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['shrimp-seed']\n")
    summary = {}
    for domain in sorted(by_domain):
        rows = [f for f in files if f["domain"] == domain]
        summary[domain] = {
            "photos": {s: len({f["group"] for f in rows if f["split"] == s}) for s in ("train", "valid", "test")},
            "train_files_before_repeat": sizes.get(domain, 0),
            "repeat": repeats.get(domain, 0),
            "train_files": sum(1 for f in rows if f["split"] == "train"),
            "train_boxes": sum(f["boxes"] for f in rows if f["split"] == "train"),
            "tile": int(spec["domains"][domain].get("tile", 0)),
        }
    (args.out / "domain_manifest.json").write_text(
        json.dumps({"spec": spec, "args": {k: str(v) for k, v in vars(args).items()},
                    "summary": summary, "files": files}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'domain':<8}{'photos tr/va/te':<18}{'tile':>5}{'train files':>13}{'repeat':>8}{'after':>7}{'train boxes':>13}")
    for domain, s in summary.items():
        p = s["photos"]
        print(f"{domain:<8}{p['train']}/{p['valid']}/{p['test']:<14}{s['tile']:>5}{s['train_files_before_repeat']:>13}"
              f"{s['repeat']:>8}{s['train_files']:>7}{s['train_boxes']:>13}")


if __name__ == "__main__":
    main()
