# PL Detection + custom shrimp dataset

Active splits: train 691 images / 19424 boxes; valid 31 / 919; test 34 / 806.
Class: 0 = shrimp-seed.
PL Detection v1: 748 images, Teramina, MIT. See source README files.
Custom images: 8 image/label pairs copied from pending/labels after the user confirmed review. 6 train, 1 valid, 1 test; assignments and label hashes are recorded in custom_inclusion.json.

pending retains working copies; data.yaml reads only the active train/valid/test directories. Do not move an included image into another split without removing its former active copy. Future edits to pending labels must be synchronized explicitly; training does not read pending.

Custom annotation correctness was confirmed by the user; automated checks verify format, class, coordinates and absence of exact cross-split duplicates. Capture-session provenance remains unknown; session independence and all near duplicates are not certified. With only one custom test image, performance on user images is preliminary.

No training has started. Source datasets and API model are unchanged.
