# Dataset v2

Copied the existing train/valid/test image and label splits without modifying the originals.
12 new user-provided photos are in pending/images. They are NOT included in data.yaml and have no ground-truth labels yet. No training has been run.

See pending/contact_sheet.jpg and pending/manifest.json for numbered images, source paths and review status.
Use ../dataset_expansion/annotate.html to draw boxes and export YOLO labels. Apply ../dataset_expansion/LABELING_GUIDE.md consistently. Review existing label semantics before mixing new labels.
After complete manual review, assign each source photo/session to ONE split and move its image/label pair into that split. Do not scatter crops or near-duplicate photographs across splits. Do not create empty labels for shrimp-containing photos.
real_002 and real_004 are especially low resolution; obtain better originals or exclude them when individual shrimp cannot be resolved reliably.

The training YAML currently selects ONLY the old copied data. Train only after the new reviewed pairs have been included. pending is deliberately excluded.
