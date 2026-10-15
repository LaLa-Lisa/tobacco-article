# Data Directory

Place the project dataset here or pass an external dataset path to the scripts.
Do not commit large private datasets unless publication and consent constraints
allow it.

Expected YOLO-style layout for detector experiments:

```text
data/tobacco_yolo/
  images/
    *.jpg
  labels/
    *.txt
```

Expected layout for weed-mask validation:

```text
data/mask_validation/
  images/
    *.jpg
  gt_masks/
    *.png
```

The final article should report the exact dataset source, annotation policy,
train/validation/test split, and sharing status.
