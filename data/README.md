# Data Directory

Place the project dataset here or set an external dataset path in the notebook configuration cells.
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
