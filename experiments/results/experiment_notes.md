# Experiment Notes

## Environment
- OS: Windows-11-10.0.26100-SP0
- Python: 3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026, 10:26:10) [MSC v.1944 64 bit (AMD64)]
- GPU: NVIDIA GeForce RTX 5070 Ti
- CUDA available to PyTorch: True
- PyTorch: 2.12.1+cu132
- Ultralytics: 8.4.81
- OpenCV: 4.13.0

## Dataset And Splits
- Dataset: data\tobacco_yolo
- Annotation class 0: tobacco
- Images: 110
- Boxes: 4436
- Missing labels: 0
- Split counts: {'train': 78, 'val': 16, 'test': 16}

## Detector Runs
- yolov8n: ok; batch=14; imgsz=1024; notes=
- yolov8m: ok; batch=14; imgsz=1024; notes=
- yolov8x: ok; batch=14; imgsz=1024; notes=

## Mask Ablation
- hsv_only: IoU=, precision=, recall=, dice=; missing or incomplete source file: experiments\results\mask_hsv_only.csv
- crop_exclusion_only: IoU=, precision=, recall=, dice=; missing or incomplete source file: experiments\results\mask_crop_exclusion_only.csv
- hsv_crop_exclusion: IoU=, precision=, recall=, dice=; missing or incomplete source file: experiments\results\mask_hsv_crop_exclusion.csv
- final_contour_filtering: IoU=, precision=, recall=, dice=; missing or incomplete source file: experiments\results\mask_final_contour_filtering.csv

## Deviations, Failures, And Skips
- Mask ground-truth directory data/mask_validation/gt_masks is not present; mask_metrics.csv records mask ablations as not evaluated, so the article must not report quantitative weed-mask IoU/Dice from this run.
- Coordinate-center annotations are not present; coordinate accuracy was not evaluated.

## Final Model Recommendation
- Recommended model based on available comparison artifacts: yolov8x
