"""Экспорт детекций YOLO в COCO с сохранением исходных выборок."""
import json
from pathlib import Path

from .yolo_dataset import annotations, describe_dataset


def export_coco(dataset, destination):
    from PIL import Image
    if dataset.task != 'detect':
        raise ValueError('RTMDet-Tiny requires detection labels')
    record = describe_dataset(dataset)
    destination.mkdir(parents=True)
    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(record['names'])]
    for split, info in record['splits'].items():
        images, boxes = [], []
        for image_id, member in enumerate(info['members'], start=1):
            with Image.open(member['image']) as image:
                width, height = image.size
            images.append(dict(id=image_id, file_name=member['image'], width=width, height=height))
            for cls, (cx, cy, w, h) in annotations(Path(member['label']), 'detect', len(categories)):
                bbox = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
                boxes.append(dict(id=len(boxes) + 1, image_id=image_id, category_id=cls + 1,
                                  bbox=bbox, area=bbox[2] * bbox[3], iscrowd=0))
        output = destination / f'{split}.json'
        output.write_text(json.dumps(dict(info={}, images=images, annotations=boxes, categories=categories)))
        info['ann_file'] = str(output.resolve())
    return record
