"""Подготовка структуры каталогов для загрузчика RF-DETR."""
from pathlib import Path

from .yolo_dataset import annotations, describe_dataset, digest


def stage_dataset(dataset, destination):
    import yaml
    record = describe_dataset(dataset)
    names = record['names']
    output = {"names": names, "nc": len(names)}
    for split, info in record['splits'].items():
        folder = 'valid' if split == 'val' else split
        image_dir, label_dir = destination / folder / 'images', destination / folder / 'labels'
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        for index, member in enumerate(info['members']):
            image, label = Path(member['image']), Path(member['label'])
            list(annotations(label, dataset.task, len(names)))
            stem = f'{index:08d}'
            (image_dir / (stem + image.suffix.lower())).symlink_to(image)
            target = label_dir / (stem + '.txt')
            if label.exists():
                target.symlink_to(label)
            else:
                target.write_text('')
        output[split] = f'{folder}/images'
    (destination / 'data.yaml').write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False))
    return record
