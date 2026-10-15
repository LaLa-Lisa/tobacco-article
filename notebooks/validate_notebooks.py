#!/usr/bin/env python3
"""Execute default notebooks and check portability/integration without real data."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from reproduction import frozen_assignments, new_run, repository_root, run_script, write_local_splits, check_detector_results

ROOT = repository_root(Path(__file__).parent)


def evidence_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('experiments/results', 'experiments/splits')
            for p in (ROOT / folder).rglob('*') if p.is_file()}


def check_helpers(temp):
    assignments = frozen_assignments(ROOT)
    assert sum(map(len, assignments.values())) == 110
    dataset = temp / 'dataset with spaces'
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'labels').mkdir()
    for names in assignments.values():
        for name in names:
            (dataset / 'images' / name).touch()
            (dataset / 'labels' / f'{Path(name).stem}.txt').touch()
    splits = write_local_splits(ROOT, dataset, temp / 'splits')
    for split, names in assignments.items():
        paths = (splits / f'{split}.txt').read_text().splitlines()
        assert [Path(p).name for p in paths] == names
        assert all(Path(p).is_absolute() and Path(p).exists() for p in paths)
    (dataset / 'images' / assignments['test'][0]).unlink()
    try:
        write_local_splits(ROOT, dataset, temp / 'missing-input-splits')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing data was not rejected.')
    new_run(temp, 'collision')
    try:
        new_run(temp, 'collision')
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing run was overwritten.')
    check_detector_results(ROOT / 'experiments/results/yolo/model_comparison.csv')
    failed = temp / 'failed.csv'
    failed.write_text('model_name,status,test_status,test_error\nyolov8n,ok,failed,fixture failure\n')
    try:
        check_detector_results(failed)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Failed evaluation was accepted.')
    print('PASS: portable split membership, missing inputs, run collisions, result failure checks')


def check_processing(temp):
    if importlib.util.find_spec('cv2') is None:
        print('SKIP: synthetic mask/coordinate checks require OpenCV.')
        return
    import cv2
    import numpy as np
    from argparse import Namespace
    sys.path.insert(0, str(ROOT / 'scripts'))
    import generate_masks as masks
    from evaluate_masks import compute_metrics
    args = Namespace(h_min=35, h_max=85, s_min=40, s_max=255, v_min=40, v_max=255,
                     min_area=50, max_area=0, box_padding=0, imgsz=1024, conf=0.25, device='cpu')
    # These are software fixtures only: a green square, no scientific evidence.
    images, truth, points = (temp / name for name in ('images', 'truth', 'points'))
    for folder in (images, truth, points):
        folder.mkdir()
    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    bgr[10:30, 10:30] = (0, 255, 0)
    cv2.imwrite(str(images / 'fixture.png'), bgr)
    vegetation = masks.hsv_mask(bgr, args)
    final, centers = masks.contour_filter(vegetation, args)
    assert centers == [(19, 19)]
    assert compute_metrics(final > 0, vegetation > 0)['iou'] == 1
    # Diagnostic no-model mode exercises batch file output, not trained inference.
    run_script(ROOT, 'generate_masks.py', '--images', images, '--output', temp / 'masks', '--device', 'cpu')
    cv2.imwrite(str(truth / 'fixture.png'), final)
    run_script(ROOT, 'evaluate_masks.py', '--ground-truth', truth,
               '--predicted', temp / 'masks/final_contour_filtering', '--output', temp / 'scores.csv')
    with (temp / 'scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]['image'] == '__mean__' and float(rows[-1]['iou']) == 1
    (points / 'fixture.csv').write_text('x,y\n19,19\n')
    run_script(ROOT, 'evaluate_coordinates.py', '--ground-truth', points,
               '--predicted', temp / 'masks/centers', '--output', temp / 'coordinates.csv')
    with (temp / 'coordinates.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row['mean_nearest_error_px']) == 0 and float(row['hit_rate_10px']) == 1
    print('PASS: synthetic mask and coordinate integration (not experimental evidence)')


def main():
    before = evidence_hashes()
    with tempfile.TemporaryDirectory(prefix='tobacco-notebook-validation-') as directory:
        temp = Path(directory)
        os.environ['JUPYTER_CONFIG_DIR'] = str(temp / 'jupyter-config')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temp / 'jupyter-runtime')
        os.environ['IPYTHONDIR'] = str(temp / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(temp / 'matplotlib')
        os.environ['YOLO_CONFIG_DIR'] = str(temp / 'ultralytics')
        os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
        os.environ['JUPYTER_PATH'] = str(temp / 'share/jupyter')
        from ipykernel.kernelspec import install
        from nbclient import NotebookClient
        import nbformat
        install(prefix=str(temp), kernel_name='tobacco-validation')
        for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            client = NotebookClient(notebook, timeout=180, kernel_name='tobacco-validation',
                                    resources={'metadata': {'path': str(ROOT / 'notebooks')}})
            client.execute()
            print(f'PASS: {path.name} (all default cells executed)', flush=True)
        check_helpers(temp)
        check_processing(temp)
    assert before == evidence_hashes(), 'Historical evidence changed during validation!'
    print('PASS: historical results and manifests unchanged')


if __name__ == '__main__':
    main()
