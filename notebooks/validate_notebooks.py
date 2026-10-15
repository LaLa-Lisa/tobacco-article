#!/usr/bin/env python3
"""Execute default notebooks and check portability/integration without real data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from reproduction import frozen_assignments, new_run, repository_root, write_local_splits, check_detector_results

ROOT = repository_root(Path(__file__).parent)


def evidence_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('experiments/results', 'experiments/splits', 'figures/publication')
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


def notebook_definitions(filename):
    """Load tagged definition cells for tests only; notebooks never import each other."""
    namespace = {'Path': Path, 'json': json, 'os': os}
    notebook = json.loads((ROOT / 'notebooks' / filename).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'definitions' in cell['metadata'].get('tags', []):
            exec(compile(''.join(cell['source']), filename, 'exec'), namespace)
    return namespace


def configuration(filename):
    """Execute the reader's configuration cell, without importing model dependencies."""
    namespace = dict(ROOT=ROOT, RESULTS=ROOT / 'experiments/results', Path=Path,
                     pd=__import__('pandas'), json=json, display=lambda *args: None)
    notebook = json.loads((ROOT / 'notebooks' / filename).read_text())
    for cell in notebook['cells']:
        if 'configuration' in cell['metadata'].get('tags', []):
            exec(''.join(cell['source']), namespace)
    return namespace


def must_raise(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f'Expected {error.__name__} from {function.__name__}')


def check_detector(temp):
    from types import SimpleNamespace
    d = notebook_definitions('02_reproduce_detector_experiments.ipynb')
    dataset = temp / 'audit dataset'
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'labels').mkdir()
    (dataset / 'images/a.jpg').touch()
    label = dataset / 'labels/a.txt'
    label.write_text('0 0.5 0.5 0.2 0.2\n')
    report = d['audit_dataset'](dataset)
    assert report['status'] == 'ok' and report['boxes'] == 1
    label.write_text('0 2 0.5 0.2 0.2\ninvalid\n')
    assert d['audit_dataset'](dataset)['status'] == 'fail'
    label.unlink()
    assert d['audit_dataset'](dataset)['missing_labels_count'] == 1
    assert d['audit_dataset'](temp / 'absent')['status'] == 'fail'
    options = configuration('02_reproduce_detector_experiments.ipynb')['TRAINING_OPTIONS']
    options.update(epochs=1, imgsz=32, batch=1, device='cpu')
    calls = []

    class FakeYOLO:
        def __init__(self, model):
            self.source = str(model)
            self.model = SimpleNamespace(parameters=lambda: [])
            calls.append(('load', self.source))

        def train(self, **kwargs):
            calls.append(('train', kwargs))
            assert kwargs['optimizer'] == 'SGD'
            assert (kwargs['lr0'], kwargs['momentum'], kwargs['weight_decay']) == (0.005, 0.937, 0.0005)
            assert kwargs['cos_lr'] and kwargs['amp']
            if self.source == 'broken.pt':
                raise RuntimeError('Training fixture failure')
            run = Path(kwargs['project']) / kwargs['name']
            (run / 'weights').mkdir(parents=True)
            if self.source != 'missing.pt':
                (run / 'weights/last.pt').write_bytes(b'last')
            if self.source not in ('fallback.pt', 'missing.pt'):
                (run / 'weights/best.pt').write_bytes(b'best')
            (run / 'results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.4\n')
            return SimpleNamespace(results_dict={}, speed={})

        def val(self, **kwargs):
            calls.append(('val', kwargs))
            if ('eval-failure' in self.source and kwargs['split'] == 'test'
                    or 'val-failure' in self.source and kwargs['split'] == 'val'):
                raise RuntimeError('Evaluation fixture failure')
            return SimpleNamespace(results_dict={'metrics/mAP50-95(B)': 0.5}, speed={'inference': 1})

    output = temp / 'fake training'
    models = ['good.pt', 'fallback.pt', 'broken.pt', 'eval-failure.pt', 'val-failure.pt', 'missing.pt']
    rows = d['train_detectors'](dataset, temp / 'splits', models, output,
                                options=options, model_factory=FakeYOLO)
    assert [r['status'] for r in rows] == ['ok', 'ok', 'failed', 'failed', 'failed', 'failed']
    assert rows[3]['test_status'] == 'failed' and rows[3]['train_status'] == 'ok'
    assert rows[4]['val_status'] == 'failed' and rows[4]['test_status'] == 'ok'
    assert rows[5]['val_status'] == rows[5]['test_status'] == 'failed'
    assert any(kind == 'load' and str(value).endswith('fallback/weights/last.pt') for kind, value in calls)
    assert any(kind == 'load' and str(value).endswith('good/weights/best.pt') for kind, value in calls)
    assert len([1 for kind, _ in calls if kind == 'val']) == 8
    assert (output / 'good_results.csv').is_file()
    must_raise(RuntimeError, check_detector_results, output / 'model_comparison.csv')
    success = temp / 'fake success'
    d['train_detectors'](dataset, temp / 'splits', ['good.pt'], success,
                         options=options, model_factory=FakeYOLO)
    check_detector_results(success / 'model_comparison.csv')
    metadata = json.loads((success / 'run_metadata.json').read_text())
    assert all(metadata[key] == value for key, value in options.items())
    must_raise(FileExistsError, d['train_detectors'], dataset, temp / 'splits', ['good.pt'],
               success, options=options, model_factory=FakeYOLO)
    print('PASS: notebook dataset audits, YOLO arguments, checkpoints, exports and failure handling')


def check_processing(temp):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    masks = notebook_definitions('03_masks_and_weed_coordinates.ipynb')
    scores = notebook_definitions('04_evaluate_masks_and_coordinates.ipynb')
    options = configuration('03_masks_and_weed_coordinates.ipynb')['PROCESSING_OPTIONS']
    options['device'] = 'cpu'
    crop_options = {key: options[key] for key in ('conf', 'imgsz', 'device', 'box_padding')}
    images, truth, points = (temp / name for name in ('images', 'truth', 'points'))
    for folder in (images, truth, points):
        folder.mkdir()
    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    bgr[10:30, 10:30] = (0, 255, 0)
    path = images / 'fixture.png'
    cv2.imwrite(str(path), bgr)
    vegetation = masks['hsv_mask'](bgr)
    final, centers = masks['contour_filter'](vegetation)
    assert centers == [(19, 19)]
    assert masks['contour_filter'](vegetation, min_area=1000)[1] == []
    assert masks['contour_filter'](vegetation, max_area=100)[1] == []
    assert scores['compute_metrics'](final > 0, vegetation > 0)['iou'] == 1
    empty = np.zeros_like(final, dtype=bool)
    assert scores['compute_metrics'](empty, empty)['dice'] == 1
    assert scores['compute_metrics'](final > 0, empty)['recall'] == 0
    assert scores['compute_metrics'](final > 0, ~ (final > 0))['iou'] == 0
    must_raise(ValueError, scores['compute_metrics'], empty, empty[:2])

    class Boxes:
        def cpu(self):
            return self
        def numpy(self):
            return np.array([[-3.2, -1.2, 15.4, 50.0]])
    detector = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(boxes=SimpleNamespace(xyxy=Boxes()))])
    crop = masks['crop_mask'](detector, path, (40, 40), **crop_options)
    assert np.all(crop[:, :16] == 255) and np.all(crop[:, 16:] == 0)
    assert not masks['crop_mask'](None, path, (40, 40), **crop_options).any()
    # Actual batch and evaluation functions from the notebooks, with synthetic data only.
    masks['generate_predictions'](images, detector, temp / 'masks', model_path=Path('fixture.pt'), **options)
    result = masks['process_image'](path, detector, **options)
    expected, expected_centers = result['variants'], result['centers']
    for name, array in expected.items():
        assert np.array_equal(cv2.imread(str(temp / 'masks' / name / 'fixture.png'), 0), array)
    cv2.imwrite(str(truth / 'fixture.png'), expected['final_contour_filtering'])
    rows = scores['evaluate_masks'](truth, temp / 'masks/final_contour_filtering', temp / 'scores.csv')
    assert rows[-1]['image'] == '__mean__' and float(rows[-1]['iou']) == 1
    (points / 'fixture.csv').write_text('x,y\n' + ''.join(f'{x},{y}\n' for x,y in expected_centers))
    rows = scores['evaluate_coordinates'](points, temp / 'masks/centers', temp / 'coordinates.csv')
    assert float(rows[0]['mean_nearest_error_px']) == 0
    assert scores['nearest_distances']([], [(0, 0)]) == [float('inf')]
    assert scores['nearest_distances']([(0, 0)], [(3, 4), (6, 8)]) == [5, 10]
    assert scores['hit_rate']([10, 20, 50, 51], 10) == 0.25
    assert scores['hit_rate']([10, 20, 50, 51], 20) == 0.5
    assert scores['hit_rate']([10, 20, 50, 51], 50) == 0.75
    assert scores['hit_rate']([], 10) == 1
    (temp / 'masks/centers/fixture.csv').write_text('x,y\n')
    rows = scores['evaluate_coordinates'](points, temp / 'masks/centers', temp / 'empty-centers.csv')
    assert rows[0]['mean_nearest_error_px'] == float('inf') and float(rows[0]['hit_rate_10px']) == 0
    (temp / 'masks/centers/fixture.csv').unlink()
    must_raise(FileNotFoundError, scores['evaluate_coordinates'], points, temp / 'masks/centers', temp / 'missing.csv')
    cv2.imwrite(str(truth / 'missing.png'), final)
    must_raise(FileNotFoundError, scores['evaluate_masks'], truth, temp / 'masks/final_contour_filtering', temp / 'missing-masks.csv')
    print('PASS: notebook image pipeline, batch exports, mask and coordinate edge cases')


def check_evaluation_regressions(temp):
    import cv2
    import numpy as np
    d = notebook_definitions('04_evaluate_masks_and_coordinates.ipynb')
    m = notebook_definitions('03_masks_and_weed_coordinates.ipynb')
    truth, pred = temp / 'regression-truth', temp / 'regression-pred'
    truth.mkdir(); pred.mkdir()
    for name, actual, predicted in [('a', '0,0', '3,4'), ('b', '10,10', '10,10')]:
        (truth / f'{name}.csv').write_text('x,y\n' + actual + '\n')
        (pred / f'{name}.csv').write_text('x,y\n' + predicted + '\n')
    rows = d['evaluate_coordinates'](truth, pred, temp / 'regression.csv')
    assert [row['mean_nearest_error_px'] for row in rows] == [5.0, 0.0]
    bad = temp / 'bad.csv'
    for content in ['u,v\n1,2\n', 'x,y\n1,\n', 'x,y\nnan,2\n', 'x,y\n1,inf\n']:
        bad.write_text(content)
        must_raise(ValueError, d['read_points'], bad)
    bad.write_text('x,y\n')
    assert d['read_points'](bad) == []
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    cv2.imwrite(str(truth / 'a.TIF'), mask)
    cv2.imwrite(str(pred / 'a.PNG'), mask)
    assert set(d['find_masks'](truth)) == {'a'}
    rows = d['evaluate_masks'](truth, pred, temp / 'formats.csv')
    assert rows[-1]['iou'] == 1
    cv2.imwrite(str(truth / 'a.png'), mask)
    must_raise(ValueError, d['find_masks'], truth)
    must_raise(ValueError, m['list_images'], truth)
    detector = notebook_definitions('02_reproduce_detector_experiments.ipynb')
    must_raise(ValueError, detector['image_paths'], truth)
    # Existing six-decimal output remains explicit at the serialization boundary.
    assert d['format_metric_row']({'iou': 1 / 3, 'tp': 2}) == {'iou': '0.333333', 'tp': 2}
    rounded_truth, rounded_pred = temp / 'rounded-truth', temp / 'rounded-pred'
    rounded_truth.mkdir(); rounded_pred.mkdir()
    for name, total, positive in [('one', 3, 1), ('two', 7, 2)]:
        gt = np.full((1, total), 255, dtype=np.uint8)
        pr = np.zeros_like(gt); pr[:, :positive] = 255
        cv2.imwrite(str(rounded_truth / f'{name}.png'), gt)
        cv2.imwrite(str(rounded_pred / f'{name}.png'), pr)
    rows = d['evaluate_masks'](rounded_truth, rounded_pred, temp / 'rounded.csv')
    expected_mean = (float(f'{1/3:.6f}') + float(f'{2/7:.6f}')) / 2
    assert rows[-1]['iou'] == expected_mean
    assert rows[-1]['iou'] != 3 / 10
    print('PASS: multiple coordinate files, mask extensions, duplicate stems and malformed annotations')


def execution_cells(filename, namespace, tag='execution'):
    notebook = json.loads((ROOT / 'notebooks' / filename).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and tag in cell['metadata'].get('tags', []):
            exec(compile(''.join(cell['source']), filename, 'exec'), namespace)


def check_enabled_workflows(temp):
    """Exercise notebook control flow with synthetic data and fake external models."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import pandas as pd
    import matplotlib.pyplot as plt
    workflow = temp / 'enabled workflows'
    workflow.mkdir()
    dataset = workflow / 'dataset'
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'labels').mkdir()
    with (ROOT / 'experiments/splits/items.csv').open() as handle:
        for item in csv.DictReader(handle):
            # Portable frozen names and original box counts; these are test fixtures only.
            from pathlib import PureWindowsPath
            (dataset / 'images' / PureWindowsPath(item['image']).name).touch()
            (dataset / 'labels' / (item['stem'] + '.txt')).write_text(
                '0 0.5 0.5 0.2 0.2\n' * int(item['boxes']))
    captures = []
    common = dict(ROOT=ROOT, pd=pd, plt=plt, display=lambda *args: None,
                  new_run=lambda root, name: new_run(workflow, name),
                  capture_environment=lambda path: captures.append(path),
                  write_local_splits=write_local_splits, check_detector_results=check_detector_results,
                  frozen_assignments=frozen_assignments)
    train = notebook_definitions('02_reproduce_detector_experiments.ipynb')
    train.update(common)
    class Model:
        def __init__(self, source):
            self.model = SimpleNamespace(parameters=lambda: [])
        def train(self, **kwargs):
            run = Path(kwargs['project']) / kwargs['name']
            (run / 'weights').mkdir(parents=True)
            (run / 'weights/best.pt').write_bytes(b'fixture')
            (run / 'results.csv').write_text('epoch,metrics/mAP50-95(B)\n1,0.5\n')
            return SimpleNamespace(results_dict={}, speed={})
        def val(self, **kwargs):
            return SimpleNamespace(results_dict={'metrics/mAP50-95(B)': .5}, speed={})
    train.update(configuration('02_reproduce_detector_experiments.ipynb'))
    train['TRAINING_OPTIONS'].update(epochs=1, imgsz=32, batch=1, device='cpu')
    train.update(RUN_TRAINING=True, CLASS_ZERO_CONFIRMED_TOBACCO=True,
                 DATASET=dataset, RUN_NAME='detector', MODELS=['fixture.pt'])
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                                  'ultralytics': SimpleNamespace(YOLO=Model)}):
        execution_cells('02_reproduce_detector_experiments.ipynb', train)
    assert train['training_complete']
    assert all(row['status'] == 'ok' for row in train['comparison_rows'])
    assert (train['run_dir'] / 'yolo/model_comparison.csv').is_file()
    import cv2
    import numpy as np
    images = workflow / 'images'
    images.mkdir()
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30] = (0,255,0)
    cv2.imwrite(str(images / 'fixture.png'), image)
    cv2.imwrite(str(images / 'second.png'), image)
    masks = notebook_definitions('03_masks_and_weed_coordinates.ipynb')
    masks.update(common)
    detector = SimpleNamespace(names={0: 'tobacco'}, predict=lambda **kwargs: [])
    masks.update(configuration('03_masks_and_weed_coordinates.ipynb'))
    masks['PROCESSING_OPTIONS'].update(imgsz=32, device='cpu')
    masks.update(RUN_PROCESSING=True, SHOW_PREVIEW=False, IMAGES=images,
                 WEIGHTS=train['run_dir'] / 'yolo/fixture/weights/best.pt',
                 RUN_NAME='processing', load_detector=lambda path: detector)
    execution_cells('03_masks_and_weed_coordinates.ipynb', masks)
    assert masks['processing_complete']
    # The batch above ran without any preview cell or synthetic example state.
    masks['SHOW_PREVIEW'] = True
    execution_cells('03_masks_and_weed_coordinates.ipynb', masks, tag='preview')
    must_raise(FileExistsError, execution_cells, '03_masks_and_weed_coordinates.ipynb', masks)
    assert not masks['processing_complete']
    truth, centers = workflow / 'truth', workflow / 'centers'
    truth.mkdir(); centers.mkdir()
    cv2.imwrite(str(truth / 'fixture.TIF'), masks['hsv_mask'](image))
    (centers / 'fixture.csv').write_text('x,y\n19,19\n')
    cv2.imwrite(str(truth / 'second.TIF'), masks['hsv_mask'](image))
    (centers / 'second.csv').write_text('x,y\n22,23\n')
    evaluation = notebook_definitions('04_evaluate_masks_and_coordinates.ipynb')
    evaluation.update(common)
    evaluation.update(RUN_EVALUATION=True, PREDICTIONS=masks['mask_run'] / 'masks',
                      GT_MASKS=truth, GT_CENTERS=centers, RUN_NAME='evaluation', THRESHOLD=127,
                      VARIANTS=masks['VARIANTS'])
    execution_cells('04_evaluate_masks_and_coordinates.ipynb', evaluation)
    assert (evaluation['evaluation_run'] / 'mask_metrics.csv').is_file()
    assert (evaluation['evaluation_run'] / 'coordinate_metrics.csv').is_file()
    with (evaluation['evaluation_run'] / 'coordinate_metrics.csv').open() as handle:
        measured = list(csv.DictReader(handle))
    assert [float(row['mean_nearest_error_px']) for row in measured] == [0.0, 5.0]
    assert len(captures) == 3
    plt.close('all')
    print('PASS: enabled training, processing and independent evaluation notebook cells')


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
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    assert 'run_script(' not in cell.source
                    assert "ROOT / 'scripts'" not in cell.source
            assert len(list((ROOT / 'notebooks').glob('*.ipynb'))) == 4
            client = NotebookClient(notebook, timeout=180, kernel_name='tobacco-validation',
                                    resources={'metadata': {'path': str(ROOT / 'notebooks')}})
            client.execute()
            print(f'PASS: {path.name} (all default cells executed)', flush=True)
        check_helpers(temp)
        check_detector(temp)
        check_processing(temp)
        check_evaluation_regressions(temp)
        check_enabled_workflows(temp)
    assert before == evidence_hashes(), 'Historical evidence changed during validation!'
    print('PASS: historical results and manifests unchanged')


if __name__ == '__main__':
    main()
