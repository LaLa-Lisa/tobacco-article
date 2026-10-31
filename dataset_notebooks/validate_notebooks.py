"""CPU checks of notebook definitions and inert execution; no SAM weights or training."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset_notebooks'))
from runtime import fresh_run, read_json, write_json, sha256


def definitions(name):
    namespace = {'Path': Path, 'np': np, 'Image': Image, 'ROOT': ROOT,
                 'json': json, 'read_json': read_json, 'write_json': write_json,
                 'sha256': sha256, 'SAM3_REVISION': 'synthetic-test'}
    notebook = nbformat.read(ROOT / 'dataset_notebooks' / name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, name, 'exec'), namespace)
    return namespace


def raises(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f'{function.__name__} did not raise {error.__name__}')


def evidence():
    return {str(p): sha256(p) for folder in ('experiments/results', 'experiments/splits', 'figures/publication')
            for p in (ROOT / folder).rglob('*') if p.is_file()}


def run_checks(temp):
    from pycocotools import mask as mask_api
    a = definitions('02_review_instances.ipynb')
    e = definitions('04_evaluate_and_examples.ipynb')
    semantic = np.zeros((16, 20), np.uint8)
    semantic[2:8, 2:8] = 1
    semantic[10:15, 12:18] = 2
    labels, classes = a['draft_components'](semantic)
    a['validate_instances'](labels, classes, np.zeros_like(labels, bool))
    raises(ValueError, a['validate_instances'], labels, {1: 1})
    raises(ValueError, a['validate_instances'], labels, classes, semantic > 0)
    # Holes and disconnected fragments must survive COCO RLE round trips.
    labels[4:6, 4:6] = 0
    labels[0, 0] = 1
    source = temp / 'raw'; source.mkdir()
    reviews = temp / 'reviews'; rows = []
    for index, split in enumerate(('train', 'val', 'test'), 1):
        image = source / f'{index}.png'
        Image.fromarray(np.full((16, 20, 3), index * 30, np.uint8)).save(image)
        row = {'key': f'c{index:02d}_0001', 'id': index, 'image': image.name,
               'image_sha256': sha256(image), 'split': split, 'campaign': index,
               'width': 20, 'height': 16}
        rows.append(row)
        a['save_review'](reviews, row, labels, classes, np.zeros_like(labels, bool),
                         'fixture reviewer', True, 'synthetic fixture only')
        restored = a['load_review'](reviews, row)
        assert np.array_equal(restored[0], labels) and restored[3]['approved']
    audit = {'rows': rows, 'split_review': {'approved': True}, 'cross_split_duplicates': []}
    exported = a['export_reviewed'](audit, source, reviews, temp / 'export')
    raises(FileExistsError, a['export_reviewed'], audit, source, reviews, exported)
    coco = read_json(exported / 'test/annotations.json')
    assert len(coco['annotations']) == 2
    assert np.array_equal(mask_api.decode(coco['annotations'][0]['segmentation']), labels == 1)
    predictions = [{k: ann[k] for k in ('image_id', 'category_id', 'segmentation')}
                   | {'score': 1.} for ann in coco['annotations']]
    perfect = e['evaluate_predictions'](exported / 'test/annotations.json', predictions)
    assert abs(perfect['mask_AP'] - 1) < 1e-8
    assert all(c['semantic_IoU'] == 1 for c in perfect['classes'].values())
    empty = e['evaluate_predictions'](exported / 'test/annotations.json', [])
    assert empty['mask_AP'] == 0
    assert all(c['semantic_IoU'] == 0 for c in empty['classes'].values())
    # Two deterministic output examples, including their lossless mask artifacts.
    import matplotlib
    matplotlib.use('Agg')
    second_image = {**coco['images'][0], 'id': 4, 'campaign': 4, 'source_key': 'fixture_second'}
    coco['images'].append(second_image)
    coco['annotations'] += [{**ann, 'id': ann['id'] + 100, 'image_id': 4}
                            for ann in list(coco['annotations'])]
    write_json(exported / 'test/annotations.json', coco)
    examples_predictions = predictions + [{**ann, 'image_id': 4} for ann in predictions]
    e['save_examples'](exported / 'test', examples_predictions, temp / 'examples', {'fixture': True})
    assert len(list((temp / 'examples').glob('*.png'))) == 2
    for path in (temp / 'examples').glob('*.npz'):
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays['masks'].shape == (2, 16, 20)
    a['save_review'](reviews, rows[-1], labels, classes, np.zeros_like(labels, bool), '', False, 'revoked')
    raises(ValueError, a['export_reviewed'], audit, source, reviews, temp / 'incomplete')
    bad = np.full((16, 20), 3)
    raises(ValueError, a['validate_instances'], bad, {3: 9})
    # Non-square masks exercise all inverse rotations and reflections.
    for rotation in range(4):
        for flip in (False, True):
            assert np.array_equal(e['restore_mask'](e['transform_image'](labels, rotation, flip), rotation, flip), labels)
    coverage = np.zeros((73, 95), bool)
    for x0, y0, x1, y1 in e['tile_boxes'](95, 73, 32):
        coverage[y0:y1, x0:x1] = True
    assert coverage.all()
    assert e['tile_boxes'](10, 8, 32) == [(0, 0, 10, 8)]
    raises(ValueError, e['tile_boxes'], 10, 10, 0)
    m = labels == 1
    candidates = [{'mask': m, 'category_id': 1, 'score': .9},
                  {'mask': m.copy(), 'category_id': 1, 'score': .8},
                  {'mask': m.copy(), 'category_id': 2, 'score': .7},
                  {'mask': labels == 2, 'category_id': 2, 'score': .6}]
    selected = e['suppress_and_resolve'](candidates, .5, .1)
    assert len(selected) == 2 and not (selected[0]['mask'] & selected[1]['mask']).any()
    encoded = e['encode_predictions'](candidates, 1)
    encoded_selected = e['suppress_and_resolve'](encoded, .5, .1)
    assert len(encoded_selected) == 2
    for dense, encoded_item in zip(selected, encoded_selected):
        assert np.array_equal(dense['mask'], mask_api.decode(encoded_item['segmentation']))
    assert e['suppress_and_resolve']([], .5, .1) == []
    fresh_run(temp, 'fixture')
    raises(FileExistsError, fresh_run, temp, 'fixture')
    raises(ValueError, fresh_run, temp, '../escape')
    # Editor construction tests widget wiring without requiring a browser.
    editor = a['review_widget'](Image.fromarray(np.zeros((16, 20, 3), np.uint8)),
                               semantic, labels, classes, np.zeros_like(labels, bool), lambda *x: {})
    assert editor.width == 20
    print('PASS: approval gate, revisions, COCO holes/fragments, AP, empty predictions, transforms, tiling, conflicts, editor, collisions')


def main():
    before = evidence()
    with tempfile.TemporaryDirectory() as directory:
        run_checks(Path(directory))
        temporary = Path(directory)
        for key, folder in [('JUPYTER_CONFIG_DIR', 'jupyter-config'),
                            ('JUPYTER_RUNTIME_DIR', 'jupyter-runtime'),
                            ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR', 'matplotlib')]:
            os.environ[key] = str(temporary / folder)
        os.environ['JUPYTER_PATH'] = str(temporary / 'share/jupyter')
        from ipykernel.kernelspec import install
        install(prefix=str(temporary), kernel_name='tobacco-sam3-validation')
        for path in sorted((ROOT / 'dataset_notebooks').glob('*.ipynb')):
            nb = nbformat.read(path, as_version=4); nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == 'code': compile(cell.source, str(path), 'exec')
            NotebookClient(nb, timeout=180, kernel_name='tobacco-sam3-validation', resources={'metadata': {'path': str(ROOT)}},
                           record_timing=False).execute()
            print('PASS: inert fresh-kernel execution:', path.name)
    assert evidence() == before, 'Historical evidence changed'
    print('PASS: historical evidence unchanged')
    print('NOT RUN: real SAM 3 imports, CUDA forward/backward, native Windows kernels, real training, trained predictions.')


if __name__ == '__main__':
    main()
