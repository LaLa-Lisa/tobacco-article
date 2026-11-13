#!/usr/bin/env python3
"""Check dataset notebooks with isolated synthetic inputs, not real experiments."""
import json
import os
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'dataset_notebooks_simple'


def definitions(name):
    namespace = {'Path': Path, 'json': json}
    notebook = json.loads((FOLDER / name).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), name, 'exec'), namespace)
    return namespace


def check_defaults(temp):
    from ipykernel.kernelspec import install
    os.environ['JUPYTER_CONFIG_DIR'] = str(temp / 'jupyter-config')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(temp / 'jupyter-runtime')
    os.environ['IPYTHONDIR'] = str(temp / 'ipython')
    os.environ['JUPYTER_PATH'] = str(temp / 'share/jupyter')
    install(prefix=str(temp), kernel_name='tobacco-dataset-validation')
    for name, flag in (('03_migrate_dataset.ipynb', 'RUN_MIGRATION'),
                       ('04_create_labels.ipynb', 'RUN_LABELING'),
                       ('05_review_dataset.ipynb', 'RUN_REVIEW'),
                       ('06_inspect_sam3_predictions.ipynb', 'RUN_INFERENCE')):
        notebook = nbformat.read(FOLDER / name, as_version=4)
        nbformat.validate(notebook)
        configuration = [cell for cell in notebook.cells
                         if 'configuration' in cell.metadata.get('tags', [])]
        if len(configuration) != 1:
            raise ValueError(f'Expected one configuration cell in {name}')
        configuration[0].source += f'\n{flag} = False\n'
        NotebookClient(notebook, timeout=120, kernel_name='tobacco-dataset-validation',
                       resources={'metadata': {'path': str(FOLDER)}}).execute()
        print(f'PASS: {name} runs with its workflow disabled')


def write_pair(folder, value, mask, name='1.png', image_folder='data'):
    (folder / image_folder).mkdir(parents=True, exist_ok=True)
    (folder / 'mask').mkdir(parents=True, exist_ok=True)
    image = np.full((*mask.shape, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(folder / image_folder / name)
    Image.fromarray(mask.astype(np.uint8)).save(folder / 'mask' / name)


def must_raise(error, function, *args):
    try:
        function(*args)
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def check_migration(temp):
    migration = definitions('03_migrate_dataset.ipynb')
    source = temp / 'original data'
    version2 = source / 'Tobacco Aerial Dataset V2' / 'Tobacco Aerial Dataset'
    version1 = source / 'Tobacco Aerial Dataset V1' / 'Ready for traintest tobacco data 352x480'
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:3, 1:4] = 2
    mask[4, 5] = 1
    for campaign in range(1, 9):
        folder = version2 / f'Campaign no. {campaign}' / ('Patch Images' if campaign == 2 else 'Patch images')
        write_pair(folder, campaign, mask)
    campaign1 = version2 / 'Campaign no. 1'
    for folder, suffix in (('Original 1080P images', '.jpg'), ('Labelled 1080P images', '.png')):
        directory = campaign1 / folder
        directory.mkdir()
        for number in range(1, 73):
            (directory / f'{number}{suffix}').touch()
    for folder in ('data', 'mask'):
        directory = campaign1 / 'Patch images' / folder
        for number in range(2, 865):
            shutil.copyfile(directory / '1.png', directory / f'{number}.png')
    mapping = migration['campaign1_mask_names'](campaign1)
    assert [mapping[f'{number}.png'] for number in (1, 12, 13, 24, 100, 864)] == [
        '1.png', '12.png', '133.png', '144.png', '856.png', '840.png']
    missing = campaign1 / 'Patch images' / 'mask' / '13.png'
    missing.unlink()
    must_raise(ValueError, migration['campaign1_mask_names'], campaign1)
    shutil.copyfile(campaign1 / 'Patch images' / 'mask' / '1.png', missing)
    original = campaign1 / 'Original 1080P images' / '72.jpg'
    original.rename(original.with_name('73.jpg'))
    must_raise(ValueError, migration['campaign1_mask_names'], campaign1)
    original.with_name('73.jpg').rename(original)
    write_pair(version1 / '119', 2, mask)
    write_pair(version1 / '120' / 'test' / 'RGB22', 9, mask)
    write_pair(version1 / '120', 10, np.ones((6, 8), dtype=np.uint8), image_folder='original')
    pairs = migration['source_pairs'](source)
    assert len(pairs) == 874
    assert len(migration['source_pairs'](source, False)) == 871
    assert next(item for item in pairs if item[0] == campaign1 / 'Patch images' / 'data' / '13.png')[1].name == '133.png'
    run = temp / 'migrated'
    run.mkdir()
    manifest = migration['migrate'](source, run)
    assert len(manifest['images']) == 10
    assert len(manifest['images'][1]['sources']) == 2
    assert 'Tobacco Aerial Dataset V2' in manifest['images'][1]['sources'][0]['image']
    assert sum(item['mask_type'] == 'binary' for item in manifest['images']) == 1
    assert json.loads((run / 'summary.json').read_text())['status'] == 'complete'
    conflict = mask.copy()
    conflict[0, 0] = 2
    write_pair(version1 / '119', 2, conflict)
    bad_run = temp / 'conflict'
    bad_run.mkdir()
    must_raise(ValueError, migration['migrate'], source, bad_run)
    assert not (bad_run / 'summary.json').exists()
    write_pair(version1 / '119', 2, conflict, name='2.png')
    retained = temp / 'retained'
    retained.mkdir()
    retained_manifest = migration['migrate'](source, retained, 'keep_separate')
    assert len(retained_manifest['images']) == 11
    assert json.loads((retained / 'summary.json').read_text())['conflicting_rgb_groups'] == 1
    selected = temp / 'selected'
    selected.mkdir()
    selected_manifest = migration['migrate'](source, selected, 'keep_first')
    assert len(selected_manifest['images']) == 10
    selected_record = selected_manifest['images'][1]
    assert 'Tobacco Aerial Dataset V2' in selected_record['sources'][0]['image']
    assert len(selected_record['discarded_sources']) == 2
    assert selected_record['discarded_sources'][0]['weed_sha256'] != selected_record['weed_sha256']
    assert np.array_equal(np.asarray(Image.open(selected / 'source_masks' / selected_record['file_name'])), mask)
    selected_summary = json.loads((selected / 'summary.json').read_text())
    assert selected_summary['conflicting_rgb_groups'] == 1
    assert selected_summary['discarded_conflict_sources'] == 2
    assert selected_summary['source_pairs'] == 875
    (version1 / '119' / 'mask' / '1.png').unlink()
    must_raise(ValueError, migration['source_pairs'], source)
    print('PASS: source discovery, deduplication, provenance, conflict and missing-mask checks')
    return run


def check_checkpoint_loading():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import torch
    import sam3.model_builder

    load_model = definitions('04_create_labels.ipynb')['load_model']
    model = MagicMock()
    model.backbone.vision_backbone.trunk.blocks = []
    model.cuda.return_value.eval.return_value = model
    weights = {'detector.example': 'weight',
               'detector.backbone.vision_backbone.sam2_convs.example': 'excluded',
               'other.example': 'excluded'}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
         patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
         patch.object(sam3.model_builder, 'build_sam3_image_model', return_value=model), \
         patch.object(torch, 'load') as load:
        load.return_value = weights
        loaded, resolution, epoch = load_model('original.pt')
        assert loaded is model and resolution == 1008 and epoch is None
        model.load_state_dict.assert_called_with({'example': 'weight'}, strict=True)
        load.assert_called_with('original.pt', map_location='cpu', weights_only=True)
        load.return_value = {'model': weights, 'options': {'resolution': 640}, 'epoch': 12}
        loaded, resolution, epoch = load_model('trained.pt')
        assert loaded is model and resolution == 640 and epoch == 12
        model.load_state_dict.assert_called_with({'example': 'weight'}, strict=True)
        load.return_value = SimpleNamespace()
        must_raise(ValueError, load_model, 'invalid.pt')
    print('PASS: original and trained checkpoint loading and metadata')


def check_labels(temp, migration):
    from unittest.mock import patch
    labeling = definitions('04_create_labels.ipynb')
    shape = (6, 8)
    original = np.zeros(shape, dtype=np.uint8)
    original[1:3, 1:4] = 2
    tobacco = np.zeros(shape, dtype=bool)
    tobacco[1:3, 2:5] = True
    lower = np.zeros(shape, dtype=bool)
    lower[2:4, 4:6] = True
    weed = np.zeros(shape, dtype=bool)
    weed[0:2, 0:3] = True
    semantic, instances = labeling['assemble_labels'](
        shape, [(0.6, lower), (0.9, tobacco)], [], original)
    assert semantic[1, 2] == 1 and semantic[1, 1] == 2
    assert len(instances) == 3 and instances[0]['score'] == 0.9
    encoded, boxes = labeling['encode_instances'](1, instances, 8, 6, 1)
    assert len(boxes) == 2 and boxes[0] == '0 0.43750000 0.33333333 0.37500000 0.33333333'
    reconstructed = np.zeros(shape, dtype=np.uint8)
    for annotation in encoded:
        decoded = mask_api.decode(annotation['segmentation']).astype(bool)
        assert not reconstructed[decoded].any()
        reconstructed[decoded] = annotation['category_id']
    assert np.array_equal(reconstructed, semantic)
    binary_semantic, binary_instances = labeling['assemble_labels'](
        shape, [(0.5, tobacco)], [(0.99, weed)])
    assert binary_semantic[1, 2] == 1 and binary_semantic[0, 0] == 2
    assert [item['category_id'] for item in binary_instances] == [1, 2]
    empty, empty_instances = labeling['assemble_labels'](shape, [], [])
    assert not empty.any() and not empty_instances

    def predict(processor, image, mask_type, threshold):
        assert threshold == 0.5
        if mask_type == 'binary':
            return [], [(0.99, weed)]
        return [(0.9, tobacco)], []

    output = temp / 'labels'
    output.mkdir()
    progress = []

    def track_progress(records, desc, unit):
        assert desc == 'Labeling images' and unit == 'image'
        for record in records:
            yield record
            progress.append(record['id'])

    with patch('tqdm.auto.tqdm', track_progress):
        coco = labeling['generate_dataset'](migration, output, None, predict, {'confidence': 0.5})
    assert len(coco['images']) == 10
    assert progress == [image['id'] for image in coco['images']]
    assert len(list((output / 'labels').glob('*.txt'))) == 10
    binary = next(item for item in json.loads((migration / 'manifest.json').read_text())['images']
                  if item['mask_type'] == 'binary')
    assert not (output / 'labels' / (Path(binary['file_name']).stem + '.txt')).read_text()
    assert np.asarray(Image.open(output / 'semantic_masks' / binary['file_name']))[0, 0] == 2
    assert json.loads((output / 'summary.json').read_text())['status'] == 'complete'
    check_review(output, binary['id'])
    altered_mask = migration / 'source_masks' / binary['file_name']
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(altered_mask)
    invalid = temp / 'invalid-labels'
    invalid.mkdir()
    must_raise(ValueError, labeling['generate_dataset'], migration, invalid, None, predict,
               {'confidence': 0.5})
    assert not (invalid / 'summary.json').exists()
    print('PASS: overlap, instances, COCO RLE, YOLO boxes and fake-inference integration')


def check_review(dataset, empty_image_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    review = definitions('05_review_dataset.ipynb')
    image, mask, boxes = review['read_example'](dataset, 1)
    assert image.shape == (6, 8, 3) and mask.shape == (6, 8)
    assert set(np.unique(mask)) == {0, 1, 2} and len(boxes) == 1
    figure = review['plot_example'](image, mask, boxes, 1)
    assert len(figure.axes) == 3 and len(figure.axes[2].patches) == 1
    assert len(figure.axes[1].images) == 2
    plt.close(figure)
    _, _, empty_boxes = review['read_example'](dataset, empty_image_id)
    assert not empty_boxes
    must_raise(ValueError, review['read_example'], dataset, 0)
    must_raise(FileNotFoundError, review['read_example'], dataset, 999999)
    box_path = dataset / 'labels' / 'image_000001.txt'
    original = box_path.read_text(encoding='utf-8')
    box_path.write_text('0 0.5 0.5 2 0.1\n', encoding='utf-8')
    must_raise(ValueError, review['read_example'], dataset, 1)
    box_path.write_text(original, encoding='utf-8')
    print('PASS: image review, colored masks, boxes, empty and invalid labels')


def check_reference_prompt():
    from types import SimpleNamespace

    inspect = definitions('06_inspect_sam3_predictions.ipynb')
    normalize = inspect['normalize_reference_box']
    box = (300, 165, 175, 180)
    expected = [387.5 / 480, 255 / 352, 175 / 480, 180 / 352]
    assert np.allclose(normalize(box, (480, 352)), expected)
    for invalid in ((-1, 0, 1, 1), (0, 0, 0, 1), (470, 0, 11, 1),
                    (0, 350, 1, 3), (0, 0, float('nan'), 1), (1, 2, 3)):
        must_raise(ValueError, normalize, invalid, (480, 352))

    class FakeModel:
        def _encode_prompt(self, backbone_out, find_input, geometric_prompt):
            assert find_input == 'find stage'
            assert backbone_out['image'].name == 'reference'
            return ('encoded reference', geometric_prompt, backbone_out)

        def _get_dummy_prompt(self):
            return 'empty target geometry'

    class FakeProcessor:
        def __init__(self):
            self.model = FakeModel()
            self.find_stage = 'find stage'
            self.fail = False

        def set_image(self, image):
            return {'backbone_out': {'image': image}}

        def add_geometric_prompt(self, state, box, label):
            assert label is True and np.allclose(box, expected)
            state['geometric_prompt'] = 'reference geometry'
            return state

        def _forward_grounding(self, state):
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                state['backbone_out'], self.find_stage, state['geometric_prompt'])
            assert prompt == 'encoded reference' and prompt_mask == 'reference geometry'
            assert backbone_out['image'].name == 'target'
            assert state['geometric_prompt'] == 'empty target geometry'
            if self.fail:
                raise RuntimeError('synthetic inference failure')
            width, height = backbone_out['image'].size
            return {'masks': np.zeros((1, height, width), dtype=bool), 'scores': [0.9]}

    reference = SimpleNamespace(name='reference', size=(480, 352))
    target = SimpleNamespace(name='target', size=(64, 32), width=64, height=32)
    processor = FakeProcessor()
    result = inspect['predict_with_reference'](processor, reference, target, box)
    assert result['masks'].shape == (1, 32, 64) and result['scores'] == [0.9]
    assert processor.model._encode_prompt.__func__ is FakeModel._encode_prompt
    processor.fail = True
    must_raise(RuntimeError, inspect['predict_with_reference'], processor, reference, target, box)
    assert processor.model._encode_prompt.__func__ is FakeModel._encode_prompt

    labeling = definitions('04_create_labels.ipynb')
    normalize = labeling['normalize_reference_box']
    assert np.allclose(normalize(box, reference.size), expected)
    for invalid in ((-1, 0, 1, 1), (0, 0, 0, 1), (470, 0, 11, 1),
                    (0, 350, 1, 3), (0, 0, float('nan'), 1), (1, 2, 3)):
        must_raise(ValueError, normalize, invalid, reference.size)

    class FakeMask:
        def __init__(self, height, width):
            self.array = np.ones((height, width), dtype=bool)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class LabelProcessor(FakeProcessor):
        def __init__(self):
            super().__init__()
            self.text_prompts = []
            self.reference_forwards = 0

        def _forward_grounding(self, state):
            super()._forward_grounding(state)
            self.reference_forwards += 1
            width, height = state['backbone_out']['image'].size
            return {'masks': [FakeMask(height, width)], 'scores': [0.9]}

        def reset_all_prompts(self, state):
            assert state['backbone_out']['image'].name == 'target'

        def set_text_prompt(self, prompt, state):
            self.text_prompts.append(prompt)
            width, height = state['backbone_out']['image'].size
            return {'masks': [FakeMask(height, width), FakeMask(height, width)],
                    'scores': [0.8, 0.1]}

    processor = LabelProcessor()
    reference_prompt = labeling['encode_reference_prompt'](processor, reference, box)
    for mask_type in ('three_class', 'binary'):
        tobacco, weed = labeling['predict_image'](
            processor, target, mask_type, 0.5, reference_prompt=reference_prompt)
        assert len(tobacco) == 1 and tobacco[0][1].shape == (32, 64)
        assert len(weed) == (1 if mask_type == 'binary' else 0)
    assert processor.reference_forwards == 2 and processor.text_prompts == ['weed plant']
    assert processor.model._encode_prompt.__func__ is FakeModel._encode_prompt
    processor.fail = True
    must_raise(RuntimeError, labeling['predict_with_reference'], processor,
               processor.set_image(target), reference_prompt)
    assert processor.model._encode_prompt.__func__ is FakeModel._encode_prompt
    processor.fail = False
    tobacco, weed = labeling['predict_image'](processor, target, 'binary', 0.5,
                                              tobacco_prompt='custom tobacco')
    assert len(tobacco) == len(weed) == 1
    assert processor.text_prompts[-2:] == ['custom tobacco', 'weed plant']
    print('PASS: reference transfer, text and weed prompts, filtering, and adapter cleanup')


def main():
    with tempfile.TemporaryDirectory(prefix='tobacco-dataset-validation-') as directory:
        temp = Path(directory)
        check_defaults(temp)
        check_checkpoint_loading()
        check_reference_prompt()
        migration = check_migration(temp)
        check_labels(temp, migration)


if __name__ == '__main__':
    main()
