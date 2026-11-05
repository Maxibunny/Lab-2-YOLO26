"""Convert official Ultralytics HomeObjects-3K ZIP to furniture-only YOLO dataset.

Source https://docs.ultralytics.com/datasets/detect/homeobjects-3k/
Source class IDs: 0=bed, 1=sofa, 2=chair, 3=table, remaining 4-11 excluded.
Target class IDs: 0=chair, 1=table, 2=bed, 3=couch.

Supports source layouts `images/train`+`labels/train` and
`train/images`+`train/labels`; also val/valid; preserves original train/val split.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

SOURCE_TO_TARGET = {0: 2, 1: 3, 2: 0, 3: 1}
CLASSES = ['chair', 'table', 'bed', 'couch']
SOURCE_CLASSES = ['bed', 'sofa', 'chair', 'table']
IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def find_layout(folder: Path):
    """Return (root, split, prefix-mode) for a supported tree."""
    found = []
    for root in [folder, *[d for d in folder.rglob('*') if d.is_dir()]]:
        for structure in ('split-first', 'type-first'):
            for val_name in ('val', 'valid'):
                if structure == 'split-first':
                    pairs = [(root/'train/images', root/'train/labels'), (root/val_name/'images', root/val_name/'labels')]
                else:
                    pairs = [(root/'images/train', root/'labels/train'), (root/'images'/val_name, root/'labels'/val_name)]
                if all(im.is_dir() and lab.is_dir() for im, lab in pairs):
                    found.append((root, pairs))
    if len(found) != 1:
        raise ValueError(f'Expected one HomeObjects-3K dataset tree, found {len(found)}. '
                         'Expected train/images + train/labels + valid/images + valid/labels, '
                         'or images/train + labels/train + images/val + labels/val.')
    return found[0]


def check_metadata(root: Path):
    for name in ('data.yaml', 'HomeObjects-3K.yaml'):
        path = root/name
        if path.exists():
            cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            values = cfg.get('names')
            if values is not None:
                names = list(values.values()) if isinstance(values, dict) else values
                if [str(x).lower() for x in names[:4]] != SOURCE_CLASSES:
                    raise ValueError(f'Unexpected class IDs in {path}: {names[:4]}')
            break


def convert_label(raw_path: Path, target: Path, tally: Counter) -> bool:
    kept = []
    for line_num, line in enumerate(raw_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        bits = line.split()
        if len(bits) != 5:
            raise ValueError(f'{raw_path}:{line_num}: expected 5 columns; got {len(bits)}')
        source_id = int(bits[0])
        if not 0 <= source_id <= 11:
            raise ValueError(f'{raw_path}:{line_num}: invalid HomeObjects class {source_id}')
        if source_id not in SOURCE_TO_TARGET:
            continue
        coords = [float(v) for v in bits[1:]]
        if not (all(0 <= v <= 1 for v in coords) and coords[2] > 0 and coords[3] > 0):
            raise ValueError(f'{raw_path}:{line_num}: invalid normalized bbox')
        target_id = SOURCE_TO_TARGET[source_id]
        tally[CLASSES[target_id]] += 1
        kept.append(str(target_id) + ' ' + ' '.join(bits[1:]))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')
    return bool(kept)


def import_homeobjects(source_zip: Path, out: Path):
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    with tempfile.TemporaryDirectory(prefix='homeobjects-') as d:
        folder = Path(d)
        with zipfile.ZipFile(source_zip) as arc:
            for obj in arc.infolist():
                to = (folder/obj.filename).resolve()
                if not to.is_relative_to(folder.resolve()):
                    raise ValueError(f'Unsafe ZIP path: {obj.filename}')
                if ((obj.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f'Symlinks are not allowed in ZIP: {obj.filename}')
            arc.extractall(folder)
        root, pairs = find_layout(folder)
        check_metadata(root)
        if out.exists():
            shutil.rmtree(out)
        tally = Counter()
        image_count = Counter()
        for split, (images, labels) in zip(('train', 'val'), pairs):
            for im in sorted(images.rglob('*')):
                if not im.is_file() or im.suffix.lower() not in IMAGES:
                    continue
                relative = im.relative_to(images)
                label = labels/relative.with_suffix('.txt')
                # Empty images are allowed, but every source image must have a label file
                # in this pre-annotated dataset to avoid silently missing ground truth.
                if not label.is_file():
                    raise ValueError(f'Missing YOLO annotation for {im}: expected {label}')
                sha = hashlib.sha256(f'{split}/{relative.as_posix()}'.encode()).hexdigest()[:10]
                file_name = f'{split}_{im.stem[:65]}_{sha}'
                dest_img = out/'images'/split/(file_name + im.suffix.lower())
                dest_lab = out/'labels'/split/(file_name + '.txt')
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(im, dest_img)
                convert_label(label, dest_lab, tally)
                image_count[split] += 1
        if not image_count['train'] or not image_count['val']:
            raise ValueError('Both train and validation image sets must be nonempty.')
        if not all(tally[name] > 0 for name in CLASSES):
            raise ValueError(f'One or more furniture classes are missing: {tally}')
        yaml_text = 'path: {}\ntrain: images/train\nval: images/val\nnames:\n'.format(out.resolve().as_posix())
        yaml_text += ''.join(f'  {idx}: {name}\n' for idx, name in enumerate(CLASSES))
        (out/'data.yaml').write_text(yaml_text, encoding='utf-8')
        (out/'docker_data.yaml').write_text(yaml_text.replace(
            out.resolve().as_posix(), '/workspace/dataset/prepared'), encoding='utf-8')
        print('Imported official HomeObjects-3K (furniture only).')
        print('Images:', dict(image_count), 'Labeled objects:', dict(tally))
        print('YOLO config:', out/'data.yaml')
        return dict(image_count), dict(tally)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('dataset/prepared'))
    args = p.parse_args()
    import_homeobjects(args.zip, args.out)
