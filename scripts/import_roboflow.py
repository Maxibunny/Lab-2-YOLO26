"""Import Roboflow YOLOv8 object-detection ZIP into this project's 4-class dataset.

Keeps the provider's train/valid/test splits, converts class indices to:
0 chair, 1 table, 2 bed, 3 couch, and produces a ready-to-train data.yaml.

Only use verified human annotations; Roboflow dataset content is unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

CLASSES = ('chair', 'table', 'bed', 'couch')
ALIASES = {
    'chair': 'chair', 'chairs': 'chair',
    'table': 'table', 'dining table': 'table', 'dining-table': 'table',
    'bed': 'bed', 'beds': 'bed',
    'sofa': 'couch', 'sofas': 'couch', 'couch': 'couch', 'couches': 'couch',
}
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'test': 'test'}


def names_from_yaml(path: Path) -> dict[int, str]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    raw_names = config.get('names')
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(k): v for k, v in raw_names.items()}
    else:
        raise ValueError('data.yaml must have names as list or dictionary')
    converted = {}
    for old, text in names.items():
        name = ALIASES.get(str(text).strip().lower())
        if name is None:
            raise ValueError(f'Unsupported source class {text!r}. Dataset must have only Chair/Table/Bed/Sofa.')
        converted[old] = CLASSES.index(name)
    if set(converted.values()) != set(range(4)):
        raise ValueError(f'All four classes are required. Source labels: {names!r}')
    return converted


def load_label(path: Path, remap: dict[int, int]) -> str:
    if not path.exists():
        # Empty targets are valid; they might represent background images.
        return ''
    lines = []
    for i, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{i}: Expected YOLO detect class cx cy w h, got {raw!r}')
        old = int(parts[0])
        if old not in remap:
            raise ValueError(f'{path}:{i}: Unexpected class id {old}')
        coords = [float(x) for x in parts[1:]]
        cx, cy, w, h = coords
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'{path}:{i}: Coordinates must be normalized to [0,1]')
        lines.append(f'{remap[old]} ' + ' '.join(f'{x:.8f}' for x in coords))
    return '\n'.join(lines) + ('\n' if lines else '')


def find_dataset_root(folder: Path) -> Path:
    candidates = [f.parent for f in folder.rglob('data.yaml') if f.is_file()]
    candidates = [p for p in candidates if (p / 'train' / 'images').is_dir()]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one Roboflow dataset data.yaml/train/images, found {len(candidates)} in {folder}')
    return candidates[0]


def import_dataset(source_zip: Path, out: Path) -> dict[str, int]:
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    with tempfile.TemporaryDirectory() as td:
        extracted = Path(td)
        with zipfile.ZipFile(source_zip) as z:
            for item in z.infolist():
                target = (extracted / item.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError(f'Unsafe path in ZIP: {item.filename}')
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlinks are not supported in the input ZIP')
            z.extractall(extracted)
        root = find_dataset_root(extracted)
        remap = names_from_yaml(root / 'data.yaml')
        if out.exists():
            shutil.rmtree(out)
        counts = {'train': 0, 'val': 0, 'test': 0}
        for source_split, split in SPLITS.items():
            images = root / source_split / 'images'
            labels = root / source_split / 'labels'
            if not images.is_dir():
                continue
            for img in sorted(images.rglob('*')):
                if not img.is_file() or img.suffix.lower() not in EXTS:
                    continue
                rel = img.relative_to(images)
                digest = hashlib.sha256((source_split + '/' + rel.as_posix()).encode()).hexdigest()[:12]
                name = f'{source_split}_{img.stem[:60]}_{digest}'
                target_img = out / 'images' / split / (name + img.suffix.lower())
                target_label = out / 'labels' / split / (name + '.txt')
                target_img.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, target_img)
                target_label.write_text(load_label(labels / rel.with_suffix('.txt'), remap), encoding='utf-8')
                counts[split] += 1
        if not counts['train'] or not counts['val']:
            raise ValueError(f'Need train and validation images, got {counts}')
        header = f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n'
        if counts['test']:
            header += 'test: images/test\n'
        header += 'names:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASSES))
        (out / 'data.yaml').write_text(header, encoding='utf-8')
        (out / 'docker_data.yaml').write_text(header.replace(
            f'path: {out.resolve().as_posix()}', 'path: /workspace/dataset/prepared'), encoding='utf-8')
        print(f'Imported Roboflow YOLOv8 ZIP, remapped to chair/table/bed/couch: {counts}')
        print(f'Data config: {out / "data.yaml"}')
        return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, required=True, help='Roboflow Download Dataset: YOLOv8 zip')
    parser.add_argument('--out', type=Path, default=Path('dataset/prepared'))
    args = parser.parse_args()
    import_dataset(args.zip, args.out)
