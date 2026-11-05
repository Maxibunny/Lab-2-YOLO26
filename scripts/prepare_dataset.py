"""Make reproducible train/val splits from annotated raw folder, preserving group distribution."""
import argparse
import collections
import hashlib
import shutil
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
CLASSES = ['chair', 'table', 'bed', 'couch']


def run(images_root: Path, labels_root: Path, out: Path, val_fraction: float):
    if not (0 < val_fraction < 0.5):
        raise ValueError('Expected 0 < val_fraction < 0.5')
    groups = collections.defaultdict(list)
    for p in images_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            rel = p.relative_to(images_root)
            label = labels_root / rel.with_suffix('.txt')
            if not label.exists():
                raise FileNotFoundError(f'No annotation for {p}: expected {label}. Run autolabel then review.')
            for number, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                cols = line.split()
                if len(cols) != 5:
                    raise ValueError(f'{label}:{number}: must contain class cx cy w h')
                if cols[0] not in ('0', '1', '2', '3'):
                    raise ValueError(f'{label}:{number}: class must be 0..3')
                xywh = [float(x) for x in cols[1:]]
                if not (all(0 <= x <= 1 for x in xywh) and xywh[2] > 0 and xywh[3] > 0):
                    raise ValueError(f'{label}:{number}: coordinates must be normalized to [0,1]')
            group = rel.parts[0] if len(rel.parts) > 1 else '_root'
            groups[group].append((p, label, rel))
    if not groups:
        raise SystemExit(f'No images in {images_root}')
    if out.exists():
        shutil.rmtree(out)
    n_train = n_val = 0
    for group, files in sorted(groups.items()):
        files.sort(key=lambda f: hashlib.sha256(f[2].as_posix().encode()).digest())
        nval = max(1, round(len(files) * val_fraction)) if len(files) > 1 else 0
        for k, (img, lab, rel) in enumerate(files):
            split = 'val' if k < nval else 'train'
            # stable unique output names across all subfolders
            key = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:10]
            stem = f'{rel.stem[:70]}_{key}'
            image_dst = out / 'images' / split / (stem + img.suffix.lower())
            label_dst = out / 'labels' / split / (stem + '.txt')
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, image_dst)
            shutil.copy2(lab, label_dst)
            if split == 'train': n_train += 1
            else: n_val += 1
        print(f'{group}: {len(files)} images ({nval if len(files)>1 else 0} val)')
    if n_train < 4 or n_val < 4:
        raise SystemExit(f'Too few images: train={n_train}, val={n_val}. Aim for dozens/hundreds of real annotated images per class.')
    yaml = f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n'
    for i, name in enumerate(CLASSES): yaml += f'  {i}: {name}\n'
    (out / 'data.yaml').write_text(yaml, encoding='utf-8')
    # Fixed path inside Docker volume; Windows host paths from data.yaml do not exist there.
    (out / 'docker_data.yaml').write_text(yaml.replace(f'path: {out.resolve().as_posix()}', 'path: /workspace/dataset/prepared'), encoding='utf-8')
    print(f'Split: train={n_train}, val={n_val}; config={out / "data.yaml"}')
    print('Use an INDEPENDENT human-labeled test set, not pseudo-labeled validation images, to demonstrate mAP@0.5.')

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--images', type=Path, default=Path('dataset/raw/images'))
    a.add_argument('--labels', type=Path, default=Path('dataset/raw/labels'))
    a.add_argument('--out', type=Path, default=Path('dataset/prepared'))
    a.add_argument('--val-fraction', type=float, default=0.2)
    x = a.parse_args()
    run(x.images, x.labels, x.out, x.val_fraction)
