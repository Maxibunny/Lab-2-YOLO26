"""Recreate data.yaml + docker_data.yaml using stable 4-class HomeObjects mapping.

The original prepared.zip may contain an absolute /content/... path (Colab),
and may omit docker_data.yaml when it was generated after prepared.zip.
Run this AFTER extracting the archive on the local PC.
"""
from pathlib import Path
import argparse
import json

CLASSES = ['chair', 'table', 'bed', 'couch']
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.webp'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('dataset/prepared'))
    a = p.parse_args()
    root = a.dataset.resolve()
    for split in ('train','val'):
        images = root/'images'/split
        labels = root/'labels'/split
        if not images.is_dir() or not labels.is_dir():
            raise SystemExit(f'Missing images/labels folders: {images}, {labels}')
        images_found = sum(f.suffix.lower() in EXTENSIONS for f in images.rglob('*') if f.is_file())
        if images_found == 0:
            raise SystemExit(f'No images found in {images}')
        print(f'{split}: {images_found} images')
    names = '\n'.join(f'  {i}: {name}' for i,name in enumerate(CLASSES))
    def yaml(path: str) -> str:
        return ('path: ' + json.dumps(path,ensure_ascii=False) + '\n'
            + 'train: images/train\nval: images/val\nnames:\n' + names + '\n')
    (root/'data.yaml').write_text(yaml(root.as_posix()),encoding='utf-8')
    (root/'docker_data.yaml').write_text(yaml('/workspace/dataset/prepared'),encoding='utf-8')
    print('UPDATED:',root/'data.yaml')
    print('UPDATED:',root/'docker_data.yaml')
    print('Expected class order:',CLASSES)

if __name__ == '__main__':
    main()
