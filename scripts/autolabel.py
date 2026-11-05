"""Draft pseudo-labels for variant 5 using pretrained COCO YOLO26.
NOT ground truth: verify EVERY image, especially val/test.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

# COCO ids -> local ids; furniture classes.
COCO_TO_LOCAL = {56: 0, 60: 1, 59: 2, 57: 3}
EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, default=Path('dataset/raw/images'))
    p.add_argument('--labels', type=Path, default=Path('dataset/raw/labels'))
    p.add_argument('--weights', default='yolo26n.pt')
    p.add_argument('--conf', type=float, default=0.18)
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing annotations (CAUTION).')
    args = p.parse_args()
    images = sorted(f for f in args.images.rglob('*') if f.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f'No images found under {args.images}. Put images in chair/table/bed/couch subfolders.')
    args.labels.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    done = kept = empty = 0
    for f in images:
        target = args.labels / f.relative_to(args.images).with_suffix('.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            kept += 1
            continue
        result = model.predict(str(f), conf=args.conf, imgsz=640, verbose=False)[0]
        lines = []
        if result.boxes is not None:
            for b in result.boxes:
                coco_id = int(b.cls.item())
                if coco_id not in COCO_TO_LOCAL:
                    continue
                x, y, w, h = [float(t) for t in b.xywhn[0].tolist()]
                lines.append(f'{COCO_TO_LOCAL[coco_id]} {x:.7f} {y:.7f} {w:.7f} {h:.7f}')
        target.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        done += 1
        if not lines:
            empty += 1
    print(f'Auto-annotated: {done}; preserved existing: {kept}; empty: {empty}')
    print('IMPORTANT: these are unverified PSEUDO-LABELS, not ground truth! Manually inspect/correct labels before training and metric reporting.')

if __name__ == '__main__':
    main()
