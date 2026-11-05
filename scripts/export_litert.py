"""Export trained 4-class YOLO26n model to raw FP32 LiteRT output (NMS outside model)."""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, default=Path('runs/best.pt'))
    p.add_argument('--assets', type=Path, default=Path('android/app/src/main/assets'))
    p.add_argument('--imgsz', type=int, default=640)
    x = p.parse_args()
    if not x.weights.exists(): raise SystemExit(f'Missing weights: {x.weights}; train first')
    model = YOLO(str(x.weights))
    names = [model.names[i] for i in range(len(model.names))]
    assert names == ['chair', 'table', 'bed', 'couch'], f'Unexpected class ordering: {names}'
    # nms=None selects conventional one-to-many predictions; NMS performed in Android.
    exported = Path(model.export(format='litert', imgsz=x.imgsz, nms=None, device='cpu'))
    assert exported.exists() and exported.suffix == '.tflite', exported
    x.assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, x.assets / 'best.tflite')
    (x.assets / 'labels.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    print(f'LiteRT model exported: {x.assets / "best.tflite"} ({(x.assets / "best.tflite").stat().st_size} bytes)')
    print('Expected input: [1,640,640,3] FP32; output: raw [1,8,N] or [1,N,8]. Check in Android logs before interpreting.')

if __name__ == '__main__': main()
