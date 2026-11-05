"""Evaluate on independent instructor's fully human-labeled test dataset YAML."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument('--weights', type=Path, default=Path('runs/best.pt'))
p.add_argument('--test-data', type=Path, required=True, help='YOLO YAML with test split and 4 class names')
p.add_argument('--out', type=Path, default=Path('runs/test_metrics.json'))
a = p.parse_args()
if not a.test_data.is_file(): raise SystemExit(f'Missing independent test YAML: {a.test_data}')
metric = YOLO(str(a.weights)).val(data=str(a.test_data.resolve()), split='test', imgsz=640, plots=True)
result = {'map50_test': float(metric.box.map50), 'map50_95_test': float(metric.box.map),
          'target_0_8_met': bool(metric.box.map50 >= 0.8)}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(result)
