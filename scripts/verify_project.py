#!/usr/bin/env python3
"""Verify local artifacts downloaded from the completed Colab session.
Usage: python scripts/verify_project.py
"""
from pathlib import Path
import json, sys, hashlib, zipfile

ROOT = Path(__file__).resolve().parent.parent
names = ["chair", "table", "bed", "couch"]
required = {
    "training weights": "runs/best.pt",
    "validation metrics": "runs/metrics.json",
    "Ultralytics commit": "runs/ultralytics_commit.txt",
    "LiteRT model": "android/app/src/main/assets/best.tflite",
    "labels": "android/app/src/main/assets/labels.txt",
    "dataset config": "dataset/prepared/data.yaml",
    "Docker dataset config": "dataset/prepared/docker_data.yaml",
    "training images": "dataset/prepared/images/train",
    "validation images": "dataset/prepared/images/val",
    "training labels": "dataset/prepared/labels/train",
    "validation labels": "dataset/prepared/labels/val",
}
errors = 0
for label, rel in required.items():
    p = ROOT / rel
    ok = p.is_dir() and any(p.iterdir()) if rel.startswith("dataset/prepared/") and rel.count("/") == 3 else p.is_file()
    if ok and p.is_file() and p.stat().st_size == 0: ok = False
    print(("OK " if ok else "MISSING ") + label + " => " + rel)
    if not ok: errors += 1

labels = ROOT / 'android/app/src/main/assets/labels.txt'
if labels.is_file():
    actual = labels.read_text(encoding='utf-8').splitlines()
    if actual != names:
        print('ERROR: wrong labels order:',actual)
        errors += 1
metrics = ROOT/'runs/metrics.json'
if metrics.is_file():
    try:
        o = json.loads(metrics.read_text(encoding='utf-8'))
        print('validation mAP50 =', o.get('map50_validation'), '(not independent test)')
    except (OSError, ValueError) as exc:
        print('ERROR invalid metrics JSON:', exc); errors += 1
model = ROOT/'android/app/src/main/assets/best.tflite'
if model.is_file():
    print('best.tflite bytes =',model.stat().st_size, 'SHA256 =', hashlib.sha256(model.read_bytes()).hexdigest())
print('RESULT:', 'OK: files present' if errors == 0 else f'INCOMPLETE: {errors} checks failed')
sys.exit(0 if errors == 0 else 1)
