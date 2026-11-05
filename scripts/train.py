"""Fine-tune YOLO26n; save weights and validation metrics to mounted runs directory."""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('dataset/prepared/data.yaml'))
    p.add_argument('--runs', type=Path, default=Path('runs'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='auto')
    p.add_argument('--weights', default='yolo26n.pt')
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--name', default='furniture_yolo26n_640_v2')
    args = p.parse_args()
    if not args.data.exists(): raise SystemExit(f'Missing {args.data}; run prepare_dataset.py')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 0 if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    args.runs.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, seed=args.seed, deterministic=True, workers=args.workers, patience=args.patience,
                amp=True, save=True, verbose=True,
                project=str(args.runs.resolve()), name=args.name, exist_ok=True, plots=True)
    best = args.runs.resolve() / args.name / 'weights' / 'best.pt'
    if not best.is_file(): raise FileNotFoundError(best)
    shutil.copy2(best, args.runs / 'best.pt')
    # measure validation on the annotated validation split; test is independent and not provided.
    metrics = YOLO(str(best)).val(data=str(args.data.resolve()), imgsz=args.imgsz, device=device,
                                        project=str(args.runs.resolve()), name='validation', exist_ok=True, plots=True)
    summary = {'map50_validation': float(metrics.box.map50), 'map50_95_validation': float(metrics.box.map),
               'epochs_requested': args.epochs, 'image_size': args.imgsz, 'weights': str(best), 'device': str(device), 'patience': args.patience, 'seed': args.seed, 'batch': args.batch, 'run_name': args.name}
    (args.runs / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('VALIDATION METRICS: ', summary)
    print('NOT the instructor withheld test mAP: test data must be independently labeled and evaluated separately.')

if __name__ == '__main__': main()
