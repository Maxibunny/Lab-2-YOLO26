"""Run a synthetic tensor through the EXPORTED LiteRT model; no training data required.

Requires `pip install numpy ai-edge-litert` in Linux/Colab. This checks graph
execution, not detection accuracy or Android runtime compatibility.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        raise SystemExit('Install ai-edge-litert or TensorFlow to run this test.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('android/app/src/main/assets/best.tflite'))
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f'Exported model missing: {args.model}')
    interpreter = Interpreter(model_path=str(args.model.resolve()))
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f'Unexpected tensor count: in={len(inputs)} out={len(outputs)}')
    x, y = inputs[0], outputs[0]
    ish, osh = tuple(map(int, x['shape'])), tuple(map(int,y['shape']))
    print('input:',ish, 'dtype:',x['dtype'])
    print('output:',osh,'dtype:',y['dtype'])
    if ish not in ((1,3,640,640), (1,640,640,3)):
        raise SystemExit('Input shape incompatible with expected 640x640 Android project')
    if osh not in ((1,8,8400),(1,8400,8),(1,300,6)):
        raise SystemExit('Output shape incompatible with implemented decoder')
    if x['dtype'] != np.float32 or y['dtype'] != np.float32:
        raise SystemExit('Unexpected tensor dtype; project currently handles FP32 only')
    dummy = np.full(ish, 114/255, dtype=np.float32)
    interpreter.set_tensor(x['index'], dummy)
    interpreter.invoke()
    pred = interpreter.get_tensor(y['index'])
    print('Synthetic inference OK. Shape:',pred.shape, 'all_finite:',bool(np.isfinite(pred).all()))
    if not np.isfinite(pred).all():
        raise SystemExit('Nonfinite model output detected')
    print('This does not certify mobile runtime, inference speed, or model accuracy.')


if __name__ == '__main__':
    main()
