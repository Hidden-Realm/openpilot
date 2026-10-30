# World map detector model

`yolo11n_448x256.onnx` is Ultralytics YOLO11n (COCO-pretrained, 80 classes), exported to ONNX at a fixed
448x256 input (opset 17, no NMS). On the device it is compiled for the GPU with tinygrad at build time
(`../compile_detector.py`, `../../SConscript`), the same way the driver monitoring model is.

Only these classes are used: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign.

License: the YOLO11 weights are published by Ultralytics under the AGPL-3.0 license
(https://github.com/ultralytics/ultralytics/blob/main/LICENSE). This fork is public and its source is open,
which is what AGPL asks for. Anyone redistributing this branch in a closed product would need to replace the
model or get an Ultralytics license.
