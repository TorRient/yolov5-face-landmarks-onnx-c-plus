# Yolov5-face using ONNX Runtime on C++

Face landmarks example using ONNX Runtime C++, support dynamic batchsize.

## Dependencies

* ONNX Runtime
* CMake 3.13.3
* OpenCV 4.6.0

## ONNX Runtime Installation
```bash
bash scripts/pull_onnx_lib.sh
```

## Opencv Installation: Recommend version 4.6.0
https://github.com/opencv/opencv

## Download pretrained yolov5s-face
https://github.com/deepcam-cn/yolov5-face

## Convert model Pytorch to ONNX
```bash
$ bash scripts/convert_onnx.sh
```

## Usages
```bash
$ mkdir build & cd build
$ cmake ..
$ make
$ ./yoloface
```

## References
Many thanks to these following projects
https://github.com/deepcam-cn/yolov5-face
https://github.com/hpc203/yolov5-face-landmarks-opencv-v2
