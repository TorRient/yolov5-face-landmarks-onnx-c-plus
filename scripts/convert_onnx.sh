git clone https://github.com/deepcam-cn/yolov5-face
cp custom_export_onnx.py yolov5-face
cd yolov5-face
python custom_export_onnx.py --weights ../model/yolov5s-face.pt
rm -rf ../yolov5-face