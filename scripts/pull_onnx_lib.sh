wget https://github.com/microsoft/onnxruntime/releases/download/v1.11.1/onnxruntime-linux-x64-1.11.1.tgz
tar -xvf onnxruntime-linux-x64-1.11.1.tgz
mv onnxruntime-linux-x64-1.11.1/include/* include
mv onnxruntime-linux-x64-1.11.1/lib/* lib
rm -rf onnxruntime-linux-x64-1.11.1*