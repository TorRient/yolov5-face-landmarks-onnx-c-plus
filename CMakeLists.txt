project(yoloface)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories (${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(yoloface src/yolov5face_onnx.cpp)

find_package(OpenCV REQUIRED)
include_directories ("/usr/local/include/opencv4/")
target_link_libraries(yoloface PRIVATE onnxruntime ${OpenCV_LIBRARIES})
