#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include "onnxruntime_cxx_api.h"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  // Object Confidence threshold
	std::string modelPath;
};

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(std::vector<cv::Mat>& images);
	cv::Mat faceAlign(cv::Mat src_image, cv::Mat& landmark_from);
private:
	const float anchors[3][6] = { {4,5,  8,10,  13,16}, {23,29,  43,55,  73,105},{146,217,  231,300,  335,433} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;

	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<cv::Point2f> landmark);
	void sigmoid(Mat* out, int length);

	// ORT Environment
    std::shared_ptr<Ort::Env> mEnv;
    // Session
    std::shared_ptr<Ort::Session> session;

    int embedding_size;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    size_t inputTensorSize;
    size_t outputTensorSize;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
};