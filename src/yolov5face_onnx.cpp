#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include "yolov5face_onnx.h"

using namespace cv;
using namespace std;

using clock_time = std::chrono::system_clock;
using sec = std::chrono::duration<double>;


static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

YOLO::YOLO(Net_config config){
	cout << "Net use " << config.modelPath << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
    std::string instanceName{"Yolov5face c++"};
    this->mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                    instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    // sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = std::make_shared<Ort::Session>(*this->mEnv, config.modelPath.c_str(),
                                            sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    const char* inputName = this->session->GetInputName(0, allocator);
    const char* outputName = this->session->GetOutputName(0, allocator);

    this->inputShape = this->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    this->outputShape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    this->inputNames = std::vector<const char*>{inputName};
    this->outputNames = std::vector<const char*>{outputName};
	this->outputShape.at(1) = 25200;
	this->outputShape.at(2) = 16;
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<cv::Point2f> landmark)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);
	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	for (int i = 0; i < 5; i++)
	{
		
    	circle(frame, landmark[i], 1, Scalar(0, 255, 0), -1);
	}
}

void YOLO::detect(std::vector<cv::Mat> & images){
	// Change batchsize
	std::vector<cv::Mat> images_processed;
	int batchsize = images.size();
	for (int i=0; i<batchsize; i++){
		Mat blob;
		cv::dnn::blobFromImage(images[i], blob, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
		images_processed.push_back(blob);
	}

    this->inputShape.at(0) = batchsize;
    this->outputShape.at(0) = batchsize;
    size_t inputTensorSize = vectorProduct(inputShape);
    std::vector<float> inputTensorValues(inputTensorSize);
    size_t outputTensorSize= vectorProduct(outputShape);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

	// Copy image processed to InputTensorValues
    for (int i=0; i < batchsize; ++i)
    {
        std::copy(images_processed[i].begin<float>(),
                  images_processed[i].end<float>(),
                  inputTensorValues.begin() + i * inputTensorSize / batchsize);
    }

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        this->inputShape.data(), this->inputShape.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        this->outputShape.data(), this->outputShape.size()));

    this->session->Run(Ort::RunOptions{nullptr}, this->inputNames.data(),
                inputTensors.data(), 1 /*Number of inputs*/, outputNames.data(),
                outputTensors.data(), 1 /*Number of outputs*/);

	/////generate proposals
	for(int b=0; b < batchsize; b++){
		vector<float> confidences;
		vector<Rect> boxes;
		vector< vector<cv::Point2f>> landmarks;
		float ratioh = (float)images[b].rows / this->inpHeight, ratiow = (float)images[b].cols / this->inpWidth;
		int n = 0, q = 0, i = 0, j = 0, nout = 16, row_ind = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,x1,y1, ... ,x5,y5,face_score
		for (n = 0; n < 3; n++)
		{
			int num_grid_x = (int)(this->inpWidth / this->stride[n]);
			int num_grid_y = (int)(this->inpHeight / this->stride[n]);
			for (q = 0; q < 3; q++)    ///anchor
			{
				const float anchor_w = this->anchors[n][q * 2];
				const float anchor_h = this->anchors[n][q * 2 + 1];
				for (i = 0; i < num_grid_y; i++)
				{
					for (j = 0; j < num_grid_x; j++)
					{
						float* pdata = &outputTensorValues[b*25200*16] + row_ind * nout;
						float box_score = sigmoid_x(pdata[4]);
						if (box_score > this->objThreshold)
						{
							float face_score = sigmoid_x(pdata[15]);
							//if (face_score > this->confThreshold)
							//{ 
							float cx = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * this->stride[n];  ///cx
							float cy = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * this->stride[n];   ///cy
							float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  ///h
							int left = (cx - 0.5*w)*ratiow; 
							int top = (cy - 0.5*h)*ratioh;   

							confidences.push_back(face_score);
							boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
							vector<cv::Point2f> landmark;
							for (k = 5; k < 15; k+=2)
							{
								const int ind = k - 5;
								int x = (int)(pdata[k] * anchor_w + j * this->stride[n])*ratiow;
								int y = (int)(pdata[k + 1] * anchor_h + i * this->stride[n])*ratioh;
								cv::Point2f point = cv::Point2f{ (float)x, (float)y};
								landmark.push_back(point);
							}
							landmarks.push_back(landmark);
							//}
						}
						row_ind++;
					}
				}
			}
		}
		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		vector<int> indices;
		std::cout << "LM: " << landmarks.size() << std::endl;
		std::cout << "boxes: " << boxes.size() << std::endl;
		cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
		cv::Mat dst;
		std::cout << "LM: " << landmarks.size() << std::endl;
		std::cout << "BOX: " << indices.size() << std::endl;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];
			this->drawPred(confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, images[b], landmarks[idx]);
		}
	}
};

int main()
{
	Net_config yolo_nets = {0.3, 0.5, 0.3, "../model/yolov5s-face.onnx"};  ///choice = [yolov5s, yolov5m, yolov5l]
	YOLO yolo_model(yolo_nets);
	string imgpath = "../selfie.jpg";
	Mat srcimg = imread(imgpath);
	vector<Mat> images;
	images.push_back(srcimg);
  	const auto before2 = clock_time::now();
	yolo_model.detect(images);
	const sec duration2 = clock_time::now() - before2;
	std::cout << "The postprocessing takes " << duration2.count() << "s"
              << std::endl;
	cv::imwrite("output.jpg", srcimg);
}