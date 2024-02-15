#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define log(X) std::cout<<X<<std::endl;


//constants 
const float INPUT_HEIGHT = 640.0;
const float INPUT_WIDTH = 640.0;
const float SCORE_THRESHOLD = 0.45;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

//text parameters
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1; 

//colors
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);

std::vector<std::string> readClasses(std::string &class_path) {
	std::vector<std::string> classes;
	std::ifstream ifs(std::string(class_path).c_str());
	std::string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}
	return classes;
}

std::vector<cv::Mat> pre_process(cv::Mat &frame, cv::dnn::Net model) {

	
	cv::Mat blob;
	cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

	model.setInput(blob);

	std::vector<cv::Mat>outputs;

	model.forward(outputs, model.getUnconnectedOutLayersNames());

	return outputs;
}

void draw_label(cv::Mat& frame, std::string label, int left, int top) {
	int baseline;
	//display the label at the top of the box
	cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseline);
	top = cv::max(top, label_size.height);
	//top left corner
	cv::Point tlc = cv::Point(left, top);
	//botton right corner
	cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseline);
	//draw black rectangule 
	cv::rectangle(frame, tlc, brc, BLACK, cv::FILLED);
	//put the label on the black rectangule
	cv::putText(frame, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

cv::Mat post_process(cv::Mat &input_frame, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_names) {
	std::vector<int> classes_id;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	///resizing factor
	float x_factor = input_frame.cols / INPUT_WIDTH;
	float y_factor = input_frame.rows / INPUT_HEIGHT;
	float *data = (float *)outputs[0].data;

	//dimensions
	const int dimensions = 85;
	const int rows = 25200;
	for (int i = 0; i < rows; i++) {
		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD) 
		{
			float *class_scores = data + 5;
			cv::Mat scores(1, class_names.size(), CV_32FC1, class_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD) 
			{
				confidences.push_back(confidence);
				classes_id.push_back(class_id.x);
				//center
				float cx = data[0];
				float cy = data[1];
				//box dimension 
				float w = data[2];
				float h = data[3];
				//bounding box coordinates
				int left =  int( (cx - 0.5 * w) * x_factor);
				int top = int( (cy - 0.5 * h) * y_factor);
				int height = int(h * y_factor);
				int width = int(w * x_factor);
				boxes.push_back(cv::Rect(left, top, width, height)); 
			}

		}
		//after iterating all the classes jump to the next row
		data += 85;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences,SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++) {
		int idx = indices[i];
		cv::Rect box = boxes[i];
		int left = box.x;
		int top = box.y;
		int width = box.width;
		int height = box.height;

		//draw bounding box 
		cv::rectangle(input_frame, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);
		std::string label = cv::format("%.2f", confidences[idx]);
		label = class_names[classes_id[idx]] + ":" + label;
		draw_label(input_frame, label, left, top);
	}
	return input_frame;
}



int main(int argc, char* argv[])
{

	std::string path_classes = "./models/coco-classes.txt";
	std::vector<std::string> classes;
	if (classes.empty()) {
		classes = readClasses(path_classes);
	}
	//loading the pre-trained model 
	std::string model_path = "./models/YOLOV5/yolov5s.onnx";
	cv::dnn::Net model_net = cv::dnn::readNet(model_path);
	
	//reading camera
	cv::VideoCapture cap;
	cap.open(0);
	cv::Mat frame;

	while(cap.isOpened()) {

		cap.read(frame);
		std::vector<cv::Mat> detections = pre_process(frame, model_net);
		cv::Mat frame_detected = post_process(frame, detections, classes);
		std::vector<double> layersTimes;
		double freq = cv::getTickFrequency() / 1000;
		double t = model_net.getPerfProfile(layersTimes) / freq;
		std::string label = cv::format("Inference time : %.2f ms", t);
		cv::putText(frame_detected, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
		cv::imshow("Real time frame object detection",frame_detected);

		if (cv::waitKey(5) >= 0) {
			break;
		}
	}
	cap.release();
	cv::destroyAllWindows();
	return 0;
}
