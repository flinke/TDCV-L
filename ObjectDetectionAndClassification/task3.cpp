#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>



#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task3.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void drawBox(Mat& img, Rect& window, int classID, float confidence)
{
	// just if you want to customize the window frame color; else it's red
	int r = 0;
	int g = 0;
	int b = 0;
	if (confidence >= 0.5 && classID < 6)
	{
		r = 255;
		//Point topLeft(window.x, window.y);
		//Point bottomRight(window.x + window.width, window.y + window.height);
		rectangle(img, window, Scalar(r , g, b));
	}
}

int main() {

	// Setup Forest
	int treeCount = 350;
	int maxDepth = 15;
	int cvFolds = 1;
	int minSampleCount = 1;
	int maxCategories = 15;
	forest = RandomForest(treeCount, maxDepth, cvFolds, minSampleCount, maxCategories);

	// Setup Training Data
	vector<list<vector<float>>> trainTemp = getAllDescriptors("data/task3/train/");
	vector<Mat> allTrainingDescriptors = convertToMatVector(trainTemp);
	Mat train;

	// Train Forest
	cout << "training our forest ..." << endl;
	forest.train(allTrainingDescriptors, 100);


	// ########## START test drawBox ###################################
	Mat img = imread("data/task1/obj1000.jpg");
	Rect window = Rect(0, 0, img.cols, img.rows);
	int classID = 1;
	float confidence = 0.5f;
	drawBox(img, window, classID, confidence);
	// ########### END test drawBox #################################

	cout << "DONE" << endl;
	return 0;
}