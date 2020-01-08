#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>



#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task3.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void drawBox(Mat img, Rect window, int classID, float confidence)
{
	if (confidence >= 0.5)
	{

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



	cout << "DONE" << endl;
	return 0;
}