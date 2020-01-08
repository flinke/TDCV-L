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
	// just if you want to customize the window frame color; else it's red
	int r = 0, g = 0, b = 0;
	if (confidence >= 0.5)
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
	RandomForest forest = RandomForest(treeCount, maxDepth, cvFolds, minSampleCount, maxCategories); // Can also be called with ::create (For tutors)

	// Setup Training Data
	vector<list<vector<float>>> trainTemp = getAllDescriptors("data/task3/train/");
	vector<Mat> allTrainingDescriptors = convertToMatVector(trainTemp);
	Mat train;

	// Train Forest
	cout << "training our forest ..." << endl;
	forest.train(allTrainingDescriptors, 100);

	// Get Testfilenames
	vector<string> filenames = list_dir("data/task3/test/");

	/* Pseudo:::@@
	
	For each file in filenames:
		Get Vector<Rect> & their confidence: -> slidingWindows
		
		For each Rect in Vecor:
			-> drawBox	
	
	@@:::Pseudo */

	cout << "DONE" << endl;
	return 0;
}