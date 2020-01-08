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
	// For text
	const int fontFace = cv::FONT_ITALIC;
	const double fontScale = 0.4;
	const int thickness_font = 1;

	// for window frame color: beautiful coral
	int r = 255, g = 100, b = 100;

	if (confidence >= 0.5)
	{
		Scalar color = Scalar(b, g, r);
		Point topLeft(window.x, window.y + 10); // position of text
		//Point bottomRight(window.x + window.width, window.y + window.height);
		
		// draw rectangle
		rectangle(img, window, color); 

		// draw text
		std::ostringstream ss;
		ss << "class: " << classID << "; " << confidence;
		string text = ss.str();
		putText(img, text, topLeft, fontFace, fontScale, color, thickness_font, 8);
		
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



	// ########## START test drawBox ###################################
	//Mat img = imread("data/task1/obj1000.jpg");
	//Rect window = Rect(10, 10, img.cols - 20, img.rows - 20);
	//int classID = 1;
	//float confidence = 0.5f;
	//drawBox(img, window, classID, confidence);
	//
	//imshow("Image with drawn window", img);
	//
	//waitKey(0);
	// ########### END test drawBox #################################


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