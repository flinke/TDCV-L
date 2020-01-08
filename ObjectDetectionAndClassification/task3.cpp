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

	if (confidence >= 0.5 && classID < 3) // background class won't be drawn
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

int SlidingWindow(int window_size, int stride)//, int num_windows = 1)
{
    //vector<string> filenames = list_dir("data/task3/test/");
    Mat LoadedImage;
    //LoadedImage = imread(filepath, IMREAD_COLOR); 
    LoadedImage = imread("data/task3/test/0000.jpg", IMREAD_COLOR);
    // Show what is in the Mat after load
    if (!LoadedImage.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    


    // namedWindow("Step 1 image loaded", WINDOW_AUTOSIZE);
    //imshow("Step 1 image loaded", LoadedImage);
    // waitKey(1000);

    imwrite("Step1.JPG", LoadedImage);

    // Parameters of your slideing window

    int windows_n_rows = window_size;
    int windows_n_cols = window_size;
    // Step of each window
    Mat DrawResultGrid = LoadedImage.clone();
    Mat LoadedImagePadded;
    // Make padding to fit 64,64
    copyMakeBorder(DrawResultGrid, LoadedImagePadded,0 ,(window_size-(DrawResultGrid.rows%window_size))%window_size,0 , (window_size - (DrawResultGrid.cols % window_size)) % window_size, BORDER_REPLICATE);

    for (int row = 0; row <= LoadedImagePadded.rows - windows_n_rows; row += stride)
    {
        for (int col = 0; col <= LoadedImagePadded.cols - windows_n_cols; col += stride)
        {
            Rect windows(col, row, windows_n_rows, windows_n_cols);
            Mat DrawResultHere = LoadedImagePadded.clone();

            // Draw only rectangle
            rectangle(DrawResultHere, windows, Scalar(255), 1, 8, 0);
            // Draw grid
            rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

            // Show  rectangle
            //namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
            //imshow("Step 2 draw Rectangle", DrawResultHere);
            //waitKey(100);
            //imwrite("Step2.JPG", DrawResultHere);


            // Show grid
            namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
            imshow("Step 3 Show Grid", DrawResultGrid);
            waitKey(100);
            imwrite("Step3.JPG", DrawResultGrid);

            // Select windows roi
            Mat Roi = LoadedImagePadded(windows);

            //Show ROI
            namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
            imshow("Step 4 Draw selected Roi", Roi);
            waitKey(100);
            imwrite("Step4.JPG", Roi);


        }
    }

}

int main() {

    SlidingWindow(64, 64);

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

	//for (string filename : filenames) {
	//	vector<Rect> boxesWithMatches;
	//	vector<float> confidencesForMatches;
	//	void slidingWindows(filename, &boxesWithMatches, &confidencesForMatches, winParams);

	//	for (Rect singleBox : boxesWithMatches) {
	//		drawBox(filename, rect, confidence);
	//	}
	//}
	cout << "DONE" << endl;
	return 0;
}