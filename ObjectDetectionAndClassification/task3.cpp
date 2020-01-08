#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>



#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task1.h"
#include "task2.h"
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
