
#ifndef TASK1_H
#define TASK1_H


#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"
#include "hog_visualization.h"

using namespace cv;
using namespace std;

// function that returns all descriptors as a list after performing different rotations and extracting the descriptors
list<vector<float>> imageToDescriptionList(string filepath, bool visualizeHog = false, bool getRotatedSamples = true);

// function that returns single description vector
vector <float> getHOGDescriptorVector(cv::HOGDescriptor& hog, Mat imageGrayscaleResizedWithPadding, bool visualizeHog = false);
Mat _cropImageToSquare(Mat uncroppedImage);
Mat downscaleAndCropImage(Mat originalImage, Size hogWinSize);
Mat rotateImage(Mat originalImage, int rotCodeInt);
Mat flipImage(Mat originalImage);
void display(Mat image);
#endif