#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include <string.h>

#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task1.h"
#include "task2.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void SlidingWindow(Mat LoadedImage, int window_size, int stride, std::vector<float>& predictedConfidence, std::vector<int>& predictedClass, vector<Rect>& rects);
void drawBoxes(Mat& img, vector<Rect>& window, vector<int>& classID, vector<float>& confidence, int imgNumber);
Mat convertToMat(vector<float> descriptor);
void filterBoxes(vector<Rect>& Frames, vector<float>& Confidence, vector<float>& classID);
RandomForest forestTask3;
vector<float> GT_Test(vector<Rect> Prognose, vector<int> Klasse, string file_loc);
vector<int> StringToIntArray(string text);