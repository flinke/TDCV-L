#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task1.h"

using namespace std;
using namespace cv;

vector<string> list_dir(string path);
vector<list<vector<float>>> getAllDescriptors(string filepath, bool getRotatedSamples = true);
vector<Mat> convertToMatVector(vector<list<vector<float>>>& allDescriptors);
vector<vector<string>> getAllClassPaths(string filepath);
void mergeVectorToSingleMats(Mat& train, Mat& test, vector<Mat>& train_data, vector<Mat>& test_data);
void mergeVectorToSingleMats(Mat& data, vector<Mat>& train_data);
//RandomForest forest;
