
#ifndef VISUALIZEHOG_H
#define VISUALIZEHOG_H


#include <opencv2/opencv.hpp>

void visualizeHOG(cv::Mat img, std::vector<float>& feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);

#endif