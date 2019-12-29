
#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"




int main(){
    cv::Mat im = cv::imread("data/task1/obj1000.jpg");
	cv::imshow("Display Window", im);
	cv::waitKey(0);

	//Fill Code here

    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */
    return 0;
}