
#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"
#include "hog_visualization.h"


using namespace cv;
using namespace std;


//int main() {
//	list<vector<float>> List = imageToDescriptionList("data/task1/obj1000.jpg");
//	//list<vector<float>> List = imageToDescriptionList("data/task2/test/01/0075.jpg");
//	return 0;
//}

// Function that returns list of DescriptorVectors given an Image
list<vector<float>> imageToDescriptionList(string filepath, bool visualizeHog, bool getRotatedSamples) {

	// Read image and convert to grayscale
	Mat input = imread(filepath, IMREAD_GRAYSCALE);
	Mat imageGrayscale = input;
	// display(input);
	// cvtColor(input, imageGrayscale, COLOR_RGB2GRAY); Not needed anymore

	// Set parameters for HOG-Extraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// Params for HOGDescriptor
	int border = 1;
	Size winSize = Size(64, 64);
	Size blockSize = Size(32, 32);
	Size blockStride = Size(16, 16); //Zu Überlappung von Blöcken: (Überlappung Prozent = 1 - Size/Stride; bei 32/32 => Keine Überlappung, bei 16/32 = 50% ÜL
	Size cellSize = Size(16, 16);
	Size padding = Size(border, border);
	int nbins = 9;

	// Init HOG
	cv::HOGDescriptor hog = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

	// Params for image operations
	bool useFlip = true;

	// Init List of DescriptorVectors that will be returned in the end
	list<vector<float>> imageDescriptorList;
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// Reminder: OriginalGrayscaleImage => Mat imageGrayscale;
	

	imageDescriptorList.push_back(getHOGDescriptorVector(hog, downscaleAndCropImage(imageGrayscale, hog.winSize), visualizeHog));
	if (getRotatedSamples == false) {
		return imageDescriptorList; 
	}
	if (useFlip == true) {
		imageDescriptorList.push_back(getHOGDescriptorVector(hog, downscaleAndCropImage(flipImage(imageGrayscale), hog.winSize), visualizeHog));
	}

	for (int rotCode = 0; rotCode <= 2; rotCode++) { // Iterate over RotCodes (see: cv::RotateFlags)
		Mat tempImage = downscaleAndCropImage(rotateImage(imageGrayscale, rotCode), hog.winSize);
		imageDescriptorList.push_back(getHOGDescriptorVector(hog, tempImage, visualizeHog));
		if (useFlip == true) {
			imageDescriptorList.push_back(getHOGDescriptorVector(hog, flipImage(tempImage), visualizeHog));
		}
	}

	return imageDescriptorList;
}

// Function returns HOGDescriptorVector if a single given image
vector <float> getHOGDescriptorVector(cv::HOGDescriptor& hog, Mat imageGrayscaleResized, bool visualizeHog) {

	vector<float> descriptorVector;
	hog.compute(imageGrayscaleResized, descriptorVector, hog.winSize, Size(0, 0));
	if (visualizeHog == true) {
		visualizeHOG(imageGrayscaleResized, descriptorVector, hog, 5);
	}
	return descriptorVector;
}

//  Function returns squared-crop of the uncropped image
Mat _cropImageToSquare(Mat uncroppedImage) {

	int minCR = std::min(uncroppedImage.cols, uncroppedImage.rows);
	Rect rCrop = Rect((uncroppedImage.cols - minCR) / 2,
		(uncroppedImage.rows - minCR) / 2,
		minCR,
		minCR);
	Mat croppedImage = uncroppedImage(rCrop);
	return croppedImage;
}

// Function returns downscaled and cropped version of the original image
Mat downscaleAndCropImage(Mat originalImage, Size hogWinSize) {
	int minCR = std::min(originalImage.cols, originalImage.rows);
	Mat downsizedAndCroppedImage;
	float rescaleFactor = (float)hogWinSize.width / (float)minCR;
	cv::resize(_cropImageToSquare(originalImage), downsizedAndCroppedImage, cv::Size(), rescaleFactor, rescaleFactor);
	return downsizedAndCroppedImage;
}

// Function to rotate image by given 90*x degrees
Mat rotateImage(Mat originalImage, int rotCodeInt) {
	Mat rotatedImage;
	cv::rotate(originalImage, rotatedImage, rotCodeInt);
	return rotatedImage;
}

// Function to flip image
Mat flipImage(Mat originalImage) {
	Mat flippedImage;
	flip(originalImage, flippedImage, 1);
	return flippedImage;
}
// Function to display image (shortcut for imshow)
void display(Mat image) {
	cv::imshow("Display Window", image);
	cv::waitKey(0);
	return;
}


// TODO?: Paddings??? Aber wofür?
//Mat imageGrayscaleResizedWithPadding;
//copyMakeBorder(imageGrayscaleResized, imageGrayscaleResizedWithPadding, border, border, border, border, BORDER_REPLICATE);