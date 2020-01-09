#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <Windows.h>

#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task3.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;


#include <fstream>
/*	zum testen der GT funktion
	// Test GT
	vector<Rect> Prognose;
	vector<int> Klasse;
	string file_loc = "data\task3\gt\0000.gt.txt";
	for (int i = 0; i < 3; i++){
		 Prognose[i] = Rect(1, 2, 3, 4);
		 Klasse[i] = i;
	}
	GT_Test(Prognose, Klasse, file_loc);
	// Test GT end
*/
string ExePath() {
	char buffer[MAX_PATH];
	GetCurrentDirectoryA(256, buffer);
	string::size_type pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}

vector<int> StringToIntArray(string text)
{

	//char str[] = text.c_str;
	//
	//// Returns first token  
	//char* token = strtok(str, " ");
	//
	//// Keep printing tokens while one of the 
	//// delimiters present in str[]. 
	//while (token != NULL)
	//{
	//	printf("%s", token); 
	//	token = strtok(NULL, " ");
	//}

	vector<string> resultString;
	vector<int> resultInt;

	string str = text;

	char* cstr = new char[str.length() + 1];
	std::strcpy(cstr, str.c_str());

	// cstr now contains a c-string copy of str

	char* p = std::strtok(cstr, " ");
	while (p != 0)
	{
		resultString.push_back(p);
		//std::cout << p << '\n';
		p = std::strtok(NULL, " ");
	}

	delete[] cstr;


	for (string s : resultString)
	{

		//cout << s << endl;
		resultInt.push_back(stoi(s));
	}
	return resultInt;
}


vector<float> GT_Test(vector<Rect> Prognose, vector<int> Klasse, string file_loc) {

	//Class_0 x_1 x_2 x_3 x_4
	//Class_1 x_1 x_2 x_3 x_4
	//Class_2 x_1 x_2 x_3 x_4
	vector<Rect> GT_rect; // Auslesen ausm GT file
	vector<int> GT_Klasse; // Auslesen ausm GT file

	fstream newfile;
	vector<string> data;
	newfile.open(file_loc, ios::in); //open a file to perform read operation using file object
	if (newfile.is_open()) {   //checking whether the file is open
		string tp;
		int i = 0;
		while (getline(newfile, tp)) { //read data from file object and put it into string.
			data.push_back(tp); //print the data of the string
			i++;
		}
		//data has 3 rows, row 1 = class 0, row 2 = class 1, row 3 = class 2.
	}
	newfile.close(); //close the file object.

		// Hier müssen die strings in Data in GT_Klasse (erste zahl pro Zeile) und GT_rect (letzten 4 Zahlen pro Zeile) aufgeteilt werden
	for (int i = 0; i < data.size(); i++) {
		string text = data[i];
		vector classes_and_Recs = StringToIntArray(data[i]);
		GT_Klasse.push_back(classes_and_Recs[0]);
		GT_rect.push_back(Rect(classes_and_Recs[1], classes_and_Recs[2], classes_and_Recs[3], classes_and_Recs[4]));
	}

	//	Erinnerung :
	//Rect Kasten = Rect(int left, int top, int right, int bottom); // bzw. (x_1 x_2 x_3 x_4)

	//Vergleiche jedes unserer Rect_prognose mit Rect_Test der gleichen Klasse, falls
	int falsch = Prognose.size();
	int richtig = 0;
	float Cutoff = 0.5; //soll man zum plotten durchstimmen

	for (int i = 0; i < Prognose.size(); i++) {
		for (int j = 0; j < GT_Klasse.size(); j++) {
			if (Klasse[i] == GT_Klasse[j]) {
				Rect Vereinigung = Prognose[i] & GT_rect[j];
				float Quotient = (float)(Vereinigung.height * Vereinigung.width) / ((Prognose[i]).height * (Prognose[i]).width);
				if (Quotient >= Cutoff) {
					falsch--;
					richtig++;
				}
			}
		}
	}

	float Precision = (float) richtig / (richtig + falsch);
	float Recall = (float) richtig / 3;
	if (Recall > 1) Recall = 1;
	vector<float> PrecRec = { Precision, Recall };
	return PrecRec;
}


void filterBoxes(vector<Rect>& Frames, vector<float>& Confidence, vector<int>& classID) {
	// vect<Rect> Frames.Jedes Element dieses vects ist ein Rect mit Konfidenz > 50 % .

	float threshhold = 0.25; // Flächenquotient ab dem wir kleinere rects beim überlapp löschen
	vector<Rect> newframes;
	vector<float> newconfidences;
	vector<int> newClassId;

	for (int i = 0; i < Frames.size(); i++) {
		int signal = 0;
		for (int j = 0; j < Frames.size(); j++) { // wir vergleichen jedes rect mit jedem anderen und löschen bei überlapp größer threshold das rect mit weniger confidence
			
			if (i == j) continue;
			int Nenner_Area = Frames[i].width * Frames[i].height; // Fläche des Rects
			Rect test = Frames[i] & Frames[j]; // returns Rect welches die Vereinigung aus Rect1 ^ Rect2 ist.
			int Zaehler_Area = test.height * test.width; // Fläche des Rects
			if (0 == test.height * test.width) {
				continue;
			}
			float Quotient_Area = (float) Zaehler_Area / Nenner_Area; // Flächenverhältnis der Rects
			if (Quotient_Area > threshhold&& Confidence[i] < Confidence[j]) {
				signal = 1;
				break;
			}
			if (Confidence[i]== Confidence[j]) {
				if (Frames[i].height < Frames[j].height) {
					signal = 1;
					break;
				}
			}
		}
		if (signal == 1) {
			continue;
		}
		newframes.push_back(Frames[i]); // übernehme das rect mit weniger confidence
		newconfidences.push_back(Confidence[i]); // und die dazugehörige confidence
		newClassId.push_back(classID[i]);
	}
	Frames = newframes;
	Confidence = newconfidences;
	classID = newClassId;
}

void drawBoxes(Mat& img, vector<Rect>& window, vector<int>& classID, vector<float>& confidence, int imgNumber)
{
	// For text
	const int fontFace = cv::FONT_ITALIC;
	const double fontScale = 0.4;
	const int thickness_font = 1;

	// for window frame color: beautiful coral
	int r = 255, g = 100, b = 100;


	Scalar color = Scalar(b, g, r);


	for (int i = 0; i < window.size(); i++) {
		switch (classID[i]) {
		case 0: color = Scalar(100, 100, 255); break;
		case 1: color = Scalar(0, 200, 255); break;
		case 2: color = Scalar(255, 100, 100); break;
		default: color = Scalar(255, 255, 255);
		}

		Point topLeft(window[i].x, window[i].y + 10); // position of text
		//Point bottomRight(window.x + window.width, window.y + window.height);

		// draw rectangle
		rectangle(img, window[i], color);

		// draw text
		std::ostringstream ss;
		ss << "class: " << classID[i] << "; " << confidence[i];
		string text = ss.str();
		putText(img, text, topLeft, fontFace, fontScale, color, thickness_font, 8);
		imwrite("results/result" + std::to_string(imgNumber) + ".jpg", img);
	}
}

void SlidingWindow(Mat LoadedImage, int window_size, int stride, std::vector<float>& predictedConfidence, std::vector<int>& predictedClass, vector<Rect>& rects)//, int num_windows = 1)
{
	//init hot
	Size winSize = Size(64, 64);
	Size blockSize = Size(32, 32);
	Size blockStride = Size(16, 16); //Zu Überlappung von Blöcken: (Überlappung Prozent = 1 - Size/Stride; bei 32/32 => Keine Überlappung, bei 16/32 = 50% ÜL
	Size cellSize = Size(16, 16);
	int nbins = 9;

	// Init HOG
	cv::HOGDescriptor hog = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
    //vector<string> filenames = list_dir("data/task3/test/");
    //LoadedImage = imread(filepath, IMREAD_COLOR); 
    // Show what is in the Mat after load

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
	
    // Make padding to fit winsize
    copyMakeBorder(DrawResultGrid, LoadedImagePadded,0 ,(window_size-(DrawResultGrid.rows%window_size))%window_size,0 , (window_size - (DrawResultGrid.cols % window_size)) % window_size, BORDER_REPLICATE);
	
	// Copy again
	Mat imageOnlyResults = LoadedImagePadded.clone();


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


            //// Show grid
            //namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
            //imshow("Step 3 Show Grid", DrawResultGrid);
            //waitKey(10);
            //imwrite("Step3.JPG", DrawResultGrid);

            // Select windows roi
            Mat Roi = LoadedImagePadded(windows);

            //// Show ROI
            //namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
            //imshow("Step 4 Draw selected Roi", Roi);
            //waitKey(10);
            //imwrite("Step4.JPG", Roi);

			// Calculate Descriptor
			Mat downscaledRoi = downscaleAndCropImage(Roi, hog.winSize);
			cv::cvtColor(downscaledRoi, downscaledRoi, cv::COLOR_BGR2GRAY);
			vector<float> descriptor = getHOGDescriptorVector(hog, downscaledRoi);

			Mat predictedOutput;
			vector<float> tempConf;
			vector<int> tempClass;
			Mat roiAsMat = convertToMat(descriptor);
			forestTask3.predict(roiAsMat, predictedOutput, cv::ml::DTrees::PREDICT_MAX_VOTE, tempConf, tempClass);
			//cout << " " << predictedOutput << endl;
			if (predictedOutput.at<int>(0, 0) != 3 && tempConf[0]>=0.4) {
				rectangle(imageOnlyResults, windows, Scalar(255), 1, 8, 0);
				//cout << " " << predictedOutput << endl;
				//namedWindow("Step 5 Draw Match", WINDOW_AUTOSIZE);
				//imshow("Step 5 Draw Match", Roi);
				//waitKey(1);
				imwrite("Step5.JPG", imageOnlyResults);
				predictedConfidence.push_back(tempConf.at(0));
				predictedClass.push_back(tempClass.at(0));
				rects.push_back(windows);
			}


        }
    }

}

Mat convertToMat(vector<float> descriptor) { // + 1 because predict expects [descriptor, classidentifier] format
	Mat asMat = Mat(1, descriptor.size() + 1, CV_32FC1);
	for (int i = 0; i < descriptor.size(); i++) {
		asMat.at<float>(0, i) = descriptor[i];
	}
	asMat.at<float>(0, descriptor.size()) = 0; // Set classidentifier to 0 so it has input
	return asMat;
}

void plotPrecisionRecall(Mat& img, vector<float> precRec, int num) {
	// draw text
	std::ostringstream ss;
	ss << "Recall: " << precRec[1] << "; ";
	string text = ss.str();
	putText(img, text, Point(img.cols-180, img.rows-5), cv::FONT_ITALIC, 0.5, Scalar(0, 255, 0), 1, 8);
	std::ostringstream sss;
	sss << "Precision: " << precRec[0] << "; ";
	text = sss.str();
	putText(img, text, Point(img.cols - 180, img.rows-20), cv::FONT_ITALIC, 0.5, Scalar(0, 255, 0), 1, 8);
	imwrite("results/result" + std::to_string(num) + ".jpg", img);
}
int main() {

	// Setup Forest
	int treeCount = 50;
	int maxDepth = 20;
	int cvFolds = 1;
	int minSampleCount = 10;
	int maxCategories = 15;
	int samples = 800; 
	forestTask3 = RandomForest(treeCount, maxDepth, cvFolds, minSampleCount, maxCategories); // Can also be called with ::create (For tutors)
	
	// Setup Training Data
	vector<list<vector<float>>> trainTemp = getAllDescriptors("data/task3/train/");
	vector<Mat> allTrainingDescriptors = convertToMatVector(trainTemp);
	Mat train;
	
	// Train Forest
	cout << "training our forest ..." << endl;
	mergeVectorToSingleMats(train, allTrainingDescriptors);
	//forestTask3.train(allTrainingDescriptors, samples);
	forestTask3.train(train, samples);


	// Test Forest 

	// Get Testfilenames
	vector<string> filenames = list_dir("data/task3/test/");
	vector<string> gtnames = list_dir("data/task3/gt/");

	for (int k = 0; k < filenames.size(); k++) {
		// init variables
		cout << "calculating for img: " << k + 1 << endl;
		vector<Rect> rects;
		vector<float> confidence;
		vector<int>predictedClasses;

		Mat loadedImage = imread(filenames[k], IMREAD_COLOR);
		for (int i = 8; i < 12; i++) {
			SlidingWindow(loadedImage, 2 * loadedImage.rows / i, 2 * loadedImage.rows / (2 * i), confidence, predictedClasses, rects);
		}

		filterBoxes(rects, confidence, predictedClasses);
		drawBoxes(loadedImage, rects, predictedClasses, confidence, k);
		vector<float> precisionRecall = GT_Test(rects, predictedClasses, gtnames[k]);
		plotPrecisionRecall(loadedImage, precisionRecall, k);
	}
	   	
	cout << "DONE" << endl;
	return 0;
}