
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>



#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task1.h"
#include "task2.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;


template<class ClassifierType>
void performanceEval(ClassifierType& treeOrForest, Mat data) {

	// Masks 
	Rect rCrop = Rect(0, 0, data.cols - 1, data.rows);
	Rect rvCrop = Rect(data.cols - 1, 0, 1, data.rows);

	// Get response Vector
	Mat testResponseVector;
	data(rvCrop).convertTo(testResponseVector, CV_32S);

	Mat predictOutput;
	treeOrForest.predict(data(rCrop), predictOutput, cv::ml::DTrees::PREDICT_MAX_VOTE);

	// Output to console
	//cout << "predictOutput = " << endl << " " << predictOutput.t() << endl << endl; //Output transposed
	//cout << "groundTruth = " << endl << " " << testResponseVector.t() << endl << endl; //Output transposed

	// Calc Misses vs Ground truth
	int hits = 0;
	int misses = 0;
	for (int i = 0; i < testResponseVector.rows; i++) {
		if (predictOutput.at<int>(i, 0) == testResponseVector.at<int>(i, 0)) {
			hits++;
		}
		else {
			misses++;
		}
	}
	cout << "Hits: " << hits << "; Misses: " << misses << "; Accuracy: " << (float)hits / (misses + hits) << endl;

}


void testDTrees(vector<Mat>  train_data, vector<Mat> test_data) {

	//int num_classes = 6;
	Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
	tree->setCVFolds(1); //Geht nicht mit anderen Zahlen größer als 1 (warum?)
	tree->setMaxCategories(10); //Standard 10;
	tree->setMaxDepth(INT8_MAX);
	tree->setMinSampleCount(2); //Standard 10; Weniger = besser (zumindest hier?)

	// Fügt alle Mats aus dem vector<Mat> zusammen (unschön geschrieben)
	Mat train;
	Mat test;
	mergeVectorToSingleMats(train, test, train_data, test_data); //quick and dirty

	// Masks
	Rect rCropForTrain = Rect(0, 0, train.cols - 1, train.rows); //Mask to get only the data
	Rect rvCropForTrain = Rect(train.cols - 1, 0, 1, train.rows); //Mask to get only the classLables

	// Extract responseVector and convert to CV_32S (else will not be recognized as a lable)
	Mat trainResponseVector;
	train(rvCropForTrain).convertTo(trainResponseVector, CV_32S);

	// Train with data + reponseVector
	cout << "training our tree ..." << endl;
	tree->train(train(rCropForTrain), cv::ml::ROW_SAMPLE, trainResponseVector);

	performanceEval<cv::ml::DTrees>(*tree, train);
	performanceEval<cv::ml::DTrees>(*tree, test);
}


void testForest(vector<Mat> training_data, vector<Mat> test_data) {

	cout << "training our forest ..." << endl;

	//int num_classes = 6;
	int treeCount = 350;
	int maxDepth = 15;
	int cvFolds = 1;
	int minSampleCount = 1;
	int maxCategories = 15;

	forest = RandomForest(treeCount, maxDepth, cvFolds, minSampleCount, maxCategories); //treecount, maxdepth (default: intmax), cvfolds (default: 10), minsamplecount (default: 10), maxcategories (default: 10)
	//Ptr<RandomForest> forestPtr = &forest;

	Mat train;
	Mat test;
	mergeVectorToSingleMats(train, test, training_data, test_data); //quick and dirty

	// Mask
	Rect rCropForTrain = Rect(0, 0, train.cols - 1, train.rows); //Mask to get only the data
	Rect rvCropForTrain = Rect(train.cols - 1, 0, 1, train.rows); //Mask to get only the classLables

	// Extract responseVector and convert to CV_32S (else will not be recognized as a lable)
	Mat trainResponseVector;
	train(rvCropForTrain).convertTo(trainResponseVector, CV_32S);

	Mat predictOutput;
	//forest.train(train, 150);
	forest.train(training_data, 100); // Bei 300 Samples + 350 Trees auch gute ergebnisse; Dauert aber ewig lol

	performanceEval<RandomForest>(forest, train);
	performanceEval<RandomForest>(forest, test);
}


int main() {

	//single descriptors can be accessed via vector[class(0-5)][image*8 or image] depending on if rotatedSamples = true/false
	//TODO Convert into training data
	cout << "computing descriptors ..." << endl;
	vector<list<vector<float>>> trainTemp = getAllDescriptors("data/task2/train/");
	vector<list<vector<float>>> testTemp = getAllDescriptors("data/task2/test/", false);
	vector<Mat> allTrainingDescriptors = convertToMatVector(trainTemp);
	vector<Mat> allTestingDescriptors = convertToMatVector(testTemp);
	Mat train;
	Mat test;
	//mergeVectorToSingleMats(train, test, allTrainingDescriptors, allTestingDescriptors);
	//testDTrees(allTrainingDescriptors, allTestingDescriptors);
	testForest(allTrainingDescriptors, allTestingDescriptors);

	cout << "DONE" << endl;

	return 0;
}

Mat getRandomSample(Mat& entireData, int sampleSize) {
	return entireData;
}

void mergeVectorToSingleMats(Mat& train, Mat& test, vector<Mat>& train_data, vector<Mat>& test_data) {

	int num_classes = train_data.size(); // Is this equal to the size of train_data / test_data? else set it back to 6

	// Fügt alle Mats aus dem vector<Mat> zusammen (sehr schön geschrieben) -> Deskriptoren werden zeilenweise aneinandergereiht
	train = train_data[0];
	test = test_data[0];
	for (int i = 1; i < num_classes; i++)
	{
		vconcat(train, train_data[i], train);
		vconcat(test, test_data[i], test);
	}
}



//function that gets all descriptors for all (output = [class[0], class[1], ...] with class[0] = [hogforimg1, hogforimg2, ...]
vector<list<vector<float>>> getAllDescriptors(string filepath, bool getRotatedSamples) {
	// Get all img_paths for all training classes
	vector<vector<string>> imgOfEachClass = getAllClassPaths(filepath);

	vector<list<vector<float>>> descriptorsForAllClasses;
	for (vector<string> filenameArrayOfSingleClass : imgOfEachClass) {
		list<vector<float>> temp;
		for (string filename : filenameArrayOfSingleClass) {
			list<vector<float>> descriptors = imageToDescriptionList(filename, false, getRotatedSamples);
			for (vector<float> singleDescriptor : descriptors) {
				temp.push_back(singleDescriptor);
			}
		}
		descriptorsForAllClasses.push_back(temp);
	}
	return descriptorsForAllClasses;
}

vector<Mat> convertToMatVector(vector<list<vector<float>>>& allDescriptors) {

	vector<Mat> descriptorsAsMatVector;

	for (float classNumber = 0; classNumber < allDescriptors.size(); classNumber++)
	{
		list<vector<float>> singleClass = allDescriptors[classNumber];
		int rows = singleClass.size();
		int cols = singleClass.front().capacity();
		Mat tempMat = Mat(rows, cols + 1, CV_32FC1);
		int j = 0;
		for (vector<float> singleDescriptor : singleClass) {
			for (int i = 0; i < cols; i++) {
				tempMat.at<float>(j, i) = singleDescriptor[i];
			}
			tempMat.at<float>(j, cols) = classNumber;
			j++;
		}
		//cout << "M = " << endl << " " << tempMat << endl << endl;
		descriptorsAsMatVector.push_back(tempMat);
	}
	return descriptorsAsMatVector;
}

//function that returns all filenames (output = [class[0], class[1], ...] with class[0] = ["~/00/img1", "~/00/img2", ...]
vector<vector<string>> getAllClassPaths(string filepath) {
	vector<vector<string>> output;
	vector<string> directories = list_dir(filepath);
	for (std::string element : directories) {
		output.push_back(list_dir(element));
	}
	return output;
}

//function to return all elements in folder (eg. input = "data/task2/train/" -> output = ["data/task2/train/00", "~/01", ...]
vector<string> list_dir(string path) {
	vector <string> img_list;
	for (const auto& entry : fs::directory_iterator(path))
		img_list.push_back(entry.path().string());
	return img_list;
}
