
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
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> test_data) {

	/*
		TODO
	*/


}


void testDTrees(vector<Mat>  train_data, vector<Mat> test_data) {

    int num_classes = train_data.size(); // Is this equal to the size of train_data / test_data? else set it back to 6

	Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
	tree->setCVFolds(1); //Geht nicht mit anderen Zahlen größer als 1 (warum?)
	tree->setMaxCategories(10); //Standard 10;
	tree->setMaxDepth(INT8_MAX);
	tree->setMinSampleCount(2); //Standard 10; Weniger = besser (zumindest hier?)

	// Mask
	Rect rCropForTrain = Rect(0, 0, train.cols - 1, train.rows); //Mask to get only the data
	Rect rvCropForTrain = Rect(train.cols - 1, 0, 1, train.rows); //Mask to get only the classLables

	// Extract responseVector and convert to CV_32S (else will not be recognized as a lable)
	Mat trainResponseVector;
	train(rvCropForTrain).convertTo(trainResponseVector, CV_32S);

	// Train with data + reponseVector
	cout << "training our tree ..." << endl;
	tree->train(train(rCropForTrain), cv::ml::ROW_SAMPLE, trainResponseVector);

	// Masks
	Rect rCropForTest = Rect(0, 0, test.cols - 1, test.rows);
	Rect rvCropForTest = Rect(test.cols - 1, 0, 1, test.rows); 

	// Convert to CV_32S (previously class defined as float)
	Mat testResponseVector;
	test(rvCropForTest).convertTo(testResponseVector, CV_32S);

	Mat predictOutput;
	tree->predict(test(rCropForTest), predictOutput, cv::ml::DTrees::PREDICT_MAX_VOTE);
	cout << "predictOutput = " << endl << " " << predictOutput.t() << endl << endl; //Output transposed
	cout << "groundTruth = " << endl << " " << testResponseVector.t() << endl << endl; //Output transposed

	// Calc Misses vs Ground truth
	int hits = 0;
	int misses = 0;
	for (int i = 0; i < testResponseVector.rows; i++) {
		if (predictOutput.at<int>(i, 0) == testResponseVector.at<int>(i,0)) {
			hits++;
		}
		else {
			misses++;
		}
	}
	cout << "Hits: " << hits << "; Misses: " << misses << endl << endl;
	cout << "Using: Predict_max_vote" << endl << endl;

    //performanceEval<cv::ml::DTrees>(tree, train_data);
    //performanceEval<cv::ml::DTrees>(tree, test_data);

}


void testForest(vector<Mat> training_data, vector<Mat> test_data){
	
    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & Count parameters, to see how it affects the performance

    */

    //performanceEval<RandomForest>(forest, train_data);
    //performanceEval<RandomForest>(forest, test_data);
}


int main(){

	forest = RandomForest(100, 10, 1, 50, 10); //treecount, maxdepth (default: intmax), cvfolds (default: 10), minsamplecount (default: 10), maxcategories (default: 10)

	//single descriptors can be accessed via vector[class(0-5)][image*8 or image] depending on if rotatedSamples = true/false
	//TODO Convert into training data
	cout << "computing descriptors ..." << endl;
	vector<list<vector<float>>> trainTemp = getAllDescriptors("data/task2/train/");
	vector<list<vector<float>>> testTemp = getAllDescriptors("data/task2/test/", false);
	vector<Mat> allTrainingDescriptors = convertToMatVector(trainTemp);
	vector<Mat> allTestingDescriptors = convertToMatVector(testTemp);
	testDTrees(allTrainingDescriptors, allTestingDescriptors);
    //testForest();
    return 0;
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
		for (vector<float> singleDescriptor : singleClass) 
		{
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
