
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>



#include "HOGDescriptor.h"
#include "RandomForest.h"
#include "task1.h"


using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

vector<string> list_dir(string path);
vector<list<vector<float>>> getAllDescriptors(string filepath, bool getRotatedSamples = true);
vector<vector<string>> getAllClassPaths(string filepath);
RandomForest forest;


template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> test_data) {

	/*
		TODO
	*/


}


void testDTrees(cv::Ptr<cv::ml::TrainData>  train_data, cv::Ptr<cv::ml::TrainData> test_data) {

    int num_classes = 6;
	Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
	tree->setCVFolds(10);
	tree->setMaxCategories(10);
	tree->setMaxDepth(10);
	tree->setMinSampleCount(10);

	cv::Ptr<cv::ml::TrainData> sample; //TODO
	tree->train(sample);
    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */

    //performanceEval<cv::ml::DTrees>(tree, train_data);
    //performanceEval<cv::ml::DTrees>(tree, test_data);

}


void testForest(vector<list<vector<float>>>& training_data, vector<list<vector<float>>>& test_data){
	
    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */

    //performanceEval<RandomForest>(forest, train_data);
    //performanceEval<RandomForest>(forest, test_data);
}


int main(){

	forest = RandomForest(100, 10, 10, 50, 10); //treecount, maxdepth (default: intmax), cvfolds (default: 10), minsamplecount (default: 10), maxcategories (default: 10)

	//single descriptors can be accessed via vector[class(0-5)][image*8 or image] depending on if rotatedSamples = true/false
	//TODO Convert into training data
	vector<list<vector<float>>> allTrainingDescriptors = getAllDescriptors("data/task2/train/");
	vector<list<vector<float>>> allTestingDescriptors = getAllDescriptors("data/task2/test/", false);
    //testDTrees();
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
