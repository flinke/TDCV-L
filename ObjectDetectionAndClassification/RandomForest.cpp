#include "RandomForest.h"
#include <cstdlib>

using namespace std;
using namespace cv;

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
	mTrees = RandomForest::createTrees();
}

RandomForest::~RandomForest()
{
}

Ptr<RandomForest> RandomForest::create(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories) {
	RandomForest forest = RandomForest(treeCount, maxDepth, CVFolds, minSampleCount, maxCategories);
	Ptr<RandomForest> ptr = &forest;
	return ptr;
}

vector<Ptr<cv::ml::DTrees>> RandomForest::createTrees() {
	vector<Ptr<cv::ml::DTrees>> tempTrees;
	for (int i = 0; i < mTreeCount; i++) {
		Ptr<cv::ml::DTrees> tempTree = cv::ml::DTrees::create();		
		tempTree->setCVFolds(mCVFolds);
		tempTree->setMaxCategories(mMaxCategories);
		tempTree->setMaxDepth(mMaxDepth);
		tempTree->setMinSampleCount(mMinSampleCount);
		tempTrees.push_back(tempTree);
	}
	return tempTrees;
}


void RandomForest::setTreeCount(int treeCount)
{
	mTreeCount = treeCount;
}

void RandomForest::setMaxDepth(int maxDepth)
{
    mMaxDepth = maxDepth;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFolds)
{
	mCVFolds = cvFolds;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMaxDepth(mCVFolds);
}

void RandomForest::setMinSampleCount(int minSampleCount)
{
	mMinSampleCount = minSampleCount;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMaxDepth(mMinSampleCount);
}

void RandomForest::setMaxCategories(int maxCategories)
{
	mMaxCategories = maxCategories;
	for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
		mTrees[treeIdx]->setMaxDepth(mMaxCategories);
}



Mat RandomForest::getSample(Mat entireData, int sampleSize) {
	Mat sample = Mat(sampleSize, entireData.cols, CV_32FC1);
	int randomIndex;
	int rd;
	int dataSize = entireData.rows;
	for (int i = 0; i < sampleSize; i++) {
		rd = rand();
		randomIndex = rd % dataSize;
		for (int j = 0; j < entireData.cols; j++) {
			sample.at<float>(i, j) = entireData.at<float>(randomIndex, j);
		}
	}
	//cout << " " << sample << endl << endl;
	return sample;
}


Mat RandomForest::getSample(vector<Mat> entireData, int sampleSize) {
	int numberOfClasses = entireData.size();
	int samplesPerClass = sampleSize / numberOfClasses;
	Mat sample = Mat(sampleSize, entireData.at(0).cols, CV_32FC1);
	int randomIndex;
	int rd;
	for (int k = 0; k < numberOfClasses; k++) {
		int dataSize = entireData.at(k).rows;
		for (int i = 0; i < samplesPerClass; i++) {
			rd = rand();
			randomIndex = rd % dataSize;
			for (int j = 0; j < entireData.at(0).cols; j++) {
				sample.at<float>(i+k*samplesPerClass, j) = entireData.at(k).at<float>(randomIndex, j);
			}
		}
	}
	//cout << " " << sample << endl << endl;
	return sample;
}

bool RandomForest::train(vector<Mat> entireTrainingSet, int numsamp) {
	Mat sample;
	int counterForConsole = 0;
	cout << "	-> generating samples ... ";
	for (Ptr<cv::ml::DTrees> tree : mTrees) {
		sample = getSample(entireTrainingSet, numsamp);

		if (counterForConsole % 5 == 0) cout << counterForConsole << " ... ";
		counterForConsole++;
		// Maks
		Rect rCropForTrain = Rect(0, 0, sample.cols - 1, sample.rows); //Mask to get only the data
		Rect rvCropForTrain = Rect(sample.cols - 1, 0, 1, sample.rows); //Mask to get only the classLables

		// Extract responseVector and convert to CV_32S (else will not be recognized as a lable)
		Mat trainResponseVector;
		sample(rvCropForTrain).convertTo(trainResponseVector, CV_32S);

		// Train with data + reponseVector
		tree->train(sample(rCropForTrain), cv::ml::ROW_SAMPLE, trainResponseVector);
	}
	return true;
}

bool RandomForest::train(Mat entireTrainingSet, int numsamp) {
	Mat sample;
	int counterForConsole = 0;
	cout << "	-> generating samples ... ";
	for (Ptr<cv::ml::DTrees> tree : mTrees) {
		sample = getSample(entireTrainingSet, numsamp);

		if (counterForConsole % 3 == 0) cout << counterForConsole << " ... ";
		counterForConsole++;
		// Maks
		Rect rCropForTrain = Rect(0, 0, sample.cols - 1, sample.rows); //Mask to get only the data
		Rect rvCropForTrain = Rect(sample.cols - 1, 0, 1, sample.rows); //Mask to get only the classLables

		// Extract responseVector and convert to CV_32S (else will not be recognized as a lable)
		Mat trainResponseVector;
		sample(rvCropForTrain).convertTo(trainResponseVector, CV_32S);

		// Train with data + reponseVector
		tree->train(sample(rCropForTrain), cv::ml::ROW_SAMPLE, trainResponseVector);
	}
	return true;
}

float RandomForest::predict(Mat testData, Mat& predictOutput, int flag) {
	cout << "predicting output ..." << endl;
	// Masks 
	Rect rCrop = Rect(0, 0, testData.cols - 1, testData.rows);
	Rect rvCrop = Rect(testData.cols - 1, 0, 1, testData.rows);
	vector<Mat> tempOutputVec; 
	for (Ptr <cv::ml::DTrees> tree : mTrees) {
		Mat tempOutput; 
		tree->predict(testData(rCrop), tempOutput, flag);
		tempOutputVec.push_back(tempOutput);
	}
	predictOutput = calcResponseVector(tempOutputVec);
	return 0;
}

float RandomForest::predict(Mat testData, Mat& predictOutput, int flag, vector<float>& predictedConfidence, vector<int>& predictedClass) {
	//cout << "predicting output ..." << endl;
	// Masks 
	Rect rCrop = Rect(0, 0, testData.cols - 1, testData.rows);
	Rect rvCrop = Rect(testData.cols - 1, 0, 1, testData.rows);
	vector<Mat> tempOutputVec;
	for (Ptr <cv::ml::DTrees> tree : mTrees) {
		Mat tempOutput;
		tree->predict(testData(rCrop), tempOutput, flag);
		tempOutputVec.push_back(tempOutput);
	}
	predictOutput = calcResponseVector(tempOutputVec, predictedConfidence, predictedClass);
	return 0;
}

Mat RandomForest::calcResponseVector(vector<Mat> vec) {
	cout << "calculating responses ..." << endl;
	Mat response = Mat(vec[0].rows,1, CV_32S);
	for (int i = 0; i < vec[0].rows; i++) {
		int classes[6] = { 0,0,0,0,0,0 };
		for (int j = 0; j < vec.size(); j++) {
			int element = vec[j].at<int>(i, 0);
			switch(element) {
				case 0:
					classes[0]++;
					break;
				case 1:
					classes[1]++;
					break;
				case 2:
					classes[2]++;
					break;
				case 3:
					classes[3]++;
					break;
				case 4:
					classes[4]++;
					break;
				case 5:
					classes[5]++;
					break;
			}
		}
		cout << classes[0] << " " << classes[1] << " " << classes[2] << " " << classes[3] << " " << classes[4] << " " << classes[5] << " " << endl;
		const int N = sizeof(classes) / sizeof(int);
		int majorityClass = std::distance(classes, max_element(classes, classes + N)); //Get MajClass
		response.at<int>(i, 0) = majorityClass;
	}
	cout << " " << response << endl << endl;
	return response;
}

Mat RandomForest::calcResponseVector(vector<Mat> vec, vector<float>& predictedConfidence, vector<int>& predictedClass) {
	//cout << "calculating responses ..." << endl;
	Mat response = Mat(vec[0].rows, 1, CV_32S);
	for (int i = 0; i < vec[0].rows; i++) {
		int classes[4] = { 0,0,0,0 };
		for (int j = 0; j < vec.size(); j++) {
			int element = vec[j].at<int>(i, 0);
			switch (element) {
			case 0:
				classes[0]++;
				break;
			case 1:
				classes[1]++;
				break;
			case 2:
				classes[2]++;
				break;
			case 3:
				classes[3]++;
				break;
			}
		}
		//cout << classes[0] << " " << classes[1] << " " << classes[2] << " " << classes[3] << endl;
		const int N = sizeof(classes) / sizeof(int);
		int majorityClass = std::distance(classes, max_element(classes, classes + N)); //Get MajClass
		response.at<int>(i, 0) = majorityClass;
		predictedClass.push_back(majorityClass);
		int totalvotes = 0;
		for (int num : classes) {
			totalvotes += num;
		}
		predictedConfidence.push_back((float)classes[majorityClass] / (float) totalvotes);
	}
	//cout << " " << response << endl << endl;
	return response;
}

void mergeVectorToSingleMat(Mat& dataAsMat, vector<Mat>& data) {
	// SEHR SEHR SEHR UNSCHÖN UND FUNKTIONIERT NUR WENN KLASSENZAHL BLEIBT; BITTE FIXEN
	Mat temp;
	vconcat(data[0], data[1], temp);
	vconcat(temp, data[2], temp);
	vconcat(temp, data[3], temp);
	vconcat(temp, data[4], temp);
	vconcat(temp, data[5], temp);
	dataAsMat = temp;
}

