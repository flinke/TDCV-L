#include "RandomForest.h"

using namespace std;
using namespace cv;

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
	mTrees = RandomForest::create();
}

RandomForest::~RandomForest()
{
}

vector<Ptr<cv::ml::DTrees>> RandomForest::create() {
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



void RandomForest::train(/* Fill */)
{
    // Fill
}

float RandomForest::predict(/* Fill */)
{
    // Fill
	return 0;
}

