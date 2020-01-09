

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>

class RandomForest
{
public:
	RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

	std::vector <cv::Ptr<cv::ml::DTrees>> createTrees();
    static cv::Ptr<RandomForest> create(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);
    bool train(cv::Mat entireTrainingSet, int numsamp);
    float predict(cv::Mat testData, cv::Mat& predictOutput, int flag);
    float predict(cv::Mat testData, cv::Mat& predictOutput, int flag, std::vector<float>& predictedConfidence, std::vector<int>& predictedClass);
    cv::Mat calcResponseVector(std::vector<cv::Mat> vec);
    cv::Mat calcResponseVector(std::vector<cv::Mat> vec, std::vector<float>& predictedConfidence, std::vector<int>& predictedClass);
    cv::Mat getSample(cv::Mat entireData, int sampleSize);
    bool train(std::vector<cv::Mat> entireTrainingSet, int numsamp);
    cv::Mat getSample(std::vector<cv::Mat> entireData, int sampleSize);


private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;

    // M-Trees for constructing the forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
};

#endif //RF_RANDOMFOREST_H
