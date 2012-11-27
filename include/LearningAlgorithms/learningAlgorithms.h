#ifndef __LEARNING_ALGORITHMS_H_INCLUDED
#define __LEARNING_ALGORITHMS_H_INCLUDED

#include <opencv2/opencv.hpp>
using namespace cv;
#include <Infrastructure/exceptions.h>

enum learningAlgorithm {ANN, SVM_ML, RT};

CvStatModel* learningAlgorithmSetup(int featureVectorSize,
    int numCategories,
    learningAlgorithm lA);

CvStatModel* readModel(const string& inputFile, learningAlgorithm lA);

void learningAlgorithmTrain(CvStatModel* model,
    const Mat& trainData,
    const Mat& responses,
    int numCategories,
    learningAlgorithm lA);

void learningAlgorithmPredict(CvStatModel* model,
    const Mat& featureData,
    Mat& responses,
    int numCategories,
    learningAlgorithm lA);

float learningAlgorithmComputeErrorRate(const Mat& predictedResponses,
    const Mat& actualResponses);

void writeModel(const CvStatModel* model, const string& outputFile);

#endif //__LEARNING_ALGORITHMS_H_INCLUDED
