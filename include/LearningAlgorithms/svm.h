#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __SVM_H
#define __SVM_H

void svmtrain(const Mat& trainData, const Mat& categoryData);
float svmtest(const Mat& testData, const Mat& categoryData);

#endif // __EXTRACT_MOMENTS_H
