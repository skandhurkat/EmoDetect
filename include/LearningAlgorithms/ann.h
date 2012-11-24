#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __ANN_H
#define __ANN_H

void annSetup(const Mat& trainData, const Mat& categoryData, int numCategories);
void anntrain(const Mat& trainData, const Mat& categoryData);
float anntest(const Mat& testData, const Mat& categoryData, int numCategories);

#endif // __EXTRACT_MOMENTS_H
