#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __RT_H
#define __RT_H

void rttrain(const Mat& trainData, const Mat& categoryData);
float rttest(const Mat& testData, const Mat& categoryData);

#endif // __EXTRACT_MOMENTS_H
