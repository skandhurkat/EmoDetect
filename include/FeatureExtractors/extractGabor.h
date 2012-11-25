#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __EXTRACT_GABOR_H
#define __EXTRACT_GABOR_H

#define NUM_GABOR_FEATURES 10240

void extractGaborFeatures(const IplImage* img, Mat& gb);
void DisposeKernels();

#endif // __EXTRACT_GABOR_H
