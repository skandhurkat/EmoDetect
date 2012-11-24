#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __EXTRACT_HAAR_H
#define __EXTRACT_HAAR_H

#define NUM_HAAR_FEATURES 3025
#define IMAGE_RESIZE 100
#define INTEGRAL_SIZE 101

void extractHaarFeatures(const IplImage* img, Mat& rVal);

#endif // __EXTRACT_HAAR_H