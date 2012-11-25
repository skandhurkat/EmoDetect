#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __EXTRACT_MOMENTS_H
#define __EXTRACT_MOMENTS_H

#define NUM_MOMENTS 24
#define MOMENTS_RESIZE 64

void extractMoments(const Mat& inputImage, Mat& rVal);

#endif // __EXTRACT_MOMENTS_H
