#ifndef PHOG_3_H_INCLUDED
#define PHOG_3_H_INCLUDED

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

#define NUM_PHOG 144
#define NUM_BINS 16
#define NUM_DIVS 3

void extractPHoG(const Mat& img, Mat& PHOG);

#endif // PHOG_3_H_INCLUDED
