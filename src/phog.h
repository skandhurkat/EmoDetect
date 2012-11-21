#ifndef PHOG_3_H_INCLUDED
#define PHOG_3_H_INCLUDED

#include <cmath>
#include <vector>
#include <opencv/cv.h>
#include <opencv/cvaux.h>

using namespace cv;

class PHoG_exception
{
	//exception class
};

void PHoG(const Mat& img, vector<float>& PHOG, int div = 3, int bins = 16);

#endif // PHOG_3_H_INCLUDED
