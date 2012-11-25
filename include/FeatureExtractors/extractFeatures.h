#ifndef __EXTRACT_FEATURES_H_INCLUDED
#define __EXTRACT_FEATURES_H_INCLUDED

#include <opencv2/opencv.hpp>
using namespace std;
#include <FeatureExtractors/extractHaar.h>
#include <FeatureExtractors/extractGabor.h>
#include <FeatureExtractors/extractMoments.h>
#include <FeatureExtractors/extractPHoG.h>

enum featureExtractor {HAAR, GABOR, MOMENTS, PHOG};

void extractFeatures(const Mat& input, Mat& featureVector,
    featureExtractor fEx);

#endif //__EXTRACT_FEATURES_H_INCLUDED
