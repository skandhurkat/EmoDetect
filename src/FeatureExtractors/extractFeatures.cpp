#include <FeatureExtractors/extractFeatures.h>
#include <Infrastructure/exceptions.h>

void extractFeatures(const IplImage* inputImage, Mat& featureVector,
    featureExtractor fEx)
{
  if(!inputImage) throw EMPTY_IMAGE_EXCEPTION;
  switch(fEx)
  {
  case HAAR:
    extractHaarFeatures(inputImage, featureVector);
    break;
  case GABOR:
    extractGaborFeatures(inputImage, featureVector);
    break;
  case MOMENTS:
    extractMoments(inputImage, featureVector);
    break;
  case PHOG:
    extractPHoG(inputImage, featureVector);
    break;
  default:
    throw(INVALID_FEATURE_EXTRACTOR);
  }
}
