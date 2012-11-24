#include <FeatureExtractors/extractFeatures.h>
#include <Infrastructure/exceptions.h>

void extractFeatures(const IplImage* inputImage, Mat& featureVector,
    featureExtractor fEx)
{
  if(!IplImage) throw EMPTY_IMAGE_EXCEPTION;
  switch(fEx)
  {
  case HAAR:
    extractHaar(inputImage, featureVector);
    break;
  case GABOR:
    extractGabor(inputImage, featureVector);
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
