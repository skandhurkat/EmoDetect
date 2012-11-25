#include <FeatureExtractors/extractFeatures.h>
#include <Infrastructure/exceptions.h>

void extractFeatures(const Mat& inputImage, Mat& featureVector,
    featureExtractor fEx)
{
  if(inputImage.empty()) throw EMPTY_IMAGE_EXCEPTION;
  switch(fEx)
  {
  case HAAR:
    {
    IplImage img = inputImage;
    extractHaarFeatures(&img, featureVector);
    break;
    }
  case GABOR:
    {
    IplImage img = inputImage;
    extractGaborFeatures(&img, featureVector);
    break;
    }
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
