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
    IplImage* Img = &img;
    extractHaarFeatures(Img, featureVector);
    cvReleaseImage(&Img);
    break;
    }
  case GABOR:
    {
    IplImage img = inputImage;
    IplImage* Img = &img;
    extractGaborFeatures(Img, featureVector);
    cvReleaseImage(&Img);
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
