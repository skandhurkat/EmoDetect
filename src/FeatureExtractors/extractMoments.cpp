#include <FeatureExtractors/extractMoments.h>
#include <Infrastructure/exceptions.h>

void extractMoments(const Mat& img, Mat& rVal)
{
  if(!img.data)
    throw emptyImageException();
  rVal.create(1, NUM_MOMENTS, CV_64FC1);
  Mat scaledImg;
  resize(img, scaledImg, Size(MOMENTS_RESIZE,MOMENTS_RESIZE));
  Moments m = moments(scaledImg);
  rVal.at<double>(0,0) = m.m00;
  rVal.at<double>(0,1) = m.m10;
  rVal.at<double>(0,2) = m.m01;
  rVal.at<double>(0,3) = m.m20;
  rVal.at<double>(0,4) = m.m11;
  rVal.at<double>(0,5) = m.m02;
  rVal.at<double>(0,6) = m.m30;
  rVal.at<double>(0,7) = m.m21;
  rVal.at<double>(0,8) = m.m12;
  rVal.at<double>(0,9) = m.m03;
  rVal.at<double>(0,10) = m.mu20;
  rVal.at<double>(0,11) = m.mu11;
  rVal.at<double>(0,12) = m.mu02;
  rVal.at<double>(0,13) = m.mu30;
  rVal.at<double>(0,14) = m.mu21;
  rVal.at<double>(0,15) = m.mu12;
  rVal.at<double>(0,16) = m.mu03;
  rVal.at<double>(0,17) = m.nu20;
  rVal.at<double>(0,18) = m.nu11;
  rVal.at<double>(0,19) = m.nu02;
  rVal.at<double>(0,20) = m.nu30;
  rVal.at<double>(0,21) = m.nu21;
  rVal.at<double>(0,22) = m.nu12;
  rVal.at<double>(0,23) = m.nu03;
}
