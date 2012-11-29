//    This file is part of EmoDetect.
//
//    EmoDetect is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    EmoDetect is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with EmoDetect. If not, see <http://www.gnu.org/licenses/>.
#include <FeatureExtractors/extractMoments.h>
#include <Infrastructure/exceptions.h>
#include <iostream>
using namespace std;

void extractMoments(const Mat& img, Mat& rVal)
{
  if(!img.data)
    throw EMPTY_IMAGE_EXCEPTION;
  rVal.create(1, NUM_MOMENTS, CV_32F);  
  Mat scaledImg;
  resize(img, scaledImg, Size(MOMENTS_RESIZE,MOMENTS_RESIZE));
  Moments m = moments(scaledImg);
  rVal.at<float>(0,0) = static_cast<float>(m.m00);
  rVal.at<float>(0,1) = static_cast<float>(m.m10);
  rVal.at<float>(0,2) = static_cast<float>(m.m01);
  rVal.at<float>(0,3) = static_cast<float>(m.m20);
  rVal.at<float>(0,4) = static_cast<float>(m.m11);
  rVal.at<float>(0,5) = static_cast<float>(m.m02);
  rVal.at<float>(0,6) = static_cast<float>(m.m30);
  rVal.at<float>(0,7) = static_cast<float>(m.m21);
  rVal.at<float>(0,8) = static_cast<float>(m.m12);
  rVal.at<float>(0,9) = static_cast<float>(m.m03);
  rVal.at<float>(0,10) = static_cast<float>(m.mu20);
  rVal.at<float>(0,11) = static_cast<float>(m.mu11);
  rVal.at<float>(0,12) = static_cast<float>(m.mu02);
  rVal.at<float>(0,13) = static_cast<float>(m.mu30);
  rVal.at<float>(0,14) = static_cast<float>(m.mu21);
  rVal.at<float>(0,15) = static_cast<float>(m.mu12);
  rVal.at<float>(0,16) = static_cast<float>(m.mu03);
  rVal.at<float>(0,17) = static_cast<float>(m.nu20);
  rVal.at<float>(0,18) = static_cast<float>(m.nu11);
  rVal.at<float>(0,19) = static_cast<float>(m.nu02);
  rVal.at<float>(0,20) = static_cast<float>(m.nu30);
  rVal.at<float>(0,21) = static_cast<float>(m.nu21);
  rVal.at<float>(0,22) = static_cast<float>(m.nu12);
  rVal.at<float>(0,23) = static_cast<float>(m.nu03);
}
