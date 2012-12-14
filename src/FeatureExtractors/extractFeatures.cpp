//    File name: extractFeatures.cpp
//    (c) Rishabh Animesh, Skand Hurkat, Abhinandan Majumdar, Aayush Saxena, 2012

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
