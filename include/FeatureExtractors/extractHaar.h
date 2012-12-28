//    File name: extractHaar.h
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

#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef __EXTRACT_HAAR_H
#define __EXTRACT_HAAR_H

#define NUM_HAAR_FEATURES 3025
#define IMAGE_RESIZE 100
#define INTEGRAL_SIZE 101

void extractHaarFeatures(const IplImage* img, Mat& rVal);

#endif // __EXTRACT_HAAR_H
