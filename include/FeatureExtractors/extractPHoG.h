//    File name: extracPHoG.h
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

#ifndef PHOG_3_H_INCLUDED
#define PHOG_3_H_INCLUDED

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

#define NUM_PHOG 144
#define NUM_BINS 16
#define NUM_DIVS 3

void extractPHoG(const Mat& inputImage, Mat& PHOG);

#endif // PHOG_3_H_INCLUDED
