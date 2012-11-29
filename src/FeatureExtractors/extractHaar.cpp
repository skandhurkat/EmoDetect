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
#include <FeatureExtractors/extractHaar.h>
#include <Infrastructure/exceptions.h>

float getIntegralRectValue(IplImage* img, int top, int left, int bottom, int right)
{
  float res = static_cast<float>( ((double*)(img->imageData + img->widthStep*bottom))[right]);
  res -= static_cast<float>(((double*)(img->imageData + img->widthStep*bottom))[left]);
  res -= static_cast<float>(((double*)(img->imageData + img->widthStep*top))[right]);
  res += static_cast<float>(((double*)(img->imageData + img->widthStep*top))[left]);
  return res;
}

void extractHaarFeatures(const IplImage* img, Mat& haar)
{
  CvSize size = cvSize(IMAGE_RESIZE, IMAGE_RESIZE);
  CvSize size2 = cvSize(INTEGRAL_SIZE, INTEGRAL_SIZE);
  CvSize img_size = cvGetSize(img);
  IplImage*	ipl=	cvCreateImage(img_size,8,0);
  if(img->nChannels==3)
    {
      cvCvtColor(img,ipl,CV_BGR2GRAY);
    }
  else
    {
      cvCopy(img,ipl,0);
    }

  if((size.width!=img_size.width)|| (size.height!=img_size.height))
    {
      IplImage* tmpsize=cvCreateImage(size,IPL_DEPTH_8U,0);
      cvResize(ipl,tmpsize,CV_INTER_LINEAR);
      cvReleaseImage( &ipl);
      ipl=cvCreateImage(size,IPL_DEPTH_8U,0);
      cvCopy(tmpsize,ipl,0);
      cvReleaseImage( &tmpsize);
    }

  IplImage* temp = cvCreateImage(size, IPL_DEPTH_64F, 0);
  cvCvtScale(ipl,temp);
  cvNormalize(temp, temp, 0, 1, CV_MINMAX);
  haar.release();
  haar = Mat::zeros(1, NUM_HAAR_FEATURES, CV_32FC1);
    
  IplImage* integral=cvCreateImage(size2,IPL_DEPTH_64F,0);
  CvMat * sqSum = cvCreateMat(temp->height + 1, temp->width + 1, CV_64FC1);
  cvIntegral(temp, integral, sqSum);
  cvReleaseMat(&sqSum);

//  IplImage* tem = cvCreateImage(size2, IPL_DEPTH_64F, 0);
//  cvCvtScale(integral,tem);
//  cvNormalize(tem, tem, 0, 255, CV_MINMAX);

  int actualSize = 0;
  // top left
  for(int i = 0; i < 100; i+= 10) {
    for(int j = 0; j < 100; j+= 10) {
      // bottom right
      for(int m = i+10; m<=100; m+=10) {
	for(int n = j+10; n<=100; n+=10) {
	  haar.at<float>(0, actualSize++) = getIntegralRectValue(integral, i, j, m, n);
	}
      }
    }
  }
  cvReleaseImage(&ipl);
  cvReleaseImage(&temp);
  cvReleaseImage(&integral);
}
