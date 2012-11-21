#include <FeatureExtractors/extractPHoG.h>
#include <Infrastructure/exceptions.h>

void extractPHoG(const Mat& img, Mat& PHOG)
{
  int bins = NUM_BINS;
  int div = NUM_DIVS;
	if(!img.data)
		throw emptyImageException();

  PHOG.release();
  PHOG = Mat::zeros(1,div*div*bins,CV_32FC1);

	Mat dx, dy;

	Sobel(img, dx, CV_16S, 1, 0, 3);
	Sobel(img, dy, CV_16S, 0, 1, 3);

	unsigned int area_img = img.rows*img.cols;

	short _dx, _dy;
	float angle;
	float grad_value;

  //Loop to find gradients
  int l_x = img.cols/div, l_y = img.rows/div;
  for(int m=0; m<div; m++)
  {
    for(int n=0; n<div; n++)
    {
      for(int i=0; i<l_x; i++)
      {
        for(int j=0; j<l_y; j++)
        {
          _dx = dx.at<int>(m*l_x+i, n*l_y+j);
          _dy = dy.at<int>(m*l_x+i, n*l_y+j);
          grad_value = std::sqrt(_dx*_dx + _dy*_dy)/area_img;
          //*grad_value_it = sqrt(_dy*_dy + _dx*_dx)/area_img;

          angle = std::atan2(_dy, _dx);
          if(angle < 0)
            angle += 2*CV_PI;
          angle *= bins/(2*CV_PI);
          PHOG.at<float>(0,(m*div+n)*bins+static_cast<int>(angle))
            += grad_value;
        }
      }
    }
  }
  float max = 0;
  for(int i = 0; i < bins; i++)
  {
    if(PHOG.at<float>(0,i) > max)
      max = PHOG.at<float>(0,i);
  }

	for(int i = 0; i < bins; i++)
	{
		PHOG.at<float>(0,i) /= max;
	}

	dx.release();
	dy.release();
}
