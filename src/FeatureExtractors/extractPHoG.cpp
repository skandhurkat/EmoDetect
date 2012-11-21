#include "phog.h"

void PHoG(const Mat& img, vector<float>& PHOG, int div, int bins)
{
	if(bins < 0)
		throw PHoG_exception();

	if(img.type() != CV_8U)
		throw PHoG_exception();

	if(!img.data)
		throw PHoG_exception();

    PHOG.clear();

	Mat dx, dy;

	Sobel(img, dx, CV_16S, 1, 0, 3);
	Sobel(img, dy, CV_16S, 0, 1, 3);

	PHOG.resize(bins*div*div, 0.0);

	unsigned int area_img = img.rows*img.cols;

	short _dx, _dy;
	float angle;
	float grad_value;

	//Loop to find gradients
	int l_x = img.cols/div, l_y = img.rows/div;
	for(int m=0; m<div; m++)
        for(int n=0; n<div; n++)
            for(int i=0; i<l_x; i++)
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
                    PHOG[(m*div+n)*bins+static_cast<int>(angle)] += grad_value;
                }

	float max = 0;
	for(int i = 0; i < bins; i++)
	{
		if(PHOG[i] > max)
			max = PHOG[i];
	}

	for(int i = 0; i < bins; i++)
	{
		PHOG[i] /= max;
	}

	dx.release();
	dy.release();
}
