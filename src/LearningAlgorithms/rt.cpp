#include <LearningAlgorithms/rt.h>
#include <Infrastructure/exceptions.h>
#include <iostream>
using namespace std;


CvRTrees rtObj;

void rttrain(const Mat& trainData, const Mat& categoryData)
{
  CvRTParams rtParams;
  rtObj.train(trainData, CV_ROW_SAMPLE, categoryData, Mat(), Mat(), Mat(), Mat(), rtParams);
}

float rttest(const Mat& testData, const Mat& categoryData) 
{
  int errCount = 0;
  for (int i = 0; i < testData.rows; i++) 
  {
    int val = rtObj.predict(testData.row(i));
    if(val != static_cast<int>(categoryData.at<float>(i,0))) 
    {
      cout << i <<":\t" << val << "\t" << categoryData.at<float>(i,0) << endl;
      errCount++;
    }
  }
  return errCount * 100.0/testData.rows;
}
