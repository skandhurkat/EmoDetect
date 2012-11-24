#include <LearningAlgorithms/svm.h>
#include <Infrastructure/exceptions.h>
#include <iostream>
using namespace std;

CvSVM svmObj;

void svmtrain(const Mat& trainData, const Mat& categoryData)
{
  SVMParams svmParams;
  svmParams.kernel_type = CvSVM::LINEAR;
  svmObj.train(trainData, categoryData, Mat(), Mat(), svmParams);
}

float svmtest(const Mat& testData, const Mat& categoryData) 
{
  int errCount = 0;
  for (int i = 0; i < testData.rows; i++) 
  {
    int val = svmObj.predict(testData.row(i));
    if(val != static_cast<int>(categoryData.at<float>(i,0))) 
    {
      cout << i <<":\t" << val << "\t" << categoryData.at<float>(i,0) << endl;
      errCount++;
    }
  }
  return errCount * 100.0/testData.rows;
}
