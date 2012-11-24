#include <LearningAlgorithms/svm.h>
#include <Infrastructure/exceptions.h>
#include <iostream>
using namespace std;

CvANN_MLP annObj;
Mat op;

void annSetup(const Mat& trainData, const Mat& categoryData, int numCategories)
{
  op = Mat::zeros(categoryData.rows, numCategories, CV_32F);
  for (int k = 0; k < categoryData.rows; k++) 
  {
    op.at<float>(k,static_cast<int>(categoryData.at<float>(k,0))) = 1.0;
  }
  int layersizeArray[3] = {trainData.cols, 128, numCategories};
  Mat layersize = Mat(1,3,CV_32S,layersizeArray);
  annObj.create(layersize,CvANN_MLP::SIGMOID_SYM,1,1);
}

void anntrain(const Mat& trainData, const Mat& categoryData)
{
  Mat annWt = Mat(categoryData.rows,1,CV_32F,1.0);
  annObj.train(trainData, op, annWt);
}

float anntest(const Mat& testData, const Mat& categoryData, int numCategories) 
{
  int errCount = 0;
  int prediction = 0;
  Mat annVal = Mat(testData.rows,numCategories,CV_32F);
  annObj.predict(testData,annVal);
  for (int i = 0; i < testData.rows; i++) 
  {
    float max = -HUGE_VAL;
    for (int k = 0; k < numCategories; k++) 
    {
      if(annVal.at<float>(i,k) > max) 
      {
        max = annVal.at<float>(i,k);
        prediction = k;
      }
    }
    if(prediction != static_cast<int>(categoryData.at<float>(i,0))) 
    {
      cout << i <<":\t" << prediction << "\t" << categoryData.at<float>(i,0) << endl;      
      errCount++;
    }
  }
  return errCount * 100.0/testData.rows;
}
