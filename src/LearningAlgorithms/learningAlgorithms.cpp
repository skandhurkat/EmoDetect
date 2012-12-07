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
#include <LearningAlgorithms/learningAlgorithms.h>

CvStatModel* learningAlgorithmSetup(int featureVectorSize,
    int numCategories,
    learningAlgorithm lA)
{
  CvStatModel* model = NULL;
  switch(lA)
  {
  case ANN:
    {
    model = new CvANN_MLP();
    int layerSizeArray[] = {featureVectorSize,
		                        featureVectorSize+numCategories,
		                        numCategories};
    Mat layerSize = Mat(1, 3, CV_32S, layerSizeArray);
    reinterpret_cast<CvANN_MLP*>(model)
      ->create(layerSize, CvANN_MLP::SIGMOID_SYM, 1, 1);
    break;
    }
  case SVM_ML:
    {
    model = new CvSVM();
    break;
    }
  case RT:
    {
    model = new CvRTrees();
    break;
    }
  default:
    throw INVALID_LEARNING_ALGORITHM;
  }
  return model;
}

CvStatModel* readModel(const string& inputFile, learningAlgorithm lA)
{
  if(inputFile.empty()) throw EMPTY_FILE_EXCEPTION;
  CvStatModel* model = NULL;
  switch(lA)
  {
  case ANN:
    {
        model = new CvANN_MLP();
        (reinterpret_cast<CvANN_MLP *>(model))->load(inputFile.c_str());
        break;
    }
  case SVM_ML:
    {
        model = new CvSVM();
        (reinterpret_cast<CvSVM *>(model))->load(inputFile.c_str());
        break;
    }
  case RT:
    {
        model = new CvRTrees();
        (reinterpret_cast<CvRTrees *>(model))->load(inputFile.c_str());
        break;
    }
  default:
    throw INVALID_LEARNING_ALGORITHM;
  }
  return model;
}

void learningAlgorithmTrain(CvStatModel* model,
    const Mat& trainData,
    const Mat& responses,
    int numCategories,
    learningAlgorithm lA)
{
  switch(lA)
  {
  case ANN:
    {
    Mat tmpOp = Mat::zeros(responses.rows, numCategories, CV_32F);
    for (int i = 0; i < responses.rows; i++)
      tmpOp.at<float>(i,static_cast<int>(responses.at<float>(i,0))) = 1;
    Mat annWt = Mat::ones(responses.rows, 1, CV_32F);
    reinterpret_cast<CvANN_MLP*>(model)->train(trainData, tmpOp, annWt);
    break;
    }
  case SVM_ML:
    {
    SVMParams svmParams;
    svmParams.kernel_type = CvSVM::LINEAR;
    reinterpret_cast<CvSVM*>(model)->train(trainData, responses,
        Mat(), Mat(), svmParams);
    break;
    }
  case RT:
    {
    reinterpret_cast<CvRTrees*>(model)->train(trainData, CV_ROW_SAMPLE,
        responses);
    break;
    }
  default:
    throw INVALID_LEARNING_ALGORITHM;
  }
}

void learningAlgorithmPredict(CvStatModel* model,
    const Mat& featureData,
    Mat& responses,
    int numCategories,
    learningAlgorithm lA)
{
  responses.create(featureData.rows,1,CV_32F);
  switch(lA)
  {
  case ANN:
    {
    Mat tmpOps(featureData.rows, numCategories, CV_32F);
    reinterpret_cast<CvANN_MLP*>(model)->predict(featureData, tmpOps);
    for(int i = 0; i < featureData.rows; i++)
    {
      float max = -HUGE_VAL;
      float prediction = 0;
      for(int j = 0; j < numCategories; j++)
      {
        if(tmpOps.at<float>(i,j) > max)
        {
          max = tmpOps.at<float>(i,j);
          prediction = j;
        }
      }
      responses.at<float>(i,0) = prediction;
    }
    break;
    }
  case SVM_ML:
    {
    CvMat fData = featureData;
    CvMat resp = responses;
    reinterpret_cast<CvSVM*>(model)->predict(&fData, &resp);
    break;
    }
  case RT:
    {
    for(int i = 0; i < featureData.rows; i++)
    {
      responses.at<float>(i,0) = reinterpret_cast<CvRTrees*>(model)
        ->predict(featureData.row(i));
    }
    break;
    }
  default:
    throw INVALID_LEARNING_ALGORITHM;
  }
}

float learningAlgorithmComputeErrorRate(const Mat& predictedResponses,
    const Mat& actualResponses)
{
  assert(predictedResponses.rows == actualResponses.rows);
  assert(predictedResponses.cols == 1 && actualResponses.cols == 1);
  assert(predictedResponses.rows != 0 && actualResponses.rows != 0);
  float error = 0;
  for(int i = 0; i < predictedResponses.rows; i++)
  {
    if(cvRound(predictedResponses.at<float>(i,0)) !=
        cvRound(actualResponses.at<float>(i,0)))
      error++;
  }
  error /= predictedResponses.rows;
  return error;
}

void writeModel(const CvStatModel* model, const string& outputFile)
{
  if(outputFile.empty()) throw EMPTY_FILE_EXCEPTION;
  model->save(outputFile.c_str());
}
