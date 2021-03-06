//    File name: test.cpp
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
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
using namespace std;
#include <getopt.h>
#include <dirent.h>

#include <FeatureExtractors/extractFeatures.h>
#include <LearningAlgorithms/learningAlgorithms.h>
#include <Infrastructure/exceptions.h>
#include <Infrastructure/defines.h>

void usage(const string programName, int exitCode)
{
    cout << "Usage\n"
         << programName << " [options]" << endl
         << "Options include:" << endl
         << "\t--inputFile=<inputFile path>" << endl
         << "\t--featureExtractor=[gabor haar moments phog]" << endl
         << "\t--learningAlgorithm=[ann rt svm]" << endl
         << "\t--loadClassifier=<file path>" << endl
         << "\t--cascadeClassifier=<cascade path>" << endl
         << "\t--verbose" << endl
         << "\t--help" << endl;
    exit(exitCode);
}

int main(int argc, char** argv)
{
  const string programName = argv[0];
  int verbose = 0;
  float percentageTestData = 10;
  const char* shortOptions = "i:f:l:L:c:h";
  const option longOptions[] =
  {
    {"inputFile", required_argument, NULL, 'i'},
    {"featureExtractor", required_argument, NULL, 'f'},
    {"learningAlgorithm", required_argument, NULL, 'l'},
    {"loadClassifier", required_argument, NULL, 'L'},
    {"cascadeClassifier", required_argument, NULL, 'c'},
    {"verbose", no_argument, &verbose, 1},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
  };

  string inputFilePath;
  featureExtractor fEx;
  learningAlgorithm lA;
  string loadClassifierLocation;
  string cascadeClassifierName;

  int opt;
  while((opt=getopt_long(argc, argv, shortOptions, longOptions, NULL))
      != -1)
  {
    switch(opt)
    {
      case 'h':
        usage(programName, EXIT_SUCCESS);
        break;
      case 'i':
        inputFilePath = optarg;
        break;
      case 'f':
        if(!strcmp(optarg,"gabor"))
          fEx = GABOR;
        else if(!strcmp(optarg,"haar"))
          fEx = HAAR;
        else if(!strcmp(optarg,"moments"))
          fEx = MOMENTS;
        else if(!strcmp(optarg,"phog"))
          fEx = PHOG;
        break;
      case 'l':
        if(!strcmp(optarg,"ann"))
          lA = ANN;
        else if(!strcmp(optarg,"rt"))
          lA = RT;
        else if(!strcmp(optarg,"svm"))
          lA = SVM_ML;
        break;
      case 'L':
        loadClassifierLocation = optarg;
        break;
      case 'c':
        cascadeClassifierName = optarg;
        break;
      case 0: break;
      case '?':
      default :
              cout << "Invalid arguments received!" << endl;
              usage(programName, EXIT_FAILURE);
              break;
    }
  }

  if(inputFilePath.empty())
  {
    cerr << "Error! Input file should be specified" << endl;
    return EXIT_FAILURE;
  }
  if(cascadeClassifierName.empty())
  {
    cerr << "Error! Should specify a Haar cascade classifier" << endl;
    return EXIT_FAILURE;
  }

  string fExtractor = (fEx==GABOR)?"Gabor":
    (fEx==HAAR)?"Haar":
    (fEx==MOMENTS)?"Moments":
    (fEx==PHOG)?"PHoG":"Error!";
  string lAlgorithm = (lA==ANN)?"ANN":
    (lA==RT)?"RT":
    (lA==SVM_ML)?"SVM":"Error!";

  cout << "Received arguments" << endl
    << "\tProgram Name      : " << programName << endl
    << "\tInput File Path   : " << inputFilePath << endl
    << "\tFeature Extractor : " << fExtractor << endl
    << "\tLearning Algorithm: " << lAlgorithm << endl
    << "\tLoad Classifier at: " << loadClassifierLocation << endl
    << "\tCascade Classifier: " << cascadeClassifierName << endl;

  string imagePath;
  Mat imageFeatureData;
  Mat categoryData;

  CascadeClassifier haarCascade(cascadeClassifierName);
  assert(!haarCascade.empty());
  cout << "Classifier loaded from " << cascadeClassifierName << endl;

  vector<Rect> detected;

  int numCategories = 0;
  int numImages = 0;

  freopen (inputFilePath.c_str() ,"r", stdin);

  string filename;
  int label;
  while(cin >> label)
  {
    cin >> filename;
    if(verbose)
      cout << "Attempting to open " << filename << endl;
    Mat img= imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    Mat m;
    if(img.data)
    {
      haarCascade.detectMultiScale(img, detected, 1.1, 3,
          CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH);
      if(detected.size() != 1)
      {
        cerr << "Error! Need exactly one face in " << filename << endl;
        continue;
      }
      Mat cropped = img(detected[0]);
      extractFeatures(cropped, m, fEx);
      imageFeatureData.push_back(m);
      categoryData.push_back(static_cast<float>(label));
      numImages++;
      if(!(numImages%100) && !verbose)
        cout << numImages << " images processed" << endl;
    }
    else if(verbose)
      cout << "\tFailed to open " << filename << endl;
    if(label > numCategories)
      numCategories = label;
  }
  numCategories += 1;
  DisposeKernels(); //TODO: Repair this cheap hack

  CvStatModel* model = readModel(loadClassifierLocation, lA);
  Mat responses;
  learningAlgorithmPredict(model, imageFeatureData, responses,
      numCategories, lA);
  Mat confusionMatrix(NUM_CATEGORIES, NUM_CATEGORIES, CV_32FC1,
      Scalar(0.0));
  float error = 0;
  for(int i = 0; i < categoryData.rows; i++)
  {
    confusionMatrix.at<float>(cvRound(categoryData.at<float>(i,0)),
        cvRound(responses.at<float>(i,0))) += 1.0;
    if(cvRound(categoryData.at<float>(i,0)) !=
        cvRound(responses.at<float>(i,0)))
    {
      error += 1.0;
    }
  }
  error /= categoryData.rows;
  cout << "Testing error is " << error*100 << "\%" << endl;

  cout << "Confusion matrix is" << endl;
  for(int i = 0; i < NUM_CATEGORIES; i++)
  {
    if(i == ANGER)
      cout << "& Anger ";
    else if(i == DISGUST)
      cout << "& Disgust ";
    else if(i == FEAR)
      cout << "& Fear "; 
    else if(i == HAPPINESS)
      cout << "& Happiness ";
    else if(i == NEUTRAL)
      cout << "& Neutral ";
    else if(i == SADNESS)
      cout << "& Sadness ";
    else if(i == SURPRISE)
      cout << "& Surprise ";
  }
  cout << "\\\\" << endl;
  for(int i = 0; i < NUM_CATEGORIES; i++)
  {
    if(i == ANGER)
      cout << "Anger ";
    else if(i == DISGUST)
      cout << "Disgust ";
    else if(i == FEAR)
      cout << "Fear "; 
    else if(i == HAPPINESS)
      cout << "Happiness ";
    else if(i == NEUTRAL)
      cout << "Neutral ";
    else if(i == SADNESS)
      cout << "Sadness ";
    else if(i == SURPRISE)
      cout << "Surprise ";
    for(int j = 0; j < NUM_CATEGORIES; j++)
    {
        cout << "& "<< cvRound(confusionMatrix.at<float>(i,j)) << " ";
    }
    cout << "\\\\" << endl;
  }

  DisposeKernels();
  delete model;

  return EXIT_SUCCESS;
}
