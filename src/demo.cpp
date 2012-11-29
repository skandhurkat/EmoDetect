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
#include <fstream>
#include <vector>
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
      case 0:
        break;
      case '?':
      default :
        cout << "Invalid arguments received!" << endl;
        usage(programName, EXIT_FAILURE);
        break;
    }
  }
  if(cascadeClassifierName.empty())
  {
    cout << "ERROR! Need a Haar cascade classifier" << endl;
    usage(programName, EXIT_FAILURE);

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
    << "\tLoad Classifier   : " << loadClassifierLocation << endl
    << "\tCascade Classifier: " << cascadeClassifierName << endl;

  string imagePath;
  Mat imageFeatureData;
  Mat categoryData;

  CascadeClassifier haarCascade(cascadeClassifierName);
  if(haarCascade.empty())
  {
    cerr << "ERROR! Could not load classifier from "
         << cascadeClassifierName << endl;
    return EXIT_FAILURE;
  }
  cerr << "Classifier loaded from " << cascadeClassifierName << endl;

  Mat features;
  vector<Rect> detected;

  CvStatModel* model = readModel(loadClassifierLocation, lA);
  if(model)
  {
    cerr << "Model loaded from " << loadClassifierLocation << endl;
  }
  else
  {
    cerr << "ERROR! Could not load model from "
         << loadClassifierLocation << endl;
    return EXIT_FAILURE;
  }

  if(inputFilePath.empty())
  {
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
      cerr << "Something went wrong. I cannot open the capture device"
           << endl;
      return EXIT_FAILURE;
    }
    namedWindow("Video Input", CV_WINDOW_AUTOSIZE);
    
    while(true)
    {
      Mat img, gray;
      cap >> img;
      cvtColor(img, gray, CV_BGR2GRAY);
      imshow("Video Input", img);
      int c = waitKey(30);
      c &= 0xFF;
      if(c == KEY_ESCAPE)
      {
        destroyAllWindows();
        delete model;
        DisposeKernels();
        return EXIT_SUCCESS;
      }
      else if(c == KEY_SPACEBAR)
      {
        haarCascade.detectMultiScale(gray, detected, 1.1, 3,
            CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH);
        if(detected.size() != 1)
        {
          cerr << "Error! Need exactly one face in the picture" << endl;
          continue;
        }
        Mat cropped = gray(detected[0]);
        equalizeHist(cropped, cropped);
        extractFeatures(cropped, features, fEx);
        Mat m = features;
		
		namedWindow("Cropped Face", CV_WINDOW_AUTOSIZE);
        Mat resized;
        resize(cropped, resized, Size(), 480.0/cropped.cols,
            480.0/cropped.cols, INTER_LANCZOS4);
        imshow("Cropped Face", resized);
        // For the demo only
		if (fEx == GABOR) {
			Mat featurekernel;
            featurekernel = Mat::zeros(16*5, 16*8, CV_32FC1);
            int n = 128*128/64;
            float minVal = 0, maxVal = 0;
            for(int loop1 =0; loop1 < 5; loop1++) {
               for(int loop2 = 0; loop2 < 8; loop2++){
			           for (int lx=4; lx<128; lx+=8)
                           for (int ly=4; ly<128; ly+=8){
								minVal = std::min(minVal, m.at<float>(0, ((loop1*8+loop2)*n+lx*2+ly/8)));
				   }
               }
			}
			if (minVal < 0) minVal *= -1.0;
            for(int loop1 =0; loop1 < 5; loop1++) {
               for(int loop2 = 0; loop2 < 8; loop2++){
			           for (int lx=4; lx<128; lx+=8)
                           for (int ly=4; ly<128; ly+=8){
                           featurekernel.at<float>(loop1*8 + lx/8, loop2*8 + ly/8) = minVal + m.at<float>(0, ((loop1*8+loop2)*n+lx*2+ly/8));
                    }
                }
            }
           normalize(featurekernel, featurekernel, 0, 255, NORM_MINMAX, CV_32FC1);
           //displayOverlay("AllKernels", "Feature Kernels");
		   imshow("AllKernels", featurekernel);
           imwrite("gaborImage.jpg", featurekernel);
       featurekernel.release();
          // waitKey();
		} 
//    destroyWindow("AllKernels");
        Mat result;
        learningAlgorithmPredict(model, features, result,
            NUM_CATEGORIES, lA);
        string emotion = 
          (cvRound(result.at<float>(0,0))==ANGER)?"Anger":
          (cvRound(result.at<float>(0,0))==DISGUST)?"Disgust":
          (cvRound(result.at<float>(0,0))==FEAR)?"Fear":
          (cvRound(result.at<float>(0,0))==HAPPINESS)?"Happiness":
          (cvRound(result.at<float>(0,0))==NEUTRAL)?"Neutral":
          (cvRound(result.at<float>(0,0))==SADNESS)?"Sadness":
          (cvRound(result.at<float>(0,0))==SURPRISE)?"Surprise":
          "ERROR!";
        cout << "Detected emotion :" << emotion << endl;
        displayOverlay("Cropped Face", emotion, 0);
        waitKey(5000);
        destroyWindow("Cropped Face");
      }
    }
    destroyAllWindows();
  }
  else
  {
    freopen (inputFilePath.c_str() ,"r", stdin);

    string filename;
    int label;
    while(cin >> label)
    {
      cin >> filename;
      if(verbose)
        cout << "Attempting to open " << filename << endl;
      Mat gray= imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
      Mat m;
      if(gray.data)
      {
        haarCascade.detectMultiScale(gray, detected, 1.1, 3,
            CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH);
        if(detected.size() != 1)
        {
          cerr << "Error! Need exactly one face in the picture" << endl;
          continue;
        }
        Mat cropped = gray(detected[0]);
        equalizeHist(cropped, cropped);
        extractFeatures(cropped, features, fEx);
        namedWindow(filename, CV_WINDOW_AUTOSIZE);
        Mat resized;
        resize(cropped, resized, Size(), 480.0/cropped.cols,
            480.0/cropped.cols, INTER_LANCZOS4);
        imshow(filename, resized);
        Mat result;
        learningAlgorithmPredict(model, features, result,
            NUM_CATEGORIES, lA);
        string emotion = 
          (cvRound(result.at<float>(0,0))==ANGER)?"Anger":
          (cvRound(result.at<float>(0,0))==DISGUST)?"Disgust":
          (cvRound(result.at<float>(0,0))==FEAR)?"Fear":
          (cvRound(result.at<float>(0,0))==HAPPINESS)?"Happiness":
          (cvRound(result.at<float>(0,0))==NEUTRAL)?"Neutral":
          (cvRound(result.at<float>(0,0))==SADNESS)?"Sadness":
          (cvRound(result.at<float>(0,0))==SURPRISE)?"Surprise":
          "ERROR!";
        cout << "Detected emotion :" << emotion << endl;
        displayOverlay(filename, emotion, 0);
        int c;
        while(c = waitKey())
        {
          if(c == KEY_ESCAPE)
          {
            destroyAllWindows();
            delete model;
            DisposeKernels();
            return EXIT_SUCCESS;
          }
          else if(c == KEY_SPACEBAR)
            break;
        }
        destroyWindow(filename);
      }
      else if(verbose)
        cout << "\tFailed to open " << filename << endl;
    }
  }

  delete model;
  DisposeKernels();

  return EXIT_SUCCESS;
}
