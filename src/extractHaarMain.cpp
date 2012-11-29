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
#include <cstring>
using namespace std;
#include <getopt.h>
#include <dirent.h>

#include <FeatureExtractors/extractFeatures.h>
#include <LearningAlgorithms/learningAlgorithms.h>
#include <Infrastructure/exceptions.h>

void usage(const string programName, int exitCode)
{
    cout << "Usage\n"
         << programName << " [options]" << endl
         << "Options include:" << endl
         << "\t--inputFile=<inputFile_path>" << endl
         << "\t--output=<output_path>" << endl;
    exit(exitCode);
}

void shuffle(Mat &data, Mat &catg)
{
    Mat data_backup = data.clone();
    Mat catg_backup = catg.clone();

    data.pop_back(data_backup.rows);
    catg.pop_back(catg_backup.rows);

    vector<int> myvector;
    vector<int>::iterator it;
    for(int k=0; k<data_backup.rows; k++)
    {
        myvector.push_back(k);
    }
    random_shuffle(myvector.begin(),myvector.end());

    for (it = myvector.begin(); it!=myvector.end(); ++it)
    {
        data.push_back(data_backup.row(*it));
        catg.push_back(catg_backup.row(*it));
    }

    data_backup.release();
    catg_backup.release();
}

int main(int argc, char** argv)
{
    const string programName = argv[0];
    const char* shortOptions = "i:o:h";
    const option longOptions[] =
    {
        {"inputFile", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    string inputFilePath, outputPath;

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
        case 'o':
            outputPath = optarg;
            break;
        case '?':
        default :
            usage(programName, EXIT_FAILURE);
            break;
        }
    }
    assert(!inputFilePath.empty());

    cout << "Received arguments" << endl
         << "\tProgram Name  : " << programName << endl
         << "\tInput File Path: " << inputFilePath << endl
         << "\tOutput Path   : " << outputPath << endl;

    string imagePath;
    Mat imageFeatureData;
    Mat categoryData;

    int numCategories = 0;

    freopen (inputFilePath.c_str() ,"r", stdin);

    string filename;
    int label;
    while(cin >> label)
    {
      cin >> filename;
      IplImage* img= cvLoadImage(filename.c_str(),0);
      Mat m;
      if(img!=NULL)
      {
        extractFeatures(img, m, HAAR);
        imageFeatureData.push_back(m);
        categoryData.push_back(static_cast<float>(label));
      }
      numCategories = label;
      cvReleaseImage(&img);
    }
    numCategories += 1;

    shuffle(imageFeatureData,categoryData);

    float percentageTestData = 50;

    Mat trainData = imageFeatureData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
    Mat categoryTrainData = categoryData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
    Mat testData = imageFeatureData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());
    Mat categoryTestData = categoryData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());

  Mat responses;
  CvStatModel* model = learningAlgorithmSetup(imageFeatureData.cols,
      numCategories, SVM_ML);
  float svmTestErr;
  learningAlgorithmTrain(model,trainData, categoryTrainData, numCategories,
      SVM_ML);
  learningAlgorithmPredict(model, testData, responses, numCategories, SVM_ML);
  svmTestErr = learningAlgorithmComputeErrorRate(responses,
      categoryTestData);
  cout << "SVM Test Error " << svmTestErr*100 << "\%" << endl;
  delete model;
  
  float rtTestErr;
  model = learningAlgorithmSetup(imageFeatureData.cols, numCategories, RT);
  learningAlgorithmTrain(model,trainData, categoryTrainData, numCategories,
      RT);
  learningAlgorithmPredict(model, testData, responses, numCategories, RT);
  rtTestErr = learningAlgorithmComputeErrorRate(responses,
      categoryTestData);
  cout << "RT Test Error " << rtTestErr*100 << "\%" << endl;
  delete model;
  
  float annTestErr;
  model = learningAlgorithmSetup(imageFeatureData.cols, numCategories, ANN);
  learningAlgorithmTrain(model,trainData, categoryTrainData, numCategories,
      ANN);
  learningAlgorithmPredict(model, testData, responses, numCategories, ANN);
  annTestErr = learningAlgorithmComputeErrorRate(responses,
      categoryTestData);
  cout << "ANN Test Error " << annTestErr*100 << "\%" << endl;
  delete model;
  
  return EXIT_SUCCESS;
}
