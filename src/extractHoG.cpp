#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
using namespace std;
#include <getopt.h>
#include <dirent.h>
#include "phog.h"

#define NUM_IMAGES 281

void usage(const string programName, int exitCode)
{
  cout << "Usage\n"
       << programName << " [options]" << endl
       << "Options include:" << endl
       << "\t--directory=<directory_path>" << endl
       << "\t--output=<output_path>" << endl;
  exit(exitCode);
}

int main(int argc, char** argv)
{
  const string programName = argv[0];
  const char* shortOptions = "d:o:h";
  const option longOptions[] =
  {
    {"directory", required_argument, NULL, 'd'},
    {"output", required_argument, NULL, 'o'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
  };

  string directoryPath, outputPath;

  int opt;
  while((opt=getopt_long(argc, argv, shortOptions, longOptions, NULL)) 
          != -1)
  {
    switch(opt)
    {
      case 'h': usage(programName, EXIT_SUCCESS);
                break;
      case 'd': directoryPath = optarg;
                break;
      case 'o': outputPath = optarg;
                break;
      case '?':
      default : usage(programName, EXIT_FAILURE);
                break;
    }
  }
  assert(!directoryPath.empty());
  assert(!outputPath.empty());

  if(directoryPath[directoryPath.length()-1] != '/')
    directoryPath += "/";

  cout << "Received arguments" << endl
       << "\tProgram Name  : " << programName << endl
       << "\tDirectory Path: " << directoryPath << endl
       << "\tOutput Path   : " << outputPath << endl;

  DIR* dir;
  DIR* subdir;
  dirent* ent;
  dirent* subent;

  dir = opendir(directoryPath.c_str());
  assert(dir);

  fstream file;
  file.open(outputPath.c_str(), ios::out);
  cout << file.is_open() << endl;
  assert(file.is_open());

  string imagePath;
  float **trainData = new float *[NUM_IMAGES];
  float categoryData[NUM_IMAGES];
  float hogImages[NUM_IMAGES][128];
  int i=0;
  float j=0.0;
  unsigned char isFile =0x8;
  vector<int> myvector;
  vector<int>::iterator it;
  vector<float> PHOG;
  
  /*Another Test */
  HOGDescriptor d;
  vector<Point> locations;
  
  for(int k=0;k<NUM_IMAGES;k++)
    myvector.push_back(k);
  random_shuffle(myvector.begin(),myvector.end());
 
  it=myvector.begin();
  ++it;
  while((ent=readdir(dir)))
  {
    if(strcmp(ent->d_name, ".")==0 || strcmp(ent->d_name, "..")==0)
      continue;
    if(ent->d_type != isFile) {
      string fullPath = directoryPath + ent->d_name;
      if(fullPath[fullPath.length()-1] != '/')
        fullPath += "/";
      subdir = opendir(fullPath.c_str());
      while((subent=readdir(subdir)))
      {
        if(strcmp(subent->d_name, ".")==0 || strcmp(subent->d_name, "..")==0)
          continue;
        imagePath = fullPath + subent->d_name;
        cout << "Attempting to open " << imagePath << endl;
        Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.data)
        {
          i=*it;
          categoryData[i] = j;
          PHoG(img, PHOG);
          //d.compute(img,PHOG,Size(0,0),Size(0,0),locations);
          //cout << "size = " << PHOG.size() << endl;
          trainData[i] = new float[PHOG.size()];
          for (int k = 0; k < PHOG.size(); k++) {
            trainData[i][k] = PHOG[k];
            //cout << PHOG[k] << "\t";
          }
          //exit(1);
          it++;
          //exit(1);
          //namedWindow(imagePath);
          //imshow(imagePath, img);
          //waitKey();
          //destroyAllWindows();
        }
        img.release();
      }
      j+=1.0;
      closedir(subdir);
    }
    else
    {
      cout << "Could not read image" << endl;
    }
  }
  int skipval = 50;
  int startimg = (skipval==NUM_IMAGES)?0:skipval;
  Mat trainDataM = Mat(NUM_IMAGES,PHOG.size(),CV_32F,trainData);
  Mat categoryDataM = Mat(NUM_IMAGES,1,CV_32F, categoryData);
  Mat trDM = trainDataM(Range(0,skipval),Range::all());
  Mat teDM = trainDataM(Range(skipval,NUM_IMAGES),Range::all());
  Mat catrDM = categoryDataM(Range(0,skipval),Range::all());
  Mat cateDM = categoryDataM(Range(skipval,NUM_IMAGES),Range::all());
  
  cout << trainDataM.rows << "x" << trainDataM.cols << endl;
  /* SVM */
  SVMParams svmParams;
  //svmParams.kernel_type = CvSVM::LINEAR;
  CvSVM svm;
  svm.train(trDM, catrDM, Mat(), Mat(), svmParams);
  int svmErrCount = 0;
  int testimages = 0;
  for (i = startimg; i < NUM_IMAGES; i++) {
    testimages++;
    int val = svm.predict(trainDataM.row(i));
    
    if(val != categoryData[i]) {
      cout << i <<":\t" << val << "\t" << categoryData[i] << endl;
      svmErrCount++;
    }
  }
  cout << "SVMError = " << svmErrCount*100.0/(testimages) << "%" << endl;
  
  /*Random Trees*/
  CvRTParams rtParams;
  //rtParams.kernel_type = CvSVM::LINEAR;
  CvRTrees rt;
  rt.train(trDM, CV_ROW_SAMPLE, catrDM, Mat(), Mat(), Mat(), Mat(), rtParams);
  int rtErrCount = 0;
  testimages = 0;
  for (i = startimg; i < NUM_IMAGES; i++) {
    testimages++;
    int val = rt.predict(trainDataM.row(i));
    
    if(val != categoryData[i]) {
      cout << i <<":\t" << val << "\t" << categoryData[i] << endl;
      rtErrCount++;
    }
  }
  cout << "RTError = " << rtErrCount*100.0/(testimages) << "%" << endl;
 
  /*ANN*/
  CvANN_MLP_TrainParams annParams;
  Mat op = Mat::zeros(skipval, 7, CV_32F);
  for (int k = 0; k < skipval; k++) {
    op.at<float>(k,static_cast<int>(categoryData[k]))=1.0;
  }
  int layersizeArray[3] = {PHOG.size(), 128, 7};
  Mat layersize = Mat(1,3,CV_32S,layersizeArray);
  CvANN_MLP ann(layersize,CvANN_MLP::SIGMOID_SYM,1,1);
  Mat annWt = Mat(skipval,1,CV_32F,1.0);
  ann.train(trDM, op, annWt);
  int annErrCount = 0;
  testimages = 0;
  Mat annVal = Mat(NUM_IMAGES,7,CV_32F);
  ann.predict(trainDataM,annVal);
  int prediction = 0;
  for (i = startimg; i < NUM_IMAGES; i++) {
    testimages++;
    float max = (float)-HUGE_VAL;
    for (int k = 0; k < 7; k++) {
      if(annVal.at<float>(i,k) > max) {
        max = annVal.at<float>(i,k);
        prediction = k;
      }
    }
    if(prediction != static_cast<int>(categoryData[i])) {
      cout << i <<":\t" << prediction << "\t" << categoryData[i] << endl;      
      annErrCount++;
   }
  }
  cout << "Error = " << annErrCount*100.0/(testimages) << "%" << endl;
  
  file.close();
  closedir(dir);

  return EXIT_SUCCESS;
}
