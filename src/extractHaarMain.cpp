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

#include <FeatureExtractors/extractHaar.h>
#include <LearningAlgorithms/svm.h>
#include <LearningAlgorithms/rt.h>
#include <LearningAlgorithms/ann.h>
#include <Infrastructure/exceptions.h>

void usage(const string programName, int exitCode)
{
  cout << "Usage\n"
       << programName << " [options]" << endl
       << "Options include:" << endl
       << "\t--directory=<directory_path>" << endl
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
  for(int k=0;k<data_backup.rows;k++)
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
  unsigned char isFile = 0x8;

  dir = opendir(directoryPath.c_str());
  assert(dir);

  fstream file;
  file.open(outputPath.c_str(), ios::out);
  cout << file.is_open() << endl;
  assert(file.is_open());

  string imagePath;
  Mat imageFeatureData;
  Mat categoryData;
  
  int numImages = 0;
  int numCategories = 0;


  /* Linux Version for directory parsing */
  while((ent=readdir(dir)))
  {
    if(strcmp(ent->d_name, ".")==0 || strcmp(ent->d_name, "..")==0)
      continue;
    if(ent->d_type != isFile) 
    {
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
        IplImage* img = cvLoadImage(imagePath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        Mat m;
        if(img!=NULL)
        {
          extractHaarFeatures(img, m);
          imageFeatureData.push_back(m);
          categoryData.push_back(static_cast<float>(numCategories));
        }
        cvReleaseImage(&img);
      }
      numCategories++;
      closedir(subdir);
    }
    else
    {
      cout << "Could not read image" << endl;
    }
  }
  
  shuffle(imageFeatureData,categoryData);
  
  float percentageTestData = 10;
  
  Mat trainData = imageFeatureData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
  Mat categoryTrainData = categoryData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
  Mat testData = imageFeatureData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());
  Mat categoryTestData = categoryData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());
  
  float svmTestErr;
  svmtrain(trainData,categoryTrainData);
  svmTestErr = svmtest(testData,categoryTestData);
  cout << "SVM Test Error " << svmTestErr << "\%" << endl;
  
  float rtTestErr;
  rttrain(trainData,categoryTrainData);
  rtTestErr = rttest(testData,categoryTestData);
  cout << "RT Test Error " << rtTestErr << "\%" << endl;
  
  float annTestErr;
  annSetup(trainData, categoryTrainData, numCategories);
  anntrain(trainData,categoryTrainData);
  annTestErr = anntest(testData, categoryData, numCategories);
  cout << "ANN Test Error " << annTestErr << "\%" << endl;
  
  file.close();
  closedir(dir);

  return EXIT_SUCCESS;
}
