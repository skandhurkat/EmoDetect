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

#include <FeatureExtractors/extractMoments.h>
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
  dirent* ent;

  dir = opendir(directoryPath.c_str());
  assert(dir);

  fstream file;
  file.open(outputPath.c_str(), ios::out);
  cout << file.is_open() << endl;
  assert(file.is_open());

  string imagePath;

  while((ent=readdir(dir)))
  {
    if(strcmp(ent->d_name, ".")==0 || strcmp(ent->d_name, "..")==0)
      continue;
    imagePath = directoryPath + ent->d_name;
    cout << "Attempting to open " << imagePath << endl;
    Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
    Mat m;
    if(img.data)
    {
      extractMoments(img, m);
      file << m.at<double>(0,0);
      for(int i = 0; i < NUM_MOMENTS; i++)
        file << '\t' << m.at<double>(0,i);
      file << endl;
    }
    else
    {
      cout << "Could not read image" << endl;
    }
  }

  file.close();
  closedir(dir);

  return EXIT_SUCCESS;
}
