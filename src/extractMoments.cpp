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
    if(img.data)
    {
      Moments m = moments(img);
      file << m.m00 << '\t' << m.m10 << '\t' << m.m01 << '\t' << m.m20 << '\t' << m.m11 << '\t' << m.m02 << '\t' << m.m30 << '\t' << m.m21 << '\t' << m.m12 << '\t' << m.m03 << '\t' << m.mu20 << '\t' << m.mu11 << '\t' << m.mu02 << '\t' << m.mu30 << '\t' << m.mu21 << '\t' << m.mu12 << '\t' << m.mu03 << '\t' << m.nu20 << '\t' << m.nu11 << '\t' << m.nu02 << '\t' << m.nu30 << '\t' << m.nu21 << '\t' << m.nu12 << '\t' << m.nu03 << endl;
      namedWindow(imagePath);
      imshow(imagePath, img);
      waitKey();
      destroyAllWindows();
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
