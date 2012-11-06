#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <cstdlib>
#include <string>
#include <iostream>
using namespace std;
#include <getopt.h>

void usage(const string programName, int exitCode)
{
  cout << "Usage\n"
       << programName << " --image=<image_path>" << endl;
  std::exit(exitCode);
}

int main(int argc, char** argv)
{
  const string programName = argv[0];
  const char* shortOptions = "i:h";
  const option longOptions[] =
  {
    {"image", required_argument, NULL, 'i'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
  };

  string imagePath;

  int opt;
  while((opt=getopt_long(argc, argv, shortOptions, longOptions, NULL)) 
          != -1)
  {
    switch(opt)
    {
      case 'h': usage(programName, EXIT_SUCCESS);
                break;
      case 'i': imagePath = optarg;
                break;
      case '?':
      default : usage(programName, EXIT_FAILURE);
                break;
    }
  }
  cout << "Received arguments" << endl
       << "\tProgram Name: " << programName << endl
       << "\tInput Image : " << imagePath << endl;

  assert(!imagePath.empty());

  Mat img = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
  Moments m = moments(img);
  cout << m.m00 << '\t' << m.m10 << '\t' << m.m01 << '\t' << m.m20 << '\t' << m.m11 << '\t' << m.m02 << '\t' << m.m30 << '\t' << m.m21 << '\t' << m.m12 << '\t' << m.m03 << '\t' << m.mu20 << '\t' << m.mu11 << '\t' << m.mu02 << '\t' << m.mu30 << '\t' << m.mu21 << '\t' << m.mu12 << '\t' << m.mu03 << '\t' << m.nu20 << '\t' << m.nu11 << '\t' << m.nu02 << '\t' << m.nu30 << '\t' << m.nu21 << '\t' << m.nu12 << '\t' << m.nu03 << endl;
  namedWindow(imagePath);
  imshow(imagePath, img);
  waitKey();
  destroyAllWindows();

  return EXIT_SUCCESS;
}
