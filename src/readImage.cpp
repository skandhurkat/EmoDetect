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
    {"help", no_argument, NULL, 'h'}
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

  Mat img = imread(imagePath);
  namedWindow(imagePath);
  imshow(imagePath, img);
  waitKey();
  destroyAllWindows();

  return EXIT_SUCCESS;
}
