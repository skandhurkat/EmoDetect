//    File name: readImage.cpp
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

  Mat img = imread(imagePath);
  namedWindow(imagePath);
  imshow(imagePath, img);
  waitKey();
  destroyAllWindows();

  return EXIT_SUCCESS;
}
