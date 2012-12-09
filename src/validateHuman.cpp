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
#include <ctime>
using namespace std;
#include <getopt.h>
#include <dirent.h>

#include <Infrastructure/exceptions.h>
#include <Infrastructure/defines.h>

#define NUM_IMAGES 50

struct EmoStruct
{
	Mat face;
	int emotion;
	EmoStruct(Mat& m, int e): face(m), emotion(e)
	{}
};
void usage(const string programName, int exitCode)
{
    cout << "Usage\n"
         << programName << " [options]" << endl
         << "Options include:" << endl
         << "\t--inputFile=<inputFile path>" << endl
         << "\t--cascadeClassifier=<cascade path>" << endl
				 << "\t--outputPath=<output path>" << endl
         << "\t--verbose" << endl
         << "\t--help" << endl;
    exit(exitCode);
}

int main(int argc, char** argv)
{
	const string programName = argv[0];
	int verbose = 0;
	float percentageTestData = 10;
	const char* shortOptions = "i:c:o:h";
	const option longOptions[] =
	{
		{"inputFile", required_argument, NULL, 'i'},
		{"cascadeClassifier", required_argument, NULL, 'c'},
		{"outputPath", required_argument, NULL, 'o'},
		{"verbose", no_argument, &verbose, 1},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}
	};

	string inputFilePath;
	string outputFilePath;
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
			case 'c':
				cascadeClassifierName = optarg;
				break;
			case 'o':
				outputFilePath = optarg;
			case 0: break;
			case '?':
			default :
							cout << "Invalid arguments received!" << endl;
							usage(programName, EXIT_FAILURE);
							break;
		}
	}
	assert(!inputFilePath.empty());
	assert(!cascadeClassifierName.empty());

	cout << "Received arguments" << endl
		<< "\tProgram Name      : " << programName << endl
		<< "\tInput File Path   : " << inputFilePath << endl
		<< "\tCascade Classifier: " << cascadeClassifierName << endl
		<< "\tOutput File Path  : " << outputFilePath << endl;

	string imagePath;

	CascadeClassifier haarCascade(cascadeClassifierName);
	assert(!haarCascade.empty());
	cout << "Classifier loaded from " << cascadeClassifierName << endl;

	vector<Rect> detected;
	vector<EmoStruct> images;

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
			equalizeHist(cropped, cropped);
			images.push_back(EmoStruct(cropped,label));
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

	srand(time(NULL));
	random_shuffle(images.begin(), images.end());

  Mat confusionMatrix(NUM_CATEGORIES, NUM_CATEGORIES, CV_32FC1,
      Scalar(0.0));
  float error = 0;
	namedWindow("Image", CV_WINDOW_AUTOSIZE);
  for(int i = 0; i < NUM_IMAGES && i < images.size(); i++)
  {
		imshow("Image", images[i].face);
		cout << "Please classify face" << endl
         << "Anger: " << ANGER << endl
				 << "Disgust: " << DISGUST << endl
				 << "Fear: " << FEAR << endl
				 << "Happiness: " << HAPPINESS << endl
				 << "Neutral: " << NEUTRAL << endl
				 << "Sadness: " << SADNESS << endl
				 << "Surprise: " << SURPRISE << endl;

		int response;
		response = waitKey() & 0xFF;
		response -= static_cast<int>('0');

		cout << "You recognised this image as ";
		if(response == ANGER)
			cout << "Anger ";
		else if(response == DISGUST)
			cout << "Disgust ";
		else if(response == FEAR)
			cout << "Fear "; 
		else if(response == HAPPINESS)
			cout << "Happiness ";
		else if(response == NEUTRAL)
			cout << "Neutral ";
		else if(response == SADNESS)
			cout << "Sadness ";
		else if(response == SURPRISE)
			cout << "Surprise ";
		else
		{
			i--;
			cout << "Sorry, could not get that image" << endl;
			continue;
		}

		cout << endl;
		cout << "Is this correct? " << flush;
		int y = waitKey() & 0xFF;
		if(y == 'y' || y == 'Y')
			cout << 'y' << endl;
		else
		{
			cout << 'n' << endl;
			cout << "Retrying" << endl;
			i--;
			continue;
		}

		confusionMatrix.at<float>(images[i].emotion,response) += 1.0;
	}

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

	if(!outputFilePath.empty())
	{
		cout << "Opening file " << outputFilePath << endl;
		fstream file(outputFilePath.c_str(), ios::out);
		if(file.fail())
		{
			cerr << "Error opening file " << outputFilePath << endl;
			return EXIT_FAILURE;
		}
		for(int i = 0; i < NUM_CATEGORIES; i++)
		{
			if(i == ANGER)
				file << ", Anger ";
			else if(i == DISGUST)
				file << ", Disgust ";
			else if(i == FEAR)
				file << ", Fear "; 
			else if(i == HAPPINESS)
				file << ", Happiness ";
			else if(i == NEUTRAL)
				file << ", Neutral ";
			else if(i == SADNESS)
				file << ", Sadness ";
			else if(i == SURPRISE)
				file << ", Surprise ";
		}
		file << endl;
		for(int i = 0; i < NUM_CATEGORIES; i++)
		{
			if(i == ANGER)
				file << "Anger ";
			else if(i == DISGUST)
				file << "Disgust ";
			else if(i == FEAR)
				file << "Fear "; 
			else if(i == HAPPINESS)
				file << "Happiness ";
			else if(i == NEUTRAL)
				file << "Neutral ";
			else if(i == SADNESS)
				file << "Sadness ";
			else if(i == SURPRISE)
				file << "Surprise ";
			for(int j = 0; j < NUM_CATEGORIES; j++)
			{
				file << ", "<< cvRound(confusionMatrix.at<float>(i,j)) << " ";
			}
			file << endl;
		}
		file.close();
		cout << "Confusion matrix written to " << outputFilePath << endl;
	}
  return EXIT_SUCCESS;
}
