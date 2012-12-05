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
         << "\t--inputFile=<inputFile path>" << endl
         << "\t--featureExtractor=[gabor haar moments phog]" << endl
         << "\t--saveClassifier=<file path>" << endl
         << "\t--cascadeClassifier=<cascade path>" << endl
         << "\t--verbose" << endl
				 << "\t--perSubjectCV" << endl
         << "\t--help" << endl;
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
  int verbose = 0;
	int perSubject = 0;
  float percentageTestData = 10;
  const char* shortOptions = "i:f:s:c:h";
  const option longOptions[] =
  {
    {"inputFile", required_argument, NULL, 'i'},
    {"featureExtractor", required_argument, NULL, 'f'},
    {"saveClassifier", required_argument, NULL, 's'},
    {"cascadeClassifier", required_argument, NULL, 'c'},
    {"verbose", no_argument, &verbose, 1},
		{"perSubjectCV", no_argument, &perSubject, 1},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
  };

  string inputFilePath;
  featureExtractor fEx;
  string saveClassifierLocation;
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
			case 's':
				saveClassifierLocation = optarg;
				break;
			case 'c':
				cascadeClassifierName = optarg;
				break;
			case 0: break;
			case '?':
			default :
							cout << "Invalid arguments received!" << endl;
							usage(programName, EXIT_FAILURE);
							break;
		}
	}
	assert(!inputFilePath.empty());
	assert(percentageTestData > 0);
	assert(!cascadeClassifierName.empty());

	string fExtractor = (fEx==GABOR)?"Gabor":
		(fEx==HAAR)?"Haar":
		(fEx==MOMENTS)?"Moments":
		(fEx==PHOG)?"PHoG":"Error!";

	cout << "Received arguments" << endl
		<< "\tProgram Name      : " << programName << endl
		<< "\tInput File Path   : " << inputFilePath << endl
		<< "\tFeature Extractor : " << fExtractor << endl
		<< "\tSave Classifier at: " << saveClassifierLocation << endl
		<< "\tCascade Classifier: " << cascadeClassifierName << endl
	  << "\tPer Subject CV    : " << (perSubject?"yes":"no") << endl
		<< "\tVerbose           : " << (verbose?"yes":"no") << endl;

	string imagePath;
	Mat imageFeatureData;
	Mat categoryData;
	vector<Mat> imageFeaturesBySubject;
	vector<Mat> labelsBySubject;

	CascadeClassifier haarCascade(cascadeClassifierName);
	assert(!haarCascade.empty());
	cout << "Classifier loaded from " << cascadeClassifierName << endl;

	vector<Rect> detected;

	int numCategories = 0;
	int numImages = 0;

	freopen (inputFilePath.c_str() ,"r", stdin);

	string filename;
	int label;
	int subId;
	int pSubId = -1;
	int cSubject = -1;
	while(cin >> label)
	{
		cin >> subId;
		if(subId != pSubId)
		{
			pSubId = subId;
			cSubject++;
			imageFeaturesBySubject.push_back(Mat());
			labelsBySubject.push_back(Mat());
		}
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
			extractFeatures(cropped, m, fEx);
			//        imageFeatureData.push_back(m);
			imageFeaturesBySubject[cSubject].push_back(m);
			//        categoryData.push_back(static_cast<float>(label));
			labelsBySubject[cSubject].push_back(static_cast<float>(label));
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
	int numSubjects = cSubject+1;
	DisposeKernels(); //TODO: Repair this cheap hack
	cout << "Finished reading images and extracting features" << endl;

	CvRTParams bestRTParams;
	float bestValidationError = HUGE_VAL;
	
	if(perSubject)
	{
		cout << "Now, cross validating across subjects" << endl;

		for(int depth = 3; depth <=30; depth += 4)
		{
			CvRTParams rtParams;
			rtParams.max_depth = depth;
			float validationError = 0;
			for(int i = 0; i < numSubjects; i++)
			{
				Mat validationData = imageFeaturesBySubject[i];
				Mat validationResponses = labelsBySubject[i];
				Mat trainData;
				Mat trainResponses;
				for(int j = 0; j < numSubjects; j++)
				{
					if(j == i)
						continue;
					trainData.push_back(imageFeaturesBySubject[j]);
					trainResponses.push_back(labelsBySubject[j]);
				}
				CvRTrees rt;
				rt.train(trainData, CV_ROW_SAMPLE, trainResponses, Mat(), Mat(),
						Mat(), Mat(), rtParams);
				Mat responses;
				learningAlgorithmPredict(&rt, validationData, responses, 
						numCategories, RT);
				validationError += learningAlgorithmComputeErrorRate(responses,
						validationResponses) * validationData.rows / numImages;
			}
			cout << "Validation error is " << validationError*100
				<< "\% for max_depth=" << rtParams.max_depth << endl;
			if(validationError < bestValidationError)
			{
				bestValidationError = validationError;
				bestRTParams = rtParams;
			}
		}
	}
	else
	{
		for(int i = 0; i < numSubjects; i++)
		{
			imageFeatureData.push_back(imageFeaturesBySubject[i]);
			categoryData.push_back(labelsBySubject[i]);
		}
		shuffle(imageFeatureData,categoryData);
    cout << "Shuffled feature data. Now cross validating" << endl;

    int splitSize = imageFeatureData.rows/10;
		for(int depth = 3; depth <=30; depth += 4)
		{
			CvRTParams rtParams;
			rtParams.max_depth = depth;
			float validationError = 0;
      for(int i = 0; i < 10; i++)
      {
        Mat validationData = imageFeatureData.rowRange(i*splitSize,
            (i+1)*splitSize);
        Mat validationResponses = categoryData.rowRange(i*splitSize,
            (i+1)*splitSize);
        Mat trainData;
        Mat trainResponses;
        if(i)
        {
          trainData = imageFeatureData.rowRange(0,i*splitSize);
          trainResponses = categoryData.rowRange(0,i*splitSize);
          trainData.push_back(imageFeatureData.rowRange((i+1)*splitSize,
                imageFeatureData.rows));
          trainResponses.push_back(categoryData.rowRange((i+1)*splitSize,
                categoryData.rows));
        }
        else
        {
          trainData = imageFeatureData.rowRange((i+1)*splitSize,
              imageFeatureData.rows);
          trainResponses = categoryData.rowRange((i+1)*splitSize,
              categoryData.rows);
        }
				CvRTrees rt;
				rt.train(trainData, CV_ROW_SAMPLE, trainResponses, Mat(), Mat(),
						Mat(), Mat(), rtParams);
				Mat responses;
				learningAlgorithmPredict(&rt, validationData, responses, 
						numCategories, RT);
				validationError += learningAlgorithmComputeErrorRate(responses,
						validationResponses) * validationData.rows / numImages;
			}
			cout << "Validation error is " << validationError*100
				<< "\% for max_depth=" << rtParams.max_depth << endl;
			if(validationError < bestValidationError)
			{
				bestValidationError = validationError;
				bestRTParams = rtParams;
			}
		}
	}

	cout << "Cross validation completed" << endl;
	cout << "Best max_depth=" << bestRTParams.max_depth 
	     << " gives validation error "
		   << bestValidationError*100 << "\%" << endl;

	if(!saveClassifierLocation.empty())
	{
		Mat trainData;
		Mat trainResponses;
		for(int i = 0; i < numSubjects; i++)
		{
			trainData.push_back(imageFeaturesBySubject[i]);
			trainResponses.push_back(labelsBySubject[i]);
		}

		CvRTrees rt;
		
		rt.train(trainData, CV_ROW_SAMPLE, trainResponses, Mat(), Mat(), 
				Mat(), Mat(), bestRTParams);

		cout << "Saving classifier to " << saveClassifierLocation << endl;
		rt.save(saveClassifierLocation.c_str());
		cout << "Classifer saved" << endl;
	}

  return EXIT_SUCCESS;
}
