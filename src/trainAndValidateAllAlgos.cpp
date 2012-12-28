//    File name: trainAndValidateAllAlgos.cpp
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
#include <fstream>
#include <vector>
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
         << "\t--learningAlgorithm=[ann rt svm]" << endl
         << "\t--saveClassifier=<file path>" << endl
         << "\t--percentageValidationData=<percentage of validation data>"
         << endl
         << "\t--cascadeClassifier=<cascade path>" << endl
         << "\t--verbose" << endl
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
    float percentageTestData = 10;
    const char* shortOptions = "i:f:l:s:p:c:h";
    const option longOptions[] =
    {
        {"inputFile", required_argument, NULL, 'i'},
        {"featureExtractor", required_argument, NULL, 'f'},
        {"learningAlgorithm", required_argument, NULL, 'l'},
        {"saveClassifier", required_argument, NULL, 's'},
        {"percentageValidationData", required_argument, NULL, 'p'},
        {"cascadeClassifier", required_argument, NULL, 'c'},
        {"verbose", no_argument, &verbose, 1},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    string inputFilePath;
    featureExtractor fEx;
    vector<learningAlgorithm> lA;
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
        case 'l':
            if(!strcmp(optarg,"ann"))
                lA.push_back(ANN);
            else if(!strcmp(optarg,"rt"))
                lA.push_back(RT);
            else if(!strcmp(optarg,"svm"))
                lA.push_back(SVM_ML);
            break;
        case 's':
            saveClassifierLocation = optarg;
            break;
        case 'p':
            percentageTestData = atof(optarg);
            break;
        case 'c':
            cascadeClassifierName = optarg;
            break;
        case 0:
            break;
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
    vector<string> lAlgorithm;
    for(int i = 0; i< lA.size(); i++)
    {
        lAlgorithm.push_back((lA[i]==ANN)?"ANN ":
                             (lA[i]==RT)?"RT ":
                             (lA[i]==SVM_ML)?"SVM ":"Error!");
    }

    cout << "Received arguments" << endl
         << "\tProgram Name      : " << programName << endl
         << "\tInput File Path   : " << inputFilePath << endl
         << "\tFeature Extractor : " << fExtractor << endl
         << "\tLearning Algorithm/s: " ;
    for(int i=0; i<lAlgorithm.size(); i++)
    {
        cout << lAlgorithm[i] << " ";
    }
    cout << endl << "\tSave Classifier at: " << saveClassifierLocation << endl
         << "\tPercentageTestData: " << percentageTestData << endl
         << "\tCascade Classifier: " << cascadeClassifierName << endl;

    string imagePath;
    Mat imageFeatureData;
    Mat categoryData;

    CascadeClassifier haarCascade(cascadeClassifierName);
    assert(!haarCascade.empty());
    cout << "Classifier loaded from " << cascadeClassifierName << endl;

    vector<Rect> detected;

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
            extractFeatures(cropped, m, fEx);
            imageFeatureData.push_back(m);
            categoryData.push_back(static_cast<float>(label));
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
    DisposeKernels(); 

    shuffle(imageFeatureData,categoryData);

    for(int i = 0; i < lA.size(); i++)
    {
        Mat trainData = imageFeatureData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
        Mat categoryTrainData = categoryData(Range(0,static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100),Range::all());
        Mat testData = imageFeatureData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());
        Mat categoryTestData = categoryData(Range(static_cast<int>((100-percentageTestData)*imageFeatureData.rows)/100,imageFeatureData.rows),Range::all());

        Mat responses;
        CvStatModel* model = learningAlgorithmSetup(imageFeatureData.cols,
                             numCategories, lA[i]);
        float testErr;
        learningAlgorithmTrain(model,trainData, categoryTrainData, numCategories,
                               lA[i]);
        learningAlgorithmPredict(model, testData, responses, numCategories, lA[i]);
        testErr = learningAlgorithmComputeErrorRate(responses,
                  categoryTestData);
        cout << lAlgorithm[i] << " validation error " << testErr*100 << "\%"
             << endl;
        delete model;
    }
    return EXIT_SUCCESS;
}
