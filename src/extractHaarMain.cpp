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
         << "\t--inputFile=<inputFile_path>" << endl
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
    const char* shortOptions = "i:o:h";
    const option longOptions[] =
    {
        {"inputFile", required_argument, NULL, 'd'},
        {"output", required_argument, NULL, 'o'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    string inputFilePath, outputPath;

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
        case 'o':
            outputPath = optarg;
            break;
        case '?':
        default :
            usage(programName, EXIT_FAILURE);
            break;
        }
    }
    assert(!inputFilePath.empty());

    /*
        assert(!outputPath.empty());
    */

    cout << "Received arguments" << endl
         << "\tProgram Name  : " << programName << endl
         << "\tInput File Path: " << inputFilePath << endl
         << "\tOutput Path   : " << outputPath << endl;

    /*
        fstream file;
        file.open(outputPath.c_str(), ios::out);
        assert(file.is_open());
    */

    string imagePath;
    Mat imageFeatureData;
    Mat categoryData;

    int numCategories = 0;

    freopen (inputFilePath.c_str() ,"r", stdin);

    string filename;
    int label;
    while(cin >> label)
    {
        cin >> filename;
        IplImage* img= cvLoadImage(filename.c_str(),0);
        Mat m;
        if(img!=NULL)
        {
            extractHaarFeatures(img, m);
            imageFeatureData.push_back(m);
            /*
                for(int i = 0; i < NUM_GABOR_FEATURES; i++)
                {
                    file << '\t' << m.at<float>(0,i);
                }
                file << endl;
            */
            categoryData.push_back(static_cast<float>(label));
        }
        numCategories = label;
        cvReleaseImage(&img);
    }
    numCategories += 1;

    shuffle(imageFeatureData,categoryData);

    float percentageTestData = 50;

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

    /*
        file.close();
    */

    return EXIT_SUCCESS;
}
